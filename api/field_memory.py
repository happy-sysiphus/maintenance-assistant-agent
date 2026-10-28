"""현장 기억(field memory): 닫힌 케이스의 사람 기록을 라벨(오알람 / 실제 고장)과 집계로 묶는다.

FIELD_MEMORY=1일 때만 쓴다 (부를 때마다 환경 변수를 읽는다). 끄면 api는 지금과 똑같이 동작한다.
순서를 바꾸는 규칙은 R1 하나: 같은 설비 · 같은 규칙 코드에서 오알람이 실제 고장보다 많으면 S10을 맨 앞에 둔다.
나머지(지난 확정 원인 · 점검 결과)는 참고 정보로만 붙이고, 매뉴얼 근거(cite)로 내보내지 않는다.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import guidance

# 원인 → 고장 연결 (RSW 정의서 p.4-5). 매뉴얼 기반 추정이라 후보 순서에만 쓴다.
LINKS = Path(__file__).resolve().parent / "cause_links.json"
# 오알람 판정에 쓰는 상황 (설정값 변경 · 잔고장 반복)
ALARM_TRUTH = "S10"
# 후보 밖 원인을 직접 적었을 때의 결과 situation_id (main.MANUAL_CAUSE와 같음)
MANUAL_CAUSE = "MANUAL"

_links: dict[str, list[str]] | None = None


def enabled() -> bool:
    return os.environ.get("FIELD_MEMORY") == "1"


def links() -> dict[str, list[str]]:
    """고장 sid → 원인 sid 목록 (파일 순서). 처음 부를 때 읽어서 들고 있는다."""
    global _links
    if _links is None:
        out: dict[str, list[str]] = {}
        for e in json.loads(LINKS.read_text(encoding="utf-8"))["edges"]:
            out.setdefault(e["to"], []).append(e["from"])
        _links = out
    return _links


def code_of(event: dict[str, Any]) -> str | None:
    """규칙 코드가 없으면(모델 신호만) 분류의 종료 코드 (guidance.guidance_for와 같은 규칙)."""
    return (event.get("trigger") or {}).get("rule_code") or (event.get("fault_class") or {}).get("terminal_code")


def alarm_key(event: dict[str, Any]) -> str:
    """오알람 집계 단위: 설비 | 규칙 코드 (모델 신호만이면 코드 자리가 빈다)."""
    return f"{event['gun_id']}|{(event.get('trigger') or {}).get('rule_code') or ''}"


def decisive(case: dict[str, Any]) -> str | None:
    """결정 원인: 마지막 resolved 결과의 sid, 없으면 마지막 yes 판단의 sid."""
    r = case["records"]
    last = next((x for x in reversed(r["results"]) if x["outcome"] == "resolved"), None)
    if last:
        return last["situation_id"]
    return next((j["situation_id"] for j in reversed(r["judgments"]) if j["verdict"] == "yes"), None)


def alarm_label(case: dict[str, Any]) -> str | None:
    """false_alarm 닫기 또는 S10으로 해결 → "false_alarm", 그 밖의 해결 → "real". 미해결 · 인계 · 열림은 None."""
    outcome = (case["records"].get("closure") or {}).get("outcome")
    if outcome == "false_alarm":
        return "false_alarm"
    if outcome == "resolved":
        return "false_alarm" if decisive(case) == ALARM_TRUTH else "real"
    return None


def build(cases: list[dict[str, Any]], exclude: str | None = None) -> dict[str, Any]:
    """닫힌 케이스만 센다. exclude는 지금 보고 있는 케이스처럼 자기 자신을 빼고 싶을 때."""
    alarms: dict[str, dict[str, Any]] = {}
    causes: dict[str, dict[str, int]] = {}
    checks: dict[str, dict[str, int]] = {}
    manual_causes: list[dict[str, Any]] = []
    missed_checks: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for d in cases:
        r = d["records"]
        if not r.get("closure") or d["case_id"] == exclude:
            continue
        e = d["event"]
        code = code_of(e)
        # PUT 본문을 그대로 저장하므로 모양이 다른 기록(dict 아닌 점검, 글 아닌 missed_checks)은 건너뛴다
        for cid, rec in r.get("checks", {}).items():
            result = rec.get("result") if isinstance(rec, dict) else None
            if result in ("abnormal", "normal"):
                checks.setdefault(cid, {"abnormal": 0, "normal": 0})[result] += 1
        text = (r.get("log") or {}).get("missed_checks")
        text = text.strip() if isinstance(text, str) else ""
        if text:
            missed_checks.append({"case_id": d["case_id"], "code": code, "text": text})
        label = alarm_label(d)
        if label is None:
            continue
        labels[d["case_id"]] = label
        a = alarms.setdefault(
            alarm_key(e),
            {
                "gun_id": e["gun_id"],
                "code": (e.get("trigger") or {}).get("rule_code"),
                "false_alarm": 0,
                "real": 0,
                "case_ids": [],
            },
        )
        a[label] += 1
        a["case_ids"].append(d["case_id"])
        if label != "real":
            continue
        sid = decisive(d)
        if sid == MANUAL_CAUSE:
            manual_causes.append({"case_id": d["case_id"], "code": code, "cause": (r.get("manual") or {}).get("cause")})
            continue
        fault = guidance.CODE_TO_SITUATION.get(code or "")
        if fault and sid and sid != ALARM_TRUTH:
            causes.setdefault(fault, {})
            causes[fault][sid] = causes[fault].get(sid, 0) + 1
    return {
        "alarms": alarms,
        "causes": causes,
        "checks": checks,
        "manual_causes": manual_causes,
        "missed_checks": missed_checks,
        "labels": labels,
        "labeled": len(labels),
    }


def _baseline(fault: str) -> list[str]:
    """기본 후보 순서: [현상] + 연결 원인(파일 순서) + [S10]."""
    return [fault, *links().get(fault, []), ALARM_TRUTH]


def _past_causes(counts: dict[str, int], baseline: list[str]) -> list[dict[str, Any]]:
    """지난 확정 원인: 건수 내림차순, 같으면 기본 후보 순서 (기본 후보 밖이면 뒤로)."""

    def at(sid: str) -> int:
        return baseline.index(sid) if sid in baseline else len(baseline)

    order = sorted(counts, key=lambda sid: (-counts[sid], at(sid)))
    return [
        {"situation_id": sid, "name": guidance.SITUATIONS.get(sid, (sid,))[0], "confirmed": counts[sid]}
        for sid in order
    ]


def _causes_text(past: list[dict[str, Any]]) -> str:
    return ", ".join(f"{p['name']}({p['situation_id']}) {p['confirmed']}건" for p in past)


def guidance_for(event: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """guidance.guidance_for의 현장 기억판: 현상 + 연결 원인 + S10 후보에 사람 기록을 붙인다.

    ingest가 부를 때 새 케이스는 아직 저장 전이라 exclude 없이 센다.
    """
    code = code_of(event)
    fault = guidance.CODE_TO_SITUATION.get(code or "")
    if fault is None or guidance.situation(fault) is None:
        return {"status": "no_evidence", "candidates": []}
    m = build(cases)
    key = alarm_key(event)
    a = m["alarms"].get(key) or {"false_alarm": 0, "real": 0}
    fa, real = a["false_alarm"], a["real"]
    baseline = _baseline(fault)
    # 모델 신호만(규칙 코드 없음)이면 키 "gun|"을 모든 코드가 같이 쓰므로 R1과 오알람 문장은 건너뛴다
    rule = (event.get("trigger") or {}).get("rule_code")
    applied = ["R1"] if rule and fa > real else []
    # R1: 같은 설비 · 코드에서 오알람이 더 많으면 S10만 맨 앞으로, 나머지 순서는 그대로
    order = [ALARM_TRUTH, *(s for s in baseline if s != ALARM_TRUTH)] if applied else baseline
    counts = m["causes"].get(fault, {})
    candidates: list[dict[str, Any]] = []
    for sid in order:
        c = guidance.candidate(sid, len(candidates) + 1)
        if c is None:
            continue
        if sid == fault:
            c["basis"] = "code"
        elif sid == ALARM_TRUTH:
            c["basis"] = "alarm_truth"
        else:
            c["basis"] = "manual_link"
        if sid != ALARM_TRUTH:
            c["field"] = {"confirmed": counts.get(sid, 0)}
        if c["basis"] == "manual_link":
            c["evidence_level"] = "field" if c["field"]["confirmed"] >= 1 else "manual"
        candidates.append(c)
    past = _past_causes(counts, baseline)
    notes: list[str] = []
    if rule and fa + real > 0:
        gun = event["gun_id"]
        if applied:
            notes.append(
                f"같은 설비·코드({gun} {rule})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 → "
                "운영 원인(S10)부터 확인. 계속 반복되면 다음 카드(현상)로."
            )
        else:
            notes.append(
                f"같은 설비·코드({gun} {rule}) 지난 기록: 오알람 {fa}건 · 고장 확인 {real}건 (순서는 매뉴얼 기준)."
            )
    if past:
        notes.append(f"지난 같은 고장({code}) 확정 원인: {_causes_text(past)} (참고, 순서는 매뉴얼 기준).")
    return {
        "status": "retrieval_only",
        "candidates": candidates,
        "memory": {
            "alarm_key": key,
            "alarm": {"false_alarm": fa, "real": real},
            "applied": applied,
            "baseline": baseline,
            "past_causes": past,
            "note_ko": " ".join(notes) or None,
        },
    }


def annotate(case: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """케이스 사본에 field_memory를 붙인다. 자기 자신은 빼고 세고, 후보 순서는 건드리지 않는다."""
    out = copy.deepcopy(case)
    m = build(cases, exclude=case["case_id"])
    e = case["event"]
    a = m["alarms"].get(alarm_key(e)) or {"false_alarm": 0, "real": 0}
    fault = guidance.CODE_TO_SITUATION.get(code_of(e) or "")
    past = _past_causes(m["causes"].get(fault, {}), _baseline(fault)) if fault else []
    # 이 케이스 후보에 있는 점검 id만 (후보 · 점검 순서대로)
    ids = [k["id"] for c in case.get("guidance", {}).get("candidates", []) for k in c.get("checks", [])]
    out["field_memory"] = {
        "alarm_key": alarm_key(e),
        "alarm": {"false_alarm": a["false_alarm"], "real": a["real"]},
        "past_causes": past,
        "checks": {cid: m["checks"][cid] for cid in ids if cid in m["checks"]},
        "labeled": m["labeled"],
    }
    return out


def snapshot(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """현장 기억 전체 요약 (/api/memory · 챗봇용). 목록은 키 순서로 정렬한다."""
    m = build(cases)
    code_by_sid = {sid: code for code, sid in guidance.CODE_TO_SITUATION.items()}
    return {
        "labeled": m["labeled"],
        "labels": m["labels"],
        "alarms": [{"key": k, **m["alarms"][k]} for k in sorted(m["alarms"])],
        "past_causes": [
            {"situation_id": f, "code": code_by_sid.get(f), "causes": _past_causes(m["causes"][f], _baseline(f))}
            for f in sorted(m["causes"])
        ],
        "checks": [{"check_id": cid, **m["checks"][cid]} for cid in sorted(m["checks"])],
        "manual_causes": m["manual_causes"],
        "missed_checks": m["missed_checks"],
    }


def chat_note(situation_id: str, snap: dict[str, Any]) -> str | None:
    """챗봇 답 끝에 붙일 현장 기록 한 줄. 고장 상황(S01~S04)만, 기록이 없으면 None."""
    code = {sid: c for c, sid in guidance.CODE_TO_SITUATION.items()}.get(situation_id)
    if code is None:
        return None
    alarms = sorted((a for a in snap["alarms"] if a["code"] == code), key=lambda a: a["gun_id"])
    past = next((p["causes"] for p in snap["past_causes"] if p["situation_id"] == situation_id), [])
    if not alarms and not past:
        return None
    parts: list[str] = []
    if alarms:
        parts.append(
            ", ".join(f"{a['gun_id']} 오알람으로 닫음 {a['false_alarm']} · 고장 확인 {a['real']}" for a in alarms)
        )
    if past:
        parts.append(f"지난 확정 원인: {_causes_text(past)} (참고)")
    return f"현장 기록(사람 판정, 매뉴얼 근거 아님): {code} 알림 — " + " / ".join(parts)
