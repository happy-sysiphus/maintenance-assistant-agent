"""현장 기억(field memory): 닫힌 케이스의 사람 기록을 라벨(오알람 / 고장 확인)과 집계로 묶는다.

FIELD_MEMORY=1일 때만 쓴다 (부를 때마다 환경 변수를 읽는다). 끄면 api는 RAG 담당 판과 똑같이 동작한다.
RAG 안내 카드(guidance)는 바꾸지 않는다. 순서도 내용도 그대로다.
현장 기억은 참고 문장(note_ko)만 만든다: 같은 설비 · 같은 규칙 코드의 오알람 기록, 이 카드 점검 항목의 지난 결과.
오알람이 고장 확인보다 많으면 "오알람(운영 원인)인지부터 확인" 문장을 쓴다(R1). 매뉴얼 근거(cite)로 내보내지 않는다.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import guidance

# 후보 밖 원인을 직접 적었을 때의 결과 situation_id (main.MANUAL_CAUSE와 같음)
MANUAL_CAUSE = "MANUAL"


def enabled() -> bool:
    return os.environ.get("FIELD_MEMORY") == "1"


def alarm_key(event: dict[str, Any]) -> str:
    """오알람 집계 단위: 설비 | 규칙 코드 (모델 신호만이면 코드 자리가 빈다)."""
    return f"{event['gun_id']}|{(event.get('trigger') or {}).get('rule_code') or ''}"


def decisive(case: dict[str, Any]) -> str | None:
    """결정 원인: 마지막 resolved 결과의 situation_id, 없으면 마지막 yes 판단의 situation_id.

    POST 본문을 그대로 저장하므로 키가 빠졌거나 dict가 아닌 기록은 건너뛴다.
    """
    r = case["records"]
    results = [x for x in r.get("results") or [] if isinstance(x, dict)]
    last = next((x for x in reversed(results) if x.get("outcome") == "resolved"), None)
    if last:
        return last.get("situation_id")
    judgments = [j for j in r.get("judgments") or [] if isinstance(j, dict)]
    return next((j.get("situation_id") for j in reversed(judgments) if j.get("verdict") == "yes"), None)


def alarm_label(case: dict[str, Any]) -> str | None:
    """false_alarm 닫기 → "false_alarm", 해결 종료(MANUAL 포함) → "real". 미해결 · 인계 · 열림은 None."""
    outcome = (case["records"].get("closure") or {}).get("outcome")
    if outcome == "false_alarm":
        return "false_alarm"
    if outcome == "resolved":
        return "real"
    return None


def build(cases: list[dict[str, Any]], exclude: str | None = None) -> dict[str, Any]:
    """닫힌 케이스만 센다. exclude는 지금 보고 있는 케이스처럼 자기 자신을 빼고 싶을 때."""
    alarms: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, int]] = {}
    manual_causes: list[dict[str, Any]] = []
    missed_checks: list[dict[str, Any]] = []
    labels: dict[str, str] = {}
    for d in cases:
        r = d["records"]
        if not r.get("closure") or d["case_id"] == exclude:
            continue
        e = d["event"]
        code = guidance.event_code(e)
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
        if label == "real" and decisive(d) == MANUAL_CAUSE:
            manual_causes.append({"case_id": d["case_id"], "code": code, "cause": (r.get("manual") or {}).get("cause")})
    return {
        "alarms": alarms,
        "checks": checks,
        "manual_causes": manual_causes,
        "missed_checks": missed_checks,
        "labels": labels,
        "labeled": len(labels),
    }


def _card_check_ids(case: dict[str, Any]) -> list[str]:
    """이 케이스 카드의 점검 id (카드 · 점검 순서대로, 겹치면 한 번)."""
    ids: list[str] = []
    for c in (case.get("guidance") or {}).get("candidates") or []:
        for k in c.get("checks") or []:
            cid = k.get("id")
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def _checks_text(checks: dict[str, dict[str, int]]) -> str:
    return ", ".join(f"{cid} 이상 {c['abnormal']} · 정상 {c['normal']}" for cid, c in checks.items())


def annotate(case: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """케이스 사본에 field_memory를 붙인다. 자기 자신은 빼고 센다. 안내 카드는 건드리지 않는다."""
    out = copy.deepcopy(case)
    m = build(cases, exclude=case["case_id"])
    e = case["event"]
    key = alarm_key(e)
    a = m["alarms"].get(key) or {"false_alarm": 0, "real": 0}
    fa, real = a["false_alarm"], a["real"]
    # 모델 신호만(규칙 코드 없음)이면 키 "gun|"을 모든 코드가 같이 쓰므로 R1과 오알람 문장은 건너뛴다 (R-8)
    rule = (e.get("trigger") or {}).get("rule_code")
    applied = ["R1"] if rule and fa > real else []
    checks = {cid: m["checks"][cid] for cid in _card_check_ids(case) if cid in m["checks"]}
    notes: list[str] = []
    if rule and fa + real > 0:
        gun = e["gun_id"]
        if applied:
            notes.append(
                f"같은 설비·코드({gun} {rule})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 → "
                "오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
            )
        else:
            notes.append(f"같은 설비·코드({gun} {rule}) 지난 기록: 오알람 {fa}건 · 고장 확인 {real}건 (참고).")
    if checks:
        code = guidance.event_code(e)
        notes.append(f"지난 점검 기록({code}): {_checks_text(checks)} (참고, 점검 순서는 매뉴얼 기준).")
    out["field_memory"] = {
        "alarm_key": key,
        "alarm": {"false_alarm": fa, "real": real},
        "applied": applied,
        "note_ko": " ".join(notes) or None,
        "checks": checks,
        "labeled": m["labeled"],
    }
    return out


def snapshot(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """현장 기억 전체 요약 (/api/memory · 챗봇용). 목록은 키 순서로 정렬한다."""
    m = build(cases)
    return {
        "labeled": m["labeled"],
        "labels": m["labels"],
        "alarms": [{"key": k, **m["alarms"][k]} for k in sorted(m["alarms"])],
        "checks": [{"check_id": cid, **m["checks"][cid]} for cid in sorted(m["checks"])],
        "manual_causes": m["manual_causes"],
        "missed_checks": m["missed_checks"],
    }


def chat_note(code: str | None, snap: dict[str, Any]) -> str | None:
    """챗봇 답 끝에 붙일 현장 기록 한 줄 (code = 답의 E 코드). 기록이 없으면 None."""
    if not code:
        return None
    alarms = sorted((a for a in snap["alarms"] if a["code"] == code), key=lambda a: a["gun_id"])
    checks = {c["check_id"]: c for c in snap["checks"] if c["check_id"].startswith(f"{code}-")}
    if not alarms and not checks:
        return None
    parts: list[str] = []
    if alarms:
        parts.append(
            ", ".join(f"{a['gun_id']} 오알람으로 닫음 {a['false_alarm']} · 고장 확인 {a['real']}" for a in alarms)
        )
    if checks:
        parts.append(f"지난 점검: {_checks_text(checks)} (참고)")
    return f"현장 기록(사람 판정, 매뉴얼 근거 아님): {code} 알림 — " + " / ".join(parts)
