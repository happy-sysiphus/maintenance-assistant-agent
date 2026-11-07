"""현장 기억(field memory): 닫힌 케이스의 사람 기록을 라벨(오알람 / 고장 확인)과 집계로 묶는다.

FIELD_MEMORY=1일 때만 쓴다 (부를 때마다 환경 변수를 읽는다). 끄면 api는 RAG 담당 판과 똑같이 동작한다.
참고 문장(note_ko): 같은 설비 · 같은 규칙 코드의 오알람 기록, 이 카드 점검 항목의 지난 결과.
오알람이 고장 확인보다 많으면 "오알람(운영 원인)인지부터 확인" 문장을 쓴다(R1). 매뉴얼 근거(cite)로 내보내지 않는다.
R2: 해결 종료된 케이스의 '이상' 점검 기록으로 새 케이스 카드의 점검 순서를 정한다(order_for · apply_order).
ingest가 케이스를 만들 때 한 번 정하고, 카드의 글 · 근거 · 조치는 바꾸지 않는다. annotate는 카드를 바꾸지 않는다.
설계: ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md
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


# ---- R2: 해결 종료된 케이스의 '이상' 점검 기록으로 새 케이스 카드의 점검 순서를 정한다 ----
# 설계: ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md (3 · 4절)

ORDER_LEAD = 3  # 앞에 있는 모든 항목보다 이만큼 더 확인돼야 앞세운다
ORDER_GUNS = 2  # 앞세울 항목의 확인이 나와야 하는 설비 수


def mapping_checks(code: str | None) -> list[str]:
    """매핑의 점검 id (원인 순, 그 안의 점검 순). 매핑에 없는 코드면 []."""
    item = guidance.mapping().get(code) if code else None
    if not item:
        return []
    return [c["id"] for cause in item["causes"] for c in cause["checks"]]


def order_counts(code: str, cases: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, set[str]]]:
    """매핑 점검마다 확인 수와 그 확인이 나온 설비 (스펙 3.1).

    확인 1건 = 해결 종료된 같은 코드 케이스(카드 밖 원인으로 해결한 것은 빼고)에서 '이상'으로 기록된 점검.
    원인 판단은 쓰지 않는다: 원인 번호는 카드 위치라, 카드가 다시 만들어지면 다른 원인을 가리킨다.
    """
    ids = mapping_checks(code)
    counts = {cid: 0 for cid in ids}
    guns: dict[str, set[str]] = {cid: set() for cid in ids}
    for d in cases:
        if alarm_label(d) != "real" or decisive(d) == MANUAL_CAUSE or guidance.event_code(d["event"]) != code:
            continue
        checks = d["records"].get("checks")
        for cid, rec in (checks if isinstance(checks, dict) else {}).items():
            if cid in counts and isinstance(rec, dict) and rec.get("result") == "abnormal":
                counts[cid] += 1
                guns[cid].add(d["event"]["gun_id"])
    return counts, guns


def order_for(code: str | None, cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """새 케이스가 받을 순서 결정 (스펙 3.2). 앞세울 항목이 없으면 None.

    후보는 확인이 가장 많은 항목 하나(같으면 매핑에서 앞선 것)다. 후보가 매핑의 첫 항목이 아니고,
    앞의 모든 항목보다 ORDER_LEAD건 이상 많고, 설비 ORDER_GUNS대 이상에서 나왔을 때만 앞세운다.
    """
    ids = mapping_checks(code)
    if len(ids) < 2 or code is None:
        return None
    counts, guns = order_counts(code, cases)
    lead = max(ids, key=lambda cid: counts[cid])  # max는 같은 값 중 첫 항목을 고른다
    i = ids.index(lead)
    if i == 0 or counts[lead] - max(counts[c] for c in ids[:i]) < ORDER_LEAD or len(guns[lead]) < ORDER_GUNS:
        return None
    rest = " · ".join(f"{c} {counts[c]}건" for c in ids if c != lead)
    note = (
        "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 "
        f"{lead} {counts[lead]}건 · {rest}(설비 {len(guns[lead])}대)이라 {lead}부터 보도록 순서를 바꿈. "
        "내용은 매뉴얼 그대로."
    )
    return {"lead": lead, "counts": counts, "guns": len(guns[lead]), "note_ko": note}


def _checks_of(cause: Any) -> list[Any]:
    checks = cause.get("checks") if isinstance(cause, dict) else None
    return checks if isinstance(checks, list) else []


def _holds(cause: Any, lead: str) -> bool:
    return any(isinstance(k, dict) and k.get("id") == lead for k in _checks_of(cause))


def _lead_first(checks: list[Any], lead: str) -> list[Any]:
    i = next(i for i, k in enumerate(checks) if isinstance(k, dict) and k.get("id") == lead)
    return [checks[i], *checks[:i], *checks[i + 1 :]]


def _move_causes(causes: list[Any], lead: str) -> list[Any] | None:
    """lead가 든 첫 원인을 맨 앞으로, 그 안에서 lead를 맨 앞으로. lead가 없거나 조치 보호에 걸리면 None.

    조치 보호: 앞 원인의 조치를 앞세울 원인이 모두(같은 항목으로) 갖고 있지 않으면 옮기지 않는다. 조치 화면은
    '맞아요'로 고른 원인의 조치만 보여서(ui/src/lib/caseFlow.ts activeCause), 밀린 원인에만 있는 조치(E029의
    재시작 주의)를 볼 기회가 줄기 때문이다. 앞세울 원인도 그 조치를 가지면 어느 원인을 골라도 보인다 (11-08 완화).
    """
    i = next((i for i, c in enumerate(causes) if _holds(c, lead)), None)
    if i is None:
        return None
    kept = causes[i].get("actions") or []
    if any(a not in kept for c in causes[:i] if isinstance(c, dict) for a in c.get("actions") or []):
        return None
    return [{**causes[i], "checks": _lead_first(_checks_of(causes[i]), lead)}, *causes[:i], *causes[i + 1 :]]


def apply_order(g: dict[str, Any], lead: str | None) -> dict[str, Any]:
    """카드에 순서를 적용한다 (스펙 3.3). lead를 카드의 첫 점검으로 만들 수 없으면 g를 그대로 돌려준다.

    글 · 근거 · 기준 · 조치는 그대로 두고 위치만 바꾼다. 입력은 고치지 않는다. 다시 적용해도 결과가 같다.
    """
    if not lead:
        return g
    causes = g.get("causes") or []
    if causes:
        moved = _move_causes(causes, lead)
        if moved is None:
            return g
        out = copy.deepcopy(g)
        out["causes"] = copy.deepcopy(moved)
        for cand in out.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            sub = _move_causes(cand.get("causes") or [], lead)
            if sub is not None:
                cand["causes"] = sub
            if _holds(cand, lead):
                cand["checks"] = _lead_first(cand["checks"], lead)
        return out
    # 원인 목록이 빈 카드: 화면이 candidates를 그대로 쓴다 (ui/src/lib/guidance.ts expand)
    if not any(_holds(c, lead) for c in g.get("candidates") or []):
        return g
    out = copy.deepcopy(g)
    for cand in out["candidates"]:
        if _holds(cand, lead):
            cand["checks"] = _lead_first(cand["checks"], lead)
    return out


def first_check(g: dict[str, Any]) -> str | None:
    """보이는 첫 점검 id: 첫 원인의 첫 점검. 원인 목록이 비면 candidates[0]의 첫 점검. 없으면 None."""
    causes = g.get("causes") or []
    if causes:
        checks = _checks_of(causes[0])
    else:
        cands = g.get("candidates") or []
        checks = _checks_of(cands[0]) if cands else []
    first = checks[0] if checks else None
    return first.get("id") if isinstance(first, dict) else None


def order_pinned(case: dict[str, Any]) -> bool:
    """R2로 순서를 정한 케이스에 판단 · 조치 · 결과 기록이 있으면 카드를 다시 만들지 않는다 (스펙 4.3).

    원인 번호는 카드 위치라, 기록 뒤 카드 묶음이 바뀌면 R2가 맨 앞에 두는 원인이 달라질 수 있다.
    점검 기록은 점검 id로 저장되므로 고정 조건이 아니다.
    """
    r = case.get("records") or {}
    return bool(case.get("field_order")) and any(r.get(k) for k in ("judgments", "actions", "results"))


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
    r1 = bool(rule) and fa > real
    # R2: ingest가 정한 순서(field_order)가 카드에 실제로 보일 때만 (조치 보호로 못 옮긴 카드는 아니다)
    order = case.get("field_order") or {}
    r2 = bool(order) and first_check(case.get("guidance") or {}) == order.get("lead")
    applied = [name for name, on in (("R1", r1), ("R2", r2)) if on]
    checks = {cid: m["checks"][cid] for cid in _card_check_ids(case) if cid in m["checks"]}
    notes: list[str] = []
    if rule and fa + real > 0:
        gun = e["gun_id"]
        if r1:
            notes.append(
                f"같은 설비·코드({gun} {rule})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 → "
                "오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
            )
        else:
            notes.append(f"같은 설비·코드({gun} {rule}) 지난 기록: 오알람 {fa}건 · 고장 확인 {real}건 (참고).")
    if r2:
        notes.append(order["note_ko"])
    if checks:
        code = guidance.event_code(e)
        tail = "(참고)." if r2 else "(참고, 점검 순서는 매뉴얼 기준)."
        notes.append(f"지난 점검 기록({code}): {_checks_text(checks)} {tail}")
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
