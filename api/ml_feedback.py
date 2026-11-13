"""GET /api/ml/feedback: 닫힌 조기 경보 케이스를 ML 모델 루프에 내준다 (D-12).

사실만 준다: 어떻게 닫혔는지, 어느 경보들이었는지, 점검에서 이상이 나왔는지. 그 케이스가 ML의 "정답"인지는
ML이 정한다 (ml/.py/model_feedback.py). 읽기 전용이고, 기존 주소의 응답은 바꾸지 않는다.

설계: ml/docs/superpowers/specs/2026-11-12-feedback-labels-design.md 3절
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter

import store

SCHEMA_VERSION = "1.0"
CLOSED = ("resolved", "false_alarm")
RESULTS = ("abnormal", "normal", "unchecked")

log = logging.getLogger(__name__)
router = APIRouter()


def _dict(value: Any) -> dict[str, Any]:
    """객체여야 할 자리에 다른 것이 있으면 빈 객체로 읽는다. POST /api/ml/handoffs는 어떤 모양이든 받아 저장하므로,
    손으로 보낸 핸드오프의 필드 하나가 이상해도 나머지 사실은 내준다."""
    return value if isinstance(value, dict) else {}


def _event(handoff: dict[str, Any], *, representative: bool) -> dict[str, Any]:
    """onset_at: 경보가 시작된 창의 마지막 초 (early_warning.onset_at). 한 요청에 창이 여럿이면 window_start는
    그 요청의 마지막 창이라 경보가 시작된 창보다 뒤일 수 있다. 블록이 없는 옛 핸드오프는 None."""
    return {
        "event_id": handoff.get("event_id"),
        "representative": representative,
        "gun_id": handoff.get("gun_id"),
        "window_start": handoff.get("window_start"),
        "onset_at": _dict(handoff.get("early_warning")).get("onset_at"),
        "detected_at": handoff.get("detected_at"),
        "detector_created": _dict(handoff.get("detector")).get("created"),
    }


def _result(value: Any) -> Any:
    """점검 값의 두 형식: 예전 화면은 {"result": "abnormal"}, 조기 경보 화면은 "abnormal"."""
    return value.get("result") if isinstance(value, dict) else value


def _checks(case: dict[str, Any], event_id: str) -> dict[str, int]:
    """records.checks와 그 경보의 점검 초안을 점검 ID로 합쳐 값별로 센다. 같은 ID는 초안이 이긴다."""
    merged = dict(_dict(_dict(case.get("records")).get("checks")))
    draft = _dict(_dict(_dict(case.get("warning_events")).get(event_id)).get("draft"))
    merged.update(_dict(draft.get("checks")))
    values = [_result(v) for v in merged.values()]
    return {name: sum(1 for v in values if isinstance(v, str) and v == name) for name in RESULTS}


def item(case: dict[str, Any], delivery: dict[str, Any] | None) -> dict[str, Any] | None:
    """닫힌 조기 경보 케이스 하나. 조기 경보가 아니거나 해결 · 오알람으로 닫히지 않았으면 None.

    delivery: store.warning_delivery_for_case의 값 ({"event_id", "status"}). /warning/resolve로 닫은 케이스에만 있다.
    """
    event = _dict(case.get("event"))
    closure = _dict(_dict(case.get("records")).get("closure"))
    if event.get("event_type") != "early_warning" or closure.get("outcome") not in CLOSED:
        return None
    selected = (delivery or {}).get("event_id")
    events = [_event(event, representative=True)]
    for r in case.get("repeats") or []:
        if not isinstance(r, dict):
            continue
        handoff = _dict(r.get("handoff")) or {"event_id": r.get("event_id"), "detected_at": r.get("detected_at")}
        events.append(_event(handoff, representative=False))
    return {
        "case_id": case.get("case_id"),
        "outcome": closure["outcome"],
        "closed_at": closure.get("at"),
        "resolution_status": (delivery or {}).get("status") or case.get("warning_resolution_status"),
        "selected_event_id": selected,
        "checks": _checks(case, selected or event.get("event_id")),
        "events": events,
    }


def items(cases: list[dict[str, Any]], delivery_for: Callable[[str], dict[str, Any] | None]) -> list[dict[str, Any]]:
    """닫힌 조기 경보 케이스 전부, 닫힌 순서로. 읽을 수 없는 케이스는 건너뛰고 로그에 남긴다: 하나가 나머지를
    막으면 ML은 판정을 하나도 가져가지 못한다."""
    out = []
    for case in cases:
        try:
            closure = _dict(_dict(case.get("records")).get("closure"))
            if _dict(case.get("event")).get("event_type") != "early_warning" or closure.get("outcome") not in CLOSED:
                continue  # 대상이 아닌 케이스에는 전송 기록을 묻지 않는다
            found = item(case, delivery_for(case["case_id"]))
        except Exception:  # noqa: BLE001 - 어떤 모양의 케이스가 저장돼 있을지 모른다
            log.warning("ml feedback: 케이스 %s를 읽지 못해 건너뜀", _dict(case).get("case_id"), exc_info=True)
            continue
        if found is not None:
            out.append(found)
    return sorted(out, key=lambda i: (str(i["closed_at"] or ""), str(i["case_id"] or "")))


@router.get("/api/ml/feedback")
def feedback() -> dict[str, Any]:
    # 고른 경보와 전송 상태는 한 번에 읽는다 (store.warning_delivery_for_case는 케이스마다 DB를 새로 연다)
    return {"schema_version": SCHEMA_VERSION, "items": items(store.all_cases(), store.warning_deliveries().get)}
