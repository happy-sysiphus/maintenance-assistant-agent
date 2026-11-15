"""이전 기록(field memory): 같은 설비 · 같은 규칙 코드로 닫힌 케이스의 사람 기록을 찾는다.

FIELD_MEMORY=1일 때만 쓴다 (부를 때마다 환경 변수를 읽는다). 끄면 api는 RAG 담당 판과 똑같이 동작한다.
기록을 보여 주기만 한다. 안내 카드 · 점검 순서 · 챗봇 답은 바꾸지 않는다.
"""

from __future__ import annotations

import os
from typing import Any


def enabled() -> bool:
    return os.environ.get("FIELD_MEMORY") == "1"


def alarm_key(event: dict[str, Any]) -> str:
    """이전 기록을 묶는 단위: 설비 | 규칙 코드 (모델 신호만이면 코드 자리가 빈다).

    조기 경보(event_type early_warning)는 "설비|early_warning"으로 따로 묶는다. 모델 단독 critical과 섞지 않는다 (11/08 FE).
    """
    if event.get("event_type") == "early_warning":
        return f"{event['gun_id']}|early_warning"
    return f"{event['gun_id']}|{(event.get('trigger') or {}).get('rule_code') or ''}"


def alarm_label(case: dict[str, Any]) -> str | None:
    """false_alarm 닫기 → "false_alarm", 해결 종료 → "real". 미해결 · 인계 · 열림은 None."""
    outcome = (case["records"].get("closure") or {}).get("outcome")
    if outcome == "false_alarm":
        return "false_alarm"
    if outcome == "resolved":
        return "real"
    return None


def past(case: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """case와 같은 설비 · 코드로 오알람 · 해결 종료된 케이스. 자기 자신은 뺀다."""
    key = alarm_key(case["event"])
    return [
        c
        for c in cases
        if c["case_id"] != case["case_id"] and alarm_key(c["event"]) == key and alarm_label(c) is not None
    ]
