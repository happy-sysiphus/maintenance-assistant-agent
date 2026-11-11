"""Bounded field-record context, separate from chat and manual evidence."""

import hashlib
import json
from datetime import datetime, timezone

from .documents import ERROR_CODES


def normalize_context(records, code):
    if not records or code not in ERROR_CODES:
        return []
    if not isinstance(records, list):
        raise ValueError("maintenance_context는 정비 기록 목록이어야 합니다.")
    selected = {}
    now = datetime.now(timezone.utc)
    for row in records:
        if not isinstance(row, dict) or row.get("error_code") != code:
            continue
        case_id, closed_at = row.get("case_id"), row.get("closed_at")
        if not isinstance(case_id, str) or not isinstance(closed_at, str):
            continue
        try:
            when = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            if when.tzinfo is None or when > now:
                continue
        except ValueError:
            continue
        outcome = row.get("outcome")
        if outcome not in {"resolved", "unresolved", "false_alarm"}:
            continue
        raw_checks = row.get("checks")
        raw_checks = raw_checks if isinstance(raw_checks,dict) else {}
        checks = {key: value for key, value in raw_checks.items()
                  if isinstance(key,str) and key.startswith(code + "-K") and isinstance(value,str)
                  and value in {"normal", "abnormal", "skipped"}}
        actions = row.get("actions")
        actions = actions if isinstance(actions,list) else []
        selected[case_id] = {"case_id": case_id[:100], "error_code": code,
            "closed_at": closed_at, "outcome": outcome, "checks": checks,
            "actions": [str(a)[:400] for a in actions[:5]],
            "note": str(row.get("note") or "")[:1200]}
    return sorted(selected.values(), key=lambda r: (r["closed_at"], r["case_id"]), reverse=True)[:5]


def context_hash(records):
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_advice(advice, records, item):
    """The LLM may cite supplied cases and registered checks, never invent IDs."""
    if not records:
        if advice is not None:
            raise ValueError("정비 이력이 없는데 이력 기반 제안이 생성됐습니다.")
        return None
    if advice is None:
        return None
    from .response import MaintenanceAdvice
    result = MaintenanceAdvice.model_validate(advice).model_dump()
    allowed_cases = {r["case_id"] for r in records}
    allowed_checks = {c["id"] for ca in (item or {}).get("causes", []) for c in ca["checks"]}
    if (not result["case_ids"] or not set(result["case_ids"]) <= allowed_cases
            or not set(result["check_ids"]) <= allowed_checks):
        raise ValueError("정비 이력 제안에 제공하지 않은 사례 또는 점검 ID가 있습니다.")
    return result
