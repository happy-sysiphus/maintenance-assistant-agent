"""
ML -> RAG handoff mapping: turns one /predict result (AnomalyResult as a dict) into the document the
ontology / RAG stage receives on a critical event (schema `rag_handoff` v1.2).

Scope (2026-10-21, D-003 / D-004 decided):
    ML (this module)   what is unusual: trigger, terminal code and fault class (E01..E04, from the data: the code a
                       failure file shows in its last 10 min), contributing sensors in words
    ontology / RAG     what it means: situations, causes, check procedures, manual sections, the written report
v1.2 drops the symptom rules (P1..P9) and every situation id (S01..S10): both were the MVP situation definition
doc's mapping, which the data could not confirm (history.md 33). The RAG side receives the controller code
(`fault_class.terminal_code`, `trigger.rule_code`) and the fault class (`fault_class.code`) as they are, plus the
per-minute sensor document (.py/sensor_features.py, sensor-features-v4).

2026-10-26 (D-7, history.md 35): c4 / c18 are the Festo fieldbus bit-13 pair - c18 the robot's "Close gun" (build up
force) command, c4 the gun's "Force reached" answer (manual A.2 p.100, Table 9.38 p.74); both 0/1, so their window
mean is the share of the minute the bit was on (wording "켜진 비율"), not a time or a set value.

Pure Python (no pandas / pydantic / fastapi) so the ontology side can import it as-is. main.py validates the
output with pydantic (RagHandoff); tests/test_rag_mapping.py keeps the keys of both in sync.
"""
from __future__ import annotations

import uuid
from typing import Any

SCHEMA_VERSION = "1.2"  # 1.1 (2026-10-08): + error_timeline, sensor_trend, times with "Z"; 1.2 (2026-10-21): - symptoms,
# situation_ids, fault_class.situation_id (no situation-doc mapping in ML)

# a contributing feature counts as a finding when it explains at least MIN_SHARE of the positive contributions
# AND its (gun-centred, z-scored) value is at least MIN_DEV away from the normal reference. Calibrated on the 132
# real events of the 80 files' last 12 h (D-5, history.md §19, results/D-5/symptom_sweep.json): the old 0.05 / 1.0 left
# 62 % of the events without any finding (median 0); 0.03 / 0.5 gives a median of 1 (mean 1.6), 25 % without
MIN_SHARE = 0.03
MIN_DEV = 0.5
MAX_FINDINGS = 6
# time-of-day features explain *when* the window is, not what the gun does - never reported as a finding
NOT_A_SYMPTOM = {"hour_sin", "hour_cos"}

SENSOR_GROUP = {
    "c1": "electrode", "c2": "force", "c3": "position", "c4": "force", "c5": "compensation",
    "c6": "friction", "c7": "position", "c8": "force", "c9": "friction", "c10": "io",
    "c13": "setpoint", "c14": "setpoint", "c15": "setpoint", "c16": "setpoint", "c17": "setpoint", "c18": "setpoint",
    "welds_delta": "activity", "pos_delta": "activity", "welds_10min": "activity", "weld_duty_10min": "activity",
    "error_share_10min": "activity", "hour_sin": "activity", "hour_cos": "activity",
}
SENSOR_KO = {
    "c1": "전극 캡 오프셋", "c2": "전극 힘", "c3": "전극 위치", "c4": "힘 도달 신호", "c5": "보정(밸런스) 압력",
    "c6": "마찰", "c7": "최대 열림 폭", "c8": "최대 전극 힘", "c9": "시작 마찰", "c10": "US2 작동 허용 신호",
    "c13": "보정 압력 설정값", "c14": "전극 힘 설정값", "c15": "전극 위치 설정값", "c16": "판 두께 설정값",
    "c17": "속도 설정값", "c18": "건 닫기·힘 형성 명령", "welds_delta": "초당 용접 수", "pos_delta": "초당 위치 이동 수",
    "welds_10min": "최근 10분 용접 수", "weld_duty_10min": "최근 10분 용접 가동률",
    "error_share_10min": "최근 10분 에러 상태 비율", "hour_sin": "시각", "hour_cos": "시각",
}
DIRECTION_KO = {"high": "평소보다 높음", "low": "평소보다 낮음", "unstable": "평소보다 흔들림"}
# 0/1 handshake bits (D-7): a window mean is the share of the minute the bit was on
ON_SHARE_BITS = {"c4", "c18"}

# fault classes of the paper (readme 1.5); terminal code = the code a failure file shows in its last 10 min (readme 5.3.4)
FAULT_CLASSES: dict[str, dict[str, str]] = {
    "E01": {"name_en": "Counterbalance timeout", "name_ko": "보정 압력 도달 지연", "terminal_code": "E012",
            "definition": "set compensating pressure not reached within 1800 ms"},
    "E02": {"name_en": "Electrode broke", "name_ko": "전극 파손", "terminal_code": "E016",
            "definition": "electrode position below the zero point of the reference travel"},
    "E03": {"name_en": "Unwanted movement", "name_ko": "원치 않는 이동", "terminal_code": "E028",
            "definition": "actual position deviates by more than 6.5 % of the cylinder stroke"},
    "E04": {"name_en": "Drift", "name_ko": "드리프트", "terminal_code": "E029",
            "definition": "locked cylinder keeps moving faster than 5 mm/min"},
}
CODE_TO_CLASS = {v["terminal_code"]: k for k, v in FAULT_CLASSES.items()}

CAVEATS = [
    "고장 유형(E01~E04)은 종료 코드(E012/E016/E028/E029)를 데이터 집계로 연결한 것 - 매뉴얼·논문이 정한 대응이 아님",
    "모델 단독 성능이 약함(테스트 AUROC 0.71, 종료 코드 이전 구간 0.67; 고장으로 이어진 지속 알람 2/8건, 둘 다 종료 코드가 "
    "뜬 마지막 10분 안) - 고장 유형 근거는 주로 종료 코드 규칙",
    "종료 코드 규칙은 다른 클래스 건에서도 뜬 적 있음(주로 E029, 테스트 critical 오트리거 0.18회/일/건)",
    "센서 방향(높음/낮음)은 이 gun의 워밍업 평균 대비 z-score 기준",
    "센서 해석(상황·원인)은 하지 않음 - 센서 편차는 매분 sensor-features 문서(v4)로 따로 전달",
]


# ------------------------------------------------------------------ steps
def utc_iso(v: Any) -> str | None:
    """A time as ISO 8601 with "Z": every time in the handoff is UTC (main.py works in naive UTC)."""
    if v is None:
        return None
    s = str(v).replace(" ", "T")
    return s if (s.endswith("Z") or "+" in s[10:] or s[10:].count("-") > 0) else s + "Z"


def sensor_findings(contributing_features: list[dict[str, Any]], min_share: float = MIN_SHARE,
                    min_dev: float = MIN_DEV) -> list[dict[str, Any]]:
    """(1)->(2): contributing window features -> directional findings ('c5 mean low'), strongest first."""
    out: list[dict[str, Any]] = []
    for c in contributing_features:
        if c["sensor"] in NOT_A_SYMPTOM or c.get("share", 0.0) < min_share or c.get("contribution", 0.0) <= 0:
            continue
        dev = float(c["value"]) - float(c["reference"])
        if abs(dev) < min_dev:
            continue
        stat = c.get("statistic", "mean")
        if stat == "std":
            if dev < 0:
                continue  # a flatter-than-usual signal is not a symptom we map
            direction = "unstable"
        else:
            direction = "high" if dev > 0 else "low"
        s = c["sensor"]
        name_ko = SENSOR_KO.get(s, s)
        # c10 is binary (on = 1): a low window mean means the enable signal dropped out
        says = "꺼짐(평소 켜짐)" if (s == "c10" and direction == "low") else DIRECTION_KO[direction]
        if s in ON_SHARE_BITS and stat == "mean":
            says = f"켜진 비율 {says}"
        out.append({"sensor": s, "sensor_name": c.get("sensor_name", s), "sensor_name_ko": name_ko,
                    "group": SENSOR_GROUP.get(s, "other"), "statistic": stat, "direction": direction,
                    "deviation_z": round(dev, 2), "share": round(float(c["share"]), 3),
                    "text_ko": f"{name_ko}({s}) {says}"})
    return out[:MAX_FINDINGS]


def resolve_class(result: dict[str, Any]) -> dict[str, Any] | None:
    """The fault class the terminal-code rule points to (rule_class_hint first, then the latest code). A repeat
    trigger (rule_repeat: the code already fired in this gun within a day) or a code outside the gun's profile
    (rule_out_of_profile, C-3) is not critical and not a rule basis."""
    ctx = result.get("context") or {}
    if (result.get("rule_triggered") and not result.get("rule_repeat") and not result.get("rule_out_of_profile")
            and result.get("rule_class_hint")):
        code, basis = result["rule_class_hint"], "rule_trigger"
    elif ctx.get("known_code_class_hint") and ctx.get("known_code_in_profile") is not False:  # C-3: not outside the profile
        code, basis = ctx["known_code_class_hint"], "latest_error_code"
    else:
        return None
    return {"code": code, **FAULT_CLASSES[code], "basis": basis}


def trigger_source(result: dict[str, Any]) -> str:
    """What made the event critical: the request-level `critical_source` (main.py, 2026-09-29) - `severity_source`
    describes only the latest window, which is 'none' when a sustained run started and ended inside the chunk."""
    src = result.get("critical_source")
    return src if src and src != "none" else result.get("severity_source", "none")


def interpret(result: dict[str, Any]) -> dict[str, Any]:
    """AnomalyResult dict -> sensor findings, fault class, one-line summary (no symptoms / situations since v1.2)."""
    findings = sensor_findings(result.get("contributing_features") or [])
    fclass = resolve_class(result)
    parts = []
    if findings:
        parts.append(", ".join(f["text_ko"] for f in findings[:2]))
    if fclass:
        parts.append(f"고장 유형 {fclass['code']}({fclass['name_ko']}) 의심, 에러 코드 {fclass['terminal_code']}")
    summary = ("; ".join(parts) or "두드러진 센서 편차 없음") + "."
    return {"summary_ko": summary, "sensor_findings": findings, "fault_class": fclass}


def build_handoff(result: dict[str, Any], event_id: str | None = None) -> dict[str, Any]:
    """Full handoff document (schema v1.2) for one critical event. `result` = AnomalyResult.model_dump(mode='json')."""
    ctx = result.get("context") or {}
    model = result.get("model") or {}
    # a repeat or an out-of-profile code (C-3) did not trigger this event
    rule = bool(result.get("rule_triggered")) and not result.get("rule_repeat") and not result.get("rule_out_of_profile")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or uuid.uuid4().hex,
        "gun_id": result["gun_id"],
        "detected_at": utc_iso(result["window_end"]),
        "window_start": utc_iso(result["window_start"]),
        "trigger": {"source": trigger_source(result),
                    "rule_code": result.get("rule_code") if rule else None,
                    "rule_trigger_time": utc_iso(result["rule_trigger_time"]) if rule and result.get("rule_trigger_time")
                    else None,
                    "anomaly_score": result.get("anomaly_score"), "threshold": result.get("threshold"),
                    "score_z": result.get("score_z"), "alarm_duration_s": result.get("alarm_duration_s", 0),
                    "sustained": bool(result.get("sustained_in_request") or result.get("sustained_alarm"))},
        **interpret(result),
        "context": {"latest_error_code": ctx.get("latest_error_code"),
                    "error_active_share": ctx.get("error_active_share"),
                    "non_welding_share": ctx.get("non_welding_share"),
                    "welds_in_window": ctx.get("welds_in_window"), "welds_10min": ctx.get("welds_10min"),
                    "alarm_held": result.get("alarm_held", False), "gun_norm": result.get("gun_norm")},
        "detector": {"name": model.get("name"), "created": model.get("created"), "window_s": model.get("window_s")},
        "caveats": CAVEATS,
    }
