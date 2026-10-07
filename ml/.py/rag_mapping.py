"""
ML -> RAG handoff mapping: turns one /predict result (AnomalyResult as a dict) into the document the
ontology / RAG stage receives (schema `rag_handoff` v1.0).

Scope (agreed split, 2026-09-25):
    ML (this module)   (1) what is unusual: score, trigger, contributing sensors
                       (2) symptom translation: "c5 mean low" -> symptom P1, fault class E01, situation ids S01..S10
    ontology / RAG     (3) causes, check procedures, manual sections   (4) the written report
The situation ids are the keys of docs "RSW 용접건 MVP 오류 상황 정의서" (S01..S10), so the ontology can look an
event up without knowing the sensor codes. This module never names parts, check steps or manual pages.

Symptom rules come from the manual guide (RSW용접건_매뉴얼_RAG_활용정리 §8), not from data: the probe in
history.md 10 found the pre-failure sensor shifts inconsistent within a class, so a symptom is a *candidate* and
confidence is at most "medium" (only when the terminal-code rule fired and agrees).

2026-10-08 (guide §8 re-check): the guide reads every pattern against the setpoints - "c5 low while the c13 setpoint
is unchanged" - and the first port dropped those conditions. Restored as `steady` (P1 c13, P2 c14, P5 c15: the
setpoint must be at its usual level, otherwise it is a recipe change, P7) and `normal` (P3: c5 must be at its usual
level, otherwise it is P2's pneumatic pattern). The level comes from context["mean_dev"] (window mean - reference for
every c-sensor, main.py) and falls back to the findings; unknown counts as steady. Not ported yet (need features the
1-min window does not have): P4 "c3 drifts slowly one way" (still the 1-min spread) and the guide's
"c6 rising over a long period + c11 accumulating" (wear, situation S06).

Pure Python (no pandas / pydantic / fastapi) so the ontology side can import it as-is. main.py validates the
output with pydantic (RagHandoff); tests/test_rag_mapping.py keeps the keys of both in sync.
"""
from __future__ import annotations

import uuid
from typing import Any

SCHEMA_VERSION = "1.0"

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
    "c1": "전극 캡 오프셋", "c2": "전극 힘", "c3": "전극 위치", "c4": "힘 형성 시간", "c5": "보정(밸런스) 압력",
    "c6": "마찰", "c7": "최대 열림 폭", "c8": "최대 전극 힘", "c9": "시작 마찰", "c10": "US2 작동 허용 신호",
    "c13": "보정 압력 설정값", "c14": "전극 힘 설정값", "c15": "전극 위치 설정값", "c16": "판 두께 설정값",
    "c17": "속도 설정값", "c18": "힘 형성 설정값", "welds_delta": "초당 용접 수", "pos_delta": "초당 위치 이동 수",
    "welds_10min": "최근 10분 용접 수", "weld_duty_10min": "최근 10분 용접 가동률",
    "error_share_10min": "최근 10분 에러 상태 비율", "hour_sin": "시각", "hour_cos": "시각",
}
DIRECTION_KO = {"high": "평소보다 높음", "low": "평소보다 낮음", "unstable": "평소보다 흔들림"}

# fault classes of the paper (readme 1.5) = situations S01..S04
FAULT_CLASSES: dict[str, dict[str, str]] = {
    "E01": {"name_en": "Counterbalance timeout", "name_ko": "보정 압력 도달 지연", "terminal_code": "E012",
            "situation_id": "S01", "definition": "set compensating pressure not reached within 1800 ms"},
    "E02": {"name_en": "Electrode broke", "name_ko": "전극 파손", "terminal_code": "E016",
            "situation_id": "S02", "definition": "electrode position below the zero point of the reference travel"},
    "E03": {"name_en": "Unwanted movement", "name_ko": "원치 않는 이동", "terminal_code": "E028",
            "situation_id": "S03", "definition": "actual position deviates by more than 6.5 % of the cylinder stroke"},
    "E04": {"name_en": "Drift", "name_ko": "드리프트", "terminal_code": "E029",
            "situation_id": "S04", "definition": "locked cylinder keeps moving faster than 5 mm/min"},
}
CODE_TO_CLASS = {v["terminal_code"]: k for k, v in FAULT_CLASSES.items()}

# symptom rules (guide §8). when: (sensor, direction) pairs; `primary` (or one of `also`) must match for a partial
# match. context "stationary" = no weld in the window. steady: setpoints that must be at their usual level (else the
# shift is a recipe change); normal: actual values that must be at their usual level. situations: keys of the
# situation definition doc (S02 for P5 / P6 as the doc's summary table lists).
SYMPTOMS: list[dict[str, Any]] = [
    {"id": "P1", "name_ko": "보정 압력 저하 + 힘 형성 지연", "primary": ("c5", "low"),
     "when": [("c5", "low"), ("c4", "high")], "steady": ["c13"], "classes": ["E01"], "situations": ["S01"]},
    {"id": "P2", "name_ko": "전극 힘 저하 + 보정 압력 저하", "primary": ("c2", "low"),
     "when": [("c2", "low"), ("c5", "low")], "steady": ["c14"], "classes": ["E01"], "situations": ["S05"]},
    {"id": "P3", "name_ko": "마찰 증가 + 전극 힘 저하", "primary": ("c6", "high"),
     "when": [("c6", "high"), ("c2", "low")], "normal": ["c5"], "classes": ["E03"], "situations": ["S06"]},
    {"id": "P4", "name_ko": "정지 중 전극 위치 흔들림", "primary": ("c3", "unstable"),
     "when": [("c3", "unstable")], "context": "stationary", "classes": ["E04"], "situations": ["S04"]},
    {"id": "P5", "name_ko": "동작 중 전극 위치·열림 폭 이탈", "primary": ("c3", "high"),
     "when": [("c3", "high")], "also": [("c3", "low"), ("c7", "low"), ("c7", "high")],
     "steady": ["c15"], "classes": ["E03", "E02"], "situations": ["S08", "S03", "S02"]},
    {"id": "P6", "name_ko": "캡 오프셋 변화 (+ 마찰 증가)", "primary": ("c1", "high"),
     "when": [("c1", "high"), ("c6", "high")], "also": [("c1", "low")], "classes": ["E02"], "situations": ["S07", "S02"]},
    {"id": "P7", "name_ko": "설정값 변화", "group": "setpoint", "classes": [], "situations": ["S10"]},
    {"id": "P8", "name_ko": "에러 상태 비율 증가", "primary": ("error_share_10min", "high"),
     "when": [("error_share_10min", "high")], "classes": [], "situations": ["S10"]},
]
# rule fired but no sensor stands out: the gun motion looks normal
NO_SIGNAL = {"id": "P9", "name_ko": "건 센서 정상 (규칙만 발화)", "classes": [], "situations": ["S09"]}

CAVEATS = [
    "증상 규칙은 매뉴얼 기반이며 데이터로 학습된 관계가 아님 - 원인 확정이 아니라 조회 후보",
    "모델 단독 성능이 약함(테스트 AUROC 0.64, 고장 1시간 전 지속 알람 0/8건) - 고장 유형 근거는 주로 종료 코드 규칙",
    "종료 코드 규칙은 다른 클래스 건에서도 뜬 적 있음(주로 E029, 테스트 오트리거 0.25회/일/건)",
    "센서 방향(높음/낮음)은 이 gun의 워밍업 평균 대비 z-score 기준",
    "P1·P2·P5는 해당 설정값(c13·c14·c15)이 평소와 같을 때만, P3은 c5가 평소와 같을 때만 (활용정리 8절)",
]


# ------------------------------------------------------------------ steps
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
        out.append({"sensor": s, "sensor_name": c.get("sensor_name", s), "sensor_name_ko": name_ko,
                    "group": SENSOR_GROUP.get(s, "other"), "statistic": stat, "direction": direction,
                    "deviation_z": round(dev, 2), "share": round(float(c["share"]), 3),
                    "text_ko": f"{name_ko}({s}) {says}"})
    return out[:MAX_FINDINGS]


def _has(findings: list[dict[str, Any]], sensor: str, direction: str) -> bool:
    return any(f["sensor"] == sensor and f["direction"] == direction for f in findings)


def _shifted(sensor: str, findings: list[dict[str, Any]], context: dict[str, Any], min_dev: float = MIN_DEV) -> bool:
    """The sensor's window mean is away from its usual level: context["mean_dev"] (all c-sensors, main.py) first,
    else any finding on it. Unknown -> not shifted, so an older caller keeps the previous behaviour."""
    dev = (context.get("mean_dev") or {}).get(sensor)
    if dev is not None:
        return abs(float(dev)) >= min_dev
    return any(f["sensor"] == sensor and f["statistic"] == "mean" for f in findings)


def match_symptoms(findings: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Symptom rules -> matched symptoms with the findings that support them (full match first)."""
    welds = context.get("welds_in_window")
    stationary = welds is not None and float(welds) <= 0.0
    hits: list[dict[str, Any]] = []
    for p in SYMPTOMS:
        if "group" in p:
            ev = [f for f in findings if f["group"] == p["group"]]
            if ev:
                hits.append({"rule": p, "match": "full", "evidence": ev})
            continue
        if p.get("context") == "stationary" and not stationary:
            continue
        if any(_shifted(s, findings, context) for s in p.get("steady", []) + p.get("normal", [])):
            continue  # guide §8: the setpoint moved too (recipe change) / the excluded sensor is not normal
        if not (_has(findings, *p["primary"]) or any(_has(findings, *a) for a in p.get("also", []))):
            continue
        n_when = sum(_has(findings, *c) for c in p["when"])
        ev = [f for f in findings if any(f["sensor"] == s and f["direction"] == d
                                         for s, d in p["when"] + p.get("also", []))]
        hits.append({"rule": p, "match": "full" if n_when == len(p["when"]) else "partial", "evidence": ev})
    return sorted(hits, key=lambda h: (h["match"] != "full", -sum(e["share"] for e in h["evidence"])))


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
    """AnomalyResult dict -> sensor findings, fault class, symptoms, situation ids, one-line summary."""
    findings = sensor_findings(result.get("contributing_features") or [])
    fclass = resolve_class(result)
    rule = trigger_source(result) in ("rule", "model+rule")

    symptoms: list[dict[str, Any]] = []
    for h in match_symptoms(findings, result.get("context") or {}):
        p = h["rule"]
        agrees = fclass is not None and fclass["code"] in p["classes"]
        symptoms.append({"id": p["id"], "name_ko": p["name_ko"], "match": h["match"],
                         "evidence": [e["text_ko"] for e in h["evidence"]],
                         "related_classes": p["classes"], "situation_ids": p["situations"],
                         "agrees_with_fault_class": agrees,
                         "confidence": "medium" if (agrees and rule) else "low"})
    if not findings and rule:
        # with a fault class the silent sensors mostly mean the model has no signal (test.md), not that the gun is
        # fine - so S09 (peripheral equipment) is only suggested when no class is known
        symptoms.append({"id": NO_SIGNAL["id"], "name_ko": NO_SIGNAL["name_ko"], "match": "no_sensor_signal",
                         "evidence": [], "related_classes": [],
                         "situation_ids": [] if fclass else NO_SIGNAL["situations"],
                         "agrees_with_fault_class": False, "confidence": "low"})
    symptoms.sort(key=lambda s: s["confidence"] != "medium")  # rule-agreeing first, stable otherwise

    situations: list[str] = [fclass["situation_id"]] if fclass else []
    for s in symptoms:
        situations += [x for x in s["situation_ids"] if x not in situations]

    parts = []
    if findings:
        parts.append(", ".join(f["text_ko"] for f in findings[:2]))
    if fclass:
        parts.append(f"고장 유형 {fclass['code']}({fclass['name_ko']}) 의심, 에러 코드 {fclass['terminal_code']}")
    summary = ("; ".join(parts) or "두드러진 센서 편차 없음") + "."
    return {"summary_ko": summary, "sensor_findings": findings, "fault_class": fclass,
            "symptoms": symptoms, "situation_ids": situations}


def build_handoff(result: dict[str, Any], event_id: str | None = None) -> dict[str, Any]:
    """Full handoff document (schema v1.0) for one critical event. `result` = AnomalyResult.model_dump(mode='json')."""
    ctx = result.get("context") or {}
    model = result.get("model") or {}
    # a repeat or an out-of-profile code (C-3) did not trigger this event
    rule = bool(result.get("rule_triggered")) and not result.get("rule_repeat") and not result.get("rule_out_of_profile")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id or uuid.uuid4().hex,
        "gun_id": result["gun_id"],
        "detected_at": str(result["window_end"]),
        "window_start": str(result["window_start"]),
        "trigger": {"source": trigger_source(result),
                    "rule_code": result.get("rule_code") if rule else None,
                    "rule_trigger_time": str(result["rule_trigger_time"]) if rule and result.get("rule_trigger_time")
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
