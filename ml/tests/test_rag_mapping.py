"""
Unit tests of .py/rag_mapping.py (ML -> RAG handoff, steps (1)-(2) only). Pure Python: no model, no server, < 1 s.
The API side (handoff on /predict, /handoffs endpoints, replay) is covered in test_pipeline.py.
"""
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))

import rag_mapping as rm  # noqa: E402


def feat(sensor, stat, value, reference=0.0, share=0.3, contribution=0.05):
    return {"feature": f"{sensor}_{stat}", "sensor": sensor, "sensor_name": sensor, "statistic": stat,
            "value": value, "reference": reference, "contribution": contribution, "share": share}


BASE = {
    "gun_id": "G1", "window_start": "2021-09-05T02:22:21", "window_end": "2021-09-05T02:23:20",
    "anomaly_score": 0.61, "threshold": 0.54, "score_z": 4.6, "severity": "critical",
    "severity_source": "rule", "rule_triggered": True, "rule_code": "E012", "rule_class_hint": "E01",
    "rule_trigger_time": "2021-09-05T02:23:00", "sustained_alarm": False, "sustained_in_request": False,
    "alarm_duration_s": 0, "alarm_held": False, "gun_norm": "gun",
    "contributing_features": [],
    "context": {"latest_error_code": "0", "error_active_share": 0.1, "non_welding_share": 0.0,
                "welds_in_window": 3, "welds_10min": 30, "weld_duty_10min": 0.1, "known_code_class_hint": None},
    "model": {"name": "baseline_iforest", "created": "2026-09-23", "window_s": 60},
}


def test_findings_direction_and_filters():
    fs = rm.sensor_findings([feat("c5", "mean", -3.1), feat("c2", "std", 2.0), feat("c6", "std", -2.0),
                             feat("c4", "mean", 0.4), feat("c3", "mean", 5.0, share=0.01),
                             feat("c1", "mean", 4.0, contribution=-0.01)])
    assert {(f["sensor"], f["direction"]) for f in fs} == {("c5", "low"), ("c2", "unstable")}
    assert fs[0]["text_ko"] == "보정(밸런스) 압력(c5) 평소보다 낮음"


def test_rule_and_symptom_agree():
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c5", "mean", -3.1, share=0.4), feat("c4", "mean", 2.2, share=0.2)]
    h = rm.build_handoff(r, "ev1")
    assert h["fault_class"]["code"] == "E01" and h["fault_class"]["situation_id"] == "S01"
    s = h["symptoms"][0]
    assert s["id"] == "P1" and s["match"] == "full" and s["confidence"] == "medium" and s["agrees_with_fault_class"]
    assert h["situation_ids"] == ["S01"]
    assert h["summary_ko"] == ("보정(밸런스) 압력(c5) 평소보다 낮음, 힘 형성 시간(c4) 평소보다 높음; "
                               "고장 유형 E01(보정 압력 도달 지연) 의심, 에러 코드 E012.")


def test_symptom_without_rule_is_low():
    r = copy.deepcopy(BASE)
    r.update(severity_source="model", rule_triggered=False, rule_code=None, rule_class_hint=None,
             rule_trigger_time=None, sustained_in_request=True)
    r["contributing_features"] = [feat("c6", "mean", 2.5), feat("c2", "mean", -1.8)]
    h = rm.build_handoff(r)
    assert h["fault_class"] is None and h["symptoms"][0]["id"] == "P3"
    assert all(s["confidence"] == "low" for s in h["symptoms"])
    # c2 low also opens P2 (partial: c5 is not low) -> S05 second
    assert [s["id"] for s in h["symptoms"]] == ["P3", "P2"] and h["symptoms"][1]["match"] == "partial"
    assert h["situation_ids"] == ["S06", "S05"] and h["trigger"]["sustained"] is True


def test_rule_only_no_signal():
    h = rm.build_handoff(copy.deepcopy(BASE))
    # silent sensors at a rule trigger = no model signal, not "gun fine": no S09 when the class is known
    assert [s["id"] for s in h["symptoms"]] == ["P9"] and h["situation_ids"] == ["S01"]
    assert h["summary_ko"] == "고장 유형 E01(보정 압력 도달 지연) 의심, 에러 코드 E012."


def test_time_of_day_is_not_a_symptom():
    assert rm.sensor_findings([feat("hour_cos", "mean", 2.0), feat("hour_sin", "mean", -2.0)]) == []


def test_c10_reads_as_dropout():
    fs = rm.sensor_findings([feat("c10", "mean", -1.0, reference=1.0)])
    assert fs[0]["text_ko"] == "US2 작동 허용 신호(c10) 꺼짐(평소 켜짐)"


def test_drift_needs_standstill():
    r = copy.deepcopy(BASE)
    r.update(rule_code="E029", rule_class_hint="E04")
    r["contributing_features"] = [feat("c3", "std", 3.0)]
    assert "P4" not in [s["id"] for s in rm.build_handoff(r)["symptoms"]]
    r["context"]["welds_in_window"] = 0
    h = rm.build_handoff(r)
    assert h["symptoms"][0]["id"] == "P4" and h["symptoms"][0]["confidence"] == "medium"
    assert h["situation_ids"] == ["S04"]


def test_rule_agreeing_symptom_first():
    r = copy.deepcopy(BASE)  # a strong setpoint change must not outrank the E01 compensation symptom
    r["contributing_features"] = [feat("c16", "mean", 4.0, share=0.6), feat("c5", "mean", -2.0, share=0.2)]
    h = rm.build_handoff(r)
    assert [s["id"] for s in h["symptoms"]] == ["P1", "P7"] and h["situation_ids"] == ["S01", "S10"]


def test_no_causes_or_manuals_in_handoff():
    """Scope (1)-(2): parts, checks and manual pages belong to the ontology."""
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c5", "mean", -3.1)]
    text = str(rm.build_handoff(r))
    for word in ("manual", "festo", "check_items", "keywords", "subsystem"):
        assert word not in text.lower()


def test_situation_ids_valid():
    ok = {f"S{i:02d}" for i in range(1, 11)}
    assert all(set(p["situations"]) <= ok for p in rm.SYMPTOMS + [rm.NO_SIGNAL])
    assert {v["situation_id"] for v in rm.FAULT_CLASSES.values()} == {"S01", "S02", "S03", "S04"}


def test_schema_keys_match_main():
    """The keys main.RagHandoff validates. Changing one side without the other breaks the contract."""
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c5", "mean", -3.1)]
    h = rm.build_handoff(r)
    assert set(h) == {"schema_version", "event_id", "gun_id", "detected_at", "window_start", "trigger", "summary_ko",
                      "sensor_findings", "fault_class", "symptoms", "situation_ids", "context", "detector", "caveats"}
    assert set(h["symptoms"][0]) == {"id", "name_ko", "match", "evidence", "related_classes", "situation_ids",
                                     "agrees_with_fault_class", "confidence"}
    assert set(h["sensor_findings"][0]) == {"sensor", "sensor_name", "sensor_name_ko", "group", "statistic",
                                            "direction", "deviation_z", "share", "text_ko"}
    assert set(h["fault_class"]) == {"code", "name_en", "name_ko", "terminal_code", "situation_id", "definition",
                                     "basis"}


def test_repeat_rule_is_not_a_rule_basis():
    """A model-critical event whose request also held a repeat rule trigger (not critical) must not be reported as
    rule-based: the class falls back to the latest code and the trigger carries no rule code."""
    r = copy.deepcopy(BASE)
    r.update(severity_source="model", rule_repeat=True)
    r["context"]["known_code_class_hint"] = None
    h = rm.build_handoff(r)
    assert h["fault_class"] is None and h["trigger"]["rule_code"] is None and h["trigger"]["rule_trigger_time"] is None
    r["context"]["known_code_class_hint"] = "E01"
    assert rm.build_handoff(r)["fault_class"]["basis"] == "latest_error_code"


def test_trigger_source_is_request_level():
    """A sustained run that started and ended inside one chunk: the latest window is 'none', the request 'model'."""
    r = copy.deepcopy(BASE)
    r.update(severity_source="none", critical_source="model", rule_triggered=False, rule_code=None,
             rule_class_hint=None, sustained_in_request=True)
    ho = rm.build_handoff(r)
    assert ho["trigger"]["source"] == "model" and ho["trigger"]["sustained"] is True
    del r["critical_source"]  # an older AnomalyResult without the field falls back to severity_source
    assert rm.build_handoff(r)["trigger"]["source"] == "none"


def _ids(r):
    return [s["id"] for s in rm.build_handoff(r)["symptoms"]]


def test_setpoint_guard_from_context():
    """Guide §8: 'c5 low while the c13 setpoint is unchanged'. A moved setpoint is a recipe change, not P1."""
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c5", "mean", -2.0), feat("c4", "mean", 1.5)]
    r["context"]["mean_dev"] = {"c5": -2.0, "c4": 1.5, "c13": 0.1}
    assert _ids(r)[0] == "P1"
    r["context"]["mean_dev"]["c13"] = 1.4
    assert "P1" not in _ids(r)


def test_setpoint_guard_falls_back_to_findings():
    r = copy.deepcopy(BASE)  # no mean_dev (older caller): a c13 finding still blocks P1 and shows as P7
    r["contributing_features"] = [feat("c5", "mean", -2.0), feat("c13", "mean", 1.2)]
    ids = _ids(r)
    assert "P1" not in ids and "P7" in ids
    r["contributing_features"] = [feat("c5", "mean", -2.0)]
    assert _ids(r)[0] == "P1"


def test_friction_needs_normal_c5():
    """Guide §8: c2 low + c6 high is friction (P3) only with c5 normal; with c5 low too it is pneumatic (P2)."""
    r = copy.deepcopy(BASE)
    r.update(rule_code="E028", rule_class_hint="E03")
    r["contributing_features"] = [feat("c6", "mean", 2.0), feat("c2", "mean", -1.5)]
    assert "P3" in _ids(r)
    r["contributing_features"].append(feat("c5", "mean", -1.0))
    ids = _ids(r)
    assert "P3" not in ids and "P2" in ids


def test_position_needs_steady_c15():
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c3", "mean", 2.0)]
    r["context"]["mean_dev"] = {"c3": 2.0, "c15": 0.0}
    assert "P5" in _ids(r)
    r["context"]["mean_dev"]["c15"] = 2.1  # position setpoint moved with it: a program change
    assert "P5" not in _ids(r)


def test_s02_listed_for_p5_p6():
    """Situation doc summary table: S02 (electrode broke) <- P5, P6."""
    sit = {p["id"]: p["situations"] for p in rm.SYMPTOMS}
    assert "S02" in sit["P5"] and "S02" in sit["P6"]


def test_handoff_times_are_utc_with_z():
    """FE request 4: every handoff time carries Z (main.py works in naive UTC)."""
    r = copy.deepcopy(BASE)
    r["contributing_features"] = [feat("c5", "mean", -3.1)]
    h = rm.build_handoff(r)
    assert h["schema_version"] == "1.1"
    assert h["detected_at"] == "2021-09-05T02:23:20Z" and h["window_start"] == "2021-09-05T02:22:21Z"
    assert h["trigger"]["rule_trigger_time"] == "2021-09-05T02:23:00Z"
    assert rm.utc_iso("2021-09-05T02:23:20Z") == "2021-09-05T02:23:20Z"
    assert rm.utc_iso("2021-09-05T02:23:20+00:00") == "2021-09-05T02:23:20+00:00"
    assert rm.utc_iso(None) is None


def test_caveats_carry_current_test_numbers():
    """FE note: the caveats are shown on screen - they must match models/baseline_ensemble_test_metrics.json."""
    text = " ".join(rm.CAVEATS)
    assert "0.64" not in text and "0/8" not in text and "0.71" in text and "2/8" in text
