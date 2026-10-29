"""
Unit tests of .py/sensor_features.py (per-minute sensor document, sensor-features-v4, D-6).
Pure Python + jsonschema: no model, no server, < 1 s. The API side (/predict field, GET, push) is in test_pipeline.py.
"""
import json
import os
import sys

import jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))

import sensor_features as sf

with open(os.path.join(ROOT, "docs", "sensor_features.schema.json"), encoding="utf-8") as fh:
    SCHEMA = json.load(fh)

# the 09:03 minute of test_3 (results/D-6 raw check): idle, 9 welds of different programs, idle
MINUTE = [
    {"c2": 64, "c4": 0, "c10": "on", "c13": 14.99, "c14": 2700, "c15": 100.0, "c16": 8.7, "c18": 0},
    {"c2": 2569, "c4": 1, "c10": "on", "c13": 9.03, "c14": 2600, "c15": 100.0, "c16": 8.7, "c18": 1},
    {"c2": 76, "c4": 0, "c10": "on", "c13": 14.99, "c14": 2600, "c15": 50.0, "c16": 8.7, "c18": 0},
    {"c2": 2500, "c4": 1, "c10": "on", "c13": 8.96, "c14": 2500, "c15": 50.0, "c16": 2.2, "c18": 1},
]


def z_all(v=0.0):
    return {f: v for f in sf.MEASURED_FEATURES + tuple(f"{c}_{s}" for c in sf.SETPOINTS for s in ("mean", "std"))}


def test_groups_cover_the_data_columns_once():
    sensors = set(sf.MEASURED) | set(sf.STATES) | set(sf.SETPOINTS) | set(sf.DROPPED)
    assert sensors == {f"c{i}" for i in range(1, 20)}, "every column decided exactly once"
    assert len(sensors) == len(sf.MEASURED) + len(sf.STATES) + len(sf.SETPOINTS) + len(sf.DROPPED)
    assert {"c4", "c10", "c18"} == set(sf.STATES) and not {"c4", "c10"} & set(sf.MEASURED)


def test_raw_summary_mean_std_and_on_share():
    raw = sf.raw_summary(MINUTE)
    m, s = raw["stats"]["c2"]
    assert round(m, 2) == 1302.25 and round(s, 1) == round((sum((v - m) ** 2 for v in (64, 2569, 76, 2500)) / 3) ** .5, 1)
    assert raw["on"] == {"c4": 0.5, "c10": 1.0, "c18": 0.5}
    assert raw["stats"]["c14"][0] == 2600 and raw["n"] == 4
    assert "c7" not in raw["stats"] and "c17" not in raw["stats"]
    assert sf.raw_summary([{"c10": "off"}, {"c10": True}, {"c10": 0.0}])["on"]["c10"] == 1 / 3


def test_all_measurements_sorted_by_abs_z():
    z = z_all()
    z.update({"c5_mean": -2.7, "c2_std": 2.0, "c3_mean": -1.1, "c6_std": 1.1, "c13_mean": 9.0})
    z.pop("c1_std")
    doc = sf.build_doc("G1", "2021-09-03 09:03:59", z, sf.raw_summary(MINUTE), "E003", "gun", "m@1")
    feats = [m["feature"] for m in doc["measurements"]]
    assert len(feats) == 10 and set(feats) == set(sf.MEASURED_FEATURES), "all of them, every minute"
    assert feats[:4] == ["c5_mean", "c2_std", "c3_mean", "c6_std"], "|z| order, ties in catalog order"
    assert feats[-1] == "c1_std" and doc["measurements"][-1]["z"] is None, "a missing z goes last"
    c2 = next(m for m in doc["measurements"] if m["feature"] == "c2_std")
    assert c2["value"] == round(sf.raw_summary(MINUTE)["stats"]["c2"][1], 3), "raw std of the CSV numbers"
    assert "c13_mean" not in feats, "setpoints never ranked with measurements"


def test_states_and_setpoints():
    doc = sf.build_doc("G1", "2021-09-03T09:03:59Z", z_all(0.5), sf.raw_summary(MINUTE), None, "gun")
    assert doc["states"] == [{"signal": "c4", "on_share": 0.5}, {"signal": "c10", "on_share": 1.0},
                             {"signal": "c18", "on_share": 0.5}]
    c14 = next(s for s in doc["setpoints"] if s["setpoint"] == "c14")
    assert c14["mean"] == 2600 and c14["std"] > 0 and c14["mean_z"] == 0.5 and c14["std_z"] == 0.5
    assert [s["setpoint"] for s in doc["setpoints"]] == list(sf.SETPOINTS)


def test_code_class_and_schema():
    doc = sf.build_doc("test_0", "2021-09-11T02:55:59", z_all(0.2), sf.raw_summary(MINUTE), "E012", "warming_up",
                       "m@2026")
    jsonschema.validate(doc, SCHEMA)
    assert doc["error_code"] == "E012" and doc["fault_class"] == "E01" and doc["observed_at"] == "2021-09-11T02:55:59Z"
    assert sf.build_doc("G", "2021-09-11T02:55:59", z_all(), {}, "E003", "gun")["fault_class"] is None
    assert sf.last_code(["0", "E003", "E012", "0"]) == "E012" and sf.last_code(["0"]) is None
    text = json.dumps(doc)
    assert "situation" not in text and "symptom" not in text and '"S0' not in text


def test_empty_minute_still_valid():
    doc = sf.build_doc("G", "2021-09-11T02:55:59", {}, {}, None, "gun")
    jsonschema.validate(doc, SCHEMA)
    assert all(m["z"] is None and m["value"] is None for m in doc["measurements"])


def test_utc_iso_forms():
    assert sf.utc_iso("2021-09-11 02:55:59.000") == "2021-09-11T02:55:59Z"
    assert sf.utc_iso("2021-09-11T02:55:59+00:00") == "2021-09-11T02:55:59Z"
