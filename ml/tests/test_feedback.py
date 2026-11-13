"""
Unit tests of D-12 (operator verdicts in the model loop): what train.py does with --feedback (the windows of a
confirmed span are not "normal") and .py/model_feedback.py (the verdict rule, the verdict store, the pull from the
api). No model is trained here. What needs data is in test_pipeline.py, the tick is in test_model_loop.py.
Spec: docs/superpowers/specs/2026-11-12-feedback-labels-design.md
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))

import train  # noqa: E402

T = pd.Timestamp
EW = {"enabled": True, "kind": "model_ng", "threshold": 0.5, "sustain": 1, "window_s": 60, "cooldown_s": 3600}


def _w(n=600, file="E01_3", start="2021-09-01 00:00:00"):
    """n one-minute windows of one file that ends in a failure: the last window has ttf_s 0, the last hour is the
    pre-failure window. No terminal code, so the onset T is 10 min before the end (train.RULE_LEAD_S)."""
    ttf = np.arange(n)[::-1] * 60.0
    w = pd.DataFrame({"file": file, "class": "E01", "ttf_s": ttf, "error_active": 0.0, "terminal_any": 0.0,
                      "non_welding": 0.0, "warmup": 0.0}, index=pd.date_range(start, periods=n, freq="60s"))
    w["label"] = (w["ttf_s"] <= 3600).astype("float32")
    return w


def _at(w, ttf_s):
    """Position of the window with this ttf_s."""
    return int(np.flatnonzero(w["ttf_s"].to_numpy() == ttf_s)[0])


# ------------------------------------------------------------------ train.py: --feedback
def test_feedback_load(tmp_path):
    p = tmp_path / "fb.json"
    p.write_text(json.dumps({"confirmed": {"E01_3": [["2021-09-01T10:00:00", "2021-09-01T11:00:00"]]},
                             "false_alarm": {"E02_1": [["2021-09-03T04:00:00", "2021-09-03T05:00:00"]]}}),
                 encoding="utf-8")
    fb = train.load_feedback(str(p))
    assert fb["confirmed"] == {"E01_3": [(T("2021-09-01 10:00"), T("2021-09-01 11:00"))]}
    assert fb["false_alarm"] == {"E02_1": [(T("2021-09-03 04:00"), T("2021-09-03 05:00"))]}
    p.write_text(json.dumps({"confirmed": {"E01_3": [["2021-09-01T10:00:00Z", "2021-09-01T11:00:00Z"]]}}),
                 encoding="utf-8")
    fb = train.load_feedback(str(p))
    assert fb["confirmed"] == {"E01_3": [(T("2021-09-01 10:00"), T("2021-09-01 11:00"))]}, "Z = the same naive UTC"
    assert fb["false_alarm"] == {}, "a missing key is an empty one"


def test_feedback_load_floors_to_window(tmp_path):
    """A warning's time is a row time (a window's first or last second, or a second after a gap); the windows are
    named by their clock label. With the window size the span is floored to it at both ends, so the warning's own
    window is the first one of the span - unfloored, a span from 10:00:07 would leave the 10:00 window normal and
    the confirmed warning a false alarm."""
    p = tmp_path / "fb.json"
    p.write_text(json.dumps({"confirmed": {"E01_3": [["2021-09-01T10:00:07Z", "2021-09-01T11:00:07Z"]]},
                             "false_alarm": {"E01_3": [["2021-09-01T12:00:59", "2021-09-01T13:00:59"]]}}),
                 encoding="utf-8")
    fb = train.load_feedback(str(p), 60)
    assert fb["confirmed"] == {"E01_3": [(T("2021-09-01 10:00"), T("2021-09-01 11:00"))]}
    assert fb["false_alarm"] == {"E01_3": [(T("2021-09-01 12:00"), T("2021-09-01 13:00"))]}
    assert train.load_feedback(str(p), 300)["confirmed"] == fb["confirmed"]
    assert train.load_feedback(str(p))["confirmed"] == {"E01_3": [(T("2021-09-01 10:00:07"), T("2021-09-01 11:00:07"))]}
    w = _w(600, start="2021-09-01 08:00:00")  # ends at 17:59: 10:00 is far outside the hit zone
    m = train.mark_confirmed(w, fb["confirmed"]["E01_3"])
    assert m["confirmed"].sum() == 60 and m.loc[T("2021-09-01 10:00"), "confirmed"] == 1
    assert m.loc[T("2021-09-01 11:00"), "confirmed"] == 0
    s = np.zeros(len(w))
    s[w.index.get_loc(T("2021-09-01 10:00"))] = 1.0  # the alarm of the warning itself
    assert train.early_warning_metrics(w, s, EW, n_boot=10)["per_file"]["E01_3"]["false_alarms"] == 1
    assert train.early_warning_metrics(m, s, EW, n_boot=10)["per_file"]["E01_3"]["false_alarms"] == 0


def test_feedback_mark_and_mask():
    w = _w(180, start="2021-09-01 10:00:00")
    m = train.mark_confirmed(w, [(T("2021-09-01 10:30"), T("2021-09-01 11:30"))])
    assert "confirmed" not in w.columns, "the input frame is left alone"
    assert m["confirmed"].sum() == 60 and str(m["confirmed"].dtype) == "float32"
    assert m.loc[T("2021-09-01 10:30"), "confirmed"] == 1 and m.loc[T("2021-09-01 11:30"), "confirmed"] == 0
    assert m.loc[T("2021-09-01 10:29"), "confirmed"] == 0 and m.loc[T("2021-09-01 11:29"), "confirmed"] == 1
    mask = train.confirmed_mask(m)
    assert mask.dtype == bool and mask.sum() == 60 and mask[30] and not mask[90]
    assert not train.confirmed_mask(w).any() and len(train.confirmed_mask(w)) == len(w), "no column: nothing confirmed"
    assert train.mark_confirmed(w, [])["confirmed"].sum() == 0, "no span: the column is there, all 0"


def test_feedback_mark_partial_span():
    w = _w(180, start="2021-09-01 10:00:00")  # the last window starts at 12:59
    over = train.mark_confirmed(w, [(T("2021-09-01 12:30"), T("2021-09-01 13:30"))])
    assert over["confirmed"].sum() == 30, "only the windows that exist"
    outside = train.mark_confirmed(w, [(T("2021-09-02 00:00"), T("2021-09-02 01:00"))])
    assert outside["confirmed"].sum() == 0


def test_feedback_counts():
    w = _w(600)
    a, b = _at(w, 118 * 60.0), _at(w, 59 * 60.0)  # 60 windows, the last 2 inside the pre-failure hour (ttf <= 3600)
    span = (w.index[a], w.index[b] + pd.Timedelta(seconds=60))
    assert b - a + 1 == 60 and w["label"].iloc[a:b + 1].sum() == 2
    near, far = _at(w, 40 * 60.0), _at(w, 190 * 60.0)  # 30 min / 3 h before the onset T (T = ttf 600)
    one = pd.Timedelta(minutes=1)
    fb = {"confirmed": {"E01_3": [span]},
          "false_alarm": {"E01_3": [(w.index[near], w.index[near] + one), (w.index[far], w.index[far] + one),
                                    # starts 3 h before the onset and runs into the hit zone: the warning itself was
                                    # not followed by the failure within the hour, the data agrees with the operator
                                    (w.index[far], w.index[far] + 150 * one)]}}
    c = train.feedback_counts(train.mark_confirmed(w, fb["confirmed"]["E01_3"]), fb)
    assert c == {"spans": 1, "windows_excluded": 58, "windows_already_not_normal": 2, "false_alarm_conflicts": 1}
    none = train.feedback_counts(w, {"confirmed": {}, "false_alarm": {}})
    assert none == {"spans": 0, "windows_excluded": 0, "windows_already_not_normal": 0, "false_alarm_conflicts": 0}


def test_feedback_population_excludes_confirmed():
    w = _w(600)
    pop, _ = train.early_warning_population(w)
    i = _at(w, 5 * 3600.0)
    m = train.mark_confirmed(w, [(w.index[i], w.index[i] + pd.Timedelta(minutes=60))])
    pop2, _ = train.early_warning_population(m)
    assert pop.sum() - pop2.sum() == 60 and not pop2[i] and pop[i]


def test_feedback_ew_false_alarm_removed():
    w = _w(600)
    i = _at(w, 5 * 3600.0)  # 5 h before the end: far outside the hit zone
    s = np.zeros(len(w))
    s[i] = 1.0
    before = train.early_warning_metrics(w, s, EW, n_boot=10)["per_file"]["E01_3"]
    assert before["false_alarms"] == 1
    m = train.mark_confirmed(w, [(w.index[i], w.index[i] + pd.Timedelta(minutes=1))])
    after = train.early_warning_metrics(m, s, EW, n_boot=10)["per_file"]["E01_3"]
    assert after["false_alarms"] == 0
    assert before["days"] - after["days"] == pytest.approx(1 / 1440)
    same = train.early_warning_metrics(train.mark_confirmed(w, []), s, EW, n_boot=10)["per_file"]["E01_3"]
    assert same == before, "a column of zeros changes nothing"


def test_feedback_ew_hits_unchanged():
    w = _w(600)
    i = _at(w, 40 * 60.0)  # 30 min before the onset: inside the hit zone
    s = np.zeros(len(w))
    s[i] = 1.0
    before = train.early_warning_metrics(w, s, EW, n_boot=10)["per_file"]["E01_3"]
    m = train.mark_confirmed(w, [(w.index[i], w.index[i] + pd.Timedelta(minutes=1))])
    after = train.early_warning_metrics(m, s, EW, n_boot=10)["per_file"]["E01_3"]
    assert before["hit"] == 1.0 and after["hit"] == 1.0
    assert after["false_alarms"] == before["false_alarms"] == 0 and after["days"] == before["days"]


def test_feedback_evaluate_normal():
    w = _w(600)
    s = np.linspace(0.0, 1.0, len(w))
    a, b = _at(w, 118 * 60.0), _at(w, 59 * 60.0)
    m0 = train.evaluate(w, s, 0.9, 3)
    m1 = train.evaluate(train.mark_confirmed(w, [(w.index[a], w.index[b] + pd.Timedelta(seconds=60))]), s, 0.9, 3)
    assert m0["n_normal"] - m1["n_normal"] == 58, "the 2 pre-failure windows of the span were not normal before"
    assert m1["n_pre_failure"] == m0["n_pre_failure"] == 61


# ------------------------------------------------------------------ model_feedback.py: the verdict rule
import model_feedback as mf  # noqa: E402

RANGES = {"E01_3": (T("2021-08-30"), T("2021-09-06")), "E02_1": (T("2021-08-30"), T("2021-09-06"))}


def _ev(eid="e1", rep=True, gun="E01_3", ws="2021-09-01T10:00:00Z"):
    return {"event_id": eid, "representative": rep, "gun_id": gun, "window_start": ws, "detected_at": ws,
            "detector_created": "2026-11-01T18:46:47"}


def _item(outcome="resolved", abnormal=1, selected=None, events=None, status=None):
    return {"case_id": "CASE-0001", "outcome": outcome, "closed_at": "2026-11-12T08:00:00+00:00",
            "resolution_status": status, "selected_event_id": selected,
            "checks": {"abnormal": abnormal, "normal": 0, "unchecked": 0}, "events": events or [_ev()]}


def test_feedback_verdict_confirmed():
    assert mf.verdict(_item(), RANGES) == {"verdict": "confirmed", "event_id": "e1", "file": "E01_3",
                                           "start": "2021-09-01T10:00:00", "end": "2021-09-01T11:00:00"}
    assert mf.SPAN_S == 3600 and mf.VERDICTS == ("confirmed", "false_alarm", "unknown", "unmatched")


def test_feedback_verdict_table():
    assert mf.verdict(_item(abnormal=0), RANGES)["verdict"] == "unknown", "resolved without an abnormal check"
    assert mf.verdict(_item(outcome="false_alarm"), RANGES)["verdict"] == "false_alarm", "whatever the checks say"
    assert mf.verdict(_item(outcome="false_alarm", abnormal=0), RANGES)["verdict"] == "false_alarm"
    assert mf.verdict(_item(status="not_linked"), RANGES)["verdict"] == "confirmed", "the Aura link is no condition"
    assert mf.verdict(_item(outcome="handed_over"), RANGES)["verdict"] == "unknown"
    assert mf.verdict({**_item(), "checks": None}, RANGES)["verdict"] == "unknown"


def test_feedback_verdict_selected_event():
    events = [_ev(), _ev("e2", rep=False, ws="2021-09-01T14:00:00Z")]
    v = mf.verdict(_item(selected="e2", events=events), RANGES)
    assert (v["event_id"], v["start"], v["end"]) == ("e2", "2021-09-01T14:00:00", "2021-09-01T15:00:00")
    v = mf.verdict(_item(selected="gone", events=events), RANGES)
    assert (v["event_id"], v["start"]) == ("e1", "2021-09-01T10:00:00"), "an unknown id: the representative event"
    v = mf.verdict(_item(events=list(reversed(events))), RANGES)
    assert v["event_id"] == "e1", "nothing selected: the representative event, wherever it stands"


def test_feedback_verdict_starts_at_the_onset():
    """One request can hold several windows: the handoff's window_start is then the latest one, and the warning began
    at early_warning.onset_at (the last second of its window), which the api passes on. Without it: window_start."""
    ev = {**_ev(ws="2021-09-01T10:19:00Z"), "onset_at": "2021-09-01T10:00:59Z"}
    v = mf.verdict(_item(events=[ev]), RANGES)
    assert (v["verdict"], v["start"], v["end"]) == ("confirmed", "2021-09-01T10:00:59", "2021-09-01T11:00:59")
    for bad in (None, "", "soon", "NaT"):
        v = mf.verdict(_item(events=[{**ev, "onset_at": bad}]), RANGES)
        assert (v["verdict"], v["start"]) == ("confirmed", "2021-09-01T10:19:00"), f"onset_at {bad!r}: window_start"
    only = {k: x for k, x in ev.items() if k != "window_start"}
    assert mf.verdict(_item(events=[only]), RANGES)["start"] == "2021-09-01T10:00:59"
    late = {**ev, "onset_at": "2021-09-07T10:00:59Z"}
    assert mf.verdict(_item(events=[late]), RANGES)["verdict"] == "unmatched", "the onset decides the file match"


def test_feedback_verdict_unmatched():
    cases = [_item(events=[_ev(gun="test_0")]),  # no file of that name
             _item(events=[_ev(ws="2021-09-07T10:00:00Z")]),  # after the file ended
             _item(events=[{"event_id": "e1", "representative": True, "gun_id": "E01_3", "window_start": None}]),
             _item(events=[{"event_id": "e1", "representative": True, "gun_id": None,
                            "window_start": "2021-09-01T10:00:00Z"}]),
             {**_item(), "events": []}]
    for item in cases:
        v = mf.verdict(item, RANGES)
        assert v["verdict"] == "unmatched" and v["file"] is None and v["start"] is None and v["end"] is None, item


def test_feedback_time_forms():
    for ws in ("2021-09-01T10:00:00Z", "2021-09-01T10:00:00+00:00", "2021-09-01T10:00:00",
               "2021-09-01T19:00:00+09:00"):
        assert mf.verdict(_item(events=[_ev(ws=ws)]), RANGES)["start"] == "2021-09-01T10:00:00", ws
    assert mf.naive_utc("2021-09-01T19:00:00+09:00") == T("2021-09-01 10:00:00")


def test_feedback_span_setting():
    assert mf.verdict(_item(), RANGES, span_s=1800)["end"] == "2021-09-01T10:30:00"


def test_feedback_load_ranges(tmp_path):
    (tmp_path / "manifest.csv").write_text(
        "file,class,start,end,dropped\n"
        "E01_3,E01,2021-08-30 00:00:00,2021-09-06 00:00:00,\n"
        "E01_9,E01,2021-08-30 00:00:00,2021-09-06 00:00:00,no fault at the end\n", encoding="utf-8")
    assert mf.load_ranges(str(tmp_path)) == {"E01_3": (T("2021-08-30"), T("2021-09-06"))}
    with pytest.raises(ValueError):
        mf.load_ranges(str(tmp_path / "nowhere"))


# ------------------------------------------------------------------ model_feedback.py: the store and the pull
def test_feedback_store_merge(tmp_path):
    st = mf.Store(str(tmp_path / "reg"))
    assert st.path == str(tmp_path / "reg" / "feedback.json") and st.load() is None
    a = _item()
    b = _item(outcome="false_alarm", events=[_ev("e9", gun="E02_1")])
    assert st.merge([a, b], "http://api", "2026-11-12T09:00:00Z") == {"new": 2, "changed": 0, "same": 0, "skipped": 0}
    data = st.load()
    assert data["schema_version"] == 1 and set(data["cases"]) == {"e1", "e9"}, "key: the representative event id"
    assert data["cases"]["e1"] == {**a, "received_at": "2026-11-12T09:00:00Z"}
    assert st.merge([a, b], "http://api", "2026-11-12T10:00:00Z") == {"new": 0, "changed": 0, "same": 2, "skipped": 0}
    assert st.load()["cases"]["e1"]["received_at"] == "2026-11-12T09:00:00Z", "an unchanged case keeps its time"
    a2 = {**a, "outcome": "false_alarm"}
    got = st.merge([a2, {**_item(), "events": []}], "http://other", "2026-11-12T11:00:00Z")
    assert got == {"new": 0, "changed": 1, "same": 0, "skipped": 1}
    data = st.load()
    assert data["cases"]["e1"]["outcome"] == "false_alarm" and data["cases"]["e1"]["received_at"].endswith("11:00:00Z")
    assert "e9" in data["cases"], "a case the api no longer sends stays"
    assert (data["source"], data["pulled_at"]) == ("http://other", "2026-11-12T11:00:00Z")
    assert os.listdir(st.root) == ["feedback.json"], "no temp file left"


def test_feedback_store_rejects_broken_json(tmp_path):
    st = mf.Store(str(tmp_path))
    with open(st.path, "w", encoding="utf-8") as fh:
        fh.write('{"schema_version": 1,')
    with pytest.raises(mf.FeedbackError) as e:
        st.load()
    assert st.path in str(e.value) and isinstance(e.value, ValueError)
    with pytest.raises(mf.FeedbackError):
        st.merge([_item()], "http://api", "2026-11-12T09:00:00Z")
    assert open(st.path, encoding="utf-8").read() == '{"schema_version": 1,', "the file is left as it was"
    with open(st.path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 2, "cases": {}}, fh)
    with pytest.raises(mf.FeedbackError):
        st.load()


def test_feedback_collect(tmp_path):
    items = [_item(), _item(outcome="false_alarm", events=[_ev("e2", ws="2021-09-01T12:00:00Z")]),
             _item(abnormal=0, events=[_ev("e3", gun="E02_1")]), _item(events=[_ev("e4", gun="test_0")])]
    data = {"schema_version": 1, "cases": {it["events"][0]["event_id"]: it for it in items}}
    got = mf.collect(data, RANGES, ["E01_3"])
    assert got == {"spans": {"confirmed": {"E01_3": [["2021-09-01T10:00:00", "2021-09-01T11:00:00"]]},
                             "false_alarm": {"E01_3": [["2021-09-01T12:00:00", "2021-09-01T13:00:00"]]}},
                   "counts": {"confirmed": 1, "false_alarm": 1, "unknown": 0, "unmatched": 1}, "event_ids": ["e1"]}
    both = mf.collect(data, RANGES, ["E01_3", "E02_1"])
    assert both["counts"] == {"confirmed": 1, "false_alarm": 1, "unknown": 1, "unmatched": 1}
    none = mf.collect(data, RANGES, ["E09_9"])
    assert none == {"spans": {"confirmed": {}, "false_alarm": {}},
                    "counts": {"confirmed": 0, "false_alarm": 0, "unknown": 0, "unmatched": 1}, "event_ids": []}
    assert mf.collect(None, RANGES, ["E01_3"]) is None, "no store: no feedback at all"
    # the function the loop gets: the store and the manifest are read at every call
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.csv").write_text(
        "file,start,end,dropped\nE01_3,2021-08-30 00:00:00,2021-09-06 00:00:00,\n", encoding="utf-8")
    fn = mf.make_feedback_fn(str(tmp_path / "reg"), str(tmp_path / "data"))
    assert fn(["E01_3"]) is None
    mf.Store(str(tmp_path / "reg")).merge(items, "http://api", "2026-11-12T09:00:00Z")
    assert fn(["E01_3"])["counts"] == {"confirmed": 1, "false_alarm": 1, "unknown": 0, "unmatched": 2}
    assert mf.make_feedback_fn(str(tmp_path / "reg"), str(tmp_path / "data"), 1800)(["E01_3"])["spans"]["confirmed"] \
        == {"E01_3": [["2021-09-01T10:00:00", "2021-09-01T10:30:00"]]}


def test_feedback_fetch(monkeypatch):
    import httpx

    class Reply:
        def __init__(self, code, body):
            self.status_code, self.body = code, body

        def json(self):
            if self.body is None:
                raise ValueError("not JSON")
            return self.body

    calls = []

    def get(reply):
        def fake(url, timeout=None):
            calls.append((url, timeout))
            if isinstance(reply, Exception):
                raise reply
            return reply
        monkeypatch.setattr(httpx, "get", fake)

    get(Reply(200, {"schema_version": "1.0", "items": [_item()]}))
    assert mf.fetch("http://api:9000/") == [_item()]
    assert calls == [("http://api:9000/api/ml/feedback", 30.0)]
    for bad in (httpx.ConnectError("refused"), Reply(500, {}), Reply(200, {"schema_version": "2.0", "items": []}),
                Reply(200, {"schema_version": "1.0"}), Reply(200, None)):
        get(bad)
        with pytest.raises(mf.FeedbackError) as e:
            mf.fetch("http://api:9000")
        assert "http://api:9000/api/ml/feedback" in str(e.value)


# ------------------------------------------------------------------ minors of the final review (2026-11-13)
def test_feedback_counts_spans_of_loaded_files_only():
    """train.py called by hand with verdicts on more files than it loads: "spans" says what was used."""
    w = _w(600)
    i = _at(w, 5 * 3600.0)
    span = (w.index[i], w.index[i] + pd.Timedelta(minutes=60))
    fb = {"confirmed": {"E01_3": [span], "E09_9": [span, span]}, "false_alarm": {}}
    c = train.feedback_counts(train.mark_confirmed(w, fb["confirmed"]["E01_3"]), fb)
    assert c["spans"] == 1 and c["windows_excluded"] == 60


def test_feedback_bundle_args_without_verdicts():
    """Without --feedback the bundle's args are those from before D-12: no "feedback": null."""
    import argparse

    assert train.bundle_args(argparse.Namespace(window=60, feedback=None)) == {"window": 60}
    assert train.bundle_args(argparse.Namespace(window=60, feedback="fb.json")) == {"window": 60, "feedback": "fb.json"}


def test_feedback_verdict_odd_checks():
    """A count that is no count (a hand-edited store) is no abnormal check: the case reads "unknown", nothing
    raises."""
    for checks in ({"abnormal": "many"}, {"abnormal": None}, {"abnormal": [1]}, ["abnormal"], "abnormal", 3):
        assert mf.verdict({**_item(), "checks": checks}, RANGES)["verdict"] == "unknown", checks
    assert mf.verdict({**_item(), "checks": {"abnormal": "2"}}, RANGES)["verdict"] == "confirmed"
    assert mf.verdict({**_item(), "checks": {"abnormal": 1.0}}, RANGES)["verdict"] == "confirmed"


def test_feedback_store_rejects_a_case_that_is_no_object(tmp_path):
    st = mf.Store(str(tmp_path))
    with open(st.path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "cases": {"e1": _item(), "e2": "oops"}}, fh)
    with pytest.raises(mf.FeedbackError) as e:
        st.load()
    assert st.path in str(e.value) and "e2" in str(e.value)


def test_feedback_store_merge_retries_the_rename(tmp_path, monkeypatch):
    """Windows: a reader that holds feedback.json makes the rename fail for a moment (as Registry.save)."""
    st = mf.Store(str(tmp_path / "reg"))
    st.merge([_item()], "http://api", "2026-11-12T09:00:00Z")
    real, calls = os.replace, []

    def busy(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)

    def never(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(os, "replace", busy)
    got = st.merge([_item(outcome="false_alarm", events=[_ev("e9")])], "http://api", "2026-11-12T10:00:00Z")
    assert got["new"] == 1 and len(calls) == 3 and set(st.load()["cases"]) == {"e1", "e9"}
    monkeypatch.setattr(os, "replace", never)
    with pytest.raises(PermissionError):
        st.merge([_item(events=[_ev("e7")])], "http://api", "2026-11-12T11:00:00Z")
    assert set(st.load()["cases"]) == {"e1", "e9"} and os.listdir(st.root) == ["feedback.json"], "left as it was"
