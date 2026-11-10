"""
D-9 ML feedback loop (docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md, 10.1) on synthetic tables: the
evaluator's labels, metrics and verdicts (.py/feedback_eval.py) and the loop's readers and report (.py/feedback_loop.py).
Never touches the real data folders; the lab check reads results/B-19/cache when it is there and is skipped otherwise.
"""
import importlib.util
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import feedback_eval as fe  # noqa: E402
import feedback_loop as fl  # noqa: E402

DAY_MIN = 1440
START = pd.Timestamp("2021-01-01 00:00")


def _wins(gun, start, minutes, **cols):
    """1-minute windows of one gun, in the evaluator's window-table shape."""
    d = {"gun": gun, "window_start": pd.date_range(start, periods=minutes, freq="60s"), "score": 0.0,
         "warmup": False, "non_welding_share": 0.0, "error_active": False}
    d.update(cols)
    return pd.DataFrame(d)


def _fails(*rows):
    return pd.DataFrame({"gun": [g for g, _ in rows], "T": pd.to_datetime([t for _, t in rows])})


NO_FAILS = pd.DataFrame({"gun": pd.Series([], dtype=object), "T": pd.Series([], dtype="datetime64[ns]")})


# ------------------------------------------------------------------ labels and metrics (Task 3)

def test_frame_matches_b18():
    """One failure 10 min before the end of every gun's record (the lab): the generalised frame gives B-18's labels
    exactly - kept windows, monitored days, the real and placebo zones, false alarms and hits."""
    sys.path.insert(0, os.path.join(ROOT, "results", "B-18"))
    import early_warning as EW

    rng = np.random.default_rng(7)
    parts, faults = [], {}
    for gun, start in (("g_c", "2021-03-01 00:00"), ("g_a", "2021-01-01 00:00"), ("g_b", "2021-02-01 06:00")):
        n = 3 * DAY_MIN
        w = _wins(gun, start, n, score=rng.random(n), non_welding_share=rng.random(n), warmup=np.arange(n) < 120)
        if gun == "g_b":
            w = w.drop(index=range(2000, 2030))  # a 30-min gap
        parts.append(w)
        faults[gun] = pd.Timestamp(start) + pd.Timedelta(minutes=n - 10, seconds=37)
    W = pd.concat(parts, ignore_index=True)
    fr = fe.build_frame(W, _fails(*faults.items()))

    b = W.rename(columns={"gun": "file", "window_start": "time", "score": "score_ens", "non_welding_share": "non_welding"})
    b["lead"] = (b["file"].map(faults) - (b["time"] + pd.Timedelta(seconds=60))).dt.total_seconds()
    b["cls"], b["fold"] = "E01", 0
    b = b.sort_values(["file", "time"]).reset_index(drop=True)
    bf = EW.Frame(b[b["lead"] > 0].reset_index(drop=True))

    assert list(fr.guns) == list(bf.guns) == ["g_a", "g_b", "g_c"]
    assert (fr.w["window_start"].to_numpy() == bf.w["time"].to_numpy()).all()
    assert np.array_equal(fr.days, bf.days)
    for name in ["real"] + [f"p{k}" for k in range(1, 7)]:
        z, valid = bf.zones[name]
        assert np.array_equal(np.sort(np.concatenate(fr.zones[name])), np.flatnonzero(z)), name
        assert np.array_equal(fr.valid[name], valid), name
    thr = float(np.quantile(W["score"], 0.9))
    for kind in ("model_ng", "model"):
        rr = fe.evaluate(fr, fe.condition(fr, kind, thr))
        fa, hits = bf.evaluate(bf.condition(kind, thr), 1)
        assert np.array_equal(rr.fa, fa), kind
        assert np.array_equal(rr.real, hits["real"], equal_nan=True), kind
        chance = np.nanmean(np.vstack([hits[f"p{k}"] for k in range(1, 7)]), axis=0)
        assert np.array_equal(rr.chance, chance, equal_nan=True), kind


def test_labels_hit_miss_undetermined():
    n = 2 * DAY_MIN
    T = START + pd.Timedelta(minutes=n - 5)
    zone = np.arange(n - 66, n - 15)  # window i ends at i + 1 min: lead 10-60 min -> 51 windows
    hit = _wins("g_hit", START, n, score=np.where(np.arange(n) == n - 36, 1.0, 0.0))  # alarm 30 min before T
    miss = _wins("g_miss", START, n)
    und = _wins("g_und", START, n).drop(index=zone[:27])  # 24 usable windows left in the zone
    fr = fe.build_frame(pd.concat([hit, miss, und], ignore_index=True),
                        _fails(("g_hit", T), ("g_miss", T), ("g_und", T)))
    rr = fe.evaluate(fr, fe.condition(fr, "model_ng", 0.5))
    assert [len(z) for z in fr.zones["real"]] == [51, 51, 24]
    assert np.array_equal(rr.real, [1.0, 0.0, np.nan], equal_nan=True)


def test_post_failure_exclusion_and_lead_to_next():
    T1, T2 = START + pd.Timedelta(hours=36), START + pd.Timedelta(hours=84)
    n = 84 * 60 - 5  # the record ends 5 min before T2
    i30, i90 = 36 * 60 + 30, 36 * 60 + 90  # windows starting T1 + 30 min and T1 + 90 min
    score = np.zeros(n)
    score[[i30, i90]] = 1.0
    fr = fe.build_frame(_wins("g", START, n, score=score), _fails(("g", T1), ("g", T2)))
    assert fr.excluded_windows == 60
    assert not (fr.w["window_start"] == T1 + pd.Timedelta(minutes=30)).any()
    after = fr.w.index[fr.w["window_start"] == T1 + pd.Timedelta(minutes=59)][0]
    assert fr.w.loc[after, "nxt"] == 1 and fr.w.loc[after, "lead"] == 48 * 3600 - 3600
    rr = fe.evaluate(fr, fe.condition(fr, "model_ng", 0.5))
    assert rr.fa.tolist() == [1] and rr.pending.tolist() == [0]
    assert rr.real.tolist() == [0.0, 0.0]


def test_pending_and_tail():
    n = DAY_MIN
    score = np.zeros(n)
    score[[n - 121, n - 31]] = 1.0  # onsets ending 120 and 30 min before the record end
    fr = fe.build_frame(_wins("g", START, n, score=score), NO_FAILS)
    rr = fe.evaluate(fr, fe.condition(fr, "model_ng", 0.5))
    assert rr.fa.tolist() == [1] and rr.pending.tolist() == [1]
    assert rr.days.tolist() == [(n - 60) / 1440]


def test_placebo_skips_overlap():
    T1, T2 = START + pd.Timedelta(days=2), START + pd.Timedelta(days=3)
    fr = fe.build_frame(_wins("g", START, 3 * DAY_MIN - 5), _fails(("g", T1), ("g", T2)))
    assert fr.valid["p1"].tolist() == [True, False]  # T2 - 1 day = T1: that zone is T1's last hour
    assert fr.valid["p2"].tolist() == [False, True]  # T1 - 2 days is before the record


def _calibrate():
    path = os.path.join(ROOT, "results", "D-9", "calibrate.py")
    spec = importlib.util.spec_from_file_location("d9_calibrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule(real, chance, fa, days):
    n = len(days)
    return fe.RuleResult(real=real, chance=chance, fa=fa, pending=np.zeros(n), acted=np.zeros(n), days=days,
                         onset=np.zeros(0, bool))


def test_summary_and_paired_match_calibration():
    cal = _calibrate()
    n = 12

    def run(seed):
        r = np.random.default_rng(seed)
        real = r.integers(0, 2, n).astype(float)
        real[[2, 7]] = np.nan
        return {"real": real, "chance": r.random(n), "fa": r.integers(0, 30, n).astype(float), "days": r.random(n) + 1}

    a, b = run(1), run(2)
    ev_a = fe.pool([_rule(**a)], np.arange(n), n)
    ev_b = fe.pool([_rule(**b)], np.arange(n), n)
    s, ref = fe.summary(ev_a), cal.summary([a])
    assert {k: s[k] for k in ref} == ref
    assert s["failures"] == 10 and s["guns"] == 10
    d, ci = cal.paired([a], [b])
    p = fe.paired(ev_a, ev_b)
    assert {k: p[k] for k in ("D", "ci")} == {"D": d, "ci": ci}
    one = fe.Evidence(real=np.array([1.0, 0.0]), chance=np.array([0.5, 0.0]), fail_gun=np.array([0, 0]), n_guns=1,
                      fa_per_day=1.0)
    s1 = fe.summary(one)
    assert s1["excess"] == 0.25 and s1["ci"] == [0.25, 0.25]


def test_condition_and_population():
    score = np.full(100, 0.2)
    score[[5, 20, 30]] = 0.9
    nws = np.zeros(100)
    nws[20] = 0.6
    fr = fe.build_frame(_wins("g", START, 100, score=score, non_welding_share=nws, warmup=np.arange(100) < 10),
                        NO_FAILS)
    assert np.flatnonzero(fe.condition(fr, "model_ng", 0.5)).tolist() == [20, 30]  # window 5 is in the warm-up
    assert np.flatnonzero(fe.condition(fr, "model", 0.5)).tolist() == [30]
    assert np.isnan(fe.quantile_threshold(fr, 0.5)), "every window is within a day of the record end"

    n = 3 * DAY_MIN
    s1 = np.full(n, 0.2)
    s1[50] = 0.9
    s1[2869:] = 5.0  # lead <= 1 day: outside the population
    err = np.zeros(n, bool)
    err[100] = True
    g1 = _wins("g1", START, n, score=s1, warmup=np.arange(n) < 10, error_active=err)
    g2 = _wins("g2", START, 2 * DAY_MIN, score=np.r_[np.full(DAY_MIN, 0.3), np.full(DAY_MIN, 7.0)])  # no failure
    fr = fe.build_frame(pd.concat([g1, g2], ignore_index=True), _fails(("g1", START + pd.Timedelta(minutes=n - 10))))
    pop = fe.normal_population(fr)
    assert np.bincount(fr.gun, pop).tolist() == [2869 - 10 - 1, DAY_MIN]
    assert fe.quantile_threshold(fr, 1.0) == 0.9


def test_constants_match_calibration():
    with open(os.path.join(ROOT, "results", "D-9", "calibrate.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    assert round(c["noise"]["margin"], 3) == fe.MARGIN and round(c["noise"]["cheaper"], 3) == fe.CHEAPER
    assert c["quantiles"] == {"q_minus": fe.Q_MINUS, "q_plus": fe.Q_PLUS}
    assert c["rule_v2"]["min_failures"] == fe.MIN_FAILURES
    assert (c["rule_v2"]["profile"]["k"], c["rule_v2"]["profile"]["d"]) == (fe.PROFILE_CASES, fe.PROFILE_DAYS)


# ------------------------------------------------------------------ verdicts, profile, human labels (Task 4)

def _s(fa, failures=10, guns=5):
    return {"fa_per_day": fa, "failures": failures, "guns": guns}


@pytest.mark.parametrize("ch, ci, fa_champion, fa_reference, field, want", [
    (_s(20.0), [0.01, 0.2], 16.4, 16.4, False, "pass"),                   # catches more, within the cap
    (_s(25.0), [0.01, 0.2], 16.4, 16.4, False, "fail"),                   # above 24 per gun-day
    (_s(13.0), [-0.10, 0.01], 16.4, 16.4, False, "pass"),                 # cheaper, no evidence of lost catches
    (_s(13.0), [-0.10, -0.001], 16.4, 16.4, False, "fail"),               # catches significantly fewer
    (_s(13.0), [-0.120, 0.01], 16.4, 16.4, False, "fail"),                # the margin must be exceeded
    (_s(13.9), [-0.10, 0.01], 16.4, 16.4, False, "fail"),                 # 13.9 > 0.843 x 16.4
    (_s(12.0), [-0.10, 0.01], 12.0, 20.0, False, "fail"),                 # not fewer than the current rule
    (_s(13.0, failures=9), [-0.10, 0.01], 16.4, 16.4, True, "insufficient"),
    (_s(13.0, failures=10, guns=4), [-0.10, 0.01], 16.4, 16.4, True, "insufficient"),
    (_s(13.0, failures=10, guns=5), [-0.10, 0.01], 16.4, 16.4, True, "pass"),
    (_s(13.0, failures=3, guns=1), [-0.10, 0.01], 16.4, 16.4, False, "pass"),  # the lab ignores the counts
])
def test_source_verdict_paths(ch, ci, fa_champion, fa_reference, field, want):
    d = {"D": sum(ci) / 2, "ci": ci}
    assert fe.source_verdict(ch, d, d, fa_champion, fa_reference, field) == want


def test_combine_table():
    review_diff, review_nolab = "검토 필요(실험실과 현장이 다름)", "검토 필요(실험실 확인 없음)"
    want = {
        ("pass", "pass"): ("propose", "반영 제안"),
        ("pass", "insufficient"): ("hold", "보류(현장 증거 쌓는 중)"),
        ("pass", "fail"): ("review", review_diff),
        ("fail", "pass"): ("review", review_diff),
        ("fail", "insufficient"): ("reject", "반영 안 함"),
        ("fail", "fail"): ("reject", "반영 안 함"),
        ("na", "pass"): ("review", review_nolab),
        ("na", "insufficient"): ("hold", "보류"),
        ("na", "fail"): ("reject", "반영 안 함"),
        ("pass", "na"): ("hold", "보류"),
        ("fail", "na"): ("reject", "반영 안 함"),
        ("na", "na"): ("not_evaluated", "평가 안 함"),
    }
    assert {k: fe.combine(*k) for k in want} == want


def _cc(case_id, gun, cls, outcome, t):
    return fe.CriticalCase(case_id=case_id, gun=gun, cls=cls, outcome=outcome, detected_at=pd.Timestamp(t))


def test_profile_boundaries():
    d1, d1b, d2 = "2021-09-04 10:00", "2021-09-04 23:59", "2021-09-05 00:01"
    cases = [
        _cc("A1", "g1", "E04", "false_alarm", d1),
        _cc("B1", "g2", "E04", "false_alarm", d1), _cc("B2", "g2", "E04", "false_alarm", d1b),
        _cc("C2", "g3", "E04", "false_alarm", d2), _cc("C1", "g3", "E04", "false_alarm", d1b),
        _cc("D1", "g4", "E04", "false_alarm", d1b), _cc("D2", "g4", "E04", "false_alarm", d2),
        _cc("D3", "g4", "E04", "real", d2),
        _cc("E1", "g5", "E04", "false_alarm", d1b), _cc("E2", "g5", "E04", "false_alarm", d2),
        _cc("F1", "g6", "E04", "false_alarm", d1b), _cc("F2", "g6", "E04", "false_alarm", d2),
        _cc("G3", "g7", "E04", "false_alarm", d1b), _cc("G4", "g7", "E04", "false_alarm", d2),
        _cc("G1", "g7", "E03", "false_alarm", d1b), _cc("G2", "g7", "E03", "false_alarm", d2),
        _cc("H1", "g8", None, "false_alarm", d1b), _cc("H2", "g8", None, "false_alarm", d2),
        _cc("I1", "g9", "E04", "unknown", d1b), _cc("I2", "g9", "E04", "unknown", d2),
    ]
    out = fe.profile_proposals(cases, {"g5": ["E04"], "g6": ["E01", "E02"]})
    risk = "이 유형의 진짜 고장이 나도 경고로만 남고 케이스가 생기지 않음(C-3, `main.py:659-663`)."
    assert out["watching"] == [
        {"gun": "g1", "cls": "E04", "cases": 1, "days": 1, "text": "지켜보는 중(2건 중 1건, 2일 중 1일)"},
        {"gun": "g2", "cls": "E04", "cases": 2, "days": 1, "text": "지켜보는 중(2건 중 2건, 2일 중 1일)"}]
    assert out["proposals"] == [
        {"gun": "g3", "exclude": ["E04"], "body": {"expected_fault_classes": ["E01", "E02", "E03"]},
         "cases": [{"case_id": "C1", "date": "2021-09-04"}, {"case_id": "C2", "date": "2021-09-05"}], "risk_ko": risk},
        {"gun": "g7", "exclude": ["E03", "E04"], "body": {"expected_fault_classes": ["E01", "E02"]},
         "cases": [{"case_id": "G1", "date": "2021-09-04"}, {"case_id": "G2", "date": "2021-09-05"},
                   {"case_id": "G3", "date": "2021-09-04"}, {"case_id": "G4", "date": "2021-09-05"}], "risk_ko": risk}]
    assert out["review"] == [{"gun": "g5", "exclude": ["E04"], "text": "검토 필요"}]


def test_reason_of():
    assert [fe.reason_of(r) for r in fe.REASONS] == list(fe.REASONS)
    assert fe.reason_of("아무 글") == "기타"
    assert [fe.reason_of(r) for r in (None, "", "  ")] == ["사유 없음"] * 3


def test_ew_human_and_acted():
    n = 2 * DAY_MIN
    T = START + pd.Timedelta(minutes=n - 5)
    iA, iB, iC = 300, n - 36, 900  # far (false alarm), 30 min before T (hit side), far (acted on)
    ew = np.zeros(n, bool)
    ew[[iA, iB, iC]] = True
    fr = fe.build_frame(_wins("g", START, n, early_warning=ew), _fails(("g", T)))
    at = {k: START + pd.Timedelta(minutes=i) for k, i in (("A", iA), ("B", iB), ("C", iC))}
    links = [fe.EwLink("CASE-A", "false_alarm", "설비 정지 · 전원 꺼짐", None, "g", at["A"]),
             fe.EwLink("CASE-B", "false_alarm", None, None, "g", at["B"]),
             fe.EwLink("CASE-C", "resolved", None, None, "g", at["C"]),
             fe.EwLink("CASE-O", "open", None, None, "g", at["A"]),  # no human label yet
             fe.EwLink("CASE-X", "false_alarm", None, None, "g", START - pd.Timedelta(days=9))]  # not in the log
    c = fr.w["early_warning"].to_numpy()
    rr = fe.evaluate(fr, c, fe.acted_mask(fr, c, links))
    assert rr.fa.tolist() == [1] and rr.acted.tolist() == [1] and rr.pending.tolist() == [0]
    i = {k: fe.onset_index(fr, "g", t) for k, t in at.items()}
    assert [fe.onset_label(fr, rr, i[k]) for k in "ABC"] == ["false_alarm", "hit_side", "false_alarm"]
    assert fe.onset_label(fr, rr, i["A"] + 1) == "not_onset" and fe.onset_index(fr, "nope", at["A"]) is None
    h = fe.ew_human(fr, rr, links)
    assert (h["false_alarm"], h["pending"], h["acted"], h["unmatched"]) == (1, 0, 1, 1)
    assert h["reasons"] == {"설비 정지 · 전원 꺼짐": 1}
    assert [d["case_id"] for d in h["disagreements"]] == ["CASE-B"]


def test_post_failure_bins():
    T = START + pd.Timedelta(hours=10)
    n = 12 * 60  # windows up to 2 h after T
    ends = START + pd.to_timedelta(np.arange(1, n + 1), unit="min")
    ew = np.asarray((ends > T) & (ends <= T + pd.Timedelta(minutes=30)))
    bins = fe.post_failure(_wins("g", START, n, early_warning=ew), _fails(("g", T)))
    assert [(b["from_min"], b["to_min"]) for b in bins] == [(m, m + 10) for m in range(0, 120, 10)]
    assert [b["windows"] for b in bins] == [10] * 12
    assert [b["ew_share"] for b in bins] == [1.0, 1.0, 1.0] + [0.0] * 9


def _field_table(seed=5):
    """6 guns x 3 days, a failure 1.5 days in and one 5 min before the end; the served early warning = score >= q0.95."""
    rng = np.random.default_rng(seed)
    parts, rows = [], []
    for g in range(6):
        gun, n = f"g{g}", 3 * DAY_MIN
        parts.append(_wins(gun, START, n, score=rng.random(n), non_welding_share=rng.random(n),
                           warmup=np.arange(n) < 60))
        rows += [(gun, START + pd.Timedelta(minutes=n // 2)), (gun, START + pd.Timedelta(minutes=n - 5))]
    W = pd.concat(parts, ignore_index=True)
    thr = float(np.quantile(W["score"], 0.95))
    W["early_warning"] = (W["score"] >= thr) & ~W["warmup"]
    W["ew_threshold"] = thr
    return W, _fails(*rows), thr


def test_field_evaluation():
    W, F, thr = _field_table()
    entry = {"early_warning": {"rule": "model_ng_k1", "threshold": thr, "quantile": 0.975}}
    out = fe.field_evaluation(W, F, entry, [])
    assert out["reference_basis"] == "ew_threshold" and out["mismatch"] == 0 and out["champion_threshold"] == thr
    assert out["reference"] == out["champion"] and (out["failures"], out["guns"]) == (12, 6)
    assert set(out["challengers"]) == {"GATE", "Q+", "Q-", "Q0.975"}
    assert all(v["verdict"] in ("pass", "fail") for v in out["challengers"].values())
    assert out["labels"]["failures"]["hit"] + out["labels"]["failures"]["miss"] == 12
    assert len(out["post_failure"]) == 12
    moved = fe.field_evaluation(W, F, {"early_warning": {**entry["early_warning"], "quantile": 0.98}}, [])
    assert moved["reference_basis"] == "field_q0.975"
    W2 = W.copy()
    W2.loc[1000, "early_warning"] = not W2.loc[1000, "early_warning"]
    flipped = fe.field_evaluation(W2, F, entry, [])
    assert flipped["mismatch"] == 1 and any("다른 창 1개" in x for x in flipped["warnings"])
    assert fe.field_evaluation(W.iloc[:0], F, entry, []) is None


def test_lab_evaluation_plumbing():
    runs = []
    for name, seed in (("r1", 1), ("r2", 2)):
        W = _field_table(seed)[0].drop(columns=["early_warning", "ew_threshold"])
        runs.append((name, W, _fails(*[(f"g{g}", START + pd.Timedelta(minutes=3 * DAY_MIN - 5)) for g in range(6)])))
    out = fe.lab_evaluation(iter(runs))
    assert out["runs"] == ["r1", "r2"] and out["challengers"]["Q0.975"] == {"verdict": "na"}
    assert all(out["challengers"][k]["verdict"] in ("pass", "fail") for k in ("GATE", "Q+", "Q-"))
    fa = days = 0.0
    for name, W, F in runs:
        fr = fe.build_frame(W, F)
        thr = fe.quantile_threshold(fr, 0.975)
        assert set(out["thresholds"][name]) == {"champion", "Q+", "Q-"} and out["thresholds"][name]["champion"] == thr
        rr = fe.evaluate(fr, fe.condition(fr, "model_ng", thr))
        fa, days = fa + rr.fa.sum(), days + rr.days.sum()
    assert out["champion"]["fa_per_day"] == fa / days and out["reference"] == out["champion"]


# ------------------------------------------------------------------ the loop's inputs (Task 5)

def _line(start, **kw):
    """One window-log line (spec 5.4)."""
    t = pd.Timestamp(start.rstrip("Z"))
    d = {"window_start": start, "window_end": (t + pd.Timedelta(seconds=59)).isoformat() + "Z", "model": "m@1",
         "score": 0.1, "threshold": 0.9, "alarm": False, "held": False, "sustained": False, "critical": False,
         "early_warning": False, "early_warning_onset": False, "ew_threshold": 0.95, "warmup": False,
         "gun_norm": "gun", "non_welding_share": 0.0, "error_codes": [], "error_code": None, "vec": []}
    d.update(kw)
    return d


def _state(tmp_path, lines_by_dir=None, outbox=(), profiles=None, models=None):
    """A server state dir: windows/<dir>/<day>.jsonl, windows/models.json, outbox/<id>.json, guns/<gun>.json."""
    st = tmp_path / "state"
    st.mkdir(parents=True, exist_ok=True)
    for folder, lines in (lines_by_dir or {}).items():
        d = st / "windows" / folder
        d.mkdir(parents=True, exist_ok=True)
        by_day = {}
        for x in lines:
            by_day.setdefault(x["window_start"][:10], []).append(json.dumps(x, ensure_ascii=False))
        for day, ls in by_day.items():
            (d / f"{day}.jsonl").write_text("\n".join(ls) + "\n", encoding="utf-8")
    if models is not None:
        (st / "windows").mkdir(exist_ok=True)
        (st / "windows" / "models.json").write_text(json.dumps(models), encoding="utf-8")
    if outbox:
        (st / "outbox").mkdir(exist_ok=True)
        for h in outbox:
            rec = {"handoff": h, "created_at": "2026-11-08T00:00:00Z", "delivery": "pull_only"}
            (st / "outbox" / f"{h['event_id']}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    if profiles:
        (st / "guns").mkdir(exist_ok=True)
        for gun, prof in profiles.items():
            (st / "guns" / f"{gun}.json").write_text(json.dumps({"gun_id": gun, "profile": prof}), encoding="utf-8")
    return str(st)


def _handoff(eid, gun="g", code="E029", t="2021-09-04T16:45:59Z", rule_t=None, event_type=None, onset_at=None):
    """An outbox handoff: v1.2 (no event_type) unless event_type is given; early warnings carry onset_at."""
    h = {"event_id": eid, "gun_id": gun, "detected_at": t,
         "trigger": {"source": "rule" if code else "model", "rule_code": code, "rule_trigger_time": rule_t}}
    if event_type:
        h["event_type"] = event_type
    if onset_at:
        h["early_warning"] = {"onset_at": onset_at}
    return h


def _case(cid, h, outcome=None, at=None, repeats=None, reason=None, simulated=False):
    """A normalised case as read_cases returns it."""
    closure = None if outcome is None else {"outcome": outcome, "at": at or "2021-09-04T18:00:00Z"}
    if closure is not None and reason:
        closure["reason"] = reason
    return {"case_id": cid, "event": {"event_id": h["event_id"], "gun_id": h["gun_id"], "detected_at": h["detected_at"],
                                      "trigger": h["trigger"], "event_type": h.get("event_type") or "critical"},
            "closure": closure, "repeat_event_ids": repeats, "simulated": simulated}


def test_read_windows_dedup_floor(tmp_path):
    st = _state(tmp_path, {"A": [_line("2021-09-01T10:05:00Z", score=0.1), _line("2021-09-01T10:05:03Z", score=0.2),
                                 _line("2021-09-01T10:06:00Z", score=0.3)]})
    w, _, notes = fl.read_windows(st)
    assert w["window_start"].tolist() == [pd.Timestamp("2021-09-01 10:05"), pd.Timestamp("2021-09-01 10:06")]
    assert w["score"].tolist() == [0.2, 0.3] and notes == []


def test_read_windows_skips_broken_lines(tmp_path):
    st = _state(tmp_path, {"A": [_line("2021-09-01T10:05:00Z"), _line("2021-09-01T10:06:00Z")]})
    with open(os.path.join(st, "windows", "A", "2021-09-01.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"window_start": "2021-09-01T10:07:00Z", "sco')  # the server died mid-write
    w, _, notes = fl.read_windows(st)
    assert len(w) == 2 and notes == ["읽지 못한 줄 1개를 건너뜀 (창 기록)"]


def test_read_windows_keeps_newest_model(tmp_path):
    old = [_line(f"2021-09-01T10:0{i}:00Z", model="m1@a") for i in range(5)]
    new = [_line(f"2021-09-02T10:0{i}:00Z", model="m2@b") for i in range(3)]
    w, _, notes = fl.read_windows(_state(tmp_path, {"A": old + new}))
    assert w["model"].tolist() == ["m2@b"] * 3
    assert notes == ["창 기록에 모델이 여럿이라 가장 늦은 모델(m2@b)의 창만 평가함 (뺀 창: m1@a 5개)"]


def test_read_windows_gun_dir_unquoted(tmp_path):
    st = _state(tmp_path, {"line%2F1": [_line("2021-09-01T10:05:00Z"),
                                        _line("2021-09-01T10:06:00Z", error_codes=["E003"])]},
                models={"m@1": {"model_cols": [], "threshold": 0.9, "early_warning": None}})
    w, models, _ = fl.read_windows(st)
    assert w["gun"].tolist() == ["line/1", "line/1"] and w["error_active"].tolist() == [False, True]
    assert list(models) == ["m@1"]


def _db(path, cases, events):
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE IF NOT EXISTS cases ("
                    " case_id TEXT PRIMARY KEY, gun_id TEXT, code TEXT, status TEXT, detected_at TEXT, data TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, case_id TEXT)")
        for d in cases:
            e = d["event"]
            con.execute("INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)", (d["case_id"], e["gun_id"], e["trigger"].get(
                "rule_code"), d["status"], e["detected_at"], json.dumps(d, ensure_ascii=False)))
        con.executemany("INSERT INTO events VALUES (?, ?)", events.items())
    con.close()


def _api_case(cid, h, outcome=None, repeats=()):
    """A case as api/store.py keeps it (the whole handoff, records.closure, repeats)."""
    rec = {"checks": {}, "judgments": [], "actions": [], "results": [], "field": [], "handovers": []}
    if outcome:
        rec["closure"] = {"outcome": outcome, "at": "2026-10-29T12:00:00Z"}
    return {"case_id": cid, "status": outcome or "new", "severity": "critical", "event": h, "records": rec,
            "repeats": [{"event_id": r, "detected_at": h["detected_at"]} for r in repeats]}


def test_read_cases_inputs_and_errors(tmp_path):
    db = str(tmp_path / "api.db")
    _db(db, [_api_case("CASE-0001", _handoff("e1"), "false_alarm", repeats=["e3"]),
             _api_case("CASE-0002", _handoff("e2", code="E012"))],
        {"e1": "CASE-0001", "e2": "CASE-0002", "e3": "CASE-0001"})
    cases, events = fl.read_cases(cases=db)
    assert [c["case_id"] for c in cases] == ["CASE-0001", "CASE-0002"]
    assert events == {"e1": "CASE-0001", "e2": "CASE-0002", "e3": "CASE-0001"}
    assert cases[0]["closure"]["outcome"] == "false_alarm" and cases[1]["closure"] is None
    assert cases[0]["event"]["event_type"] == "critical" and cases[0]["event"]["trigger"]["rule_code"] == "E029"

    item = {"case_id": "CASE-0009", "event": {"event_id": "w1", "gun_id": "g", "detected_at": "2021-09-04T16:45:59Z",
                                              "event_type": "early_warning", "trigger": {"source": "early_warning"}},
            "closure": {"outcome": "false_alarm", "at": "2021-09-04T16:55:59Z", "reason": "기타"},
            "repeat_event_ids": ["w2"], "simulated": True}
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps({"items": [item]}, ensure_ascii=False), encoding="utf-8")
    cases, events = fl.read_cases(history=str(hist))
    assert events is None and cases[0]["repeat_event_ids"] == ["w2"] and cases[0]["simulated"]
    assert cases[0]["event"]["event_type"] == "early_warning"

    junk = tmp_path / "junk.db"
    junk.write_text("not a database", encoding="utf-8")
    no_items = tmp_path / "no_items.json"
    no_items.write_text('{"cases": []}', encoding="utf-8")
    for kw in ({"cases": str(tmp_path / "missing.db")}, {"cases": str(junk)}, {"history": str(no_items)},
               {"history": str(junk)}, {"api": "http://127.0.0.1:9"}):
        with pytest.raises(fl.InputError):
            fl.read_cases(**kw)


def test_case_index_paths():
    a, b = _handoff("e1", t="2021-09-04T10:00:00Z"), _handoff("e2", code="E012", t="2021-09-04T10:00:00Z")
    by_events = fl.case_index([_case("A", a, repeats=["e5"]), _case("B", b)], {"e1": "A", "e9": "A", "e2": "B"}, [])
    assert {k: v["case_id"] for k, v in by_events.items()} == {"e1": "A", "e9": "A", "e2": "B"}

    by_ids = fl.case_index([_case("A", a, repeats=["e3"])], None, [])
    assert {k: v["case_id"] for k, v in by_ids.items()} == {"e1": "A", "e3": "A"}

    c = _handoff("c1", t="2021-09-04T10:00:00Z")
    c2 = _handoff("c2", t="2021-09-04T10:30:00Z")
    m = _handoff("m1", code=None, t="2021-09-04T10:00:00Z")
    d = _handoff("d1", t="2021-09-04T11:30:00Z")
    old = [_case("C", c, "false_alarm", at="2021-09-04T12:00:00Z"),            # pre-11.1: no repeat_event_ids
           _case("C2", c2, "false_alarm", at="2021-09-04T12:00:00Z"),
           _case("M", m, "resolved", at="2021-09-04T12:00:00Z"),
           _case("D", d, "false_alarm", at="2021-09-04T12:00:00Z")]
    outbox = [c, c2, m, d,
              _handoff("x1", t="2021-09-04T11:00:00Z"),                  # E029 inside both C and C2: the later case
              _handoff("x2", t="2021-09-04T13:00:00Z"),                  # after every closure
              _handoff("x3", code="E016", t="2021-09-04T11:00:00Z"),     # another code
              _handoff("m2", code=None, t="2021-09-04T11:00:00Z")]       # no code with no code
    idx = fl.case_index(old, None, outbox)
    assert {k: v["case_id"] for k, v in idx.items()} == {"c1": "C", "c2": "C2", "m1": "M", "d1": "D", "x1": "C2",
                                                         "m2": "M"}


def test_failures_from_status():
    hs = [_handoff("h1", t="2021-09-04T16:45:59Z", rule_t="2021-09-04T16:45:10Z"),
          _handoff("h2", code="E012", t="2021-09-05T02:55:59Z", rule_t="2021-09-05T02:46:08Z", event_type="critical"),
          _handoff("h3", code="E016", t="2021-09-06T01:00:59Z", rule_t="2021-09-06T01:00:02Z"),
          _handoff("h4", code=None, event_type="early_warning", onset_at="2021-09-04T15:00:59Z"),
          _handoff("h5", code=None, t="2021-09-04T17:00:59Z"),
          _handoff("h6", t="2021-09-07T19:35:59Z")]
    index = {"h1": _case("A", hs[0], "false_alarm"), "h2": _case("B", hs[1], "resolved")}
    f = fl.failures_from(hs, index)
    assert list(zip(f["event_id"], f["status"])) == [("h1", "rejected"), ("h2", "confirmed"), ("h3", "presumed"),
                                                     ("h6", "presumed")]
    assert f["T"].tolist() == [pd.Timestamp("2021-09-04 16:45:10"), pd.Timestamp("2021-09-05 02:46:08"),
                               pd.Timestamp("2021-09-06 01:00:02"), pd.Timestamp("2021-09-07 19:35:59")]
    assert f["rule_code"].tolist() == ["E029", "E012", "E016", "E029"]


def test_ew_links_and_critical_cases():
    w1 = _handoff("w1", code=None, event_type="early_warning", onset_at="2021-09-01T10:05:59Z")
    w2 = _handoff("w2", code=None, event_type="early_warning", onset_at="2021-09-01T11:20:59Z")
    r1, m1, u1 = _handoff("r1"), _handoff("m1", code=None), _handoff("u1", code="E016")
    cases = [_case("W", w1, "false_alarm", repeats=["w2", "w3"], reason="정상 작업 중 변동"),
             _case("R", r1, "resolved", repeats=[]), _case("M", m1, repeats=[]), _case("U", u1, "unresolved", repeats=[])]
    outbox = [w1, w2, r1, m1, u1]
    links = fl.ew_links(cases, fl.case_index(cases, None, outbox), outbox)
    assert [x.onset_start for x in links] == [pd.Timestamp("2021-09-01 10:05"), pd.Timestamp("2021-09-01 11:20"), None]
    assert {(x.case_id, x.outcome, x.reason, x.gun) for x in links} == {("W", "false_alarm", "정상 작업 중 변동", "g")}
    cc = fl.critical_cases(cases)
    assert [(c.case_id, c.cls, c.outcome) for c in cc] == [("R", "E04", "real"), ("M", None, "unknown"),
                                                          ("U", "E02", "unknown")]


LAB = os.path.join(ROOT, "results", "B-19", "cache")
META = os.path.join(ROOT, "results", "B-17", "cache", "meta.csv")


@pytest.mark.skipif(not (os.path.isdir(LAB) and os.path.exists(META)), reason="B-19 cache / B-17 meta not here")
def test_lab_reproduces_b20():
    """Success criterion 3: the new evaluator gives B-20's numbers and the 11-08 probe's GATE numbers on B-19's 8 runs."""
    out = fe.lab_evaluation(fl.iter_lab(LAB, META))

    def r3(x):
        return round(x, 3)

    c = out["champion"]
    assert len(out["runs"]) == 8
    assert (round(c["fa_per_day"], 2), r3(c["recall"]), r3(c["chance"]), r3(c["excess"])) == (16.40, 0.488, 0.343, 0.146)
    assert [r3(x) for x in c["ci"]] == [0.059, 0.221]
    g = out["challengers"]["GATE"]
    assert (round(g["summary"]["fa_per_day"], 2), r3(g["summary"]["excess"])) == (15.10, 0.080)
    assert [r3(x) for x in g["summary"]["ci"]] == [-0.009, 0.162]
    assert r3(g["d_champion"]["D"]) == -0.066 and [r3(x) for x in g["d_champion"]["ci"]] == [-0.128, -0.013]
    q = out["challengers"]["Q+"]
    assert round(q["summary"]["fa_per_day"], 2) == 13.59 and r3(q["d_champion"]["D"]) == -0.022
    assert [r3(x) for x in q["d_champion"]["ci"]] == [-0.050, 0.002]
    assert {k: v["verdict"] for k, v in out["challengers"].items()} == {"GATE": "fail", "Q+": "pass", "Q-": "fail",
                                                                     "Q0.975": "na"}


# ------------------------------------------------------------------ the report and the command line (Task 6)

def _iso(t):
    return pd.Timestamp(t).isoformat() + "Z"


def _synthetic_state(tmp_path):
    """The Task 4 field table as a state dir (window log, models.json, outbox) plus a simulated /api/history:
    every failure resolved; g5 has two E016 false alarms on two days; three early warnings of g0 closed as false
    alarm (two reasons) or resolved."""
    W, F, thr = _field_table()
    lines = {g: [_line(_iso(r.window_start), score=float(r.score), warmup=bool(r.warmup),
                       non_welding_share=float(r.non_welding_share), early_warning=bool(r.early_warning),
                       ew_threshold=thr) for r in grp.itertuples()] for g, grp in W.groupby("gun")}
    models = {"m@1": {"model_cols": [], "threshold": 0.9,
                      "early_warning": {"rule": "model_ng_k1", "threshold": thr, "quantile": 0.975}}}
    outbox, items = [], []

    def close(h, outcome, reason=None, simulated=False):
        closure = {"outcome": outcome, "at": _iso(pd.Timestamp(h["detected_at"].rstrip("Z")) + pd.Timedelta(minutes=10))}
        if reason:
            closure["reason"] = reason
        items.append({"case_id": f"SIM-{len(items) + 1:04d}", "status": outcome,
                      "event": {**{k: h[k] for k in ("event_id", "gun_id", "detected_at", "trigger")},
                                "event_type": h.get("event_type") or "critical"},
                      "closure": closure, "repeat_event_ids": [], "simulated": simulated})

    for i, r in enumerate(F.itertuples()):
        h = _handoff(f"f{i}", gun=r.gun, code="E029" if i % 2 == 0 else "E012", t=_iso(r.T + pd.Timedelta(minutes=1)),
                     rule_t=_iso(r.T), event_type="critical")
        outbox.append(h)
        close(h, "resolved")
    for j, hours in enumerate((10, 34)):
        t = START + pd.Timedelta(hours=hours)
        h = _handoff(f"x{j}", gun="g5", code="E016", t=_iso(t), rule_t=_iso(t), event_type="critical")
        outbox.append(h)
        close(h, "false_alarm", reason="기타")
    g0 = W[W["gun"] == "g0"].reset_index(drop=True)
    on = g0.index[g0["early_warning"] & ~g0["early_warning"].shift(fill_value=False)][:3]
    for k, (i, how) in enumerate(zip(on, (("false_alarm", "설비 정지 · 전원 꺼짐"), ("false_alarm", "센서 · 신호 문제"),
                                          ("resolved", None)))):
        end = g0.loc[i, "window_start"] + pd.Timedelta(seconds=59)
        h = _handoff(f"w{k}", gun="g0", code=None, t=_iso(end), event_type="early_warning", onset_at=_iso(end))
        outbox.append(h)
        close(h, how[0], reason=how[1], simulated=k == 0)
    st = _state(tmp_path, lines, outbox=outbox, models=models)
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    return st, str(hist)


def _report(tmp_path, tag):
    return json.loads((tmp_path / "out" / tag / "report.json").read_text(encoding="utf-8"))


def test_report_end_to_end(tmp_path):
    st, hist = _synthetic_state(tmp_path)
    assert fl.main(["--state", st, "--history", hist, "--out", str(tmp_path / "out"), "--tag", "t1"]) == 0
    rep = _report(tmp_path, "t1")
    assert list(rep) == ["schema", "meta", "criteria", "inputs", "labels", "early_warning", "critical", "reasons",
                         "post_failure", "warnings"]
    assert rep["schema"] == "feedback_report/1" and rep["meta"]["simulated"] is True
    md = (tmp_path / "out" / "t1" / "report.md").read_text(encoding="utf-8").splitlines()
    body = [x for x in md if x.strip()]
    assert body[:2] == ["# ML 피드백 루프 보고서 (D-9)", "> 시뮬레이션 라벨 포함 — 성능 아님"]
    assert [x for x in md if x.startswith("## ")] == [
        "## 1. 머리말", "## 2. 한눈에 보기", "## 3. 라벨 현황", "## 4. 조기 경보", "## 5. critical 프로필", "## 6. 사유별 메모",
        "## 7. 고장 뒤 조기 경보 진단", "## 8. 한계"]
    chs = rep["early_warning"]["challengers"]
    assert [c["name"] for c in chs] == ["GATE", "Q+", "Q-", "Q0.975"]
    assert all(c["lab_verdict"] == "na" and c["field_verdict"] in ("pass", "fail") for c in chs)
    assert all(c["verdict"] in ("review", "reject") for c in chs)  # no lab: never 반영 제안
    assert rep["labels"]["failures"]["confirmed"] == 12 and rep["labels"]["failures"]["rejected"] == 2
    assert rep["labels"]["ew_onsets"]["acted"] == 1
    assert [p["gun"] for p in rep["critical"]["proposals"]] == ["g5"]
    assert rep["critical"]["proposals"][0]["body"] == {"expected_fault_classes": ["E01", "E03", "E04"]}
    assert rep["reasons"]["설비 정지 · 전원 꺼짐"]["count"] == 1
    assert rep["reasons"]["설비 정지 · 전원 꺼짐"]["linked"]["challenger"] == "GATE"
    assert rep["reasons"]["센서 · 신호 문제"]["guns"] == ["g0"]
    assert len(rep["post_failure"]) == 12
    assert "PUT /guns/g5/profile" in "\n".join(md)


def test_report_reproducible(tmp_path):
    st, hist = _synthetic_state(tmp_path)
    reps = []
    for out in ("o1", "o2"):
        assert fl.main(["--state", st, "--history", hist, "--out", str(tmp_path / out), "--tag", "same"]) == 0
        r = json.loads((tmp_path / out / "same" / "report.json").read_text(encoding="utf-8"))
        r["meta"].pop("generated_at")
        reps.append(r)
    assert reps[0] == reps[1]


def test_exit_codes(tmp_path):
    st, out = _state(tmp_path), str(tmp_path / "out")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert fl.main(["--state", str(tmp_path / "nope"), "--out", out]) == 2
    assert fl.main(["--state", st, "--cases", str(tmp_path / "missing.db"), "--out", out]) == 2
    assert fl.main(["--state", st, "--history", str(broken), "--out", out]) == 2
    assert fl.main(["--state", st, "--lab", str(tmp_path / "no_lab"), "--out", out]) == 2
    with pytest.raises(SystemExit) as e:
        fl.main(["--state", st, "--cases", "a.db", "--history", "b.json"])
    assert e.value.code == 2
    assert fl.main(["--state", st, "--out", out, "--tag", "empty"]) == 0
    rep = _report(tmp_path, "empty")
    assert {c["verdict"] for c in rep["early_warning"]["challengers"]} == {"not_evaluated"}
    assert "창 기록 없음: 조기 경보 현장 평가는 해당 없음" in rep["warnings"]


def test_report_warns_long_log(tmp_path):
    long_ = [_line(f"2021-09-01T10:0{i}:00Z") for i in range(3)] + [_line(f"2021-10-02T10:0{i}:00Z") for i in range(3)]
    short = [_line(f"2021-09-01T10:0{i}:00Z") for i in range(3)]
    st = _state(tmp_path, {"longgun": long_, "shortgun": short})
    assert fl.main(["--state", st, "--out", str(tmp_path / "out"), "--tag", "t"]) == 0
    assert [x for x in _report(tmp_path, "t")["warnings"] if "30일" in x] == [
        "창 기록이 30일보다 긴 설비(longgun): outbox는 30일만 보존하므로 케이스 기록이 없는 오래된 고장은 빠졌을 수 있음"]


def test_report_run2_shape(tmp_path):
    days = ["2021-09-04T16:45:59Z", "2021-09-06T19:35:59Z", "2021-09-07T19:35:59Z", "2021-09-08T19:35:59Z",
            "2021-09-10T19:35:59Z"]
    hs = [_handoff(f"e{i}", gun="test_0", t=t, rule_t=t) for i, t in enumerate(days, 1)]
    hs.append(_handoff("e6", gun="test_0", code="E012", t="2021-09-11T02:55:59Z", rule_t="2021-09-11T02:46:08Z"))
    st = _state(tmp_path, outbox=hs, profiles={"test_0": None})
    db = str(tmp_path / "api.db")
    _db(db, [_api_case(f"CASE-000{i}", h, "false_alarm") for i, h in enumerate(hs[:5], 1)]
        + [_api_case("CASE-0006", hs[5], "resolved")], {h["event_id"]: f"CASE-000{i}" for i, h in enumerate(hs, 1)})
    assert fl.main(["--state", st, "--cases", db, "--out", str(tmp_path / "out"), "--tag", "r2"]) == 0
    rep = _report(tmp_path, "r2")
    assert [p["body"] for p in rep["critical"]["proposals"]] == [{"expected_fault_classes": ["E01", "E02", "E03"]}]
    assert [c["case_id"] for c in rep["critical"]["proposals"][0]["cases"]] == [f"CASE-000{i}" for i in range(1, 6)]
    assert {c["verdict"] for c in rep["early_warning"]["challengers"]} == {"not_evaluated"}


# ------------------------------------------------------------------ the final review's findings (11-08)

@pytest.mark.parametrize("ci_champion, ci_reference, fa, want", [
    ([0.01, 0.20], [-0.30, -0.20], 20.0, "pass"),    # "better" reads the current rule, not the reference
    ([-0.01, 0.20], [0.01, 0.20], 20.0, "fail"),     # a gain over the reference alone is not "better"
    ([-0.30, -0.20], [-0.10, 0.01], 13.0, "pass"),   # "cheaper" reads the fixed reference, not the current rule
    ([-0.10, 0.01], [-0.30, -0.20], 13.0, "fail"),   # a loss against the fixed reference stops it
    ([-0.10, 0.01], [-0.10, -0.001], 13.0, "fail"),  # so does a significant loss inside the margin
])
def test_source_verdict_champion_and_reference_differ(ci_champion, ci_reference, fa, want):
    dc, dr = {"D": sum(ci_champion) / 2, "ci": ci_champion}, {"D": sum(ci_reference) / 2, "ci": ci_reference}
    assert fe.source_verdict(_s(fa), dc, dr, 16.4, 16.4, False) == want


def test_paired_reports_the_change_in_recall():
    """D nets out chance, so recall can fall with D at 0; the paired change in recall shows it, on the same draws."""
    guns = np.arange(6)
    a = fe.Evidence(real=np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), chance=np.array([0.5, 0.0, 0.5, 0.0, 0.5, 0.0]),
                    fail_gun=guns, n_guns=6, fa_per_day=1.0)
    b = fe.Evidence(real=np.ones(6), chance=np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5]), fail_gun=guns, n_guns=6,
                    fa_per_day=1.0)
    p = fe.paired(a, b)
    assert p["D"] == pytest.approx(0.0) and p["recall_change"] == pytest.approx(-0.5)
    assert p["recall_change_ci"][0] <= -0.5 <= p["recall_change_ci"][1] < 0
    nothing = fe.Evidence(real=np.full(6, np.nan), chance=np.zeros(6), fail_gun=guns, n_guns=6, fa_per_day=1.0)
    assert fe.paired(nothing, nothing) == {"D": None, "ci": None, "recall_change": None, "recall_change_ci": None}


def test_combine_demotes_a_proposal_measured_against_the_apply_direction():
    assert fe.combine("pass", "pass") == ("propose", "반영 제안")
    assert fe.combine("pass", "pass", True) == ("review", "검토 필요(현장 임계값이 적용 방향과 반대)")
    assert fe.combine("pass", "insufficient", True) == fe.combine("pass", "insufficient")
    assert fe.combine("fail", "pass", True) == fe.combine("fail", "pass")


def test_field_flags_a_threshold_on_the_far_side_of_the_apply_command():
    """Applying Q+ raises the threshold and applying Q- lowers it; a field quantile on the other side of the served
    threshold (spec 7.2: the field's own normal windows, not the training ones) is not evidence for that command."""
    W, F, _ = _field_table()

    def flags(served):
        out = fe.field_evaluation(W, F, {"early_warning": {"rule": "model_ng_k1", "threshold": served,
                                                           "quantile": 0.975}}, [])
        return {k: v["against_apply"] for k, v in out["challengers"].items()}

    assert flags(0.999) == {"GATE": False, "Q+": True, "Q-": False, "Q0.975": False}
    assert flags(0.5) == {"GATE": False, "Q+": False, "Q-": True, "Q0.975": False}


def test_field_judges_q_plus_and_minus_at_the_threshold_the_apply_command_would_make():
    """`train.py --early-warning-q` recalibrates on the training data, so the field judges that threshold (the lab
    mean), not its own quantile; GATE keeps the served one and Q0.975 is the field's own quantile by definition."""
    W, F, thr = _field_table()
    entry = {"early_warning": {"rule": "model_ng_k1", "threshold": thr, "quantile": 0.975}}
    own = fe.field_evaluation(W, F, entry, [])
    out = fe.field_evaluation(W, F, entry, [], {"champion": thr, "Q+": thr + 0.01, "Q-": thr - 0.01})
    ch = out["challengers"]
    assert {k: (v["threshold"], v["threshold_basis"]) for k, v in ch.items()} == {
        "GATE": (thr, "served"), "Q+": (thr + 0.01, "lab_mean"), "Q-": (thr - 0.01, "lab_mean"),
        "Q0.975": (own["challengers"]["Q0.975"]["threshold"], "field_quantile")}
    assert ch["Q+"]["threshold"] != own["challengers"]["Q+"]["threshold"]
    assert not ch["Q+"]["against_apply"] and not ch["Q-"]["against_apply"]
    fr = fe.build_frame(W, F)
    for name, t in (("Q+", thr + 0.01), ("Q-", thr - 0.01)):
        rr = fe.evaluate(fr, fe.condition(fr, "model_ng", t))
        assert ch[name]["summary"]["fa_per_day"] == rr.fa.sum() / rr.days.sum()


def test_field_warns_when_no_lab_gives_the_apply_threshold():
    W, F, thr = _field_table()
    entry = {"early_warning": {"rule": "model_ng_k1", "threshold": thr, "quantile": 0.975}}
    own = fe.field_evaluation(W, F, entry, [])
    assert own["challengers"]["Q+"]["threshold_basis"] == "field_quantile" and fe.FIELD_QUANTILE_KO in own["warnings"]
    given = fe.field_evaluation(W, F, entry, [], {"Q+": thr + 0.01, "Q-": thr - 0.01})
    assert fe.FIELD_QUANTILE_KO not in given["warnings"]


def test_field_flags_a_lab_mean_on_the_far_side_of_the_served_threshold():
    """A lab cache that no longer matches the served bundle can put the estimate on the wrong side: not evidence."""
    W, F, thr = _field_table()
    entry = {"early_warning": {"rule": "model_ng_k1", "threshold": thr, "quantile": 0.975}}
    ch = fe.field_evaluation(W, F, entry, [], {"Q+": thr - 0.01, "Q-": thr + 0.01})["challengers"]
    assert ch["Q+"]["against_apply"] and ch["Q-"]["against_apply"]


def test_lab_mean_thresholds():
    lab = {"thresholds": {"r1": {"champion": 0.94, "Q+": 0.95}, "r2": {"champion": 0.96, "Q+": 0.97}}}
    assert fe.lab_mean_thresholds(lab) == pytest.approx({"champion": 0.95, "Q+": 0.96})
    assert fe.lab_mean_thresholds(None) == {}


def test_challenger_entry_applies_the_direction_flag_and_shows_thresholds():
    ch = next(c for c in fe.CHALLENGERS if c.name == "Q+")
    lab = {"challengers": {"Q+": {"verdict": "pass"}},
           "thresholds": {"r1": {"champion": 0.94, "Q+": 0.95}, "r2": {"champion": 0.94, "Q+": 0.96}}}
    field = {"champion_threshold": 0.946,
             "challengers": {"Q+": {"verdict": "pass", "threshold": 0.92, "against_apply": True}}}
    e = fl.challenger_entry(ch, lab, field)
    assert (e["verdict"], e["verdict_ko"]) == ("review", "검토 필요(현장 임계값이 적용 방향과 반대)")
    assert e["thresholds"] == {"served": 0.946, "field": 0.92, "lab_mean": pytest.approx(0.955)}
    field["challengers"]["Q+"]["against_apply"] = False
    assert fl.challenger_entry(ch, lab, field)["verdict"] == "propose"
    none = fl.challenger_entry(ch, None, None)
    assert none["verdict"] == "not_evaluated"
    assert none["thresholds"] == {"served": None, "field": None, "lab_mean": None}


def test_profile_unknown_case_blocks_the_proposal():
    """An unresolved or open case may be a real failure (it counts as a presumed one): that class is not proposed."""
    d1, d2 = "2021-09-04 10:00", "2021-09-05 10:00"
    cases = [_cc("A1", "g1", "E04", "false_alarm", d1), _cc("A2", "g1", "E04", "false_alarm", d2),
             _cc("A3", "g1", "E04", "unknown", d2), _cc("A4", "g1", "E04", "unknown", d2),
             _cc("B1", "g1", "E03", "false_alarm", d1), _cc("B2", "g1", "E03", "false_alarm", d2)]
    out = fe.profile_proposals(cases, {})
    assert out["review"] == [{"gun": "g1", "exclude": ["E04"], "text": "검토 필요", "why": "미확인 케이스 2건"}]
    assert [(p["exclude"], p["body"]) for p in out["proposals"]] == [
        (["E03"], {"expected_fault_classes": ["E01", "E02", "E04"]})]


def test_failures_from_cases_when_the_outbox_is_pruned():
    """The outbox keeps 30 days, the api keeps the case: its failure still labels the windows before it."""
    ok = _handoff("a", t="2021-08-01T10:01:00Z", rule_t="2021-08-01T10:00:00Z", event_type="critical")
    fa = _handoff("b", t="2021-08-02T10:01:00Z", rule_t="2021-08-02T10:00:00Z")
    opened = _handoff("c", t="2021-08-03T10:01:00Z")
    ew = _handoff("w", code=None, event_type="early_warning", onset_at="2021-08-04T10:00:59Z")
    model = _handoff("m", code=None, t="2021-08-05T10:00:00Z")
    new = _handoff("n", t="2021-09-20T10:01:00Z", rule_t="2021-09-20T10:00:00Z")
    cases = [_case("A", ok, "resolved", repeats=[]), _case("B", fa, "false_alarm", repeats=[]),
             _case("C", opened, repeats=[]), _case("W", ew, "false_alarm", repeats=[]),
             _case("M", model, "resolved", repeats=[]), _case("N", new, "resolved", repeats=[])]
    outbox = [new]
    f = fl.failures_from(outbox, fl.case_index(cases, None, outbox), cases)
    assert list(zip(f["event_id"], f["status"])) == [("a", "confirmed"), ("b", "rejected"), ("c", "presumed"),
                                                     ("n", "confirmed")]
    assert f["T"].tolist()[:3] == [pd.Timestamp("2021-08-01 10:00"), pd.Timestamp("2021-08-02 10:00"),
                                   pd.Timestamp("2021-08-03 10:01")]


def test_report_counts_a_failure_gone_from_the_outbox(tmp_path):
    h = _handoff("old", gun="A", t="2021-09-01T10:31:00Z", rule_t="2021-09-01T10:30:00Z", event_type="critical")
    st = _state(tmp_path, {"A": [_line("2021-09-01T10:00:00Z")]})
    item = {"case_id": "C1", "event": h, "closure": {"outcome": "resolved", "at": "2021-09-01T11:00:00Z"},
            "repeat_event_ids": []}
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps({"items": [item]}, ensure_ascii=False), encoding="utf-8")
    assert fl.main(["--state", st, "--history", str(hist), "--out", str(tmp_path / "out"), "--tag", "t"]) == 0
    assert _report(tmp_path, "t")["labels"]["failures"]["confirmed"] == 1


def test_ew_link_onset_from_the_case_when_the_outbox_is_pruned(tmp_path):
    w1 = _handoff("w1", code=None, event_type="early_warning", onset_at="2021-09-01T10:05:59Z")
    db = str(tmp_path / "api.db")
    _db(db, [_api_case("W", w1, "false_alarm")], {"w1": "W"})
    cases, events = fl.read_cases(cases=db)
    links = fl.ew_links(cases, fl.case_index(cases, events, []), [])
    assert [x.onset_start for x in links] == [pd.Timestamp("2021-09-01 10:05")]


def test_read_windows_skips_undecodable_bytes(tmp_path):
    st = _state(tmp_path, {"A": [_line("2021-09-01T10:05:00Z"), _line("2021-09-01T10:06:00Z")]})
    with open(os.path.join(st, "windows", "A", "2021-09-01.jsonl"), "ab") as fh:
        fh.write(b'{"window_start": "\xff\xfe')  # a power cut, not a clean cut between two characters
    w, _, notes = fl.read_windows(st)
    assert len(w) == 2 and notes == ["읽지 못한 줄 1개를 건너뜀 (창 기록)"]


def test_report_shows_thresholds_recall_change_and_review_reason(tmp_path):
    st, hist = _synthetic_state(tmp_path)
    mp = os.path.join(st, "windows", "models.json")
    with open(mp, encoding="utf-8") as fh:
        models = json.load(fh)
    models["m@1"]["early_warning"]["threshold"] = 0.999           # the served threshold above the field's quantiles
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(models, fh)
    with open(hist, encoding="utf-8") as fh:
        data = json.load(fh)
    t = _iso(START + pd.Timedelta(hours=58))
    data["items"].append({"case_id": "SIM-9999", "closure": {"outcome": "unresolved", "at": t}, "repeat_event_ids": [],
                          "event": _handoff("x9", gun="g5", code="E016", t=t, rule_t=t, event_type="critical")})
    with open(hist, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    assert fl.main(["--state", st, "--history", hist, "--out", str(tmp_path / "out"), "--tag", "t"]) == 0
    rep = _report(tmp_path, "t")
    qp = next(c for c in rep["early_warning"]["challengers"] if c["name"] == "Q+")
    assert qp["field"]["against_apply"] is True and qp["thresholds"]["served"] == 0.999
    assert qp["thresholds"]["field"] < 0.999 and qp["thresholds"]["lab_mean"] is None
    assert qp["field"]["d_champion"]["recall_change"] is not None
    assert rep["critical"]["review"] == [{"gun": "g5", "exclude": ["E02"], "text": "검토 필요", "why": "미확인 케이스 1건"}]
    md = (tmp_path / "out" / "t" / "report.md").read_text(encoding="utf-8")
    assert "적중 변화" in md and "임계값: 지금 0.999 · 현장 " in md
    assert "현장 임계값은 지금보다 낮은데 적용 명령은 임계값을 올린다" in md
    assert "- g5 E02: 검토 필요 (미확인 케이스 1건)" in md


def test_report_judges_the_field_at_the_lab_mean_thresholds(tmp_path, monkeypatch):
    st, hist = _synthetic_state(tmp_path)
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    (lab_dir / "scores_cv_x.parquet").write_bytes(b"")
    meta = tmp_path / "meta.csv"
    meta.write_text("unused", encoding="utf-8")
    with open(os.path.join(st, "windows", "models.json"), encoding="utf-8") as fh:
        thr = json.load(fh)["m@1"]["early_warning"]["threshold"]
    lab = {"runs": ["r1", "r2"], "champion": None, "reference": None,
           "challengers": {c.name: {"verdict": "na"} for c in fe.CHALLENGERS},
           "thresholds": {"r1": {"champion": thr, "Q+": thr + 0.005, "Q-": thr - 0.02},
                          "r2": {"champion": thr, "Q+": thr + 0.015, "Q-": thr - 0.04}}}
    monkeypatch.setattr(fe, "lab_evaluation", lambda runs: lab)
    assert fl.main(["--state", st, "--history", hist, "--lab", str(lab_dir), "--lab-meta", str(meta),
                    "--out", str(tmp_path / "out"), "--tag", "t"]) == 0
    rep = _report(tmp_path, "t")
    qp = next(c for c in rep["early_warning"]["challengers"] if c["name"] == "Q+")
    assert qp["field"]["threshold"] == pytest.approx(thr + 0.01) and qp["field"]["threshold_basis"] == "lab_mean"
    assert qp["thresholds"]["lab_mean"] == pytest.approx(thr + 0.01)
    assert rep["early_warning"]["confirmed_only"]["challengers"]["Q+"] == qp["field"]["summary"]  # every failure resolved
    assert fe.FIELD_QUANTILE_KO not in rep["warnings"]
    md = (tmp_path / "out" / "t" / "report.md").read_text(encoding="utf-8")
    assert f"임계값: 지금 {thr:.3f} · 현장 평가 {thr + 0.01:.3f}(실험실 평균, 적용 명령이 만들 값의 추정)" in md


CLOSED_ONLY = ("닫힌 케이스만 읽음(--history · --api): 열린 케이스가 있어도 프로필 제안이 막히지 않고 critical 미확인 수에도 "
               "들어가지 않음. 열린 케이스까지 보려면 --cases(api DB)로 돌릴 것")


def test_report_warns_when_its_input_cannot_see_open_cases(tmp_path, monkeypatch):
    days = ["2021-09-04T16:45:59Z", "2021-09-06T19:35:59Z", "2021-09-07T19:35:59Z"]
    hs = [_handoff(f"e{i}", gun="test_0", t=t, rule_t=t) for i, t in enumerate(days, 1)]
    st, out = _state(tmp_path, outbox=hs, profiles={"test_0": None}), str(tmp_path / "out")
    db = str(tmp_path / "api.db")
    _db(db, [_api_case("C1", hs[0], "false_alarm"), _api_case("C2", hs[1], "false_alarm"), _api_case("C3", hs[2])],
        {"e1": "C1", "e2": "C2", "e3": "C3"})
    hist = tmp_path / "history.json"  # GET /api/history leaves out C3, which is still open
    hist.write_text(json.dumps({"items": [{"case_id": c, "event": h, "repeat_event_ids": [],
                                           "closure": {"outcome": "false_alarm", "at": "2021-09-07T00:00:00Z"}}
                                          for c, h in zip(("C1", "C2"), hs)]}, ensure_ascii=False), encoding="utf-8")
    assert fl.main(["--state", st, "--cases", db, "--out", out, "--tag", "db"]) == 0
    assert fl.main(["--state", st, "--history", str(hist), "--out", out, "--tag", "hist"]) == 0
    assert fl.main(["--state", st, "--out", out, "--tag", "none"]) == 0
    monkeypatch.setattr(fl, "read_cases", lambda *a: ([], None))
    assert fl.main(["--state", st, "--api", "http://api", "--out", out, "--tag", "api"]) == 0

    by_db, by_history = _report(tmp_path, "db"), _report(tmp_path, "hist")
    assert by_db["critical"]["proposals"] == [] and by_db["critical"]["review"][0]["why"] == "미확인 케이스 1건"
    assert [p["gun"] for p in by_history["critical"]["proposals"]] == ["test_0"]  # the open case blocks nothing here
    warned = {tag: CLOSED_ONLY in _report(tmp_path, tag)["warnings"] for tag in ("db", "hist", "none", "api")}
    assert warned == {"db": False, "hist": True, "none": False, "api": True}
    assert f"- {CLOSED_ONLY}" in (tmp_path / "out" / "hist" / "report.md").read_text(encoding="utf-8")
