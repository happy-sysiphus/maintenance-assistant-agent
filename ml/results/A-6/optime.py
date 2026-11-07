"""
A-6 (2026-11-02): do start-up failures have a precursor when the lead is counted in operating time? A-5 found that
most E01 / E03 failures are start-up check failures. B-11 / B-14 / B-18 / B-20 / A-5 all count the lead in wall-clock
time, so for a failure at 07:10 after a 03:00 - 07:00 rest the "30 min before" window is mostly switched-off time. The
user's point: skip the rest, so T - 30 min = 02:40 (the last minutes the gun ran before it was switched off).

Inputs, no raw CSV pass and no retraining: A-5 transitions_events.csv (power-ons, failing flag, off length, cluster),
A-4 runs cache (c10-off runs), B-17 per-minute cache (sensors) and meta.csv (fault times), B-19 out-of-fold scores.

PRE-REGISTERED before computing any result below (2026-11-02):

Operating time
    off runs        c10-off runs >= 10 min (A-5's power_off .. power_on, A-4 runs); logging gaps count as operating
    op clock        seconds since the file start minus the off-run seconds before t. A minute / scoring window is
                    operating when >= 30 of its 60 s are outside off runs
    lead            counted on the op clock (a 07:10 failure after a 03:00 - 07:00 rest: T - 30 min = 02:40)

Cases and controls (train files only - the scores are B-19's out-of-fold ones)
    case            a failure whose own power-on P falls in [T - 30 min, T) (A-5 `failing`), T = first own terminal
                    code (B-17 meta), S = the shutdown that started P's off run. Primary: that off run >= 60 min
    controls        the same file's other power-ons whose off run >= 60 min, each given a pseudo-failure at the same
                    operating minutes after its power-on as the case (d = op(T) - op(P)); kept only if that pseudo-
                    failure is <= op(S) - 30 min, so no control window overlaps the case's. A stratum needs >= 2
                    controls with a value
    sensitivity     all train cases (any off length), controls = the file's power-ons after any >= 10 min off run

Windows (operating minutes)
    W1 (primary)    the last 30 min before S: [op(S) - 30, op(S)). Nothing after the restart is in it, so it asks the
                    user's question directly: did the gun already look different before it was switched off?
    W2              [op(T) - 30, op(T))          the user's example (02:40 ..), includes the minutes after power-on
    W3              [op(T) - 60, op(T) - 10]     the early-warning zone (B-18) on the op clock

Signals (fixed)
    score           B-19 out-of-fold ensemble score, 8 runs: each run's score as its rank among all that run's windows
                    (0..1), averaged over the runs; window value = max over non-warm-up windows (the early warning
                    fires on one window)
    c5_press        balance pressure while pressing (B-17 per minute); window value = mean
    c3_closed       closed electrode position (B-17 per minute); window value = mean
    a window value needs >= 5 operating minutes / windows with the signal defined, else it is missing

Test (A-5 part 2's)
    percentile of the case value among its controls ((#below + 0.5 #ties) / #controls); mean over strata, H0 0.5;
    p from 10000 permutations (a random member of each stratum plays the case), two-sided; 95% CI from 2000
    bootstraps of A-5's clusters (same physical gun or simultaneous failure group)
    primary         W1 x {score, c5_press, c3_closed} on the rest cases, Holm over 3
    supported if    Holm-adjusted p < 0.05 AND the CI excludes 0.5. Everything else is exploratory
    method placebo  pseudo-case = the case's latest control (same d; its controls = the others, same overlap rule):
                    all three primary CIs must contain 0.5, else the design is biased here and the claim is void
    power           the case W1 value shifted up by 0.5 / 1.0 sd of its controls' values: what a precursor of that
                    size would have given with these strata
    descriptive     per rest case: the share of the wall-clock early-warning zone [T - 60, T - 10] that is off time,
                    and W1's wall-clock span

ADDED AFTER THE FIRST RUN (exploratory, not pre-registered): the sensors had a value in only 3 rest cases' W1 - the
guns had stopped welding before the shutdown. W1w = the last 30 operating minutes with welding (c5_press defined)
inside the 3 operating hours before S, same signals and test; plus the welding minutes inside W1 per event.

    python results/A-6/optime.py   -> optime.json (+ optime.log via shell)

Never writes outside results/A-6/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "results", "A-5"))
import context as A5  # noqa: E402

B19 = os.path.join(ROOT, "results", "B-19", "cache")
RUNS = [(sc, s) for s in (42, 1, 2, 3) for sc in ("file", "physical")]
SEC = A5.SEC
REST_MIN = 60
OPER_S = 30
MIN_VALUES = 5
WINDOWS = {"W1": ("S", 30, 0), "W2": ("T", 30, 0), "W3": ("T", 60, 10)}   # anchor, (start, end) op-minutes before
SIGNALS = {"score": "max", "c5_press": "mean", "c3_closed": "mean"}
SHIFTS = (0.5, 1.0)
WELD_SPAN_S = 3 * 3600                                                    # exploratory W1w: look back 3 op-hours
log = A5.log


def save(name, out):
    dst = os.path.join(HERE, f"{name}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst}")


class Clock:
    """Operating seconds since t0: wall-clock seconds minus the seconds spent inside off runs."""

    def __init__(self, s, e, t0):
        self.s, self.t0 = s, t0
        self.dur = (e - s) / SEC
        self.cum = np.r_[0.0, np.cumsum(self.dur)]

    def __call__(self, t):
        t = np.asarray(t, "datetime64[ns]")
        wall = (t - self.t0) / SEC
        if not len(self.s):
            return wall
        i = np.searchsorted(self.s, t, side="right")
        j = np.maximum(i - 1, 0)
        off = np.where(i > 0, self.cum[j] + np.clip((t - self.s[j]) / SEC, 0, self.dur[j]), 0.0)
        return wall - off

    def operating(self, t):
        return self(np.asarray(t, "datetime64[ns]") + np.timedelta64(60, "s")) - self(t) >= OPER_S


def selftest():
    t = lambda s: np.datetime64(f"2021-09-01T{s}", "ns")  # noqa: E731
    c = Clock(np.array([t("03:00:00")]), np.array([t("07:00:00")]), t("00:00:00"))
    assert c(t("07:10:00")) - c(t("02:40:00")) == 1800, "07:10 minus 30 op-min must be 02:40"
    assert not c.operating(t("05:00:00")) and c.operating(t("02:50:00"))


def load_scores():
    """Per (file, window start): mean over the 8 runs of each run's score rank, plus the warm-up flag."""
    base, acc = None, None
    for sc, s in RUNS:
        d = pd.read_parquet(os.path.join(B19, f"scores_cv_{sc}_s{s}.parquet"), columns=["file", "time", "warmup", "score_ens"])
        d = d.sort_values(["file", "time"]).reset_index(drop=True)
        rank = d["score_ens"].rank(pct=True).to_numpy()
        if base is None:
            base, acc = d[["file", "time", "warmup"]], rank
        else:
            assert (d["file"].to_numpy() == base["file"].to_numpy()).all() and (d["time"].to_numpy() == base["time"].to_numpy()).all()
            acc = acc + rank
    base = base.assign(score=acc / len(RUNS))
    return {f: g for f, g in base.groupby("file")}


def file_data(f, start, scores):
    r = pd.read_parquet(os.path.join(A5.A4, f"runs_{f}.parquet"))
    off = r[(r["state"] == "c10_off") & (r["dur_s"] >= A5.LONG_OFF_S)].sort_values("start")
    clock = Clock(off["start"].to_numpy("datetime64[ns]"), off["end"].to_numpy("datetime64[ns]"), np.datetime64(start, "ns"))
    m = pd.read_parquet(os.path.join(A5.B17, f"min_{f}.parquet"), columns=["minute", "c5_press", "c3_closed"])
    tm = m["minute"].to_numpy("datetime64[ns]")
    sg = scores[f]
    sg = sg[~sg["warmup"].astype(bool)]
    ts = sg["time"].to_numpy("datetime64[ns]")
    series = {"score": (clock(ts), clock.operating(ts), sg["score"].to_numpy(float)), "score_t": ts}
    op_m, oper_m = clock(tm), clock.operating(tm)
    for c in ("c5_press", "c3_closed"):
        series[c] = (op_m, oper_m, m[c].to_numpy(float))
    return clock, series, (tm, op_m, oper_m)


def window_values(series, lo, hi):
    out = {}
    for sig, how in SIGNALS.items():
        op, oper, v = series[sig]
        sel = oper & (op >= lo) & (op < hi) & ~np.isnan(v)
        out[sig] = np.nan if sel.sum() < MIN_VALUES else float(v[sel].max() if how == "max" else v[sel].mean())
    return out


def last_weld_values(series, minutes, op_s):
    """Exploratory W1w: the last 30 operating welding minutes (c5_press defined) in the 3 op-hours before op_s."""
    tm, op_m, oper_m = minutes
    c5 = series["c5_press"][2]
    weld = oper_m & ~np.isnan(c5) & (op_m < op_s)
    idx = np.flatnonzero(weld & (op_m >= op_s - WELD_SPAN_S))[-30:]
    n_w1 = int(np.sum(weld & (op_m >= op_s - 1800)))
    if len(idx) < MIN_VALUES:
        return {sig: np.nan for sig in SIGNALS}, n_w1
    out = {c: float(np.nanmean(series[c][2][idx])) for c in ("c5_press", "c3_closed")}
    sc = series["score"][2][np.isin(series["score_t"], tm[idx])]
    out["score"] = float(sc.max()) if len(sc) >= MIN_VALUES else np.nan
    return out, n_w1


def event_windows(clock, series, minutes, P, S, d):
    """Window values of one power-on: W1 before its shutdown S, W2 / W3 before its (pseudo) failure op(P) + d."""
    op_s, op_t = float(clock(S)), float(clock(P)) + d
    anchors = {"S": op_s, "T": op_t}
    w = {w: window_values(series, anchors[a] - lo * 60, anchors[a] - hi * 60) for w, (a, lo, hi) in WINDOWS.items()}
    w["W1w"], w["weld_min_W1"] = last_weld_values(series, minutes, op_s)
    return w, op_t


def build(ev, meta, scores, rest_cases, ctrl_min_off):
    """One stratum per case: case / placebo values and their controls' values per window and signal."""
    strata = []
    for f, g in ev.groupby("file"):
        fail = g[g["failing"]]
        if not len(fail) or f not in scores:
            continue
        c = fail.iloc[0]
        if rest_cases and not c["off_min"] >= REST_MIN:
            continue
        mrow = meta.loc[f]
        clock, series, minutes = file_data(f, mrow["start"], scores)
        T = np.datetime64(mrow["fault_time"], "ns")
        P = np.datetime64(c["time"], "ns")
        S = P - np.timedelta64(int(round(c["off_min"] * 60)), "s")
        d = float(clock(T) - clock(P))
        case_w, _ = event_windows(clock, series, minutes, P, S, d)
        ctrl = []
        for _, o in g[~g["failing"] & (g["off_min"] >= ctrl_min_off)].sort_values("time").iterrows():
            Pc = np.datetime64(o["time"], "ns")
            Sc = Pc - np.timedelta64(int(round(o["off_min"] * 60)), "s")
            if Pc >= S:
                continue
            w, op_tc = event_windows(clock, series, minutes, Pc, Sc, d)
            ctrl.append({"P": Pc, "S": Sc, "op_s": float(clock(Sc)), "op_t": op_tc, "w": w})
        op_s = float(clock(S))
        ctrl = [x for x in ctrl if x["op_t"] <= op_s - 1800]
        # descriptive: wall-clock zone [T - 60, T - 10] that is off time; W1's wall-clock span
        zone_op = float(clock(T - np.timedelta64(10, "m")) - clock(T - np.timedelta64(60, "m")))
        tm, op_m, oper_m = minutes
        w1 = oper_m & (op_m >= op_s - 1800) & (op_m < op_s)
        strata.append({"file": f, "class": c["class"], "cluster": int(c["cluster"]), "off_min": float(c["off_min"]),
                       "case": case_w, "controls": ctrl, "d_min": d / 60,
                       "zone_off_share": 1 - zone_op / 3000,
                       "kst": {"T": str(pd.Timestamp(T) + A5.KST), "P": str(pd.Timestamp(P) + A5.KST),
                               "S": str(pd.Timestamp(S) + A5.KST),
                               "W1_from": str(pd.Timestamp(tm[w1].min()) + A5.KST) if w1.any() else None}})
    return strata


def test(strata, w, sig, rng, who="case", shift=0.0):
    st, cl = [], []
    for s in strata:
        if who == "case":
            x, others = s["case"][w][sig], [c["w"][w][sig] for c in s["controls"]]
        else:                                               # placebo: the latest control plays the case
            if len(s["controls"]) < 3:
                continue
            pc = s["controls"][-1]
            x = pc["w"][w][sig]
            others = [c["w"][w][sig] for c in s["controls"][:-1] if c["op_t"] <= pc["op_s"] - 1800 and c["P"] < pc["S"]]
        o = np.array(others, float)
        if np.sum(~np.isnan(o)) < 2 or np.isnan(x):
            continue
        if shift:
            x = x + shift * np.nanstd(o, ddof=1)
        st.append((x, o))
        cl.append(s["cluster"])
    return A5.percentile_test(st, cl, rng)


def main():
    selftest()
    meta = pd.read_csv(os.path.join(A5.B17, "meta.csv"), parse_dates=["fault_time", "start"])
    meta = meta[~meta["dropped"]].set_index("file")
    ev = pd.read_csv(os.path.join(ROOT, "results", "A-5", "transitions_events.csv"), parse_dates=["time"])
    ev = ev[ev["kind"] == "on"].copy()
    log("loading B-19 out-of-fold scores (8 runs)")
    scores = load_scores()
    out = {"n": {}, "primary": {}, "placebo": {}, "power": {}, "secondary": {}, "cases": []}
    rest = build(ev, meta, scores, rest_cases=True, ctrl_min_off=REST_MIN)
    allc = build(ev, meta, scores, rest_cases=False, ctrl_min_off=0)
    out["n"] = {"rest_cases": len(rest), "all_cases": len(allc),
                "rest_controls_median": float(np.median([len(s["controls"]) for s in rest])),
                "rest_clusters": len({s["cluster"] for s in rest}),
                "rest_by_class": pd.Series([s["class"] for s in rest]).value_counts().to_dict()}
    log(f"strata: {json.dumps(out['n'])}")

    rng = np.random.default_rng(6)
    prim = {sig: test(rest, "W1", sig, rng) for sig in SIGNALS}
    adj = A5.holm(np.array([prim[s].get("p", 1.0) for s in SIGNALS]))
    for (sig, s), q in zip(prim.items(), adj):
        out["primary"][sig] = {**s, "p_holm": round(float(q), 4),
                               "supported": bool(s.get("strata", 0) and q < 0.05 and not (s["ci"][0] <= 0.5 <= s["ci"][1]))}
    out["placebo"] = {sig: test(rest, "W1", sig, rng, who="placebo") for sig in SIGNALS}
    out["design_ok"] = all(p.get("strata", 0) and p["ci"][0] <= 0.5 <= p["ci"][1] for p in out["placebo"].values())
    out["power"] = {sig: {str(k): test(rest, "W1", sig, rng, shift=k) for k in SHIFTS} for sig in SIGNALS}
    for name, strata in (("rest", rest), ("all", allc)):
        out["secondary"][name] = {w: {sig: test(strata, w, sig, rng) for sig in SIGNALS} for w in WINDOWS}
    out["by_class_W1"] = {cls: {sig: test([s for s in rest if s["class"] == cls], "W1", sig, rng) for sig in SIGNALS}
                          for cls in sorted({s["class"] for s in rest})}
    out["explore_W1w"] = {name: {sig: test(strata, "W1w", sig, rng) for sig in SIGNALS}
                          for name, strata in (("rest", rest), ("all", allc))}
    out["explore_W1w_placebo"] = {sig: test(rest, "W1w", sig, rng, who="placebo") for sig in SIGNALS}
    out["weld_min_W1"] = {"case_median": float(np.median([s["case"]["weld_min_W1"] for s in rest])),
                          "control_median": float(np.median([c["w"]["weld_min_W1"] for s in rest for c in s["controls"]])),
                          "case_zero": int(sum(s["case"]["weld_min_W1"] == 0 for s in rest)),
                          "control_zero_share": round(float(np.mean([c["w"]["weld_min_W1"] == 0 for s in rest
                                                                     for c in s["controls"]])), 2)}
    for s in rest:
        row = {k: s[k] for k in ("file", "class", "cluster", "kst")}
        row.update({"off_min": round(s["off_min"], 1), "d_min": round(s["d_min"], 1),
                    "zone_off_share": round(s["zone_off_share"], 2), "n_controls": len(s["controls"])})
        for sig in SIGNALS:
            x = s["case"]["W1"][sig]
            o = np.array([c["w"]["W1"][sig] for c in s["controls"]], float)
            o = o[~np.isnan(o)]
            row[f"W1_{sig}"] = None if np.isnan(x) else round(x, 3)
            row[f"W1_{sig}_pct"] = None if np.isnan(x) or not len(o) else round(float((np.sum(o < x) + 0.5 * np.sum(o == x)) / len(o)), 2)
        out["cases"].append(row)
    out["zone_off_share_median"] = round(float(np.median([s["zone_off_share"] for s in rest])), 2)

    log("primary W1 (last 30 operating min before the shutdown), Holm over 3:")
    for sig, s in out["primary"].items():
        log(f"  {sig:10s} {json.dumps(s)}")
    log(f"method placebo (latest control as pseudo-case) - design ok: {out['design_ok']}")
    for sig, s in out["placebo"].items():
        log(f"  {sig:10s} {json.dumps(s)}")
    log("power (case W1 value + k sd of its controls):")
    for sig, d in out["power"].items():
        log(f"  {sig:10s} " + "  ".join(f"+{k} sd: {v.get('mean_percentile')} {v.get('ci')} p {v.get('p')}" for k, v in d.items()))
    for name, d in out["secondary"].items():
        log(f"secondary ({name} cases):")
        for w, dd in d.items():
            log(f"  {w}: " + "  ".join(f"{sig} {v.get('mean_percentile')} {v.get('ci')} p {v.get('p')} n {v.get('strata')}"
                                       for sig, v in dd.items()))
    log("W1 by class: " + json.dumps({c: {s: (v.get("mean_percentile"), v.get("strata")) for s, v in d.items()}
                                      for c, d in out["by_class_W1"].items()}))
    log(f"welding minutes inside W1: {json.dumps(out['weld_min_W1'])}")
    log("exploratory W1w (last 30 welding op-minutes before the shutdown):")
    for name, d in out["explore_W1w"].items():
        log(f"  {name}: " + "  ".join(f"{sig} {v.get('mean_percentile')} {v.get('ci')} p {v.get('p')} n {v.get('strata')}"
                                      for sig, v in d.items()))
    log("  placebo: " + "  ".join(f"{sig} {v.get('mean_percentile')} {v.get('ci')} n {v.get('strata')}"
                                  for sig, v in out["explore_W1w_placebo"].items()))
    log(f"wall-clock zone [T-60, T-10] off share, median over rest cases: {out['zone_off_share_median']}")
    log("rest cases:")
    for row in out["cases"]:
        log(f"  {json.dumps(row, ensure_ascii=False)}")
    save("optime", out)


if __name__ == "__main__":
    main()
