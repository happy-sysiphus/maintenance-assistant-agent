"""
A-5 follow-up (2026-11-01): does the short power blip (A-5 exploratory: a 5 s - 10 min c10-off run 10-60 min before
the failure, 36% vs 17% same-clock) add failures to the production early warning (B-20 / C-8), and does it beat
spending the same number of extra alarms on the model?

PRE-REGISTERED before computing any result below (2026-11-01):
    rules       ML     B-20 fixed rule: model_ng_k1, threshold = the 0.975 quantile of each run's normal CV scores
                BLIP   a short c10-off run (5 s <= length < 10 min, A-5 definition, A-4 runs) starts inside the window;
                       one window; warm-up windows never flag (as ML)
                UNION  ML or BLIP, one window
    metrics     B-18 exactly (results/B-18/early_warning.py Frame.evaluate): hit = flag in a window ending 10-60 min
                before T, chance = the same at T - 1..6 days, false alarm = run onsets > 60 min before T per gun-day
    runs        the 8 OOF score sets of B-19 / B-20 (64 CV guns): each gun's hit / chance averaged over the runs, gun
                bootstrap 2000, paired (one resample for both rules of a difference); 8 test guns as a check
    comparisons D1 = excess(UNION) - excess(ML)          does the blip add hits on top of the model?
                D2 = excess(UNION) - excess(ML at q*)    q* = the highest quantile of B-18's grid whose pooled false
                                                         alarms per gun-day >= UNION's: the same alarms spent on the model
    adds value  D1 CI lower bound > 0 AND D2 CI lower bound > 0 AND UNION <= 24 false alarms per gun-day (the user's
                cap) AND the test excess of UNION >= that of ML (direction check only, run mean)
    otherwise   the blip stays out of the early warning
    caveat      BLIP was picked in A-5 from 9 events x 3 windows on these same failures, so a pass is consistency with
                the model's evidence, not an independent confirmation

    python results/A-5/overlap.py     -> overlap.json (+ .log via shell)

Inputs: results/B-19/cache/scores_cv_*.parquet, results/B-11/cache/scores_test.parquet, results/A-4/cache/runs_*.parquet,
results/B-17/cache/meta.csv. Never writes outside results/A-5/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "results", "B-18"))
import early_warning as EW  # noqa: E402

A4 = os.path.join(ROOT, "results", "A-4", "cache")
B19 = os.path.join(ROOT, "results", "B-19", "cache")
RUNS = [(sc, s) for s in (42, 1, 2, 3) for sc in ("file", "physical")]
Q = 0.975
CAP = 24
SHORT_OFF_S, LONG_OFF_S = 5, 600
QS = np.r_[np.arange(0.50, 0.95, 0.025), np.arange(0.95, 0.995, 0.005), [0.995, 0.997, 0.999]]   # B-18's grid
N_BOOT = 2000


def log(msg):
    print(msg, flush=True)


def blip_starts(files):
    out = {}
    for f in files:
        r = pd.read_parquet(os.path.join(A4, f"runs_{f}.parquet"), columns=["state", "start", "dur_s"])
        r = r[(r["state"] == "c10_off") & (r["dur_s"] >= SHORT_OFF_S) & (r["dur_s"] < LONG_OFF_S)]
        out[f] = np.sort(r["start"].to_numpy())
    return out


def blip_flag(fr, starts):
    w = fr.w
    flag = np.zeros(len(w), bool)
    for g, f in enumerate(fr.guns):
        idx = np.flatnonzero(fr.gun == g)
        t = w["time"].to_numpy()[idx]
        s = starts[f]
        flag[idx] = np.searchsorted(s, t + np.timedelta64(60, "s")) > np.searchsorted(s, t)
    return flag & fr.ok


def normal_thr(fr, q):
    w = fr.w
    return np.quantile(w.loc[(w["lead"] > EW.DAY) & (w["error_active"] == 0) & ~w["warmup"], "score_ens"], q)


def per_gun(fr, c):
    fa, hits = fr.evaluate(c, 1)
    plc = np.nanmean(np.vstack([hits[f"p{k}"] for k in EW.PLACEBO_K]), axis=0)
    return {"fa": fa.astype(float), "days": fr.days, "real": hits["real"], "chance": plc}


def stack(runs):
    """runs: list of per_gun dicts (same gun order) -> per-gun means over runs + pooled false alarms per gun-day."""
    real = np.nanmean(np.vstack([r["real"] for r in runs]), axis=0)
    ch = np.nanmean(np.vstack([r["chance"] for r in runs]), axis=0)
    fa = sum(r["fa"].sum() for r in runs) / sum(r["days"].sum() for r in runs)
    return real, ch, fa


def summary(runs, rng_seed=42):
    real, ch, fa = stack(runs)
    rng = np.random.default_rng(rng_seed)
    n = len(real)
    ex = [np.nanmean(real[i]) - np.nanmean(ch[i]) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return {"fa_per_day": round(float(fa), 2), "recall": round(float(np.nanmean(real)), 3),
            "chance": round(float(np.nanmean(ch)), 3), "excess": round(float(np.nanmean(real) - np.nanmean(ch)), 3),
            "ci": [round(float(np.nanpercentile(ex, q)), 3) for q in (2.5, 97.5)]}


def paired(a, b, rng_seed=7):
    ra, ca, _ = stack(a)
    rb, cb, _ = stack(b)
    rng = np.random.default_rng(rng_seed)
    n = len(ra)
    d = lambda i: (np.nanmean(ra[i]) - np.nanmean(ca[i])) - (np.nanmean(rb[i]) - np.nanmean(cb[i]))  # noqa: E731
    boot = [d(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return {"diff": round(float(d(np.arange(n))), 3), "ci": [round(float(np.nanpercentile(boot, q)), 3) for q in (2.5, 97.5)]}


def main():
    te = EW.Frame(EW.load("test"))
    cv0 = EW.Frame(EW.load("cv", os.path.join(B19, "scores_cv_file_s42.parquet")))
    starts = blip_starts(list(cv0.guns) + list(te.guns))
    guns = list(cv0.guns)
    del cv0
    res = {"ML": [], "BLIP": [], "UNION": [], "grid": {q: [] for q in QS}}
    test = {"ML": [], "BLIP": [], "UNION": []}
    overlap = []
    bl_te = blip_flag(te, starts)
    for sc, s in RUNS:
        cv = EW.Frame(EW.load("cv", os.path.join(B19, f"scores_cv_{sc}_s{s}.parquet")))
        assert list(cv.guns) == guns
        bl = blip_flag(cv, starts)
        thr = normal_thr(cv, Q)
        ml = cv.condition("model_ng", thr)
        for name, c in (("ML", ml), ("BLIP", bl), ("UNION", ml | bl)):
            res[name].append(per_gun(cv, c))
        hm, hb = res["ML"][-1]["real"], res["BLIP"][-1]["real"]
        ok = np.isfinite(hm)
        overlap.append({"both": int(((hm == 1) & (hb == 1)).sum()), "ml_only": int(((hm == 1) & (hb == 0)).sum()),
                        "blip_only": int(((hm == 0) & (hb == 1)).sum()), "neither": int(((hm == 0) & (hb == 0)).sum()),
                        "events": int(ok.sum())})
        for q, t in zip(QS, normal_thr(cv, QS)):
            fa, _ = cv.evaluate(cv.condition("model_ng", t), 1)
            res["grid"][q].append((fa.sum(), cv.days.sum(), t))
        ml_te = te.condition("model_ng", thr)
        for name, c in (("ML", ml_te), ("BLIP", bl_te), ("UNION", ml_te | bl_te)):
            fa_t, hits_t = te.evaluate(c, 1)
            t = EW.summarise(fa_t, hits_t, te.days)
            test[name].append({"run": f"{sc}_s{s}", "fa_per_day": round(t["fa_per_day"], 2),
                               "hits": int(np.nansum(hits_t["real"])), "chance": round(t["chance"], 3),
                               "excess": round(t["excess"], 3),
                               "hit_guns": [str(g) for g, h in zip(te.guns, hits_t["real"]) if h == 1]})
        log(f"{sc}_s{s}: thr {thr:.4f}  overlap {overlap[-1]}  test hits ML {test['ML'][-1]['hits']} "
            f"BLIP {test['BLIP'][-1]['hits']} UNION {test['UNION'][-1]['hits']}")
        del cv
    out = {"criteria": __doc__.split("PRE-REGISTERED")[1].split("    python")[0].strip(), "pooled": {}, "test": {}}
    for name in ("ML", "BLIP", "UNION"):
        out["pooled"][name] = summary(res[name])
    fa_u = out["pooled"]["UNION"]["fa_per_day"]
    grid = {q: sum(a for a, _, _ in v) / sum(d for _, d, _ in v) for q, v in res["grid"].items()}
    q_star = max(q for q, fa in grid.items() if fa >= fa_u) if any(fa >= fa_u for fa in grid.values()) else min(QS)
    # re-score ML at q* per run (per-gun hits for the paired difference)
    res["ML_q*"] = []
    tq = []
    for sc, s in RUNS:
        cv = EW.Frame(EW.load("cv", os.path.join(B19, f"scores_cv_{sc}_s{s}.parquet")))
        thr = normal_thr(cv, q_star)
        res["ML_q*"].append(per_gun(cv, cv.condition("model_ng", thr)))
        fa_t, hits_t = te.evaluate(te.condition("model_ng", thr), 1)
        t = EW.summarise(fa_t, hits_t, te.days)
        tq.append({"run": f"{sc}_s{s}", "fa_per_day": round(t["fa_per_day"], 2), "hits": int(np.nansum(hits_t["real"])),
                   "excess": round(t["excess"], 3)})
        del cv
    test["ML_q*"] = tq
    out["pooled"]["ML_q*"] = summary(res["ML_q*"]) | {"q": round(float(q_star), 3)}
    out["grid_fa_per_day"] = {f"{q:.3f}": round(float(v), 2) for q, v in grid.items()}
    out["D1_union_minus_ml"] = paired(res["UNION"], res["ML"])
    out["D2_union_minus_ml_matched"] = paired(res["UNION"], res["ML_q*"])
    for name, v in test.items():
        out["test"][name] = {"runs": v, "hits_mean": round(float(np.mean([r["hits"] for r in v])), 2),
                             "excess_mean": round(float(np.mean([r["excess"] for r in v])), 3),
                             "fa_per_day_mean": round(float(np.mean([r["fa_per_day"] for r in v])), 2)}
    ov = pd.DataFrame(overlap)
    out["overlap_mean_over_runs"] = {k: round(float(ov[k].mean()), 2) for k in ov.columns}
    # by class (pooled per-gun means)
    cls = pd.read_csv(EW.META).set_index("file")["class"].reindex(guns).to_numpy()
    out["by_class"] = {}
    for name in ("ML", "BLIP", "UNION", "ML_q*"):
        real, ch, _ = stack(res[name])
        out["by_class"][name] = {c: {"recall": round(float(np.nanmean(real[cls == c])), 3),
                                     "chance": round(float(np.nanmean(ch[cls == c])), 3)} for c in sorted(set(cls))}
    d1, d2 = out["D1_union_minus_ml"], out["D2_union_minus_ml_matched"]
    checks = {"D1_ci_above_0": d1["ci"][0] > 0, "D2_ci_above_0": d2["ci"][0] > 0, "union_within_cap": fa_u <= CAP,
              "test_union_not_worse": out["test"]["UNION"]["excess_mean"] >= out["test"]["ML"]["excess_mean"]}
    out["checks"] = checks
    out["adds_value"] = bool(all(checks.values()))
    log("\npooled over 8 runs (64 CV guns): fa/gun-day  recall  chance  excess [95% CI]")
    for name, v in out["pooled"].items():
        log(f"  {name:6s} {v['fa_per_day']:6.2f}  {v['recall']:.3f}  {v['chance']:.3f}  {v['excess']:+.3f} "
            f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" + (f"  (q* {v['q']})" if "q" in v else ""))
    log(f"D1 UNION - ML        {d1}")
    log(f"D2 UNION - ML at q*  {d2}")
    log(f"overlap (CV events, mean over runs): {out['overlap_mean_over_runs']}")
    log("test (8 guns, run mean): " + json.dumps({k: {kk: v[kk] for kk in ("hits_mean", "excess_mean", "fa_per_day_mean")}
                                                   for k, v in out["test"].items()}))
    log("by class recall/chance: " + json.dumps(out["by_class"]))
    log(f"checks {checks} -> adds value: {out['adds_value']}")
    with open(os.path.join(HERE, "overlap.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log("saved overlap.json")


if __name__ == "__main__":
    main()
