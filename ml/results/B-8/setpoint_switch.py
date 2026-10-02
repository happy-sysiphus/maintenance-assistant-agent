"""B-8 on the production ensemble (history.md §20; needs `python .py/experiments.py cache --window 60`):
do alarms on normal windows come from setpoint (program) switches, and what removes them without losing the signal?

    python results/B-8/setpoint_switch.py      # -> results/B-8/setpoint_switch.json (~30 min)

Production recipe on the window cache: IsolationForest + LightGBM ensemble (train.EnsembleDetector, all normals,
gun-level OOF), q99 threshold, per-gun thresholds, non-welding gate 0.5, warm-up hold. 64-gun 4-fold; the fold models
also score the 8 test guns (confirmation only - the selection rule of techspec B-9 ranks on the CV).
variants: base | drop_sp_std (c13-c18 _std out of the input) | drop_sp (c13-c18 out) | hold_switch (a window whose
setpoints change - any c13-c16 _std > 0 - never alarms, like the non-welding gate; base scores).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import experiments as E  # noqa: E402
import train as T  # noqa: E402

SP = ["c13", "c14", "c15", "c16", "c17", "c18"]
SWITCH = ["c13_std", "c14_std", "c15_std", "c16_std"]
stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731


def log(*a):
    print(*a, flush=True)


def switch_mask(w):
    return (w[SWITCH].to_numpy() > 1e-6).any(axis=1)


def hold_switch(w):
    """The frame with switch windows marked fully non-welding: evaluate()'s gate (and the calibration filter) then
    hold them exactly as a production gate would."""
    return w.assign(non_welding=np.where(switch_mask(w), 1.0, w["non_welding"].to_numpy()))


def fit_fold(tr, cols):
    normal = E.normal_mask(tr)
    pos = tr[(tr["label"] == 1) & (tr["non_welding"] == 0)]
    fit = tr[normal]
    m = T.EnsembleDetector(random_state=E.SEED, model_cols=cols).fit(
        fit[cols].to_numpy(dtype=np.float32), pos[cols].to_numpy(dtype=np.float32), fit["file"].to_numpy(),
        pos["file"].to_numpy(), fit["ttf_s"].to_numpy())
    return m, float(np.quantile(m.fit_scores_, E.Q))


def evaluate(model, cols, thr, w, cal, hold):
    if hold:
        w, cal = hold_switch(w), hold_switch(cal)
    s = T.anomaly_score(model, w[cols].to_numpy(dtype=np.float32))
    gthr = T.gun_thresholds(model, cols, cal, thr, E.Q, E.GATE)
    return T.evaluate(w, s, thr, E.SUSTAIN, E.GATE, gthr, warmup_alarms=False), s


def main():
    D = E.without(E.load(60), ["c19"])
    base = D["model_cols"]
    variants = {"base": (base, False),
                "drop_sp_std": ([c for c in base if c not in {f"{s}_std" for s in SP}], False),
                "drop_sp": ([c for c in base if c.rsplit("_", 1)[0] not in SP], False),
                "hold_switch": (base, True)}
    folds = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)
    allw, cal, te, cte = D["all"], D["cal_all"], D["te"], D["cte"]
    res = {k: {"val": [], "test": [], "rates": {}} for k in variants}
    diag = []
    for i, va in enumerate(folds, 1):
        va_s = {stem(f) for f in va}
        tr, w_va, c_va = allw[~allw["file"].isin(va_s)], allw[allw["file"].isin(va_s)], cal[cal["file"].isin(va_s)]
        fitted = {}
        for name, (cols, hold) in variants.items():
            t0 = time.time()
            key = tuple(cols)
            if key not in fitted:
                fitted[key] = fit_fold(tr, cols)
            model, thr = fitted[key]
            for split, w, c in (("val", w_va, c_va), ("test", te, cte)):
                m, s = evaluate(model, cols, thr, w, c, hold)
                res[name][split].append(m)
                if split == "val":
                    res[name]["rates"].update({f: v["alarm_rate_normal"] for f, v in m["per_file"].items()})
                    if name == "base":  # where do the normal-window alarms of the production recipe come from?
                        a = T.alarm_mask(s, T.window_thresholds(w, thr, T.gun_thresholds(model, cols, c, thr, E.Q, E.GATE)),
                                         w["non_welding"].values, E.GATE, w["warmup"].to_numpy())
                        normal = ((w["label"] == 0) & (w["error_active"] == 0)).to_numpy()
                        diag.append(pd.DataFrame({"file": w["file"].to_numpy(), "normal": normal, "alarm": a,
                                                  "switch": switch_mask(w), "c1_std": w["c1_std"].to_numpy(),
                                                  "cls": w["class"].to_numpy() if "class" in w else ""}))
            log(f"fold {i} {name}: {time.time() - t0:.0f}s  val pre {res[name]['val'][-1]['auroc_pre_rule']:.3f}  "
                f"test pre {res[name]['test'][-1]['auroc_pre_rule']:.3f}")
    out = {}
    for name, r in res.items():
        rates = pd.Series(r["rates"])

        def agg(split, k, r=r):
            v = [m[k] for m in r[split]]
            return round(float(np.nanmean(v)), 4), round(float(np.nanstd(v)), 4)
        pc = lambda split, c, r=r: round(float(np.nanmean([m["per_class"][c]["auroc"] for m in r[split]])), 4)  # noqa: E731
        out[name] = {"cv_pre_rule": agg("val", "auroc_pre_rule"), "cv_auroc": agg("val", "auroc"),
                     "cv_folds_pre_rule": [round(m["auroc_pre_rule"], 4) for m in r["val"]],
                     "cv_alarm": agg("val", "alarm_rate_normal"), "cv_recall": agg("val", "recall_pre_failure"),
                     "cv_gun_max": round(float(rates.max()), 4), "cv_guns_over5": int((rates > 0.05).sum()),
                     "cv_top": rates.sort_values(ascending=False).head(4).round(4).to_dict(),
                     "cv_class_auroc": {c: pc("val", c) for c in ("E01", "E02", "E03", "E04")},
                     "test_pre_rule": agg("test", "auroc_pre_rule"), "test_auroc": agg("test", "auroc"),
                     "test_alarm": agg("test", "alarm_rate_normal"),
                     "test_class_auroc": {c: pc("test", c) for c in ("E01", "E02", "E03", "E04")}}
        o = out[name]
        log(f"== {name:<12} cv pre {o['cv_pre_rule'][0]:.3f}+-{o['cv_pre_rule'][1]:.3f} worst {min(o['cv_folds_pre_rule']):.3f} "
            f"alarm {o['cv_alarm'][0]:.4f} gunmax {o['cv_gun_max']:.3f} >5% {o['cv_guns_over5']} | test pre "
            f"{o['test_pre_rule'][0]:.3f} alarm {o['test_alarm'][0]:.4f} | E04 cv {o['cv_class_auroc']['E04']:.3f} "
            f"test {o['test_class_auroc']['E04']:.3f} | top {o['cv_top']}")
    d = pd.concat(diag)
    n = d[d["normal"]]
    top = n.groupby("file")["alarm"].mean().sort_values(ascending=False).head(6).index
    out["diagnosis"] = {
        "normal_windows_switch_share": round(float(n["switch"].mean()), 4),
        "alarm_rate_switch": round(float(n.loc[n["switch"], "alarm"].mean()), 4),
        "alarm_rate_no_switch": round(float(n.loc[~n["switch"], "alarm"].mean()), 4),
        "share_of_normal_alarms_in_switch_windows": round(float(n.loc[n["alarm"], "switch"].mean()), 4),
        "top_guns": {f: {"alarm": round(float(g["alarm"].mean()), 4), "switch_share": round(float(g["switch"].mean()), 4),
                         "alarms_in_switch": round(float(g.loc[g["alarm"], "switch"].mean()), 4) if g["alarm"].any() else None}
                     for f, g in n[n["file"].isin(top)].groupby("file")}}
    log("diagnosis", json.dumps(out["diagnosis"], indent=1))
    json.dump(out, open(os.path.join(HERE, "setpoint_switch.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
