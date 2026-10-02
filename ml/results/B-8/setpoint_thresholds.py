"""B-8 follow-up (history.md §20): per window-type thresholds instead of holding setpoint-switch windows.

44 % of the normal windows contain a setpoint switch and they alarm 4.6x as often (setpoint_switch.json). Holding them
removes most false alarms but also blinds the model on 44 % of the windows (pre-failure recall 8.2 -> 6.4 %).
cond_thr keeps them scored and gives each window type its own threshold: the q99 of the training normals of that
type, and per gun the max(type global, q99 of the gun's warm-up calibration windows of that type). Implemented as a
score shift s - threshold(window), so evaluate() sees threshold 0 (its AUROC is then the AUROC of the shifted score,
i.e. of the alarm decision). hold_recal: hold the switch windows but take the threshold over the non-switch
training normals (the windows that can alarm - the logic of the non-welding gate).

    python results/B-8/setpoint_thresholds.py     # -> results/B-8/setpoint_thresholds.json (~10 min)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import setpoint_switch as P  # noqa: E402

E, T = P.E, P.T


def type_thresholds(model, cols, fit_sw, cal, q=E.Q):
    """(global per type, per gun per type) from the training-normal fit scores and the calibration windows."""
    s_fit = model.fit_scores_
    glob_t = {k: float(np.quantile(s_fit[fit_sw == k], q)) for k in (False, True)}
    c = T.calibration_windows(cal, E.GATE)
    cs = T.anomaly_score(model, c[cols].to_numpy(dtype=np.float32))
    csw = P.switch_mask(c)
    gun = {}
    for f in c["file"].unique():
        for k in (False, True):
            sel = (c["file"].to_numpy() == f) & (csw == k)
            t = T.gun_threshold(cs[sel], glob_t[k], q)
            if t is not None:
                gun[(f, k)] = t
    return glob_t, gun


def shifted(model, cols, w, glob_t, gun):
    s = T.anomaly_score(model, w[cols].to_numpy(dtype=np.float32))
    sw = P.switch_mask(w)
    warm = w["warmup"].to_numpy() > 0
    thr = np.array([glob_t[k] if wu else gun.get((f, k), glob_t[k])
                    for f, k, wu in zip(w["file"].to_numpy(), sw, warm)])
    return s - thr


def main():
    D = E.without(E.load(60), ["c19"])
    cols = D["model_cols"]
    folds = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)
    allw, cal, te, cte = D["all"], D["cal_all"], D["te"], D["cte"]
    res = {"val": [], "test": [], "rates": {}}
    other = {k: {"val": [], "test": [], "rates": {}} for k in ("base", "hold_switch", "hold_recal")}  # same fold models
    for i, va in enumerate(folds, 1):
        va_s = {P.stem(f) for f in va}
        tr = allw[~allw["file"].isin(va_s)]
        model, _ = P.fit_fold(tr, cols)
        fit_sw = P.switch_mask(tr[E.normal_mask(tr)])
        for split, w, c in (("val", allw[allw["file"].isin(va_s)], cal[cal["file"].isin(va_s)]), ("test", te, cte)):
            glob_t, gun = type_thresholds(model, cols, fit_sw, c)
            thr = float(np.quantile(model.fit_scores_, E.Q))
            # hold_recal: switch windows held AND the threshold taken over the windows that can still alarm (no
            # switch) - the calibration windows of the gun thresholds are filtered the same way by hold_switch()
            for k, hold, t in (("base", False, thr), ("hold_switch", True, thr), ("hold_recal", True, glob_t[False])):
                mo, _ = P.evaluate(model, cols, t, w, c, hold)
                other[k][split].append(mo)
                if split == "val":
                    other[k]["rates"].update({f: v["alarm_rate_normal"] for f, v in mo["per_file"].items()})
            m = T.evaluate(w, shifted(model, cols, w, glob_t, gun), 0.0, E.SUSTAIN, E.GATE, None, warmup_alarms=False)
            res[split].append(m)
            if split == "val":
                res["rates"].update({f: v["alarm_rate_normal"] for f, v in m["per_file"].items()})
        P.log(f"fold {i}: thr no-switch {glob_t[False]:.4f} switch {glob_t[True]:.4f} | val pre "
              f"{res['val'][-1]['auroc_pre_rule']:.3f} recall {res['val'][-1]['recall_pre_failure']:.3f} alarm "
              f"{res['val'][-1]['alarm_rate_normal']:.4f} | test pre {res['test'][-1]['auroc_pre_rule']:.3f}")
    json.dump({"cond_thr": summary(res), **{k: summary(v) for k, v in other.items()}},
              open(os.path.join(HERE, "setpoint_thresholds.json"), "w"), indent=1)


def summary(res):
    rates = pd.Series(res["rates"])

    def agg(split, k):
        v = [m[k] for m in res[split]]
        return round(float(np.nanmean(v)), 4), round(float(np.nanstd(v)), 4)
    out = {"cv_pre_rule": agg("val", "auroc_pre_rule"), "cv_folds_pre_rule": [round(m["auroc_pre_rule"], 4) for m in res["val"]],
           "cv_auroc": agg("val", "auroc"), "cv_alarm": agg("val", "alarm_rate_normal"),
           "cv_recall": agg("val", "recall_pre_failure"), "cv_gun_max": round(float(rates.max()), 4),
           "cv_guns_over5": int((rates > 0.05).sum()), "cv_top": rates.sort_values(ascending=False).head(4).round(4).to_dict(),
           "test_pre_rule": agg("test", "auroc_pre_rule"), "test_auroc": agg("test", "auroc"),
           "test_alarm": agg("test", "alarm_rate_normal"), "test_recall": agg("test", "recall_pre_failure"),
           "cv_class_auroc": {c: round(float(np.nanmean([m["per_class"][c]["auroc"] for m in res["val"]])), 4)
                              for c in ("E01", "E02", "E03", "E04")},
           "test_class_auroc": {c: round(float(np.nanmean([m["per_class"][c]["auroc"] for m in res["test"]])), 4)
                                for c in ("E01", "E02", "E03", "E04")}}
    P.log("==", json.dumps(out))
    return out


if __name__ == "__main__":
    main()
