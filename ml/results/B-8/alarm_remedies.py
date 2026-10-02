"""B-8 remedies (history.md §17; needs the experiments.py window-60 cache), 4-fold (production IF setting) + fold models on the 8 test guns.
variants: base | drop789 (no c7-c9) | nowarm (no model alarm in the warm-up) | recentre (gun mean re-estimated every 24 h
from the previous 24 h of error-free welding windows) and combinations."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import experiments as E  # noqa: E402
import train as T  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
D = E.without(E.load(60), ["c19"])
MEAN_COLS = [f"{c}_mean" for c in D["meta"]["gun_norm"]["columns"] if c != "c19"]
stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
FOLDS = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)


def recentre(w, period_h=24, min_windows=360):
    """From warm-up end + k*period on, subtract the mean of the previous period's error-free welding windows."""
    out = w.copy()
    per = pd.Timedelta(hours=period_h)
    for f, part in w.groupby("file"):
        post = part["warmup"].to_numpy() == 0
        if not post.any() or (part["gun_norm"] != "gun").all():
            continue
        t = part.index
        ok = post & (part["error_active"].to_numpy() == 0) & (part["non_welding"].to_numpy() == 0)
        e = t[post][0] + per
        while e <= t[-1]:
            ref = part.loc[ok & (t >= e - per) & (t < e), MEAN_COLS]
            if len(ref) >= min_windows:
                sel = (out["file"] == f) & (out.index >= e) & (out.index < e + per)
                out.loc[sel, MEAN_COLS] = out.loc[sel, MEAN_COLS].to_numpy() - ref.mean().to_numpy()
            e += per
    return out


def run(name, cols, nowarm=False, rec=False):
    allw, te = D["all"], D["te"]
    if rec:
        allw, te = recentre(allw), recentre(te)
    res = {"val": [], "test": [], "rates": {}}
    for va in FOLDS:
        va_s = {stem(f) for f in va}
        tr = allw[~allw["file"].isin(va_s)]
        X = tr.loc[E.normal_mask(tr), cols].to_numpy(dtype=np.float32)
        model = T.build_model("iforest", E.SEED).fit(X)
        thr = float(np.quantile(T.anomaly_score(model, X), E.Q))
        for split, w, cal in (("val", allw[allw["file"].isin(va_s)], D["cal_all"][D["cal_all"]["file"].isin(va_s)]),
                              ("test", te, D["cte"])):
            s = T.anomaly_score(model, w[cols].to_numpy(dtype=np.float32))
            gthr = T.gun_thresholds(model, cols, cal, thr, E.Q, E.GATE)
            m = T.evaluate(w, s, thr, E.SUSTAIN, E.GATE, gthr)
            a = T.alarm_mask(s, T.window_thresholds(w, thr, gthr), w["non_welding"].values, E.GATE)
            if nowarm:
                a &= w["warmup"].to_numpy() == 0
            normal = ((w["label"] == 0) & (w["error_active"] == 0)).to_numpy()
            rates = pd.Series(a[normal]).groupby(w["file"].to_numpy()[normal]).mean()
            res[split].append({"auroc": m["auroc"], "auroc_pre_rule": m["auroc_pre_rule"], "alarm": float(a[normal].mean()),
                               "recall": float(a[(w["label"] == 1).to_numpy()].mean())})
            if split == "val":
                res["rates"].update(rates.to_dict())
            else:
                res.setdefault("test_rates", []).append(rates.to_dict())
    r = pd.Series(res["rates"])
    tr_ = pd.DataFrame(res["test_rates"]).mean()

    def agg(lst, k):
        v = [x[k] for x in lst]
        return float(np.mean(v)), float(np.std(v))
    out = {"name": name, "n_features": len(cols),
           **{f"cv_{k}": agg(res["val"], k) for k in ("auroc", "auroc_pre_rule", "alarm", "recall")},
           **{f"test_{k}": agg(res["test"], k) for k in ("auroc", "auroc_pre_rule", "alarm", "recall")},
           "cv_gun_max": float(r.max()), "cv_gun_sd": float(r.std(ddof=0)), "cv_guns_over5": int((r > 0.05).sum()),
           "cv_top": r.sort_values(ascending=False).head(5).round(3).to_dict(),
           "test_gun_max": float(tr_.max()), "cv_rates": r.round(4).to_dict()}
    print(f"== {name:<18} {len(cols)} f | cv AUROC {out['cv_auroc'][0]:.3f}+-{out['cv_auroc'][1]:.3f} pre {out['cv_auroc_pre_rule'][0]:.3f}"
          f"+-{out['cv_auroc_pre_rule'][1]:.3f} alarm {out['cv_alarm'][0]:.4f} gun max {out['cv_gun_max']:.3f} sd {out['cv_gun_sd']:.4f}"
          f" >5% {out['cv_guns_over5']} | test AUROC {out['test_auroc'][0]:.3f} pre {out['test_auroc_pre_rule'][0]:.3f} alarm "
          f"{out['test_alarm'][0]:.4f} gun max {out['test_gun_max']:.3f} | top {out['cv_top']}", flush=True)
    return out


base = D["model_cols"]
no789 = [c for c in base if not c.startswith(("c7_", "c8_", "c9_"))]
results = [run("base", base), run("drop789", no789), run("nowarm", base, nowarm=True),
           run("recentre", base, rec=True), run("drop789+nowarm", no789, nowarm=True),
           run("drop789+recentre", no789, rec=True), run("all3", no789, nowarm=True, rec=True)]
json.dump(results, open(os.path.join(SP, "alarm_remedies.json"), "w"), indent=1)
