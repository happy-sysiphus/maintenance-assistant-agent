"""
B-16 (2026-10-06): revive the paper's 2-stage design with a SOTA forecaster? Stage 1 = Chronos-2 (B-15's best,
with covariates) forecasts the next 60 s of the target sensor; stage 2 = a state classifier on that forecast.
The question is what stage 2 can see that a 1-step model does not, so the classifier input is varied on the SAME
origins (results/B-15/cache, 4,391 origins of 72 guns):

    past      the last 60 s of the context (what the 1-step pipeline already has)
    forecast  Chronos-2's forecast of the next 60 s (the 2-stage input)
    oracle    the ACTUAL next 60 s (a perfect stage 1 - the 2-stage ceiling)
    past+forecast, past+oracle

Tasks (positives vs the "normal" origins > 24 h before the failure): "fault" = the paper's origin, the last 60 s
of the file (terminal-code state + stop), and "pre" = origins whose 60 s end 10-60 min before the failure (a real
precursor). Features per 60 s series: mean, sd, min, max, last - first, share of zeros, mean |diff|; gun-level
4-fold LightGBM (results/B-14 gun_folds), pooled AUROC.

    python results/B-16/two_stage.py      -> results/B-16/two_stage.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-14"))
import feasibility as F  # noqa: E402
import train as T  # noqa: E402

B15 = os.path.join(ROOT, "results", "B-15", "cache")


def feats(x):
    d = np.diff(x, axis=1)
    return np.column_stack([x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1] - x[:, 0], (np.abs(x) < 1e-3).mean(1),
                            np.abs(d).mean(1)])


def cv_auc(X, y, files, folds):
    p = np.full(len(y), np.nan)
    for va in folds:
        te = np.isin(files, va)
        if len(set(y[~te])) < 2:
            continue
        clf = T.lgbm_classifier(int((y[~te] == 0).sum()), int((y[~te] == 1).sum()), 42)
        clf.set_params(n_estimators=200, n_jobs=4)
        clf.fit(X[~te], y[~te])
        p[te] = clf.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok]))


def main():
    z = np.load(os.path.join(B15, "origins.npz"))
    tgt, fut = z["tgt"], z["fut"]
    meta = pd.read_csv(os.path.join(B15, "origins.csv"))
    fc = np.load(os.path.join(B15, "pred_chronos2_cov.npy"))
    files = meta["file"].to_numpy()
    folds = F.gun_folds(files)
    blocks = {"past": feats(tgt[:, -60:]), "forecast": feats(fc), "oracle": feats(fut)}
    sets = {"past": ["past"], "forecast": ["forecast"], "oracle": ["oracle"], "past+forecast": ["past", "forecast"],
            "past+oracle": ["past", "oracle"]}
    out = {}
    for task in ("fault", "pre"):
        kind = "paper" if task == "fault" else "pre"
        sel = meta["kind"].isin(["normal", kind]).to_numpy()
        y = (meta["kind"].to_numpy()[sel] == kind).astype(int)
        out[task] = {}
        for name, parts in sets.items():
            X = np.hstack([blocks[b] for b in parts])[sel]
            out[task][name] = cv_auc(X, y, files[sel], folds)
        print(f"{task:5s} (n pos {y.sum()}, neg {(y == 0).sum()}): " +
              "  ".join(f"{k} {v:.3f}" for k, v in out[task].items()), flush=True)
    # how close is the forecast to the truth where it matters (the 60 s that end in the stop)?
    m = meta["kind"].to_numpy()
    out["forecast_mae"] = {k: float(np.abs(fc - fut)[m == k].mean()) for k in ("normal", "pre", "paper")}
    out["oracle_change_vs_past"] = {k: float(np.abs(fut.mean(1) - tgt[:, -60:].mean(1))[m == k].mean())
                                    for k in ("normal", "pre", "paper")}
    print("forecast MAE by origin kind:", {k: round(v, 3) for k, v in out["forecast_mae"].items()})
    print("|future mean - past 60 s mean| by kind:", {k: round(v, 3) for k, v in out["oracle_change_vs_past"].items()})
    with open(os.path.join(HERE, "two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
