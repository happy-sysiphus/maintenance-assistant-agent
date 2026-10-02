"""B-9 ensemble search (history.md §18; needs `python .py/experiments.py cache --window 60`).

Every component is scored once per outer fold (the 64-gun 4-fold of train.cv_folds) and mapped through its CDF over
the fold's training normals (supervised components: gun-level out-of-fold, as train.EnsembleDetector does); any
combination is then an average / max of those CDF scores, judged with the production alarm logic (q99 threshold,
per-gun thresholds, non-welding gate, warm-up hold) on the held-out guns and, with the same fold models, on the 8
test guns. Supervised components see a 150k subsample of the normal windows (all positives) - a common, cheaper
protocol for every candidate, so the current IF+LGBM is re-run here as the reference.

    python results/B-9/ensembles.py score      # component scores -> results/B-9/cache/ (~30 min)
    python results/B-9/ensembles.py combine    # every combination -> results/B-9/ensembles.json
"""
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
sys.path.insert(0, os.path.join(ROOT, ".py"))
import experiments as E  # noqa: E402
import train as T  # noqa: E402

NEG_SUB = 150_000
INNER = 4
LONG_LABEL_S = 6 * 3600
stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ components
class Unsup:
    def __init__(self, kind, cols, all_cols):
        self.kind, self.idx = kind, [all_cols.index(c) for c in cols]

    def fit(self, X):
        X = X[:, self.idx]
        if self.kind == "iforest":
            self.m = T.build_model("iforest", E.SEED).fit(X)
        elif self.kind == "pca":
            self.m = T.PCADetector(random_state=E.SEED).fit(X)
        elif self.kind == "maha":
            from sklearn.covariance import LedoitWolf
            self.m = LedoitWolf().fit(X)
        return self

    def score(self, X):
        X = X[:, self.idx]
        if self.kind == "maha":
            return self.m.mahalanobis(X)
        return -self.m.score_samples(X)


def make_sup(kind, n_neg, n_pos):
    if kind in ("lgbm", "lgbm6h"):
        return T.lgbm_classifier(n_neg, n_pos, E.SEED)
    if kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=150, min_samples_leaf=20, max_features=0.3, n_jobs=4,
                                    class_weight="balanced_subsample", random_state=E.SEED)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=500))
    raise ValueError(kind)


def sup_fit_oof(kind, tr, cols, rng):
    """Final model on the fold's training guns + out-of-fold scores of every training normal window (CDF grid)."""
    y = tr["label"].to_numpy() == 1 if kind != "lgbm6h" else (tr["ttf_s"].to_numpy() <= LONG_LABEL_S)
    nw_ok = tr["non_welding"].to_numpy() == 0
    normal = E.normal_mask(tr)
    pos = y & nw_ok & (tr["error_active"].to_numpy() == 0 if kind == "lgbm6h" else True)
    neg = normal & ~y
    files = tr["file"].to_numpy()
    X = tr[cols].to_numpy(dtype=np.float32)

    def fit_on(mask_files):
        neg_i = np.flatnonzero(neg & mask_files)
        if len(neg_i) > NEG_SUB:
            neg_i = rng.choice(neg_i, NEG_SUB, replace=False)
        pos_i = np.flatnonzero(pos & mask_files)
        i = np.r_[neg_i, pos_i]
        yy = np.r_[np.zeros(len(neg_i), int), np.ones(len(pos_i), int)]
        return make_sup(kind, len(neg_i), len(pos_i)).fit(X[i], yy)

    guns = np.array(sorted(set(files)))
    rng.shuffle(guns)
    oof = np.full(len(tr), np.nan)
    for k in range(INNER):
        hold = np.isin(files, guns[k::INNER])
        m = fit_on(~hold)
        idx = np.flatnonzero(hold & normal)
        oof[idx] = m.predict_proba(X[idx])[:, 1]
    final = fit_on(np.ones(len(tr), bool))
    return final, oof[normal]


def score_all():
    os.makedirs(CACHE, exist_ok=True)
    D = E.without(E.load(60), ["c19"])
    cols = D["model_cols"]
    if_cols = [c for c in cols if not c.startswith(T.EnsembleDetector.IF_DROP_PREFIXES)]
    folds = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)
    allw, cal, te, cte = D["all"], D["cal_all"], D["te"], D["cte"]
    unsup = {"iforest": if_cols, "pca": cols, "maha": cols}
    sup = ["lgbm", "lgbm6h", "et", "lr"]
    for i, va in enumerate(folds, 1):
        va_s = {stem(f) for f in va}
        tr = allw[~allw["file"].isin(va_s)]
        frames = {"val": allw[allw["file"].isin(va_s)], "cval": cal[cal["file"].isin(va_s)], "test": te, "ctest": cte}
        out = {k: pd.DataFrame(index=range(len(v))) for k, v in frames.items()}
        grids, trn = {}, pd.DataFrame()
        Xn = tr.loc[E.normal_mask(tr), cols].to_numpy(dtype=np.float32)
        for name, c in unsup.items():
            t0 = time.time()
            m = Unsup(name, c, cols).fit(Xn)
            s_n = m.score(Xn)
            grids[name] = T.EnsembleDetector._grid(s_n)
            trn[name] = T.EnsembleDetector._cdf(grids[name], s_n)
            for k, v in frames.items():
                out[k][name] = T.EnsembleDetector._cdf(grids[name], m.score(v[cols].to_numpy(dtype=np.float32)))
            log(f"fold {i} {name}: {time.time() - t0:.0f}s")
        for name in sup:
            t0 = time.time()
            m, oof = sup_fit_oof(name, tr, cols, np.random.default_rng(E.SEED + i))
            grids[name] = T.EnsembleDetector._grid(oof)
            trn[name] = T.EnsembleDetector._cdf(grids[name], oof)
            for k, v in frames.items():
                out[k][name] = T.EnsembleDetector._cdf(grids[name], m.predict_proba(v[cols].to_numpy(dtype=np.float32))[:, 1])
            log(f"fold {i} {name}: {time.time() - t0:.0f}s")
        # training normals, per row (the combined threshold is the q99 of the COMBINED score, so the components'
        # correlation matters): unsupervised in-sample CDF, supervised out-of-fold CDF - as EnsembleDetector.fit_scores_
        out["trn"] = trn
        for k, v in out.items():
            v.to_parquet(os.path.join(CACHE, f"fold{i}_{k}.parquet"))
        log(f"fold {i} saved")


# ------------------------------------------------------------------ combinations
def combos():
    base = ["iforest", "pca", "maha", "lgbm", "lgbm6h", "et", "lr"]
    out = [(c, "mean", None) for c in base]
    for r in (2, 3, 4):
        for sub in itertools.combinations(base, r):
            if any(x in sub for x in ("lgbm", "lgbm6h", "et", "lr")):
                out.append(("+".join(sub), "mean", None))
    out.append(("+".join(base), "mean", None))
    for w in (0.3, 0.7):
        out.append(("iforest+lgbm", "weighted", w))
    for sub in (("iforest", "lgbm"), ("iforest", "lgbm", "lgbm6h")):
        out.append(("+".join(sub), "max", None))
    return out


def combine_scores(df, names, how, w):
    X = df[names].to_numpy()
    if how == "mean":
        return X.mean(1)
    if how == "max":
        return X.max(1)
    return (1 - w) * X[:, 0] + w * X[:, 1]  # weighted: w on the second component


def gun_thr_from(calib, s, thr):
    ok = T.calibration_windows(calib.assign(_s=s), E.GATE)
    return {f: t for f, part in ok.groupby("file") if (t := T.gun_threshold(part["_s"].to_numpy(), thr, E.Q)) is not None}


def combine_all():
    D = E.load(60)
    folds = T.cv_folds(sorted(glob.glob(os.path.join(E.DATA, "E0*.parquet"))), 4, E.SEED)
    allw, cal = D["all"], D["cal_all"]
    fold_data = []
    for i, va in enumerate(folds, 1):
        va_s = {stem(f) for f in va}
        frames = {"val": allw[allw["file"].isin(va_s)], "cval": cal[cal["file"].isin(va_s)], "test": D["te"], "ctest": D["cte"]}
        sc = {k: pd.read_parquet(os.path.join(CACHE, f"fold{i}_{k}.parquet")) for k in list(frames) + ["trn"]}
        fold_data.append((frames, sc))
    results = []
    for names_s, how, w in combos():
        names = names_s.split("+")
        per = {"val": [], "test": []}
        rates = {}
        for frames, sc in fold_data:
            thr = float(np.quantile(combine_scores(sc["trn"], names, how, w), E.Q))
            for split, csplit in (("val", "cval"), ("test", "ctest")):
                w_ = frames[split]
                s = combine_scores(sc[split], names, how, w)
                gthr = gun_thr_from(frames[csplit], combine_scores(sc[csplit], names, how, w), thr)
                m = T.evaluate(w_, s, thr, E.SUSTAIN, E.GATE, gthr, warmup_alarms=False)
                per[split].append(m)
                if split == "val":
                    rates.update({f: v["alarm_rate_normal"] for f, v in m["per_file"].items()})
        r = pd.Series(rates)

        def agg(split, k):
            v = [m[k] for m in per[split]]
            return round(float(np.nanmean(v)), 4), round(float(np.nanstd(v)), 4)
        res = {"name": names_s, "how": how, "w": w,
               **{f"cv_{k}": agg("val", k) for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal",
                                                    "recall_pre_failure")},
               **{f"test_{k}": agg("test", k) for k in ("auroc", "auroc_pre_rule", "alarm_rate_normal")},
               "cv_fold_pre_rule": [round(m["auroc_pre_rule"], 4) for m in per["val"]],
               "cv_gun_max": round(float(r.max()), 4), "cv_guns_over5": int((r > 0.05).sum()),
               "cv_top": r.sort_values(ascending=False).head(3).round(3).to_dict()}
        results.append(res)
        tag = names_s if how == "mean" else f"{names_s} [{how} {w or ''}]"
        log(f"{tag:<34} cv pre {res['cv_auroc_pre_rule'][0]:.3f}"
            f"+-{res['cv_auroc_pre_rule'][1]:.3f} auroc {res['cv_auroc'][0]:.3f} alarm {res['cv_alarm_rate_normal'][0]:.4f}"
            f" gunmax {res['cv_gun_max']:.3f} | test pre {res['test_auroc_pre_rule'][0]:.3f} auroc {res['test_auroc'][0]:.3f}")
    json.dump(results, open(os.path.join(HERE, "ensembles.json"), "w"), indent=1)


if __name__ == "__main__":
    {"score": score_all, "combine": combine_all}[sys.argv[1]]()
