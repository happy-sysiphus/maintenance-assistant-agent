"""
Anomaly-detection baseline on the preprocessed RSW train set.

Input : preprocessed/<file>.parquet from preprocess.py (1 Hz rows, z-scored, with
        segment_id / error_active / non_welding / ttf_s / label columns).
Model : ensemble  IsolationForest (normal windows) + LightGBM (pre-failure 1 h vs normal), each score mapped
                   to its CDF over the training normals and averaged (default; the production model since
                   2026-09-29, history.md §18 - other mixes via --ensemble-components)
          iforest  IsolationForest alone, fitted on NORMAL windows only (the production model before)
          pca      PCA reconstruction error
        Rows are first aggregated into fixed windows (--window, default 60 s) inside each
        segment: mean + std of every continuous feature, max of the flags. That turns the
        37 M second-rows into ~600 k windows and gives the detector some temporal context.
Split : by FILE (= gun), stratified by class, so validation guns are never seen in training.
Normal: label == 0 (outside the pre-failure window) and error_active == 0;
        add --exclude-non-welding to also drop cap-dressing windows.
Score : higher = more anomalous. The alarm threshold is the --threshold-q quantile of the
        training-normal scores (i.e. 1 % false-alarm rate at q = 0.99).
Guns  : --gun-norm center|scale (default center) re-normalises every file / stream with the gun's
        own statistics estimated on a warm-up period (--warmup-hours, default 6) - see the
        "per-gun normalisation" section - and, unless --no-gun-threshold, judges each gun
        against its own threshold calibrated on its warm-up windows. main.py reproduces both
        online from the bundle's `gun_norm`. --gun-norm scale exists but is NOT recommended:
        a gun that idles through its warm-up gets a tiny std and alarms constantly afterwards
        (56 % normal alarm rate on one validation gun).
Gate  : windows whose non-welding share exceeds --alarm-max-non-welding (default 0.5) never
        alarm: their sensor values are carried-forward constants (cap dressing), not
        measurements. Scores / AUROC are unaffected; alarm rates, recall and alarm runs are.
        The gate is stored in the bundle and applied identically by main.py (`alarm_held`).
Eval  : on validation files - AUROC / AUPRC of pre-failure vs normal windows (pooled, and per gun: `auroc_gun_mean`
        is free of the score offsets between guns that the pooled AUROC mixes in), per class,
        alarm rate on normal windows, recall on pre-failure windows, and per file: how many
        hours before failure the final alarm run (--sustain consecutive windows, reaching the
        failure) starts, and the number of sustained false-alarm runs per day > 24 h earlier.
        Every evaluation also reports (a) the OPERATING POINT: how many files get a final alarm
        run >= OP_LEAD_MIN minutes before the failure and <= OP_FALSE_RUNS false runs per day,
        (b) the TERMINAL-CODE RULE as a baseline, exactly as main.py fires it: any class's terminal
        code (E012/E016/E028/E029), once per episode start, then a RULE_COOLDOWN_S cooldown; a trigger whose code
        already fired < RULE_REPEAT_S earlier is a repeat and not critical - hits (lead of the first critical
        trigger inside the pre-failure window) and critical false triggers per day.
        --label-window relabels the pre-failure window (default: the 3600 s preprocess.py used),
        --cv k adds a gun-level k-fold estimate.
Test  : the 8 held-out files (preprocess.py --split test -> preprocessed/test/) get the same
        evaluation, with the SAME threshold, after training (metrics["test"]) or on their own
        with --evaluate (loads the saved bundle, writes models/baseline_<model>_test_metrics.json
        and per-file scores to models/scores/). Their class is the one preprocess.py inferred
        from the terminal code, so per-class numbers on test are "by inferred class".
Save  : models/baseline_<model>.joblib - a dict with the fitted model, feature list, window,
        threshold, the scaler + preprocess config from preprocess.py and the metrics. Inference:

            from train import load_bundle, score_frame
            b = load_bundle("models/baseline_ensemble.joblib")
            out = score_frame(b, pd.read_parquet("preprocessed/E04_3.parquet"))  # -> time, score, alarm

        or  python train.py --score preprocessed/E04_3.parquet

Usage:
    python train.py --exclude-non-welding --cv 4 # the production ensemble + 4-fold, then val + test (~10 min)
    python train.py --model iforest              # IsolationForest alone (~1 min)
    python train.py --evaluate                   # test evaluation only, with the saved bundle
    python train.py --model pca --window 120
    python train.py --max-files 8                # quick check
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

# LightGBM >= 4 records feature names even when fitted on a numpy array, and sklearn then warns on EVERY predict with
# a numpy array (each /predict request of the ensemble - it flooded the serving log). Inputs are always the bundle's
# model_cols in order, so the warning carries no information here.
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["E01", "E02", "E03", "E04"]
# target operating point (test.md §1, history.md B-4): a sustained alarm at least this early, at most this many false runs
OP_LEAD_MIN = 30.0
OP_FALSE_RUNS = 1.0
META_COLS = {"file", "class", "gun", "error_code", "segment_id", "dow", "ttf_s", "label",
             "error_active", "terminal_code", "terminal_any", "terminal_idx", "non_welding", "warmup", "gun_norm"}
FLAG_COLS = ["error_active", "terminal_code", "terminal_any", "terminal_idx", "non_welding", "label", "warmup"]
# terminal-code rule, as main.py applies it: ANY class's terminal code counts (the serving layer does not know
# the gun's class), and it fires once when a terminal-code episode starts, then stays quiet for
# RULE_COOLDOWN_S. E029 in particular also shows up for hours in E01-E03 guns days before their failure, so
# a state-based rule (critical as long as the code is present) would page for hours.
TERMINAL_CODES = ("E012", "E016", "E028", "E029")
RULE_COOLDOWN_S = 1800
# A trigger whose code already fired in the same gun within RULE_REPEAT_S is a REPEAT: it is reported but not
# critical. In the 80 train + test files, 17 of the 58 false triggers were repeats and 1 of the 72 hits
# (history.md §11): an episode that did not end in a failure tends to come back.
RULE_REPEAT_S = 24 * 3600
# the terminal code appears ~10 min before the failure in 71/72 train files (history.md §11): the rule's lead
RULE_LEAD_S = 600
# the rule baseline counts a trigger as a hit inside this window before the failure (independent of --label-window)
RULE_EVAL_WINDOW_S = 3600
# c19 ("offset value in robot") is a counter that grows ~55/s through every file. Every file is
# exactly 7 days long and ends at its failure, so within a file c19 == time since start ==
# 168 h - time to failure: a label leak, not a measurement (a supervised model reaches AUROC 0.98
# with it and 0.62 without - history.md, old test.md §8). Dropped from the model input by default.
DROP_FEATURES_DEFAULT = ["c19"]


# ------------------------------------------------------------- windowing
def feature_columns(df, drop=()):
    """Continuous model inputs = every column that is not metadata / flag (minus `drop`)."""
    return [c for c in df.columns if c not in META_COLS and c not in set(drop)]


def window_features(df, window, feat_cols=None, min_rows=None):
    """1 Hz rows -> one row per (segment, window): mean & std of features, max of flags, min ttf."""
    feat_cols = feat_cols or feature_columns(df)
    if "error_code" in df.columns:
        # terminal_idx: 1..4 = which terminal code (TERMINAL_CODES order), 0 = none - the rule's repeat check needs the code
        idx = df["error_code"].map({c: i for i, c in enumerate(TERMINAL_CODES, 1)}).fillna(0).astype("float32")
        df = df.assign(terminal_any=(idx > 0).astype("float32"), terminal_idx=idx)
    agg = {c: ["mean", "std"] for c in feat_cols}
    agg.update({c: "max" for c in FLAG_COLS if c in df.columns})
    agg["non_welding"] = "mean"  # share of the window, not a 0/1 flag like the others
    if "ttf_s" in df.columns:
        agg["ttf_s"] = "min"
    g = df.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).agg(agg)
    # only the model features get the _mean/_std suffix; metadata keeps its own name, so that
    # `non_welding` (aggregated with "mean") does not silently become `non_welding_mean`.
    g.columns = [f"{a}_{b}" if a in feat_cols and b in ("mean", "std") else a for a, b in g.columns]
    g = g.droplevel(0)
    if "ttf_s" in g.columns:
        g = g[g["ttf_s"].notna()]
    g["n"] = df.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).size().droplevel(0).reindex(g.index)
    g = g[g["n"] >= (min_rows or max(2, window // 2))]  # drop windows with less than half the samples
    std_cols = [c for c in g.columns if c.endswith("_std")]
    g[std_cols] = g[std_cols].fillna(0.0)
    for c in ("file", "class", "gun"):
        if c in df.columns:
            g[c] = df[c].iloc[0]
    return g


def model_input_columns(feat_cols):
    return [f"{c}_mean" for c in feat_cols] + [f"{c}_std" for c in feat_cols]


# ------------------------------------------------------- per-gun normalisation
# Guns differ by an offset of up to ~1.5 sigma on several sensors (c1, c14, ...), so one global
# z-score + one global threshold gives per-gun normal alarm rates anywhere between 0 and 13 %.
# preprocess.py's --scale per-file cannot be reproduced online, so the per-gun statistics are
# estimated from a WARM-UP period instead: the normal welding rows of the first `warmup_s` of a
# file (online: of a gun's stream). Rows after the warm-up are re-normalised with them; the
# warm-up rows themselves keep the global scaling (that is all the serving layer has at the
# time) and are flagged `warmup`. mode "center" subtracts the gun mean only (the gun constants
# c7-c9 become 0, i.e. --drop-static); "scale" also divides by the gun std, floored at
# std_floor (in global-z units) because a gun that idles through its warm-up has std ~ 0.
# With threshold_q, the warm-up windows are scored with the gun statistics and the gun's
# threshold becomes max(global threshold, that quantile) - a per-gun 99 % operating point.
# With warmup_rows the warm-up ends once that many normal welding rows are collected, at the latest after
# warmup_s (then a cap, not a length): a gun that idles through its first hours no longer gets statistics
# from a handful of rows. Bundles without warmup_rows keep the fixed-length warm-up.
GUN_NORM_DEFAULT = {"mode": "center", "warmup_s": 6 * 3600, "warmup_rows": None, "min_rows": 600, "std_floor": 0.25,
                    "threshold_q": None}


def gun_norm_columns(feat_cols, scaler_cols=None):
    """Columns the per-gun normalisation applies to: the z-scored sensors / counter features
    (never the ratios weld_duty / error_share, nor the time-of-day features)."""
    base = scaler_cols or [c for c in feat_cols if c.startswith("c") or c in ("welds_delta", "pos_delta", "welds_10min")]
    return [c for c in feat_cols if c in base]


def gun_norm_stats(df, cols, cfg):
    """Mean/std of the normal welding rows of the warm-up - the first cfg['warmup_s'] of the frame, or up to the
    cfg['warmup_rows']-th such row when that comes first - or None when there are fewer than cfg['min_rows'] of
    them (the file then keeps the global scaling). t_end = the end of the warm-up (rows before it are warm-up)."""
    t_end = df.index[0] + pd.Timedelta(seconds=cfg["warmup_s"])
    ok = (df.index < t_end) & (df["non_welding"].to_numpy() == 0) & (df["error_active"].to_numpy() == 0)
    target = cfg.get("warmup_rows")
    if target and ok.sum() >= target:
        t_end = df.index[np.flatnonzero(ok)[target - 1]] + pd.Timedelta(seconds=1)
        ok &= df.index < t_end
    n = int(ok.sum())
    if n < cfg["min_rows"]:
        return None
    x = df.loc[ok, cols].astype("float64")
    mean = x.mean().to_numpy()
    if cfg["mode"] == "scale":
        std = np.maximum(x.std(ddof=0).fillna(0.0).to_numpy(), cfg["std_floor"])
    else:
        std = np.ones(len(cols))
    return {"mean": mean, "std": std, "t_end": t_end, "n": n}


def apply_gun_norm(df, cols, stats, rows=None):
    """(x - mean) / std on `cols`, for the rows at/after the warm-up end (default) or a boolean mask."""
    out = df.copy()
    sel = (out.index >= stats["t_end"]) if rows is None else rows
    out.loc[sel, cols] = ((out.loc[sel, cols].to_numpy(dtype="float64") - stats["mean"]) / stats["std"]).astype("float32")
    return out


def window_file(df, window, feat_cols, gn=None, cols=None):
    """Window one preprocessed file. With a gun-norm config the rows after the warm-up are normalised
    with the gun's own warm-up statistics (exactly what main.py does online) and warm-up windows are
    flagged. Returns (windows, calibration windows): the latter are the warm-up windows normalised
    with the gun statistics (they calibrate the gun threshold), None without gun normalisation."""
    stats = gun_norm_stats(df, cols, gn) if gn else None
    if stats is None:
        w = window_features(df, window, feat_cols)
        w["warmup"], w["gun_norm"] = 0.0, "global"
        return w, None
    df = df.assign(warmup=(df.index < stats["t_end"]).astype("float32"))
    w = window_features(apply_gun_norm(df, cols, stats), window, feat_cols)
    w["gun_norm"] = "gun"
    wu = df.index < stats["t_end"]
    calib = window_features(apply_gun_norm(df[wu], cols, stats, rows=np.ones(int(wu.sum()), dtype=bool)), window, feat_cols)
    return w, calib


# fewer usable warm-up windows than this -> no gun threshold (a quantile of a handful of windows is noise)
CALIB_MIN_WINDOWS = 10


def calibration_windows(calib, gate=None):
    """The warm-up windows a gun threshold is calibrated on: the ones the alarm logic could fire on, i.e. no
    error state and (with a gate) not mostly non-welding - the same population the global threshold comes from.
    Shared with main.py (Detector._finish_warmup)."""
    ok = calib["error_active"].to_numpy() == 0 if "error_active" in calib.columns else np.ones(len(calib), bool)
    if gate is not None and "non_welding" in calib.columns:
        ok &= calib["non_welding"].to_numpy() <= gate
    return calib[ok]


def gun_threshold(scores, global_thr, q):
    """max(global, q-quantile of a gun's calibration scores), None with fewer than CALIB_MIN_WINDOWS of them."""
    if q is None or len(scores) < CALIB_MIN_WINDOWS:
        return None
    return float(max(global_thr, np.quantile(scores, q)))


def gun_thresholds(model, model_cols, calib, global_thr, q, gate=None):
    """file -> alarm threshold = max(global, q-quantile of the gun-normalised warm-up window scores)."""
    if calib is None or q is None or len(calib) == 0:
        return {}
    out = {}
    for f, part in calibration_windows(calib, gate).groupby("file"):
        thr = gun_threshold(anomaly_score(model, part[model_cols].to_numpy(dtype=np.float32)), global_thr, q)
        if thr is not None:
            out[f] = thr
    return out


def window_thresholds(w, global_thr, gun_thr):
    """Per-window threshold: the gun's own after its warm-up, the global one otherwise."""
    if not gun_thr:
        return global_thr
    thr = w["file"].map(gun_thr).fillna(global_thr).to_numpy(dtype="float64")
    if "warmup" in w.columns:
        thr = np.where(w["warmup"].to_numpy() > 0, global_thr, thr)
    return thr


# ----------------------------------------------------------------- models
class PCADetector:
    """Reconstruction error of a PCA fitted on normal windows."""

    def __init__(self, n_components=0.95, random_state=42):
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def score_samples(self, X):  # sklearn convention: higher = more normal
        rec = self.pca.inverse_transform(self.pca.transform(X))
        return -((X - rec) ** 2).mean(axis=1)


def lgbm_classifier(n_neg, n_pos, seed=42):
    """The LightGBM setup of every supervised run (B-2~B-7 experiments and EnsembleDetector)."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=float(n_neg / max(n_pos, 1)),
                              random_state=seed, n_jobs=8, verbose=-1)


class EnsembleDetector:
    """Anomaly components combined on a common scale: each score is mapped through its empirical CDF over the
    normal training windows and the CDFs are averaged (B-7 history.md §13, B-9 §18). score_samples follows the
    sklearn convention (higher = more normal).

    Components: "iforest" (normal windows only; time of day and c7-c9 left out), and supervised classifiers of
    pre-failure vs normal windows - "lgbm" (pre-failure 1 h), "lgbm6h" (pre-failure 6 h: slower drift), "et"
    (ExtraTrees, 1 h). The unsupervised part keeps the ensemble honest on unseen guns: supervised-only mixes won
    the 64-gun CV but fell apart on the test guns (B-9). With exactly ("iforest", "lgbm") the weights are
    (1 - lgbm_weight, lgbm_weight), otherwise equal.

    The supervised CDFs and the fit scores (which set the global threshold) come from a gun-level out-of-fold fit:
    in-sample probabilities of the training normals are optimistically low, and a threshold on them would alarm
    far more than the designed 1 % on unseen guns. neg_sub subsamples the normal windows a supervised component is
    FIT on (all of them are scored); B-9 validated the 4-component mix with 150,000."""

    N_GRID = 2001  # CDF knots
    # kept out of the forest only: time of day and the gun constants c7-c9 (0 after the gun centring, raw during the
    # warm-up) make the forest flag unusual hours / guns, not unusual behaviour. B-7 CV pre-rule AUROC 0.585 -> 0.593.
    IF_DROP_PREFIXES = ("hour_", "c7_", "c8_", "c9_")
    SUPERVISED = ("lgbm", "lgbm6h", "et")
    LONG_LABEL_S = 6 * 3600  # "lgbm6h": positives = windows <= 6 h before the failure

    def __init__(self, random_state=42, folds=4, lgbm_weight=0.5, model_cols=None, components=("iforest", "lgbm"),
                 neg_sub=None):
        self.random_state, self.folds, self.lgbm_weight = random_state, folds, lgbm_weight
        self.components, self.neg_sub = tuple(components), neg_sub
        unknown = set(self.components) - {"iforest", *self.SUPERVISED}
        if unknown or not self.components:
            raise ValueError(f"unknown ensemble components {sorted(unknown)}")
        self.if_idx = (None if model_cols is None else
                       np.array([i for i, c in enumerate(model_cols) if not c.startswith(self.IF_DROP_PREFIXES)]))
        self.iforest = build_model("iforest", random_state) if "iforest" in self.components else None
        self.sup, self.grids = {}, {}

    def __setstate__(self, state):
        """Bundles written before B-9 hold one LightGBM as `lgbm` and the CDFs as grid_if / grid_lgbm."""
        if "components" not in state:
            state = {**state, "components": ("iforest", "lgbm"), "neg_sub": None, "sup": {"lgbm": state.pop("lgbm")},
                     "grids": {"iforest": state.pop("grid_if"), "lgbm": state.pop("grid_lgbm")}}
        self.__dict__.update(state)

    def _xf(self, X):
        return X if self.if_idx is None else X[:, self.if_idx]

    def _classifier(self, kind, n_neg, n_pos):
        if kind == "et":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=150, min_samples_leaf=20, max_features=0.3, n_jobs=4,
                                        class_weight="balanced_subsample", random_state=self.random_state)
        return lgbm_classifier(n_neg, n_pos, self.random_state)

    @staticmethod
    def _grid(x):
        return np.quantile(np.asarray(x, dtype=np.float64), np.linspace(0, 1, EnsembleDetector.N_GRID))

    @staticmethod
    def _cdf(grid, x):
        """Empirical CDF over the training normals, extended linearly past the top knot: scores beyond every
        training normal keep their order (a clamp at 1.0 would tie all of the clearly anomalous windows)."""
        x = np.asarray(x, dtype=np.float64)
        n = len(grid)
        top, q99 = grid[-1], grid[int(0.99 * (n - 1))]
        scale = max(top - q99, 1e-9) / 0.01  # score units per unit of CDF over the last percent
        return np.where(x > top, 1.0 + (x - top) / scale, np.interp(x, grid, np.linspace(0, 1, n)))

    def _fit_supervised(self, kind, X, X_pos, groups, groups_pos, ttf, rng):
        """Final classifier + out-of-fold probabilities of every normal window X (for its CDF)."""
        if kind == "lgbm6h":
            if ttf is None:
                raise ValueError("lgbm6h needs ttf (seconds to failure) of the normal windows")
            near = np.asarray(ttf) <= self.LONG_LABEL_S
            neg_i, Xp = np.flatnonzero(~near), np.vstack([X[near], X_pos])
            gp = None if groups is None else np.r_[np.asarray(groups)[near], np.asarray(groups_pos)]
        else:
            neg_i, Xp, gp = np.arange(len(X)), X_pos, None if groups is None else np.asarray(groups_pos)
        if len(neg_i) == 0 or len(Xp) == 0:
            raise ValueError(f"ensemble component {kind}: no {'negatives' if len(neg_i) == 0 else 'positives'}")
        gn = None if groups is None else np.asarray(groups)[neg_i]

        def fit_on(keep_neg, keep_pos):
            ni = neg_i[keep_neg]
            if self.neg_sub and len(ni) > self.neg_sub:
                ni = rng.choice(ni, self.neg_sub, replace=False)
            Xa = np.vstack([X[ni], Xp[keep_pos]])
            ya = np.r_[np.zeros(len(ni), int), np.ones(int(keep_pos.sum()), int)]
            return self._classifier(kind, len(ni), int(keep_pos.sum())).fit(Xa, ya)

        oof = np.full(len(X), np.nan)
        all_neg, all_pos = np.ones(len(neg_i), bool), np.ones(len(Xp), bool)
        if groups is not None and len(set(groups)) >= self.folds:
            guns = np.array(sorted(set(np.r_[np.asarray(groups), np.asarray(groups_pos)])))
            rng.shuffle(guns)
            for k in range(self.folds):
                held = guns[k::self.folds]
                m = fit_on(~np.isin(gn, held), ~np.isin(gp, held))
                idx = np.flatnonzero(np.isin(np.asarray(groups), held))
                oof[idx] = m.predict_proba(X[idx])[:, 1]
        final = fit_on(all_neg, all_pos)
        miss = np.isnan(oof)
        if miss.any():  # no groups (or too few guns): fall back to in-sample
            oof[miss] = final.predict_proba(X[miss])[:, 1]
        return final, oof

    def fit(self, X, X_pos=None, groups=None, groups_pos=None, ttf=None):
        """X: normal windows (forest fit + classifier negatives), X_pos: pre-failure (1 h) windows, groups /
        groups_pos: gun id per row for the out-of-fold fits, ttf: seconds to failure of each X row ("lgbm6h")."""
        sup = [c for c in self.components if c in self.SUPERVISED]
        if sup and (X_pos is None or len(X_pos) == 0):
            raise ValueError("supervised ensemble components need pre-failure windows (X_pos)")
        rng = np.random.default_rng(self.random_state)
        fit_scores = []
        for c in self.components:
            if c == "iforest":
                self.iforest.fit(self._xf(X))
                s = -self.iforest.score_samples(self._xf(X))
            else:
                self.sup[c], s = self._fit_supervised(c, X, X_pos, groups, groups_pos, ttf, rng)
            self.grids[c] = self._grid(s)
            fit_scores.append(s)
        self.fit_scores_ = self._combine(fit_scores)
        return self

    def _combine(self, scores):
        cdf = [self._cdf(self.grids[c], s) for c, s in zip(self.components, scores)]
        if self.components == ("iforest", "lgbm"):
            return (1 - self.lgbm_weight) * cdf[0] + self.lgbm_weight * cdf[1]
        return np.mean(cdf, axis=0)

    def score_samples(self, X):
        scores = [-self.iforest.score_samples(self._xf(X)) if c == "iforest" else self.sup[c].predict_proba(X)[:, 1]
                  for c in self.components]
        return -self._combine(scores)


def build_model(name, seed, model_cols=None, lgbm_weight=0.5, components=("iforest", "lgbm"), neg_sub=None):
    if name == "ensemble":
        return EnsembleDetector(random_state=seed, lgbm_weight=lgbm_weight, model_cols=model_cols,
                                components=components, neg_sub=neg_sub)
    if name == "iforest":
        return IsolationForest(n_estimators=300, max_samples=1024, contamination="auto",
                               random_state=seed, n_jobs=-1)
    if name == "pca":
        return PCADetector(random_state=seed)
    raise ValueError(name)


def anomaly_score(model, X):
    return -model.score_samples(X)  # flip: higher = more anomalous


# ------------------------------------------------------------- evaluation
def alarm_mask(scores, threshold, non_welding=None, gate=None, warmup=None):
    """Window-level alarm decision: score above threshold, unless the window is mostly non-welding
    (share > gate) - those windows hold carried-forward constants, not measurements - or (warmup given) it
    lies in the gun's warm-up. Warm-up windows are scored at the global scale, where a gun's constant offsets
    (c7-c9 up to 14 sigma) look anomalous: 84-96 % of the warm-up windows of E02_14 / E02_13 / E02_10 alarmed
    (history.md §17). The terminal-code rule is not affected."""
    alarm = np.asarray(scores) > threshold
    if gate is not None and non_welding is not None:
        alarm &= np.asarray(non_welding) <= gate
    if warmup is not None:
        alarm &= np.asarray(warmup) == 0
    return alarm


def warmup_flags(w, warmup_alarms):
    """The `warmup` argument of alarm_mask: None when warm-up windows may alarm (bundles before 2026-09-29)."""
    return None if warmup_alarms or "warmup" not in w.columns else w["warmup"].to_numpy()


def alarm_runs(alarm, sustain):
    """(start, end) index pairs of runs of >= `sustain` consecutive alarm windows."""
    above = np.concatenate([[False], np.asarray(alarm, dtype=bool), [False]])
    edges = np.flatnonzero(above[1:] != above[:-1])
    return [(a, b) for a, b in zip(edges[0::2], edges[1::2]) if b - a >= sustain]


def alarm_timing(alarm, ttf_s, sustain, normal_before_h=24):
    """Chronological alarm flags -> when the final alarm run (the one that reaches the failure)
    starts, in hours before failure, and how many sustained false-alarm runs per day occur
    > normal_before_h before the failure."""
    runs = alarm_runs(alarm, sustain)
    n = len(alarm)
    final_start_h = np.nan
    # the last run must reach the end of the series AND start inside the last normal_before_h: a run that has been
    # on for days is a standing alarm, not a prediction (an always-on alarm used to pass as "lead 168 h")
    if runs and runs[-1][1] >= n - sustain and ttf_s[runs[-1][0]] <= normal_before_h * 3600:
        final_start_h = ttf_s[runs[-1][0]] / 3600
    normal_days = max((ttf_s[0] - normal_before_h * 3600) / 86400, 1e-9)
    n_false = sum(1 for a, b in runs if ttf_s[a] > normal_before_h * 3600)
    return final_start_h, n_false / normal_days


def rule_triggers(terminal, ttf_s, cooldown_s=RULE_COOLDOWN_S):
    """Chronological per-window terminal-code flags -> mask of the windows where the rule fires: a
    terminal-code episode starts (0 -> 1) and the previous trigger is >= cooldown_s earlier (main.py)."""
    t = np.asarray(terminal) > 0
    onset = t & ~np.concatenate([[False], t[:-1]])
    out, last = np.zeros(len(t), dtype=bool), None
    for i in np.flatnonzero(onset):
        if last is None or ttf_s[last] - ttf_s[i] >= cooldown_s:
            out[i], last = True, i
    return out


def rule_repeats(trig, code, ttf_s, repeat_s=RULE_REPEAT_S):
    """Chronological trigger mask + per-window code id -> mask of the triggers whose code already
    triggered less than repeat_s earlier (main.py reports them as rule_repeat, not critical)."""
    out, last = np.zeros(len(trig), dtype=bool), {}
    for i in np.flatnonzero(trig):
        c = code[i]
        if c in last and ttf_s[last[c]] - ttf_s[i] < repeat_s:
            out[i] = True
        last[c] = i
    return out


def evaluate(val, scores, threshold, sustain, gate=None, gun_thr=None, warmup_alarms=True):
    """threshold: the global one; gun_thr: file -> per-gun threshold (applied after the warm-up).
    Every AUROC / AUPRC here is pre-failure windows vs NORMAL windows: error-state windows outside the pre-failure
    window are neither (until 2026-09-29 they counted as negatives - history.md §14)."""
    normal = (val["label"] == 0) & (val["error_active"] == 0)
    pre = val["label"] == 1
    y, s = pre.values.astype(int), scores
    scored = (normal | pre).values
    nw = val["non_welding"].values if "non_welding" in val.columns else None
    thr = window_thresholds(val, threshold, gun_thr)
    a = alarm_mask(s, thr, nw, gate, warmup_flags(val, warmup_alarms))
    welding = np.ones(len(val), dtype=bool) if nw is None or gate is None else nw <= gate

    def auroc(mask):
        mask = mask & scored
        return float(roc_auc_score(y[mask], s[mask])) if y[mask].any() and not y[mask].all() else np.nan

    m = {"n_windows": int(len(val)), "n_pre_failure": int(pre.sum()), "n_normal": int(normal.sum()),
         "auroc": auroc(np.ones(len(val), dtype=bool)),
         # pre-failure windows > RULE_LEAD_S before the failure only: the part the terminal-code rule cannot cover
         # (its code shows up ~10 min before). A shorter label window raises `auroc` through the last 10 min alone.
         "auroc_pre_rule": auroc(~(pre.values & (val["ttf_s"].values <= RULE_LEAD_S))),
         "auprc": float(average_precision_score(y[scored], s[scored])) if y.any() else np.nan,
         "alarm_rate_normal": float(a[normal.values].mean()),
         "recall_pre_failure": float(a[pre.values].mean()),
         "alarm_rate_error_state": float(a[(val["error_active"] == 1).values].mean())
         if (val["error_active"] == 1).any() else np.nan,
         # gate diagnostics: how much was held, and how the score separates on welding windows only
         "alarm_gate_non_welding": gate,
         "held_windows": int(((s > thr) & ~welding).sum()),
         "held_share_of_windows": float((~welding).mean()),
         "auroc_welding_windows": auroc(welding),
         # per-gun normalisation diagnostics
         "warmup_windows": int(val["warmup"].sum()) if "warmup" in val.columns else 0,
         "files_gun_normalised": int((val.groupby("file")["gun_norm"].first() == "gun").sum()) if "gun_norm" in val.columns else 0,
         "files_gun_threshold": len(gun_thr or {}),
         "alarm_rate_normal_after_warmup": float(a[normal.values & (val["warmup"].values == 0)].mean())
         if "warmup" in val.columns and (normal.values & (val["warmup"].values == 0)).any() else np.nan}
    per_class, per_file = {}, {}
    for k in CLASSES:
        sel = (val["class"] == k).values
        if sel.any() and y[sel].any() and not y[sel].all():
            per_class[k] = {"auroc": auroc(sel), "auroc_welding_windows": auroc(sel & welding),
                            "recall_pre_failure": float(a[sel & pre.values].mean()),
                            "alarm_rate_normal": float(a[sel & normal.values].mean())}
    for f, idx in val.groupby("file").indices.items():
        order = idx[np.argsort(-val["ttf_s"].values[idx])]  # chronological
        final_h, false_per_day = alarm_timing(a[order], val["ttf_s"].values[order], sustain)
        per_file[f] = {"final_alarm_run_starts_h_before_failure": float(final_h),
                       "false_alarm_runs_per_day": float(false_per_day),
                       "alarm_rate_normal": float(a[idx][normal.values[idx]].mean()),
                       "held_share_of_windows": float((~welding[idx]).mean()),
                       "gun_norm": str(val["gun_norm"].values[idx[0]]) if "gun_norm" in val.columns else "global",
                       "threshold": float((gun_thr or {}).get(f, threshold)),
                       # within-gun separation: free of the score offsets between guns that the pooled AUROC mixes in
                       "auroc": auroc(np.bincount(idx, minlength=len(val)).astype(bool))}
    rates = np.array([v["alarm_rate_normal"] for v in per_file.values()])
    m["alarm_rate_normal_per_file_min_max_std"] = [float(rates.min()), float(rates.max()), float(rates.std())]
    gun_auc = np.array([v["auroc"] for v in per_file.values()], dtype=float)
    m["auroc_gun_mean"] = float(np.nanmean(gun_auc)) if np.isfinite(gun_auc).any() else np.nan
    m["auroc_gun_min_max"] = [float(np.nanmin(gun_auc)), float(np.nanmax(gun_auc))] if np.isfinite(gun_auc).any() else [np.nan, np.nan]
    # terminal-code rule exactly as main.py fires it (any class's code, episode start, cooldown; a trigger whose
    # code already fired < RULE_REPEAT_S earlier is a repeat and not critical) - a model-free baseline / safety
    # net. Counted on the critical (non-repeat) triggers: hit = the first one inside the pre-failure window (its
    # lead); false triggers = those before that window, per day. `all_triggers` counts repeats too.
    tcol = "terminal_any" if "terminal_any" in val.columns else "terminal_code"
    terminal = val[tcol].to_numpy() if tcol in val.columns else np.zeros(len(val))
    code = val["terminal_idx"].to_numpy() if "terminal_idx" in val.columns else None
    ttf, n_trig, n_rep, n_rep_hit, all_false = val["ttf_s"].values, 0, 0, 0, []
    for f, idx in val.groupby("file").indices.items():
        order = idx[np.argsort(-ttf[idx])]
        trig = rule_triggers(terminal[order], ttf[order])
        rep = rule_repeats(trig, code[order], ttf[order]) if code is not None else np.zeros(len(order), dtype=bool)
        crit = trig & ~rep
        pre_o = ttf[order] <= RULE_EVAL_WINDOW_S  # fixed, so that --label-window does not turn hits into false triggers
        hit = crit & pre_o
        pre_start = ttf[order][pre_o].max() if pre_o.any() else 0
        normal_days = max((ttf[order][0] - pre_start) / 86400, 1e-9)
        n_trig, n_rep, n_rep_hit = n_trig + int(trig.sum()), n_rep + int(rep.sum()), n_rep_hit + int((rep & pre_o).sum())
        all_false.append(float((trig & ~pre_o).sum() / normal_days))
        per_file[f]["rule_lead_min"] = float(ttf[order][hit].max() / 60) if hit.any() else np.nan
        per_file[f]["rule_false_triggers_per_day"] = float((crit & ~pre_o).sum() / normal_days)
        per_file[f]["rule_repeats"] = int(rep.sum())
    leads = [v["rule_lead_min"] for v in per_file.values()]
    false_trig = np.array([v["rule_false_triggers_per_day"] for v in per_file.values()])
    m["rule_terminal_code"] = {
        "codes": list(TERMINAL_CODES), "cooldown_s": RULE_COOLDOWN_S, "flag": tcol, "n_triggers": n_trig,
        "repeat_s": RULE_REPEAT_S if code is not None else None, "n_repeats": n_rep, "n_repeats_in_pre_failure": n_rep_hit,
        "files_hit": int(sum(1 for v in leads if v == v)),
        "rule_lead_min_median": float(np.nanmedian(leads)) if any(v == v for v in leads) else np.nan,
        "false_triggers_per_day_mean": float(false_trig.mean()),
        "files_with_false_trigger": int((false_trig > 0).sum()),
        "files_false_triggers_ok": int((false_trig <= OP_FALSE_RUNS).sum()),
        "all_triggers_false_per_day_mean": float(np.mean(all_false))}
    # operating point
    lead = np.nan_to_num(np.array([v["final_alarm_run_starts_h_before_failure"] for v in per_file.values()]) * 60, nan=-1)
    false_runs = np.array([v["false_alarm_runs_per_day"] for v in per_file.values()])
    m["operating_point"] = {
        "lead_min_target": OP_LEAD_MIN, "false_runs_per_day_target": OP_FALSE_RUNS, "files": len(per_file),
        "files_lead_ok": int((lead >= OP_LEAD_MIN).sum()), "files_false_runs_ok": int((false_runs <= OP_FALSE_RUNS).sum()),
        "files_both_ok": int(((lead >= OP_LEAD_MIN) & (false_runs <= OP_FALSE_RUNS)).sum()),
        "files_with_final_run": int((lead >= 0).sum()),
        # the rule alone: judged by its lead (first trigger inside the pre-failure window), not by a run
        "files_rule_lead_ok": int(sum(1 for v in per_file.values() if v["rule_lead_min"] >= OP_LEAD_MIN)),
        "final_lead_min_median": float(np.median(lead[lead >= 0])) if (lead >= 0).any() else np.nan}
    m["per_class"], m["per_file"] = per_class, per_file
    return m


def relabel(w, label_window):
    """Recompute `label` (pre-failure window) from ttf_s - lets the evaluation vary the label window
    without re-running preprocess.py."""
    w = w.copy()
    w["label"] = (w["ttf_s"] <= label_window).astype("float32")
    return w


def cv_folds(files, k, seed):
    """Gun-level k-fold, stratified by class: list of k validation file lists."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in CLASSES:
        fs = sorted(f for f in files if os.path.basename(f).startswith(c))
        rng.shuffle(fs)
        for i, f in enumerate(fs):
            folds[i % k].append(f)
    return [sorted(f) for f in folds]


def print_metrics(tag, v):
    lo, hi, sd = v.get("alarm_rate_normal_per_file_min_max_std", (np.nan, np.nan, np.nan))
    print(f"{tag}: AUROC {v['auroc']:.3f} (per gun mean {v.get('auroc_gun_mean', np.nan):.3f}, "
          f"> {RULE_LEAD_S // 60} min before failure {v.get('auroc_pre_rule', np.nan):.3f})  "
          f"AUPRC {v['auprc']:.3f}  alarm@normal {v['alarm_rate_normal']:.3f}  "
          f"recall@pre-failure {v['recall_pre_failure']:.3f}  ({v['n_windows']:,} windows"
          + (f", {v['held_windows']:,} alarms held by the non-welding gate" if v.get("alarm_gate_non_welding") is not None else "")
          + f"; per-file alarm@normal {lo:.3f}..{hi:.3f} sd {sd:.3f}"
          + (f"; {v['files_gun_normalised']} files gun-normalised" if v.get("files_gun_normalised") else "")
          + (f", {v['files_gun_threshold']} with a gun threshold" if v.get("files_gun_threshold") else "") + ")")
    for k, r in v["per_class"].items():
        print(f"  {k}: AUROC {r['auroc']:.3f}  recall {r['recall_pre_failure']:.3f}  alarm@normal {r['alarm_rate_normal']:.3f}")
    op, rule = v.get("operating_point"), v.get("rule_terminal_code")
    if op and rule:
        print(f"  operating point (lead >= {op['lead_min_target']:.0f} min, false runs <= {op['false_runs_per_day_target']:.0f}/day): "
              f"{op['files_both_ok']}/{op['files']} files (lead ok {op['files_lead_ok']}, false runs ok {op['files_false_runs_ok']}); "
              f"final run in {op['files_with_final_run']} files (median lead {op['final_lead_min_median']:.1f} min)")
        print(f"  terminal-code rule (any code, episode start, {rule['cooldown_s'] // 60} min cooldown): "
              f"{rule['files_hit']}/{op['files']} files hit, median lead {rule['rule_lead_min_median']:.1f} min; "
              f"false triggers {rule['false_triggers_per_day_mean']:.2f}/day/file "
              f"({rule['files_with_false_trigger']} files with any, {rule['files_false_triggers_ok']} files <= "
              f"{op['false_runs_per_day_target']:.0f}/day); {rule.get('n_repeats', 0)} repeats not critical "
              f"({rule.get('n_repeats_in_pre_failure', 0)} in the pre-failure window), "
              f"{rule.get('all_triggers_false_per_day_mean', np.nan):.2f}/day counting them")


def test_files_in(test_dir):
    return sorted(glob.glob(os.path.join(test_dir, "test_*.parquet")))


def evaluate_test(model, feat_cols, model_cols, window, threshold, sustain, files, score_dir=None, tag="", gate=None,
                  gn=None, label_window=None, warmup_alarms=True):
    """Window + score the preprocessed test files with the frozen model/threshold (+ the bundle's gun
    normalisation) and run the same evaluation as for validation. Optionally writes per-file score
    CSVs (time, score, alarm, threshold, ...)."""
    ws, cs = [], []
    norm_cols = gun_norm_columns(feat_cols, gn.get("columns")) if gn else None
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        w, calib = window_file(df, window, feat_cols, gn, norm_cols)
        if label_window:
            w = relabel(w, label_window)
        ws.append(w)
        if calib is not None:
            cs.append(calib)
        print(f"[test {i}/{len(files)}] {os.path.basename(f)} (class {df['class'].iloc[0]}): "
              f"{len(df):,} rows -> {len(w):,} windows{'' if calib is None else ' (gun-normalised)'}", flush=True)
    test_w = pd.concat(ws)
    scores = anomaly_score(model, test_w[model_cols].to_numpy(dtype=np.float32))
    gun_thr = gun_thresholds(model, model_cols, pd.concat(cs) if cs else None, threshold, (gn or {}).get("threshold_q"), gate)
    m = evaluate(test_w, scores, threshold, sustain, gate, gun_thr, warmup_alarms)
    m["files"] = [os.path.basename(f) for f in files]
    m["file_class"] = {f: str(c) for f, c in test_w.groupby("file")["class"].first().items()}
    m["class_source"] = "inferred by preprocess.py from the terminal code in the last 10 min"
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)
        thr = window_thresholds(test_w, threshold, gun_thr)
        test_w = test_w.assign(score=scores, alarm=alarm_mask(scores, thr, test_w["non_welding"].values, gate,
                                                              warmup_flags(test_w, warmup_alarms)),
                               threshold=thr)
        for f, part in test_w.groupby("file"):
            out_cols = ["score", "alarm", "threshold"] + [c for c in SCORE_META_COLS if c in part.columns]
            part[out_cols].to_csv(os.path.join(score_dir, f"{f}_{tag}.csv"))
    return m


# ------------------------------------------------------------- inference
# window columns written next to the scores (score CSVs, score_frame)
SCORE_META_COLS = ("ttf_s", "label", "error_active", "terminal_any", "terminal_idx", "non_welding", "warmup")


def load_bundle(path):
    return joblib.load(path)


def score_frame(bundle, df):
    """Preprocessed 1 Hz frame (as written by preprocess.py) -> DataFrame[time, score, alarm, threshold, ...]."""
    gn, gate = bundle.get("gun_norm"), bundle.get("alarm_max_non_welding")
    norm_cols = gun_norm_columns(bundle["feature_cols"], gn.get("columns")) if gn else None
    w, calib = window_file(df, bundle["window"], bundle["feature_cols"], gn, norm_cols)
    label_window = (bundle.get("args") or {}).get("label_window")
    if label_window and "ttf_s" in w.columns:
        w = relabel(w, label_window)
    s = anomaly_score(bundle["model"], w[bundle["model_cols"]].to_numpy(dtype=np.float32))
    gun_thr = gun_thresholds(bundle["model"], bundle["model_cols"], calib, bundle["threshold"],
                             (gn or {}).get("threshold_q"), gate)
    thr = window_thresholds(w, bundle["threshold"], gun_thr)
    alarm = alarm_mask(s, thr, w["non_welding"].values if "non_welding" in w.columns else None, gate,
                       warmup_flags(w, bundle.get("warmup_alarms", True)))
    out = pd.DataFrame({"score": s, "alarm": alarm, "threshold": thr}, index=w.index)
    for c in SCORE_META_COLS:
        if c in w.columns:
            out[c] = w[c].values
    return out


# ------------------------------------------------------------------- main
def split_files(files, val_frac, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for k in CLASSES:
        fs = sorted(f for f in files if os.path.basename(f).startswith(k))
        rng.shuffle(fs)
        n_val = max(1, int(round(len(fs) * val_frac))) if len(fs) > 1 else 0
        val += fs[:n_val]
        train += fs[n_val:]
    return sorted(train), sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "preprocessed"))
    ap.add_argument("--test-dir", default=None, help="preprocessed test files; default <data-dir>/test. "
                    "Evaluated after training when present; pass --no-test to skip")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--evaluate", action="store_true", help="skip training; evaluate the saved bundle on --test-dir")
    ap.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--model", choices=["iforest", "pca", "ensemble"], default="ensemble",
                    help="ensemble = IsolationForest + LightGBM, the production model (needs lightgbm; history.md "
                         "§13, §18); iforest = IsolationForest alone")
    ap.add_argument("--lgbm-weight", type=float, default=0.5, help="--model ensemble: weight of the LightGBM score")
    ap.add_argument("--ensemble-components", default="iforest,lgbm",
                    help="--model ensemble: comma list of iforest, lgbm, lgbm6h, et. The production mix is "
                         "iforest,lgbm; iforest,lgbm,lgbm6h,et won the 64-gun CV but fell on the test guns (B-9)")
    ap.add_argument("--ensemble-neg-sub", type=int, default=None,
                    help="--model ensemble: normal windows each supervised component is fit on (B-9: 150000)")
    ap.add_argument("--window", type=int, default=60, help="window length in seconds")
    ap.add_argument("--val-frac", type=float, default=0.25, help="share of files per class held out")
    ap.add_argument("--threshold-q", type=float, default=0.99, help="quantile of train-normal scores")
    ap.add_argument("--sustain", type=int, default=3, help="consecutive alarm windows for a sustained alarm")
    ap.add_argument("--alarm-max-non-welding", type=lambda v: None if str(v).lower() in ("none", "off") else float(v),
                    default=0.5, help="windows with a larger non-welding share never alarm (none = no gate)")
    ap.add_argument("--exclude-non-welding", action="store_true", help="drop cap-dressing windows from training")
    ap.add_argument("--warmup-alarms", action="store_true",
                    help="let warm-up windows (global scale) alarm, as before 2026-09-29; default: held (history.md §17)")
    ap.add_argument("--drop-features", nargs="*", default=DROP_FEATURES_DEFAULT,
                    help="preprocessed columns kept out of the model input (default: c19, a time counter = label leak)")
    ap.add_argument("--gun-norm", choices=["none", "center", "scale"], default=GUN_NORM_DEFAULT["mode"],
                    help="per-gun re-normalisation from a warm-up period: center = subtract the gun mean, "
                         "scale = also divide by the gun std (floored), none = global z-score only")
    ap.add_argument("--warmup-hours", type=float, default=GUN_NORM_DEFAULT["warmup_s"] / 3600,
                    help="warm-up period per file / stream for the gun statistics (with --warmup-rows: its cap)")
    ap.add_argument("--warmup-rows", type=int, default=GUN_NORM_DEFAULT["warmup_rows"],
                    help="end the warm-up once this many normal welding rows are collected (default: fixed length)")
    ap.add_argument("--warmup-min-rows", type=int, default=GUN_NORM_DEFAULT["min_rows"],
                    help="fewer normal welding rows in the warm-up -> the file keeps the global scaling")
    ap.add_argument("--std-floor", type=float, default=GUN_NORM_DEFAULT["std_floor"], help="--gun-norm scale: min gun std (z units)")
    ap.add_argument("--no-gun-threshold", action="store_true",
                    help="judge every gun against the global threshold; default: per-gun threshold = max(global, "
                         "--threshold-q quantile of the gun-normalised warm-up window scores)")
    ap.add_argument("--label-window", type=int, default=None,
                    help="pre-failure window in seconds for training / evaluation (default: the one preprocess.py used, 3600)")
    ap.add_argument("--cv", type=int, default=None, help="also run a gun-level k-fold (metrics['cv']) before the final fit")
    ap.add_argument("--max-train-windows", type=int, default=None, help="subsample normal windows")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", default=None, help="skip training; score this preprocessed parquet with the saved model")
    args = ap.parse_args()

    if args.score:
        b = load_bundle(os.path.join(args.model_dir, f"baseline_{args.model}.joblib"))
        out = score_frame(b, pd.read_parquet(args.score))
        os.makedirs(os.path.join(args.model_dir, "scores"), exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.score))[0]
        dst = os.path.join(args.model_dir, "scores", f"{stem}_{args.model}.csv")
        out.to_csv(dst)
        print(f"{len(out)} windows, alarm rate {out['alarm'].mean():.3f}, threshold {b['threshold']:.4f} -> {dst}")
        return

    test_dir = args.test_dir or os.path.join(args.data_dir, "test")
    test_files = [] if args.no_test else test_files_in(test_dir)
    if args.evaluate:
        if not test_files:
            raise SystemExit(f"no test_*.parquet in {test_dir} - run preprocess.py --split test first")
        bundle_path = os.path.join(args.model_dir, f"baseline_{args.model}.joblib")
        if not os.path.exists(bundle_path):
            raise SystemExit(f"{bundle_path} not found - train first")
        b = load_bundle(bundle_path)
        print(f"evaluating {bundle_path} (window {b['window']}s, threshold {b['threshold']:.4f}, "
              f"sustain {b['sustain']}) on {len(test_files)} test files")
        m = evaluate_test(b["model"], b["feature_cols"], b["model_cols"], b["window"], b["threshold"],
                          b["sustain"], test_files, os.path.join(args.model_dir, "scores"), args.model,
                          gate=b.get("alarm_max_non_welding"), gn=b.get("gun_norm"),
                          label_window=(b.get("args") or {}).get("label_window"),
                          warmup_alarms=b.get("warmup_alarms", True))
        print_metrics("test", m)
        dst = os.path.join(args.model_dir, f"baseline_{args.model}_test_metrics.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump({"bundle": bundle_path, "created": b.get("created"), "threshold": b["threshold"],
                       "window": b["window"], "sustain": b["sustain"],
                       "alarm_max_non_welding": b.get("alarm_max_non_welding"), "gun_norm": b.get("gun_norm"), "test": m,
                       "evaluated": dt.datetime.now().isoformat(timespec="seconds")}, f, indent=2, default=str)
        print(f"saved {dst}")
        return

    files = sorted(glob.glob(os.path.join(args.data_dir, "E0*.parquet")))
    if not files:
        raise SystemExit(f"no E0*.parquet in {args.data_dir} — run preprocess.py first")
    if args.max_files:  # keep the classes balanced in a quick run
        per_class = [[f for f in files if os.path.basename(f).startswith(k)] for k in CLASSES]
        files = sorted(f for fs in per_class for f in fs[: -(-args.max_files // len(CLASSES))])
    train_files, val_files = split_files(files, args.val_frac, args.seed)
    print(f"train files {len(train_files)}, val files {len(val_files)}")

    scaler_path = os.path.join(args.data_dir, "scaler.json")
    config_path = os.path.join(args.data_dir, "preprocess_config.json")
    scaler = json.load(open(scaler_path, encoding="utf-8")) if os.path.exists(scaler_path) else None
    gn = None
    if args.gun_norm != "none":
        gn = {"mode": args.gun_norm, "warmup_s": int(round(args.warmup_hours * 3600)), "warmup_rows": args.warmup_rows,
              "min_rows": args.warmup_min_rows,
              "std_floor": args.std_floor, "threshold_q": None if args.no_gun_threshold else args.threshold_q}

    t0 = time.time()
    feat_cols, gn_cols, parts, calibs = None, None, {}, {}
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f)
        feat_cols = feat_cols or feature_columns(df, args.drop_features)
        if gn and gn_cols is None:
            gn_cols = gun_norm_columns(feat_cols, (scaler or {}).get("columns"))
            gn["columns"] = gn_cols
        w, calib = window_file(df, args.window, feat_cols, gn, gn_cols)
        if args.label_window:
            w = relabel(w, args.label_window)
        parts[f] = w
        if calib is not None:
            calibs[f] = calib
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {len(df):,} rows -> {len(w):,} windows"
              f"{'' if calib is None else ' (gun-normalised)'}", flush=True)
    print(f"windowed in {time.time() - t0:.0f}s" + (f", gun-norm {gn['mode']} on {gn_cols}" if gn else ""))
    model_cols = model_input_columns(feat_cols)
    q_gun = (gn or {}).get("threshold_q")

    def fit_and_eval(tr_files, va_files):
        tr = pd.concat([parts[f] for f in tr_files])
        # the warm-up windows stay in the fit at the GLOBAL scale on purpose: warm-up windows are scored that way
        # online, and without them the normal alarm rate triples (val 1.0 -> 3.2 %, test 0.5 -> 1.5 %; history.md §11)
        normal = (tr["label"] == 0) & (tr["error_active"] == 0)
        if args.exclude_non_welding:
            normal &= tr["non_welding"] == 0
        fit = tr[normal]
        if args.max_train_windows and len(fit) > args.max_train_windows:
            fit = fit.sample(args.max_train_windows, random_state=args.seed)
        X = fit[model_cols].to_numpy(dtype=np.float32)
        model = build_model(args.model, args.seed, model_cols, args.lgbm_weight,
                            tuple(args.ensemble_components.split(",")), args.ensemble_neg_sub)
        if args.model == "ensemble":
            # positives get the negatives' non-welding filter: with it on one side only, carried-forward constants
            # (std ~ 0) appear among the positives alone and LightGBM learns "cap dressing -> failure" (MEMORY §7-1)
            pos = tr[(tr["label"] == 1) & ((tr["non_welding"] == 0) if args.exclude_non_welding else True)]
            model.fit(X, pos[model_cols].to_numpy(dtype=np.float32), fit["file"].to_numpy(), pos["file"].to_numpy(),
                      fit["ttf_s"].to_numpy())
            s_tr = model.fit_scores_  # out-of-fold for the supervised half
        else:
            model.fit(X)
            s_tr = anomaly_score(model, X)
        thr = float(np.quantile(s_tr, args.threshold_q))
        m_val = None
        if va_files:
            va = pd.concat([parts[f] for f in va_files])
            cal = [calibs[f] for f in va_files if f in calibs]
            gun_thr = gun_thresholds(model, model_cols, pd.concat(cal) if cal else None, thr, q_gun, args.alarm_max_non_welding)
            m_val = evaluate(va, anomaly_score(model, va[model_cols].to_numpy(dtype=np.float32)), thr, args.sustain,
                             args.alarm_max_non_welding, gun_thr, args.warmup_alarms)
        return model, X, s_tr, thr, m_val

    cv = None
    if args.cv:
        folds = cv_folds(files, args.cv, args.seed)
        fold_metrics = []
        for i, va_files in enumerate(folds, 1):
            tr_files = [f for f in files if f not in va_files]
            _, _, _, thr_i, m_i = fit_and_eval(tr_files, va_files)
            fold_metrics.append({k: m_i[k] for k in ("auroc", "auroc_gun_mean", "auroc_pre_rule", "auprc", "recall_pre_failure",
                                                     "alarm_rate_normal", "alarm_rate_normal_per_file_min_max_std")}
                                | {"threshold": thr_i, "val_files": [os.path.basename(f) for f in va_files],
                                   "alarm_rate_normal_per_file": {f: v["alarm_rate_normal"] for f, v in m_i["per_file"].items()},
                                   "operating_point": m_i["operating_point"], "per_class": m_i["per_class"]})
            print(f"cv fold {i}/{args.cv}: AUROC {m_i['auroc']:.3f}  recall {m_i['recall_pre_failure']:.3f}  "
                  f"alarm@normal {m_i['alarm_rate_normal']:.3f}  ({len(va_files)} guns)", flush=True)
        keys = ("auroc", "auroc_gun_mean", "auroc_pre_rule", "auprc", "recall_pre_failure", "alarm_rate_normal")
        cv = {"k": args.cv, "folds": fold_metrics,
              "mean": {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys},
              "sd": {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in keys},
              "files_both_ok": int(sum(m["operating_point"]["files_both_ok"] for m in fold_metrics)),
              "files_with_final_run": int(sum(m["operating_point"]["files_with_final_run"] for m in fold_metrics))}
        print(f"cv {args.cv}-fold: AUROC {cv['mean']['auroc']:.3f} +- {cv['sd']['auroc']:.3f} "
              f"(per gun {cv['mean']['auroc_gun_mean']:.3f} +- {cv['sd']['auroc_gun_mean']:.3f})  "
              f"recall {cv['mean']['recall_pre_failure']:.3f} +- {cv['sd']['recall_pre_failure']:.3f}  "
              f"alarm@normal {cv['mean']['alarm_rate_normal']:.3f} +- {cv['sd']['alarm_rate_normal']:.3f}  "
              f"operating point {cv['files_both_ok']}/{len(files)} guns", flush=True)

    print(f"fitting {args.model} on the normal windows of {len(train_files)} files x {len(model_cols)} features")
    model, X, train_scores, threshold, m_val = fit_and_eval(train_files, val_files)
    print(f"fitted on {len(X):,} normal windows, threshold {threshold:.4f}")

    metrics = {"train": {"n_fit_windows": int(len(X)), "threshold": threshold,
                         "score_mean": float(train_scores.mean()), "score_std": float(train_scores.std())}}
    if cv:
        metrics["cv"] = cv
    if m_val is not None:
        metrics["val"] = m_val
        print_metrics("val", metrics["val"])
    if test_files:
        metrics["test"] = evaluate_test(model, feat_cols, model_cols, args.window, threshold, args.sustain, test_files,
                                        gate=args.alarm_max_non_welding, gn=gn, label_window=args.label_window,
                                        warmup_alarms=args.warmup_alarms)
        print_metrics("test", metrics["test"])
    else:
        print(f"no test files in {test_dir} - skipped test evaluation")

    os.makedirs(args.model_dir, exist_ok=True)
    bundle = {
        "model": model, "model_type": args.model, "feature_cols": feat_cols, "model_cols": model_cols,
        "window": args.window, "threshold": threshold, "threshold_q": args.threshold_q, "sustain": args.sustain,
        "alarm_max_non_welding": args.alarm_max_non_welding,
        "warmup_alarms": args.warmup_alarms,  # False: the model raises no alarm in a gun's warm-up (main.py holds it)
        # terminal-code rule (main.py fires it on an episode start, then keeps quiet for cooldown_s; a code that
        # already fired < repeat_s earlier is a repeat: reported, not critical)
        "rule": {"codes": list(TERMINAL_CODES), "cooldown_s": RULE_COOLDOWN_S, "repeat_s": RULE_REPEAT_S},
        # per-gun normalisation recipe (None = global z-score only); main.py reproduces it online
        "gun_norm": gn,
        "dropped_features": list(args.drop_features),
        # per-feature reference (median of the normal training windows): the serving layer replaces one
        # feature at a time with it to attribute a score to individual features
        "feature_reference": np.median(X, axis=0).astype(float).tolist(),
        "feature_scale": X.std(axis=0).astype(float).tolist(),
        "scaler": scaler,
        # the serving layer reproduces the preprocessing online from these (gap limit, c16 rule, resample)
        "preprocess_config": json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else None,
        "train_files": [os.path.basename(f) for f in train_files],
        "val_files": [os.path.basename(f) for f in val_files],
        "metrics": metrics, "args": vars(args), "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(args.model_dir, f"baseline_{args.model}.joblib")
    joblib.dump(bundle, path, compress=3)
    with open(os.path.join(args.model_dir, f"baseline_{args.model}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2, default=str)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    # bundles must pickle the model classes as train.X (what main.py / load_bundle import), not __main__.X
    sys.modules.setdefault("train", sys.modules["__main__"])
    for _cls in (PCADetector, EnsembleDetector):
        _cls.__module__ = "train"
    main()
