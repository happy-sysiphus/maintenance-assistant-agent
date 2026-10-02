"""
B-2 / B-4 / B-5 experiment lab (history.md, old test.md §4 and §11; 2026-09-23 / 09-26).

Windows the preprocessed files ONCE per window length - with the production per-gun
normalisation from train.py - into results/B-2/cache/, then runs the experiments on those
frames and writes one JSON per experiment to results/B-2/ (B-5: results/B-5/). Nothing here touches models/.

The cache keeps EVERY preprocessed column, c19 included: the plain names below (base60, sup_lgbm60, ...) are
the 2026-09-23 runs WITH the c19 label leak and are kept only for the record. Honest runs drop it - the
`noc19_` / `clean_` prefixes and every B-5 experiment do.

    python .py/experiments.py cache --window 60          # ~3 min, once (rebuild after train.window_features changes)
    python .py/experiments.py cache --window 300
    python .py/experiments.py cache-maint                # B-5 maintenance-history features, ~5 min
    python .py/experiments.py run all                    # or a list of experiment names
    python .py/experiments.py summary                    # markdown table of every result JSON

Experiments (the names are the JSON file names; "48 features" = 46 honest + c19_mean/_std):
    base60            IsolationForest on the 48 window features            (the 2026-09-23 production model)
    hist_std60        + rolling-history features of the 24 `_std` features  (B-2 variability / trend)
    hist_all60        + rolling-history features of all 48 features
    base300           5-minute windows                                       (B-2 `--window 300`)
    hist_std300       5-minute windows + history of the `_std` features
    ewma5 / ewma15    EWMA of the base60 score (half-life 5 / 15 windows)    (B-2 temporal memory)
    cusum             CUSUM of the standardised base60 score (k = 0.5)
    lstm_ae60         LSTM auto-encoder on 30-window sequences, trained on normal sequences
    sup_lgbm60        supervised LightGBM on `label` - the upper bound of these features (B-2)
    sup_lgbm_hist60   supervised LightGBM with the history features
    sup_lgbm300       supervised LightGBM on 5-minute windows
    e02_only60        IsolationForest trained on E02 files only, evaluated on E02 (B-2 class split)
    no_e02_60         base60 evaluated without the E02 files
    labelwin_<s>      base60 + LightGBM with the pre-failure window = <s> seconds (B-4 label sensitivity)
    cv4_base60        gun-level 4-fold of base60 (B-4 evaluation design)
  B-5 (2026-09-26, c19 always dropped, gun-level 4-fold over the 64 train guns + the 8 test guns per fold model,
  pooled AND per-gun AUROC):
    p3_labelcv        IsolationForest + LightGBM for pre-failure windows of 900 / 1800 / 2700 / 3600 / 7200 s
    p3_maint          + maintenance-history features (cap dressing / c1 cap-offset changes), IF + LightGBM
Every result carries train.evaluate()'s operating-point block and the terminal-code rule baseline.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402

ROOT = T.PROJECT_ROOT
DATA = os.path.join(ROOT, "preprocessed")
OUT = os.path.join(ROOT, "results", "B-2")
OUT3 = os.path.join(ROOT, "results", "B-5")
CACHE = os.path.join(OUT, "cache")
SEED = 42
Q = 0.99
SUSTAIN = 3
GATE = 0.5
HIST_SPANS = (5, 30, 60, 1440)  # windows: 5 min, 30 min, 1 h, 24 h (at 60 s windows)
# c19 ("offset value in robot") is a counter that grows ~55/s through every file; every file is
# exactly 7 days and ends at its failure, so after gun-centring c19 == time since start ==
# 168 h - ttf: a label leak, not a sensor. `noc19_*` drops it, `clean_*` also drops the
# time-of-day features and the gun constants c7-c9 (which only change in 3 files).
DROP = {"noc19": ["c19"], "clean": ["c19", "hour_sin", "hour_cos", "c7", "c8", "c9"]}


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ cache
def build_cache(window):
    os.makedirs(CACHE, exist_ok=True)
    scaler = json.load(open(os.path.join(DATA, "scaler.json"), encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(DATA, "E0*.parquet")))
    tests = T.test_files_in(os.path.join(DATA, "test"))
    gn = {**T.GUN_NORM_DEFAULT, "threshold_q": Q}
    feat_cols = None
    for split, fs in (("train", files), ("test", tests)):
        ws, cs = [], []
        for i, f in enumerate(fs, 1):
            df = pd.read_parquet(f)
            if feat_cols is None:
                feat_cols = T.feature_columns(df)
                gn["columns"] = T.gun_norm_columns(feat_cols, scaler["columns"])
            w, calib = T.window_file(df, window, feat_cols, gn, gn["columns"])
            ws.append(w)
            if calib is not None:
                cs.append(calib)
            log(f"[{split} {i}/{len(fs)}] {os.path.basename(f)}: {len(w):,} windows")
        for tag, frames in (("w", ws), ("calib", cs)):
            x = pd.concat(frames)
            x.index.name = "time"
            x.reset_index().to_parquet(os.path.join(CACHE, f"{tag}{window}_{split}.parquet"))
    json.dump({"feature_cols": feat_cols, "gun_norm": gn, "window": window},
              open(os.path.join(CACHE, f"meta{window}.json"), "w", encoding="utf-8"), indent=2)
    log(f"cached window={window}s in {CACHE}")


def load(window):
    meta = json.load(open(os.path.join(CACHE, f"meta{window}.json"), encoding="utf-8"))
    w = pd.read_parquet(os.path.join(CACHE, f"w{window}_train.parquet")).set_index("time")
    te = pd.read_parquet(os.path.join(CACHE, f"w{window}_test.parquet")).set_index("time")
    cal = pd.read_parquet(os.path.join(CACHE, f"calib{window}_train.parquet")).set_index("time")
    cte = pd.read_parquet(os.path.join(CACHE, f"calib{window}_test.parquet")).set_index("time")
    files = sorted(glob.glob(os.path.join(DATA, "E0*.parquet")))
    tr_files, va_files = T.split_files(files, 0.25, SEED)  # the production split
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
    tr_s, va_s = {stem(f) for f in tr_files}, {stem(f) for f in va_files}
    D = {"meta": meta, "window": window, "model_cols": T.model_input_columns(meta["feature_cols"]),
         "all": w, "cal_all": cal, "te": te, "cte": cte, "files": sorted(tr_s | va_s), "tr_files": sorted(tr_s),
         "va_files": sorted(va_s)}
    D["tr"], D["va"] = w[w["file"].isin(tr_s)], w[w["file"].isin(va_s)]
    D["cva"] = cal[cal["file"].isin(va_s)]
    log(f"loaded window={window}s: train {len(D['tr']):,} / val {len(D['va']):,} / test {len(te):,} windows, "
        f"{len(D['model_cols'])} features")
    return D


# --------------------------------------------------------- feature engineering
def add_history(w, base_cols, spans=HIST_SPANS):
    """Rolling means of `base_cols` over the past k windows of the same file (chronological), plus a
    short-vs-long trend (5 vs 60 windows) and a 1 h-vs-24 h difference. Online this needs a
    per-gun history of aligned windows (not implemented in main.py - experiment only)."""
    w = w.reset_index().sort_values(["file", "time"]).set_index("time")
    g = w.groupby("file", sort=False)[base_cols]
    roll = {k: g.rolling(k, min_periods=1).mean().to_numpy(dtype=np.float32) for k in spans}
    new = {}
    for j, c in enumerate(base_cols):
        new[f"h5_{c}"] = roll[5][:, j]
        new[f"h30_{c}"] = roll[30][:, j]
        new[f"tr_{c}"] = roll[5][:, j] - roll[60][:, j]
        new[f"d24_{c}"] = roll[60][:, j] - roll[1440][:, j]
    return pd.concat([w, pd.DataFrame(new, index=w.index)], axis=1), list(new)


def relabel_all(D, L):
    return {k: (T.relabel(v, L) if isinstance(v, pd.DataFrame) and "ttf_s" in v.columns else v) for k, v in D.items()}


# ------------------------------------------------------------------ scoring
def iforest_fit(X):
    return T.build_model("iforest", SEED).fit(X)


def iforest_score(model, df, cols):
    return T.anomaly_score(model, df[cols].to_numpy(dtype=np.float32))


def chronological(df):
    """file -> row positions in chronological order."""
    t = df.index.values
    return {f: idx[np.argsort(t[idx])] for f, idx in df.groupby("file").indices.items()}


def seq_transform(scores, df, kind, p, mu=0.0, sd=1.0):
    """EWMA (half-life p windows) or CUSUM (drift k = p, on the standardised score) of a score
    series, per file in time order. Online: one scalar of state per gun."""
    out = np.empty_like(scores, dtype=np.float64)
    for f, order in chronological(df).items():
        s = scores[order]
        if kind == "ewma":
            v = pd.Series(s).ewm(halflife=p, adjust=False).mean().to_numpy()
        else:
            z = (s - mu) / sd
            v, acc = np.empty_like(z), 0.0
            for i in range(len(z)):
                acc = max(0.0, acc + z[i] - p)
                v[i] = acc
        out[order] = v
    return out


def normal_mask(w, exclude_nw=True):
    m = (w["label"] == 0) & (w["error_active"] == 0)
    if exclude_nw:
        m &= w["non_welding"] == 0
    return m.to_numpy()


def gun_thr(model, score_fn, calib, thr, q):
    """train.gun_thresholds with any score function (same calibration windows, same minimum)."""
    out = {}
    if calib is None or len(calib) == 0:
        return out
    for f, part in T.calibration_windows(calib, GATE).groupby("file"):
        g = T.gun_threshold(score_fn(model, part), thr, q)
        if g is not None:
            out[f] = g
    return out


def evaluate_split(df, scores, thr, gthr):
    return T.evaluate(df, scores, thr, SUSTAIN, GATE, gthr)


def run_unsup(name, D, cols, fit_fn, score_fn, note="", thr_frame=None, write=True):
    """Generic unsupervised run: fit on normal training windows, threshold at the Q quantile of the
    training-normal scores (optionally of a sequence-transformed score), per-gun thresholds from
    the calibration windows, train.evaluate() on val and test."""
    t0 = time.time()
    tr = D["tr"]
    X = tr.loc[normal_mask(tr), cols].to_numpy(dtype=np.float32)
    model = fit_fn(X)
    s_tr = score_fn(model, tr if thr_frame is None else thr_frame)
    thr = float(np.quantile(s_tr[normal_mask(tr)], Q))
    res = {"name": name, "note": note, "window": D["window"], "n_fit": int(len(X)), "n_features": len(cols),
           "threshold": thr, "fit_s": round(time.time() - t0, 1)}
    for split, key, ckey in (("val", "va", "cva"), ("test", "te", "cte")):
        df = D[key]
        s = score_fn(model, df)
        res[split] = evaluate_split(df, s, thr, gun_thr(model, score_fn, D[ckey], thr, Q))
    if write:
        save(res)
    return res, model


def save(res):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{res['name']}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    line(res)


def line(r):
    def part(m):
        op = m["operating_point"]
        return (f"AUROC {m['auroc']:.3f} recall {m['recall_pre_failure']:.3f} alarm {m['alarm_rate_normal']:.3f} "
                f"| final runs {op['files_with_final_run']}/{op['files']} (median {op['final_lead_min_median']:.1f} min) "
                f"op {op['files_both_ok']}")
    log(f"== {r['name']:<16} {r.get('n_features', '-'):>4} feats thr {r.get('threshold', float('nan')):.4f} "
        f"| val {part(r['val'])} | test {part(r['test'])}")


# ------------------------------------------------------------------ models
class LSTMAE:
    """Sequence auto-encoder: LSTM encoder -> latent -> LSTM decoder, trained on sequences of
    `seq_len` consecutive normal windows; the score of a window is the reconstruction error of
    the sequence that ends with it."""

    def __init__(self, cols, seq_len=30, hidden=32, epochs=6, batch=512, max_seq=60000, seed=SEED):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.cols, self.seq_len, self.hidden, self.epochs, self.batch, self.max_seq = cols, seq_len, hidden, epochs, batch, max_seq
        n = len(cols)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.LSTM(n, hidden, batch_first=True)
                s.dec = nn.LSTM(hidden, hidden, batch_first=True)
                s.out = nn.Linear(hidden, n)

            def forward(s, x):
                _, (h, _) = s.enc(x)
                z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = s.dec(z)
                return s.out(y)

        self.net = Net()

    def _sequences(self, df, only_normal):
        """(n_seq, L, n) float tensor of consecutive windows per file + the row positions of the
        last window of each sequence."""
        torch = self.torch
        xs, pos = [], []
        norm = normal_mask(df) if only_normal else None
        vals = np.clip(df[self.cols].to_numpy(dtype=np.float32), -10, 10)
        for f, order in chronological(df).items():
            if len(order) < self.seq_len:
                continue
            x = torch.from_numpy(vals[order]).unfold(0, self.seq_len, 1).permute(0, 2, 1)  # (n-L+1, L, n)
            last = order[self.seq_len - 1:]
            if only_normal:
                ok = torch.from_numpy(norm[order]).float().unfold(0, self.seq_len, 1).min(dim=1).values > 0
                x, last = x[ok], last[ok.numpy()]
            xs.append(x)
            pos.append(last)
        return torch.cat(xs), np.concatenate(pos)

    def fit(self, df):
        torch, nn = self.torch, self.nn
        X, _ = self._sequences(df, only_normal=True)
        if len(X) > self.max_seq:
            X = X[torch.randperm(len(X))[: self.max_seq]]
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), self.batch):
                xb = X[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = loss_fn(self.net(xb), xb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(xb)
            log(f"  lstm-ae epoch {ep + 1}/{self.epochs}: mse {tot / len(X):.4f} ({len(X):,} sequences)")
        self.net.eval()
        return self

    def score(self, df):
        torch = self.torch
        X, pos = self._sequences(df, only_normal=False)
        out = np.full(len(df), np.nan)
        with torch.no_grad():
            errs = []
            for i in range(0, len(X), 4096):
                xb = X[i:i + 4096]
                errs.append(((self.net(xb) - xb) ** 2).mean(dim=(1, 2)).numpy())
        out[pos] = np.concatenate(errs) if errs else []
        # the first L-1 windows of a file have no full sequence: give them the first available score
        for f, order in chronological(df).items():
            s = out[order]
            if np.isnan(s).all():
                out[order] = 0.0
            else:
                first = s[~np.isnan(s)][0]
                s[np.isnan(s)] = first
                out[order] = s
        return out


def run_supervised(name, D, cols, note="", write=True):
    """LightGBM on label (pre-failure 1 h = 1, normal = 0), gun-level split: the AUROC a model can
    reach with these features when it is TOLD the answer. Not a candidate for production."""
    t0 = time.time()
    tr = D["tr"]
    pos, neg = (tr["label"] == 1).to_numpy(), normal_mask(tr, exclude_nw=False)
    keep = pos | neg
    X, y = tr.loc[keep, cols].to_numpy(dtype=np.float32), pos[keep].astype(int)
    model = T.lgbm_classifier(int(neg.sum()), int(pos.sum()), SEED).fit(X, y)
    score_fn = lambda m, df: m.predict_proba(df[cols].to_numpy(dtype=np.float32))[:, 1]  # noqa: E731
    # in-sample: boosted models under-score their own training normals, so this threshold alarms more than Q
    # promises on val / test (the ensemble uses out-of-fold scores - train.EnsembleDetector). AUROCs are unaffected.
    thr = float(np.quantile(score_fn(model, tr)[neg], Q))
    imp = sorted(zip(cols, model.booster_.feature_importance("gain")), key=lambda t: -t[1])[:15]
    tot = float(sum(model.booster_.feature_importance("gain"))) or 1.0
    res = {"name": name, "note": note, "window": D["window"], "supervised": True, "n_fit": int(len(X)),
           "n_pos": int(y.sum()), "n_features": len(cols), "threshold": thr, "fit_s": round(time.time() - t0, 1),
           "top_features_gain_share": [(c, round(float(g) / tot, 3)) for c, g in imp]}
    for split, key, ckey in (("val", "va", "cva"), ("test", "te", "cte")):
        df = D[key]
        res[split] = evaluate_split(df, score_fn(model, df), thr, gun_thr(model, score_fn, D[ckey], thr, Q))
    if write:
        save(res)
    return res


# ------------------------------------------------------------- experiments
def exp_base(D, name="base60"):
    return run_unsup(name, D, D["model_cols"], iforest_fit, lambda m, df: iforest_score(m, df, D["model_cols"]),
                     "IsolationForest, 48 window features, gun-centred, gun thresholds (= production recipe)")


def without(D, drop):
    """Copy of D whose model_cols exclude the `drop` base features (both _mean and _std)."""
    D2 = dict(D)
    D2["model_cols"] = [c for c in D["model_cols"] if c.rsplit("_", 1)[0] not in drop]
    D2["dropped"] = list(drop)
    return D2


def with_history(D, base):
    cols = D["model_cols"]
    base_cols = [c for c in cols if c.endswith("_std")] if base == "std" else cols
    D2 = dict(D)
    for k in ("tr", "va", "te", "cva", "cte"):
        D2[k], hist = add_history(D[k], base_cols)
    return D2, cols + hist


def exp_hist(D, base, name):
    D2, cols = with_history(D, base)
    res, _ = run_unsup(name, D2, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols),
                       f"IsolationForest + rolling history (5/30/60/1440 windows) of the {base} features")
    return D2, cols


def exp_seq(D, kind, p, name):
    cols = D["model_cols"]
    tr = D["tr"]
    base = iforest_fit(tr.loc[normal_mask(tr), cols].to_numpy(dtype=np.float32))
    s_tr = iforest_score(base, tr, cols)
    mu, sd = float(s_tr[normal_mask(tr)].mean()), float(s_tr[normal_mask(tr)].std())

    def score_fn(_, df):
        return seq_transform(iforest_score(base, df, cols), df, kind, p, mu, sd)

    return run_unsup(name, D, cols, lambda X: base, score_fn,
                     f"{kind} (p={p}) of the base60 IsolationForest score, per gun in time order")


def exp_lstm(D, name="lstm_ae60"):
    cols = D["model_cols"]
    ae = LSTMAE(cols)
    return run_unsup(name, D, cols, lambda X: ae.fit(D["tr"]), lambda m, df: m.score(df),
                     "LSTM auto-encoder, 30-window sequences of the 48 features, reconstruction MSE")


def exp_e02(D):
    e02 = {f for f in D["files"] if f.startswith("E02")}
    D2 = dict(D)
    D2["tr"] = D["tr"][D["tr"]["file"].isin(e02)]
    D2["va"], D2["cva"] = D["va"][D["va"]["file"].isin(e02)], D["cva"][D["cva"]["file"].isin(e02)]
    D2["te"], D2["cte"] = D["te"][D["te"]["class"] == "E02"], D["cte"][D["cte"]["file"].isin(set(D["te"].loc[D["te"]["class"] == "E02", "file"]))]
    cols = D["model_cols"]
    run_unsup("e02_only60", D2, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols),
              f"IsolationForest trained on the {len(e02 & set(D['tr_files']))} E02 training guns only, evaluated on E02 val + test_2")
    D3 = dict(D)
    D3["tr"] = D["tr"]  # same model as base60, evaluation without E02
    D3["va"], D3["cva"] = D["va"][~D["va"]["file"].isin(e02)], D["cva"][~D["cva"]["file"].isin(e02)]
    D3["te"], D3["cte"] = D["te"][D["te"]["class"] != "E02"], D["cte"][~D["cte"]["file"].isin(set(D["te"].loc[D["te"]["class"] == "E02", "file"]))]
    run_unsup("no_e02_60", D3, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols),
              "base60 model, val / test evaluated without the E02 guns")


def exp_labelwin(D, L, prefix=""):
    D2 = relabel_all(D, L)
    cols = D["model_cols"]
    run_unsup(f"{prefix}labelwin_{L}", D2, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols),
              f"base60 with the pre-failure window = {L} s (features: {len(cols)})")
    run_supervised(f"{prefix}labelwin_{L}_lgbm", D2, cols, f"supervised LightGBM with the pre-failure window = {L} s (features: {len(cols)})")


def exp_cv(D, k=4, name="cv4_base60"):
    files = sorted(glob.glob(os.path.join(DATA, "E0*.parquet")))
    folds = T.cv_folds(files, k, SEED)
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
    cols = D["model_cols"]
    fold_res, test_res = [], []
    for i, va in enumerate(folds, 1):
        va_s = {stem(f) for f in va}
        D2 = dict(D)
        D2["tr"], D2["va"] = D["all"][~D["all"]["file"].isin(va_s)], D["all"][D["all"]["file"].isin(va_s)]
        D2["cva"] = D["cal_all"][D["cal_all"]["file"].isin(va_s)]
        r, _ = run_unsup(f"{name}_fold{i}", D2, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols),
                         f"fold {i}/{k}: val guns {sorted(va_s)}")
        fold_res.append(r["val"])
        test_res.append(r["test"])
        os.remove(os.path.join(OUT, f"{name}_fold{i}.json"))
    keys = ("auroc", "auprc", "recall_pre_failure", "alarm_rate_normal")

    def agg(rs):
        return {"mean": {kk: float(np.nanmean([r[kk] for r in rs])) for kk in keys},
                "sd": {kk: float(np.nanstd([r[kk] for r in rs])) for kk in keys},
                "per_class_auroc_mean": {c: float(np.nanmean([r["per_class"][c]["auroc"] for r in rs if c in r["per_class"]]))
                                         for c in T.CLASSES},
                "files_with_final_run": int(sum(r["operating_point"]["files_with_final_run"] for r in rs)),
                "files_both_ok": int(sum(r["operating_point"]["files_both_ok"] for r in rs)),
                "files": int(sum(r["operating_point"]["files"] for r in rs))}

    res = {"name": name, "note": f"gun-level {k}-fold of base60 over all 64 training guns; test = the 8 test guns "
                                  f"scored by each fold model", "k": k, "val_folds": agg(fold_res),
           "test_over_folds": agg(test_res), "folds": [{"val_files": sorted({stem(f) for f in va}), **{kk: r[kk] for kk in keys},
                                                        "operating_point": r["operating_point"]} for va, r in zip(folds, fold_res)]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    v, t = res["val_folds"], res["test_over_folds"]
    log(f"== {name}: val AUROC {v['mean']['auroc']:.3f}+-{v['sd']['auroc']:.3f} recall {v['mean']['recall_pre_failure']:.3f}"
        f"+-{v['sd']['recall_pre_failure']:.3f} alarm {v['mean']['alarm_rate_normal']:.3f}+-{v['sd']['alarm_rate_normal']:.3f} "
        f"final runs {v['files_with_final_run']}/{v['files']} op {v['files_both_ok']} | test over folds AUROC "
        f"{t['mean']['auroc']:.3f}+-{t['sd']['auroc']:.3f}")
    return res


# ------------------------------------------------------------------ B-5 (2026-09-26)
MAINT_CAP_MIN = 360  # lookbacks / "time since" features stop at 6 h: longer ones would grow with the file age (= ttf, c19-style leak)
MAINT_COLS = ["min_since_dress", "dress_starts_6h", "nw_share_6h", "min_since_c1_change", "c1_changes_6h",
              "c1_vs_6h", "welds_since_dress"]


def maintenance_features(df):
    """Per-second maintenance history from a preprocessed file: cap dressing = a non-welding block (c16 <= 0),
    cap offset c1 steps when the cap is dressed / changed. Every feature looks back at most 6 h (capped)."""
    t = df.index
    nw = df["non_welding"].to_numpy() > 0
    start = nw & ~np.r_[False, nw[:-1]]
    end = ~nw & np.r_[False, nw[:-1]]
    c1 = df["c1"].to_numpy()
    step = np.r_[False, np.abs(np.diff(c1)) > 1e-6]
    welding = df["welds_delta"].to_numpy() > df["welds_delta"].min()  # a weld counted this second (z-scored delta)

    def since(ev):
        last = pd.Series(np.where(ev, t.asi8, np.nan), index=t).ffill().to_numpy()
        return np.minimum((t.asi8 - last) / 6e10, MAINT_CAP_MIN)  # ns -> min; never seen = capped (fillna below)

    s = pd.DataFrame({"start": start.astype("float32"), "nw": nw.astype("float32"), "step": step.astype("float32")}, index=t)
    r = s.rolling("6h", min_periods=1)
    blk = np.cumsum(end)  # welds since the last dressing block ended
    wsd = pd.Series(welding.astype("float32"), index=t).groupby(blk).cumsum().to_numpy()
    out = pd.DataFrame({
        "min_since_dress": since(end), "dress_starts_6h": r["start"].sum().to_numpy(), "nw_share_6h": r["nw"].mean().to_numpy(),
        "min_since_c1_change": since(step), "c1_changes_6h": r["step"].sum().to_numpy(),
        "c1_vs_6h": c1 - pd.Series(c1, index=t).rolling("6h", min_periods=1).mean().to_numpy(),
        "welds_since_dress": np.minimum(wsd, 5000)}, index=t)
    return out.fillna(MAINT_CAP_MIN).astype("float32")


def build_maint_cache(window=60):
    """Window-mean of the maintenance features, keyed like the window cache (file, time)."""
    for split, fs in (("train", sorted(glob.glob(os.path.join(DATA, "E0*.parquet")))),
                      ("test", T.test_files_in(os.path.join(DATA, "test")))):
        parts = []
        for i, f in enumerate(fs, 1):
            df = pd.read_parquet(f, columns=["file", "segment_id", "non_welding", "c1", "welds_delta"])
            m = maintenance_features(df)
            g = m.groupby([df["segment_id"], pd.Grouper(freq=f"{window}s")]).mean().droplevel(0)
            g["file"] = df["file"].iloc[0]
            parts.append(g)
            log(f"[maint {split} {i}/{len(fs)}] {os.path.basename(f)}")
        x = pd.concat(parts)
        x.index.name = "time"
        x.reset_index().to_parquet(os.path.join(CACHE, f"maint{window}_{split}.parquet"))


def add_maint(D, window=60):
    D2 = dict(D)
    for split, keys in (("train", ("all", "tr", "va", "cal_all", "cva")), ("test", ("te", "cte"))):
        m = pd.read_parquet(os.path.join(CACHE, f"maint{window}_{split}.parquet"))
        for k in keys:
            x = D[k].reset_index().merge(m, on=["file", "time"], how="left").set_index("time")
            x[MAINT_COLS] = x[MAINT_COLS].fillna(MAINT_CAP_MIN)
            D2[k] = x
    return D2


def cv_run(name, D, cols, kind, L=None, drop_warmup=False, k=4, note=""):
    """Gun-level k-fold over the 64 train guns; each fold model also scores the 8 test guns. kind = iforest | lgbm.
    Reports mean +- sd over folds of pooled AUROC, per-gun AUROC mean, recall, normal alarm rate, and the summed
    operating point. drop_warmup removes the first 6 h of every gun from fit and evaluation."""
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731
    folds = T.cv_folds(sorted(glob.glob(os.path.join(DATA, "E0*.parquet"))), k, SEED)
    base = relabel_all(D, L) if L else D
    if drop_warmup:
        # the calibration frames are warm-up windows by definition: filtering them would silently drop the gun thresholds
        base = {kk: (v[v["warmup"] == 0] if isinstance(v, pd.DataFrame) and "warmup" in v.columns
                     and kk not in ("cal_all", "cte") else v) for kk, v in base.items()}
    val_res, test_res = [], []
    for i, va in enumerate(folds, 1):
        va_s = {stem(f) for f in va}
        D2 = dict(base)
        D2["tr"], D2["va"] = base["all"][~base["all"]["file"].isin(va_s)], base["all"][base["all"]["file"].isin(va_s)]
        D2["cva"] = base["cal_all"][base["cal_all"]["file"].isin(va_s)]
        if kind == "lgbm":
            r = run_supervised(f"{name}_fold{i}", D2, cols, write=False)
        else:
            r, _ = run_unsup(f"{name}_fold{i}", D2, cols, iforest_fit, lambda m, df: iforest_score(m, df, cols), write=False)
        val_res.append(r["val"])
        test_res.append(r["test"])
    keys = ("auroc", "auroc_gun_mean", "auprc", "recall_pre_failure", "alarm_rate_normal", "auroc_pre_rule")

    def agg(rs):
        return {"mean": {kk: float(np.nanmean([r.get(kk, np.nan) for r in rs])) for kk in keys},
                "sd": {kk: float(np.nanstd([r.get(kk, np.nan) for r in rs])) for kk in keys},
                "files_with_final_run": int(sum(r["operating_point"]["files_with_final_run"] for r in rs)),
                "files_both_ok": int(sum(r["operating_point"]["files_both_ok"] for r in rs)),
                "files": int(sum(r["operating_point"]["files"] for r in rs))}

    res = {"name": name, "note": note, "kind": kind, "label_window": L or 3600, "drop_warmup": drop_warmup, "k": k,
           "n_features": len(cols), "val_folds": agg(val_res), "test_over_folds": agg(test_res),
           "folds": [{"val_files": sorted({stem(f) for f in va}), **{kk: r[kk] for kk in keys}} for va, r in zip(folds, val_res)]}
    os.makedirs(OUT3, exist_ok=True)
    with open(os.path.join(OUT3, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    v, t = res["val_folds"], res["test_over_folds"]
    log(f"== {name:<24} {len(cols):>3} feats | cv AUROC {v['mean']['auroc']:.3f}+-{v['sd']['auroc']:.3f} "
        f"per-gun {v['mean']['auroc_gun_mean']:.3f}+-{v['sd']['auroc_gun_mean']:.3f} recall {v['mean']['recall_pre_failure']:.3f} "
        f"alarm {v['mean']['alarm_rate_normal']:.3f} op {v['files_both_ok']}/{v['files']} | test AUROC {t['mean']['auroc']:.3f}"
        f"+-{t['sd']['auroc']:.3f} per-gun {t['mean']['auroc_gun_mean']:.3f} recall {t['mean']['recall_pre_failure']:.3f}")
    return res


def exp_p3_labelcv(D):
    D = without(D, ["c19"])
    for L in (900, 1800, 2700, 3600, 7200):
        for kind in ("iforest", "lgbm"):
            cv_run(f"p3_labelcv_{L}_{kind}", D, D["model_cols"], kind, L,
                   note=f"{kind}, pre-failure window {L} s, 46 features (c19 dropped)")


def exp_p3_maint(D):
    D = add_maint(without(D, ["c19"]))
    for L in (1800, 3600):
        for kind in ("iforest", "lgbm"):
            for tag, cols in (("base", D["model_cols"]), ("maint", D["model_cols"] + MAINT_COLS)):
                cv_run(f"p3_maint_{tag}_{L}_{kind}", D, cols, kind, L, drop_warmup=True,
                       note=f"{kind}, {len(cols)} features, pre-failure {L} s, first 6 h of every gun dropped (maintenance "
                            "lookbacks are incomplete there)")


ALL = ["noc19_base60", "clean_base60", "noc19_sup60", "clean_sup60", "clean_sup300", "clean_hist_std60", "clean_sup_hist60",
       "clean_labelwin_1800", "clean_labelwin_7200", "clean_labelwin_21600", "clean_cv4", "clean_ewma5", "clean_base300",
       "base60", "hist_std60", "hist_all60", "base300", "hist_std300", "ewma5", "ewma15", "cusum", "lstm_ae60",
       "sup_lgbm60", "sup_lgbm_hist60", "sup_lgbm300", "e02_only60", "labelwin_1800", "labelwin_7200", "labelwin_21600",
       "cv4_base60"]


def run(names):
    names = ALL if names == ["all"] else names
    D60 = D300 = None
    for n in names:
        t0 = time.time()
        if n.endswith("300"):
            D300 = D300 or load(300)
            D = D300
        else:
            D60 = D60 or load(60)
            D = D60
        if n in ("p3_labelcv", "p3_maint"):
            D60 = D60 or load(60)
            (exp_p3_labelcv if n == "p3_labelcv" else exp_p3_maint)(D60)
            log(f"   ({n} done in {time.time() - t0:.0f}s)")
            continue
        prefix = n.split("_", 1)[0]
        if prefix in DROP:
            D = without(D, DROP[prefix])
            rest = n.split("_", 1)[1]
            tag = f" (without {', '.join(DROP[prefix])})"
            if rest in ("base60", "base300"):
                run_unsup(n, D, D["model_cols"], iforest_fit, lambda m, df, c=D["model_cols"]: iforest_score(m, df, c),
                          f"IsolationForest{tag}")
            elif rest in ("sup60", "sup300"):
                run_supervised(n, D, D["model_cols"], f"supervised LightGBM{tag} - honest upper bound")
            elif rest == "hist_std60":
                exp_hist(D, "std", n)
            elif rest == "sup_hist60":
                D2, cols = with_history(D, "all")
                run_supervised(n, D2, cols, f"supervised LightGBM + history features{tag}")
            elif rest.startswith("labelwin_"):
                exp_labelwin(D, int(rest.split("_")[1]), prefix=prefix + "_")
            elif rest == "cv4":
                exp_cv(D, name=n)
            elif rest.startswith("ewma"):
                exp_seq(D, "ewma", int(rest[4:]), n)
            else:
                raise SystemExit(f"unknown experiment {n}")
            log(f"   ({n} done in {time.time() - t0:.0f}s)")
            continue
        if n == "base60":
            exp_base(D)
        elif n == "base300":
            exp_base(D, "base300")
        elif n == "hist_std60":
            exp_hist(D, "std", n)
        elif n == "hist_all60":
            exp_hist(D, "all", n)
        elif n == "hist_std300":
            exp_hist(D, "std", n)
        elif n.startswith("ewma"):
            exp_seq(D, "ewma", int(n[4:]), n)
        elif n == "cusum":
            exp_seq(D, "cusum", 0.5, n)
        elif n == "lstm_ae60":
            exp_lstm(D)
        elif n == "sup_lgbm60":
            run_supervised(n, D, D["model_cols"], "supervised LightGBM on the 48 features - upper bound")
        elif n == "sup_lgbm300":
            run_supervised(n, D, D["model_cols"], "supervised LightGBM on the 48 features, 5-minute windows")
        elif n == "sup_lgbm_hist60":
            D2, cols = with_history(D, "all")
            run_supervised(n, D2, cols, "supervised LightGBM on 48 + 192 history features - upper bound with temporal context")
        elif n == "e02_only60":
            exp_e02(D)
        elif n.startswith("labelwin_"):
            exp_labelwin(D, int(n.split("_")[1]))
        elif n == "cv4_base60":
            exp_cv(D)
        else:
            raise SystemExit(f"unknown experiment {n}")
        log(f"   ({n} done in {time.time() - t0:.0f}s)")


def summary():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if "val" not in r:
            continue
        v, t = r["val"], r["test"]
        ov, ot = v["operating_point"], t["operating_point"]
        lead10 = sum(1 for f in t["per_file"].values() if f["final_alarm_run_starts_h_before_failure"] * 60 >= 10)
        rows.append(f"| {r['name']} | {r.get('n_features', '-')} | {v['auroc']:.3f} | {v['recall_pre_failure']:.3f} | "
                    f"{v['alarm_rate_normal']:.3f} | {ov['files_with_final_run']}/{ov['files']} | {ov['files_both_ok']} | "
                    f"{t['auroc']:.3f} | {t['recall_pre_failure']:.3f} | {t['alarm_rate_normal']:.3f} | "
                    f"{ot['files_with_final_run']}/{ot['files']} | {lead10} | {ot['files_both_ok']} |")
    print("| 실험 | 피처 | 검증 AUROC | recall@1h | 정상 알람률 | 지속 알람 건 | 운영점 | 테스트 AUROC | recall@1h | "
          "정상 알람률 | 지속 알람 건 | 10분+ 선행 | 운영점 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--window", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+")
    sub.add_parser("cache-maint")
    sub.add_parser("summary")
    args = ap.parse_args()
    if args.cmd == "cache":
        build_cache(args.window)
    elif args.cmd == "cache-maint":
        build_maint_cache()
    elif args.cmd == "run":
        run(args.names)
    else:
        summary()


if __name__ == "__main__":
    main()
