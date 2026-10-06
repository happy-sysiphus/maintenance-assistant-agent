"""
B-15 (2026-10-06): time-series foundation models (SOTA) vs the paper's benchmark (Wang et al. 2024, Table 11).

The paper's stage 1 forecasts the next N = 60 s of the target sensor - electrode force c2 for E02 guns, balance
pressure c5 for the others - from a 20 s input chunk with the other sensors as past covariates (z-scored), and scores
MAE / MAPE / MARRE / MSE (best: TFT MAE 0.2099, RF 0.2203). Stage 2 (Table 12, state classification) is left out: a
copy of the current state already beats it (B-6, accuracy 0.999).

    build              forecast origins -> results/B-15/cache/origins.npz
                       per gun (64 train + 8 test, preprocessed/): "paper" = the last 60 s of the file (the paper's
                       hold-out), "normal" = 50 random origins > 24 h before the failure, "pre" = 10 origins whose
                       60 s end 10-60 min before the failure. Context 512 s, all inside one segment.
    run <model>        predictions -> cache/pred_<model>.npy (one model per process: ~1 GB free RAM)
                       naive, mean, rf (the paper's RF: lags 20 + covariate lags 20, 100 trees, gun 4-fold),
                       chronos2s / chronos2 (+ _cov = with the 13 covariates; _20 = 20 s context),
                       timesfm25, tirex, moirai2s (univariate; _20 = 20 s context)
    report             metrics -> results/B-15/forecast.json + markdown table on stdout
    onset_score        T2: Chronos-2-small forecast "surprise" (mean |actual - forecast| over the next 60 s) every
                       10 min, every 2 min in the last 24 h, per gun -> cache/onset/<file>.csv (venv)
    onset_eval         T2: surprise vs the production score by time-to-failure bin + placebo -> onset.json (project env)

Run with the separate env: .venv-tsfm/Scripts/python.exe results/B-15/forecast.py ... (the project env lacks the
foundation-model packages). Pretrained weights come from Hugging Face; they are models, not data about these guns.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(HERE, "cache")
CTX, H = 512, 60
SENSORS = ["c1", "c2", "c3", "c4", "c5", "c6", "c10", "c13", "c14", "c15", "c16", "c17", "c18"]  # c7-c9 constant, c19 leak
N_NORMAL, N_PRE = 50, 10
SEED = 42
PAPER_TABLE11 = {  # MAE, MAPE, MARRE, MSE (paper Table 11)
    "BayesianRidge": (0.6047, 174.0001, 15.3520, 0.7032), "LinearRegression": (0.5431, 255.9767, 13.7886, 0.5156),
    "RandomForest": (0.2203, 90.5216, 5.5921, 0.0958), "NBEATS": (0.3041, 214.3311, 7.7203, 0.1671),
    "LightGB": (0.3174, 148.2126, 8.0568, 0.2055), "RNN": (0.4008, 164.0906, 10.1751, 0.2403),
    "LSTM": (0.8479, 598.63796, 21.52503, 1.3111), "GRU": (0.6668, 604.9237, 16.9287, 1.0064),
    "TCNModel": (0.9413, 219.2912, 23.8958, 1.7227), "TFT": (0.2099, 181.1341, 5.33004, 0.1047)}


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ origins
def build():
    rng = np.random.default_rng(SEED)
    files = sorted(glob.glob(os.path.join(ROOT, "preprocessed", "E0*.parquet"))) + \
        sorted(glob.glob(os.path.join(ROOT, "preprocessed", "test", "test_*.parquet")))
    tgt, cov, fut, meta = [], [], [], []
    for f in files:
        d = pd.read_parquet(f, columns=["class", "segment_id", "ttf_s"] + SENSORS)
        cls = str(d["class"].iloc[0])
        target = "c2" if cls == "E02" else "c5"
        covs = [c for c in SENSORS if c != target]
        seg, ttf = d["segment_id"].to_numpy(), d["ttf_s"].to_numpy()
        y, X = d[target].to_numpy(np.float32), d[covs].to_numpy(np.float32)
        # o = first forecast row; [o - CTX, o + H) must lie in one segment
        ok = np.zeros(len(d), bool)
        same = np.r_[False, seg[1:] == seg[:-1]]
        run = np.zeros(len(d), int)
        for i in range(1, len(d)):
            run[i] = run[i - 1] + 1 if same[i] else 0  # rows since the segment start
        o_all = np.arange(CTX, len(d) - H + 1)
        ok_o = o_all[(run[o_all + H - 1] >= CTX + H - 1)]
        ok[ok_o] = True
        picks = []
        last = len(d) - H
        if ok[last]:
            picks.append((last, "paper"))
        end_ttf = ttf[np.minimum(ok_o + H - 1, len(d) - 1)]
        normal = ok_o[end_ttf > 24 * 3600]
        pre = ok_o[(end_ttf > 600) & (end_ttf <= 3600)]
        picks += [(o, "normal") for o in rng.choice(normal, min(N_NORMAL, len(normal)), replace=False)]
        picks += [(o, "pre") for o in rng.choice(pre, min(N_PRE, len(pre)), replace=False)]
        name = os.path.splitext(os.path.basename(f))[0]
        for o, kind in picks:
            tgt.append(y[o - CTX:o])
            cov.append(X[o - CTX:o])
            fut.append(y[o:o + H])
            meta.append({"file": name, "class": cls, "target": target, "kind": kind, "origin": int(o),
                         "ttf_end": float(ttf[o + H - 1])})
        log(f"{name} ({cls}, {target}): {len(picks)} origins")
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "origins.npz"), tgt=np.array(tgt), cov=np.array(cov), fut=np.array(fut))
    pd.DataFrame(meta).to_csv(os.path.join(CACHE, "origins.csv"), index=False)
    log(f"{len(meta)} origins, {len(files)} guns")


def load():
    z = np.load(os.path.join(CACHE, "origins.npz"))
    return z["tgt"], z["cov"], z["fut"], pd.read_csv(os.path.join(CACHE, "origins.csv"))


# ------------------------------------------------------------------ models
def gun_folds(files, k=4):
    sys.path.insert(0, os.path.join(ROOT, ".py"))
    import train as T
    names = sorted(set(files))
    folds = T.cv_folds([f for f in names if not f.startswith("test_")], k, SEED)
    for i, f in enumerate(f for f in names if f.startswith("test_")):
        folds[i % k].append(f)
    return folds


def rf_features(tgt, cov, lags=20):
    return np.hstack([tgt[:, -lags:], cov[:, -lags:, :].reshape(len(tgt), -1)])


def run_rf(tgt, cov, fut, meta):
    """The paper's RF (lags 20 + past-covariate lags 20, 100 trees), trained on other guns' random windows."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(SEED)
    files = meta["file"].to_numpy()
    pred = np.zeros_like(fut)
    # training windows: the same origin pool is small, so draw extra training windows from the cached contexts
    # (any 80 s slice of a 512 s context: 20 s input + 60 s target) of the training guns
    for va in gun_folds(files):
        tr = np.flatnonzero(~np.isin(files, va))
        Xs, Ys = [], []
        for i in tr:
            for s in rng.integers(0, CTX - 80, 4):
                Xs.append(np.r_[tgt[i, s:s + 20], cov[i, s:s + 20, :].ravel()])
                Ys.append(tgt[i, s + 20:s + 80])
        m = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=SEED, min_samples_leaf=2)
        m.fit(np.array(Xs), np.array(Ys))
        te = np.flatnonzero(np.isin(files, va))
        pred[te] = m.predict(rf_features(tgt[te], cov[te]))
        log(f"  rf fold: {len(Xs)} training windows, {len(te)} origins")
    return pred


def run_chronos2(tgt, cov, name, use_cov, ctx, batch=16):
    from chronos import Chronos2Pipeline
    p = Chronos2Pipeline.from_pretrained(name, device_map="cpu")
    out = []
    for a in range(0, len(tgt), batch):
        inputs = []
        for i in range(a, min(a + batch, len(tgt))):
            d = {"target": tgt[i, -ctx:]}
            if use_cov:
                d["past_covariates"] = {f"x{j}": cov[i, -ctx:, j] for j in range(cov.shape[2])}
            inputs.append(d)
        r = p.predict(inputs, prediction_length=H)
        for q in r:
            q = q.numpy()  # (variates, quantiles, horizon)
            out.append(q[0, q.shape[1] // 2])
    return np.array(out)


def run_timesfm(tgt, ctx, batch=32):
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=max(ctx, 32), max_horizon=64, normalize_inputs=True,
                                     use_continuous_quantile_head=True, per_core_batch_size=batch))
    out = []
    for a in range(0, len(tgt), batch):
        pt, _ = m.forecast(horizon=H, inputs=[x[-ctx:] for x in tgt[a:a + batch]])
        out.append(np.asarray(pt)[:, :H])
    return np.vstack(out)


def run_tirex(tgt, ctx, batch=64):
    import torch
    from tirex import load_model
    m = load_model("NX-AI/TiRex", device="cpu")
    out = []
    for a in range(0, len(tgt), batch):
        _, mean = m.forecast(context=torch.tensor(tgt[a:a + batch, -ctx:]), prediction_length=H)
        out.append(mean.numpy())
    return np.vstack(out)


def run_moirai(tgt, ctx, batch=64):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    mod = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    f = Moirai2Forecast(module=mod, prediction_length=H, context_length=ctx, target_dim=1, feat_dynamic_real_dim=0,
                        past_feat_dynamic_real_dim=0)
    out = []
    for a in range(0, len(tgt), batch):
        r = np.asarray(f.predict(past_target=[x[-ctx:, None] for x in tgt[a:a + batch]]))
        # (batch, quantiles, horizon) -> median quantile
        out.append(r[:, r.shape[1] // 2, :H] if r.ndim == 3 else r[:, :H])
    return np.vstack(out)


MODELS = {
    "naive": lambda t, c, f, m: np.repeat(t[:, -1:], H, axis=1),
    "mean": lambda t, c, f, m: np.repeat(t[:, -20:].mean(1, keepdims=True), H, axis=1),
    "rf": run_rf,
    "chronos2s": lambda t, c, f, m: run_chronos2(t, c, "autogluon/chronos-2-small", False, CTX),
    "chronos2s_cov": lambda t, c, f, m: run_chronos2(t, c, "autogluon/chronos-2-small", True, CTX),
    "chronos2s_cov_20": lambda t, c, f, m: run_chronos2(t, c, "autogluon/chronos-2-small", True, 20),
    "chronos2": lambda t, c, f, m: run_chronos2(t, c, "amazon/chronos-2", False, CTX),
    "chronos2_cov": lambda t, c, f, m: run_chronos2(t, c, "amazon/chronos-2", True, CTX),
    "chronos2_cov_20": lambda t, c, f, m: run_chronos2(t, c, "amazon/chronos-2", True, 20),
    "timesfm25": lambda t, c, f, m: run_timesfm(t, CTX),
    "timesfm25_20": lambda t, c, f, m: run_timesfm(t, 20),
    "tirex": lambda t, c, f, m: run_tirex(t, CTX),
    "tirex_20": lambda t, c, f, m: run_tirex(t, 20),
    "moirai2s": lambda t, c, f, m: run_moirai(t, CTX),
    "moirai2s_20": lambda t, c, f, m: run_moirai(t, 20),
}


def run(name):
    tgt, cov, fut, meta = load()
    t0 = time.time()
    pred = np.asarray(MODELS[name](tgt, cov, fut, meta), dtype=np.float32)
    assert pred.shape == fut.shape, (pred.shape, fut.shape)
    np.save(os.path.join(CACHE, f"pred_{name}.npy"), pred)
    log(f"{name}: {len(pred)} forecasts in {time.time() - t0:.0f}s, MAE {np.abs(pred - fut).mean():.4f}")


# ------------------------------------------------------------------ metrics
def metrics(a, p):
    """Per-origin MAE / MAPE (nonzero actuals, %) / MARRE (% of the actual range) / MSE, then averaged - the paper's
    darts metrics per series. MAPE / MARRE are NaN where undefined (all-zero actuals / flat actuals)."""
    err = np.abs(a - p)
    nz = np.abs(a) > 1e-6
    mape = np.where(nz.any(1), np.nansum(np.where(nz, err / np.where(nz, np.abs(a), 1), 0), 1) / np.maximum(nz.sum(1), 1)
                    * 100, np.nan)
    rng = a.max(1) - a.min(1)
    marre = np.where(rng > 1e-6, err.mean(1) / np.where(rng > 1e-6, rng, 1) * 100, np.nan)
    return {"mae": err.mean(1), "mape": mape, "marre": marre, "mse": ((a - p) ** 2).mean(1)}


def report():
    _, _, fut, meta = load()
    preds = {os.path.basename(p)[5:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(CACHE, "pred_*.npy")))}
    naive_mae = metrics(fut, preds["naive"])["mae"]
    out = {"n_origins": meta["kind"].value_counts().to_dict(), "paper_table11": PAPER_TABLE11, "models": {}}
    rows = []
    for name, p in preds.items():
        m = metrics(fut, p)
        r = {}
        for kind in ("paper", "normal", "pre"):
            sel = (meta["kind"] == kind).to_numpy()
            r[kind] = {k: float(np.nanmean(v[sel])) for k, v in m.items()}
            r[kind]["mase_vs_naive"] = float(np.nanmean(m["mae"][sel]) / max(np.nanmean(naive_mae[sel]), 1e-9))
            r[kind]["per_class_mae"] = {c: float(np.nanmean(m["mae"][sel & (meta["class"] == c).to_numpy()]))
                                        for c in sorted(meta["class"].unique())}
        out["models"][name] = r
        rows.append((name, r))
    rows.sort(key=lambda kv: kv[1]["normal"]["mae"])
    log("| model | paper origins (last 60 s): MAE / MAPE / MARRE / MSE | normal (>24 h): MAE / MARRE / vs naive | "
        "10-60 min before: MAE / vs naive |")
    log("|---|---|---|---|")
    for name, r in rows:
        a, n, q = r["paper"], r["normal"], r["pre"]
        log(f"| {name} | {a['mae']:.4f} / {a['mape']:.1f} / {a['marre']:.2f} / {a['mse']:.4f} | "
            f"{n['mae']:.4f} / {n['marre']:.2f} / {n['mase_vs_naive']:.2f} | {q['mae']:.4f} / {q['mase_vs_naive']:.2f} |")
    for k, v in PAPER_TABLE11.items():
        log(f"| paper {k} | {v[0]:.4f} / {v[1]:.1f} / {v[2]:.2f} / {v[3]:.4f} | | |")
    with open(os.path.join(HERE, "forecast.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ T2: forecast surprise as a pre-failure score
ONSET_MODEL = "autogluon/chronos-2-small"  # accuracy close to the best (MAE 0.224 vs 0.209) at a third of the cost
ONSET_CTX = 256


def onset_score():
    """Per gun, stream the file: an origin every 10 min over the whole file and every 2 min in the last 24 h. Score =
    mean |actual - forecast| over the next 60 s (the 'surprise'). Written per gun to cache/onset/<file>.csv with the
    window END time (minute label) so it joins the production windows (results/B-11 cache)."""
    from chronos import Chronos2Pipeline
    p = Chronos2Pipeline.from_pretrained(ONSET_MODEL, device_map="cpu")
    out_dir = os.path.join(CACHE, "onset")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "preprocessed", "E0*.parquet"))) + \
        sorted(glob.glob(os.path.join(ROOT, "preprocessed", "test", "test_*.parquet")))
    t0 = time.time()
    for k, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(dst):
            continue
        d = pd.read_parquet(f, columns=["class", "segment_id", "ttf_s", "c2", "c5"])
        y = d["c2" if str(d["class"].iloc[0]) == "E02" else "c5"].to_numpy(np.float32)
        seg, ttf = d["segment_id"].to_numpy(), d["ttf_s"].to_numpy()
        same = np.r_[False, seg[1:] == seg[:-1]]
        run_len = np.zeros(len(d), int)
        for i in range(1, len(d)):
            run_len[i] = run_len[i - 1] + 1 if same[i] else 0
        o_all = np.arange(ONSET_CTX, len(d) - H + 1)
        o_all = o_all[run_len[o_all + H - 1] >= ONSET_CTX + H - 1]
        end_ttf = ttf[o_all + H - 1]
        keep = np.where(end_ttf <= 24 * 3600, (o_all % 120) == 0, (o_all % 600) == 0)
        origins = o_all[keep]
        res = []
        for a in range(0, len(origins), 32):
            batch = origins[a:a + 32]
            r = p.predict([{"target": y[o - ONSET_CTX:o]} for o in batch], prediction_length=H)
            for o, q in zip(batch, r):
                q = q.numpy()
                fc = q[0, q.shape[1] // 2]
                res.append((d.index[o + H - 1], float(ttf[o + H - 1]), float(np.abs(y[o:o + H] - fc).mean()),
                            float(fc.mean() - y[o - ONSET_CTX:o].mean())))
        pd.DataFrame(res, columns=["time", "ttf_s", "surprise", "fc_shift"]).to_csv(dst, index=False)
        log(f"[{k}/{len(files)}] {name}: {len(origins)} forecasts, {time.time() - t0:.0f}s")


def onset_eval():
    """Forecast surprise vs the production ensemble score on the same windows: AUROC by time-to-failure bin against
    negatives > 24 h before the failure (error-free, not warm-up), per-gun z of the surprise (median / IQR of the
    gun's windows > 24 h out), and the same with pseudo failures 36-120 h earlier (placebo)."""
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, os.path.join(ROOT, "results", "B-11"))
    import horizon as Hz
    cols = ["file", "time", "ttf_s", "error_active", "warmup", "non_welding", "score_ens"]
    w = pd.concat([pd.read_parquet(os.path.join(ROOT, "results", "B-11", "cache", f"scores_{s}.parquet"), columns=cols)
                   for s in ("cv", "test")])
    w["minute"] = pd.to_datetime(w["time"]).dt.floor("60s")
    parts = []
    for f in sorted(glob.glob(os.path.join(CACHE, "onset", "*.csv"))):
        s = pd.read_csv(f, parse_dates=["time"])
        s["file"] = os.path.splitext(os.path.basename(f))[0]
        s["minute"] = s["time"].dt.floor("60s")
        far = s["ttf_s"] > 24 * 3600
        med, iqr = s.loc[far, "surprise"].median(), s.loc[far, "surprise"].quantile(.75) - s.loc[far, "surprise"].quantile(.25)
        s["surprise_z"] = (s["surprise"] - med) / max(iqr, 1e-6)
        parts.append(s[["file", "minute", "surprise", "surprise_z", "fc_shift"]])
    s = pd.concat(parts)
    x = w.merge(s, on=["file", "minute"], how="inner").drop_duplicates(["file", "minute"])
    log(f"{len(x):,} windows with a forecast, {x['file'].nunique()} guns")
    out = {"model": ONSET_MODEL, "context_s": ONSET_CTX, "rows": []}
    ok = (x["error_active"].to_numpy() == 0) & (x["warmup"].to_numpy() == 0)
    for shift in [0] + [a * 3600 for a in Hz.PLACEBO_H]:
        ttf = x["ttf_s"].to_numpy() - shift
        valid = ttf > 0
        neg = valid & ok & (ttf > 24 * 3600)
        for lab, (lo, hi) in zip(Hz.LABELS, Hz.BINS):
            pos = valid & (ttf > lo) & (ttf <= hi) & (x["warmup"].to_numpy() == 0)
            if pos.sum() < 10 or neg.sum() < 50:
                continue
            r = {"bin": lab, "shift_h": shift / 3600, "n_pos": int(pos.sum())}
            for col in ("surprise_z", "surprise", "score_ens"):
                v = x[col].to_numpy()
                r[col] = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[v[pos], v[neg]]))
            out["rows"].append(r)
    rows = pd.DataFrame(out["rows"])
    real, plc = rows[rows["shift_h"] == 0], rows[rows["shift_h"] > 0]
    log("| bin | n | surprise (gun z) | placebo median [q95] | production score | placebo median [q95] |")
    log("|---|---|---|---|---|---|")
    for _, r in real.iterrows():
        p_ = plc[plc["bin"] == r["bin"]]
        log(f"| {r['bin']} | {r['n_pos']} | {r['surprise_z']:.3f} | {p_['surprise_z'].median():.3f} "
            f"[{p_['surprise_z'].quantile(.95):.3f}] | {r['score_ens']:.3f} | {p_['score_ens'].median():.3f} "
            f"[{p_['score_ens'].quantile(.95):.3f}] |")
    with open(os.path.join(HERE, "onset.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "report":
        report()
    elif cmd == "onset_score":
        onset_score()
    elif cmd == "onset_eval":
        onset_eval()
    else:
        sys.exit(__doc__)
