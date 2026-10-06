"""
B-13 (2026-10-03): do several window scales TOGETHER help? The decision stays every 1 min (production windows); the
longer scales come in as causal look-back features over the gun's own 1 min windows (computable online from a
per-gun window history, <= 6 h look-back - the B-9 gate), or as a fusion of the score over time.

    base       production 46 features (1 min mean + std)
    ms_mean    + trailing mean of the 16 gun-normalised sensor/counter means over 10 min, 1 h, 6 h  (46 + 48)
    ms_shift   + (now - trailing 6 h mean) and trailing 1 h std of the same 16 means                (46 + 32)
    fuse_*     base score combined with its own trailing 10 min / 1 h mean (mean or max of the two) - no refit

Look-backs run within a gun and within its warm-up / post-warm-up part (the scale changes at the warm-up end).
Same harness as B-11/B-12: production recipe (IF + LightGBM, gun thresholds, gate), 64-gun 4-fold + all-64 fit on the
8 test guns, train.evaluate per fold, the B-11 horizon table + placebo, plus a leak check (within-gun Spearman of every
new feature with elapsed time; > 0.5 = leak, the c19 case).

    python results/B-13/multiscale.py     # ~35 min, results/B-13/multiscale.json
    python results/B-13/multiscale.py ms  # low-memory rerun of base / ms_mean / ms_shift (~1 GB free RAM):
                                          # one float32 matrix, no frame copies, normal fit windows subsampled to
                                          # FIT_NEG (base re-run under the same subsample) -> variants base_sub, ms_*
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-11"))
import horizon as H  # noqa: E402
import train as T  # noqa: E402

SCALES = {"10m": "600s", "1h": "3600s", "6h": "21600s"}


def log(msg):
    print(msg, flush=True)


def base_means(cols):
    return [c for c in cols if c.endswith("_mean") and not c.startswith(("hour_", "c7_", "c8_", "c9_", "weld_duty",
                                                                         "error_share"))]


def add_lookback(w, src, variant):
    """Causal trailing features per (gun, warm-up part), time-based on the window end."""
    w = w.sort_values(["file", "time"]).reset_index(drop=True)
    if variant == "ms_mean":
        names = [f"{c}_r{tag}" for tag in SCALES for c in src]
    else:
        names = [f"{c}_d6h" for c in src] + [f"{c}_s1h" for c in src]
    arr = np.empty((len(w), len(names)), np.float32)
    for _, idx in w.groupby(["file", "warmup"], sort=False).indices.items():
        g = pd.DataFrame(w[src].to_numpy(np.float64)[idx], columns=src, index=pd.DatetimeIndex(w["time"].to_numpy()[idx]))
        if variant == "ms_mean":
            blocks = [g.rolling(span, min_periods=1).mean().to_numpy() for span in SCALES.values()]
        else:
            blocks = [g.to_numpy() - g.rolling(SCALES["6h"], min_periods=1).mean().to_numpy(),
                      g.rolling(SCALES["1h"], min_periods=2).std().fillna(0.0).to_numpy()]
        arr[idx] = np.hstack(blocks)
    for i, n in enumerate(names):
        w[n] = arr[:, i]
    return w, names


def leak_check(w, cols):
    """Median and max over guns of |Spearman(feature, elapsed time)| on post-warm-up windows."""
    out = {}
    post = w[w["warmup"] == 0]
    for c in cols:
        rho = []
        for _, g in post.groupby("file"):
            if g[c].nunique() > 2:
                rho.append(abs(spearmanr(g[c], -g["ttf_s"]).statistic))
        out[c] = (float(np.nanmedian(rho)), float(np.nanmax(rho))) if rho else (0.0, 0.0)
    return out


def run(tr, cal_tr, te, cal_te, cols, tag):
    t0 = time.time()
    folds = T.cv_folds(sorted(tr["file"].unique()), H.K, H.SEED)
    parts, fm = [], []
    for va in folds:
        model, thr = H.fit_production(tr[~tr["file"].isin(va)], cols)
        v = tr[tr["file"].isin(va)].copy()
        v["score_ens"] = H.score_parts(model, v[cols].to_numpy(np.float32))["ens"]
        c = cal_tr[cal_tr["file"].isin(va)]
        v["alarm"] = H.alarms(v, model, cols, thr, c)
        gun_thr = T.gun_thresholds(model, cols, c, thr, H.Q, H.GATE)
        m = T.evaluate(v, v["score_ens"].to_numpy(), thr, 3, H.GATE, gun_thr, warmup_alarms=False)
        fm.append({k: m[k] for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal",
                                     "recall_pre_failure")} | {"threshold": thr})
        parts.append(v[["file", "time", "ttf_s", "label", "error_active", "terminal_any", "non_welding", "warmup",
                        "weld_duty_10min_mean", "class", "score_ens", "alarm"]])
        del model
    cv = pd.concat(parts, ignore_index=True)
    model, thr = H.fit_production(tr, cols)
    t = te.copy()
    t["score_ens"] = H.score_parts(model, t[cols].to_numpy(np.float32))["ens"]
    t["alarm"] = H.alarms(t, model, cols, thr, cal_te)
    gun_thr = T.gun_thresholds(model, cols, cal_te, thr, H.Q, H.GATE)
    mt = T.evaluate(t, t["score_ens"].to_numpy(), thr, 3, H.GATE, gun_thr, warmup_alarms=False)
    res = {"n_features": len(cols), "folds": fm,
           "cv_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in fm[0]},
           "cv_sd": {k: float(np.nanstd([f[k] for f in fm])) for k in fm[0]},
           "test_metrics": {k: mt[k] for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal",
                                               "recall_pre_failure")}}
    t = t[cv.columns]
    report(res, cv, t, tag, time.time() - t0)
    return res, cv, t


def report(res, cv, t, tag, secs):
    for name, x in (("cv", cv), ("test", t)):
        for s in ("ens", "iforest", "lgbm"):  # horizon_table expects all three; only the ensemble matters here
            if f"score_{s}" not in x.columns:
                x[f"score_{s}"] = x["score_ens"]
        rows, neg = H.horizon_table(x, per_gun=False)
        null = {a: H.horizon_table(x, a * 3600, per_gun=False)[0] for a in H.PLACEBO_H}
        for i, r in enumerate(rows):
            pv = np.array([null[a][i]["auc_ens"] for a in H.PLACEBO_H], float)
            r["placebo_median"], r["placebo_q95"] = float(np.nanmedian(pv)), float(np.nanquantile(pv, .95))
            for k in [k for k in r if k.startswith("auc_") and k not in ("auc_ens", "auc_ens_clean")]:
                r.pop(k)
        res[name] = {"rows": rows, "neg_alarm_rate": neg}
    m, s, mt = res["cv_mean"], res["cv_sd"], res["test_metrics"]
    log(f"\n== {tag} ({res['n_features']} features, {secs:.0f}s) ==")
    log(f"4-fold AUROC {m['auroc']:.3f}+-{s['auroc']:.3f}  pre-rule {m['auroc_pre_rule']:.3f}+-{s['auroc_pre_rule']:.3f}"
        f" (folds {[round(f['auroc_pre_rule'], 3) for f in res['folds']]})  per gun {m['auroc_gun_mean']:.3f}"
        f"  alarm@normal {m['alarm_rate_normal']:.4f}  recall {m['recall_pre_failure']:.3f}"
        f" | test AUROC {mt['auroc']:.3f} pre-rule {mt['auroc_pre_rule']:.3f} alarm@normal {mt['alarm_rate_normal']:.4f}")
    for name in ("cv", "test"):
        rows = res[name]["rows"]
        log(f"  {name:4s} AUROC   " + " ".join(f"{r['auc_ens']:.3f}" for r in rows))
        log(f"  {name:4s} placebo " + " ".join(f"{r['placebo_median']:.3f}" for r in rows))
        log(f"  {name:4s} plc q95 " + " ".join(f"{r['placebo_q95']:.3f}" for r in rows))


def fuse(x, span, how):
    """Score fused with its own trailing mean over `span` (per gun and warm-up part)."""
    x = x.sort_values(["file", "time"]).reset_index(drop=True)
    r = (x.set_index("time").groupby(["file", "warmup"], sort=False)["score_ens"]
         .rolling(span, min_periods=1).mean().reset_index(level=[0, 1], drop=True))
    # groupby/rolling keeps the group order, which is the sorted order here
    r = r.to_numpy()
    s = x["score_ens"].to_numpy()
    out = x.copy()
    out["score_ens"] = np.maximum(s, r) if how == "max" else (s + r) / 2
    return out


def fusion_metrics(x, folds_of):
    """Per fold AUROC / pre-rule AUROC with the same definitions as train.evaluate (scores only)."""
    pre = (x["label"] == 1).to_numpy()
    normal = ((x["label"] == 0) & (x["error_active"] == 0)).to_numpy()
    early = pre & (x["ttf_s"].to_numpy() > T.RULE_LEAD_S)
    s = x["score_ens"].to_numpy()
    out = []
    for k in folds_of:
        f = folds_of[k]
        out.append({"auroc": H.auc(s, pre & f, normal & f), "auroc_pre_rule": H.auc(s, early & f, normal & f)})
    return out


def main():
    t0 = time.time()
    w, cal, cols = H.load()
    tr, te = w["train"], w["test"]
    cal_tr, cal_te = cal["train"], cal["test"]
    del w, cal
    keep = ["file", "time", "ttf_s", "label", "error_active", "terminal_any", "terminal_idx", "non_welding", "warmup",
            "class", "gun_norm"] + cols
    tr, te = tr[[c for c in keep if c in tr.columns]], te[[c for c in keep if c in te.columns]]
    out = {"variants": {}}

    res, cv_base, te_base = run(tr, cal_tr, te, cal_te, cols, "base")
    out["variants"]["base"] = res

    # score fusion on the base scores (no refit)
    files = sorted(tr["file"].unique())
    fold_files = T.cv_folds(files, H.K, H.SEED)
    for span in ("600s", "3600s"):
        for how in ("mean", "max"):
            tag = f"fuse_{how}_{span}"
            x, xt = fuse(cv_base, span, how), fuse(te_base, span, how)
            fm = fusion_metrics(x, {k: x["file"].isin(v).to_numpy() for k, v in enumerate(fold_files)})
            mt = fusion_metrics(xt, {0: np.ones(len(xt), bool)})[0]
            res = {"n_features": len(cols), "folds": fm,
                   "cv_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in fm[0]},
                   "cv_sd": {k: float(np.nanstd([f[k] for f in fm])) for k in fm[0]}, "test_metrics": mt}
            for name, d in (("cv", x), ("test", xt)):
                rows, _ = H.horizon_table(d.assign(score_iforest=d["score_ens"], score_lgbm=d["score_ens"]), per_gun=False)
                res[name] = {"rows": [{"bin": r["bin"], "auc_ens": r["auc_ens"]} for r in rows]}
            out["variants"][tag] = res
            log(f"\n== {tag}: 4-fold AUROC {res['cv_mean']['auroc']:.3f}  pre-rule {res['cv_mean']['auroc_pre_rule']:.3f}"
                f"+-{res['cv_sd']['auroc_pre_rule']:.3f} | test {mt['auroc']:.3f} / {mt['auroc_pre_rule']:.3f}")
            log("  cv   AUROC   " + " ".join(f"{r['auc_ens']:.3f}" for r in res["cv"]["rows"]))
            log("  test AUROC   " + " ".join(f"{r['auc_ens']:.3f}" for r in res["test"]["rows"]))
    save(out)
    del cv_base, te_base

    src = base_means(cols)
    for variant in ("ms_mean", "ms_shift"):
        tr2, new = add_lookback(tr, src, variant)
        te2, _ = add_lookback(te, src, variant)
        c2, _ = add_lookback(cal_tr, src, variant)
        ct2, _ = add_lookback(cal_te, src, variant)
        leak = leak_check(pd.concat([tr2, te2], ignore_index=True), new)
        worst = sorted(leak.items(), key=lambda kv: -kv[1][0])[:3]
        log(f"\n{variant}: {len(new)} new features; elapsed-time |rho| median over guns, top: "
            + ", ".join(f"{k} {v[0]:.2f} (max {v[1]:.2f})" for k, v in worst))
        res, _, _ = run(tr2, c2, te2, ct2, cols + new, variant)
        res["leak_top"] = {k: v for k, v in worst}
        out["variants"][variant] = res
        save(out)
        del tr2, te2, c2, ct2

    out["seconds"] = round(time.time() - t0)
    log(f"saved {save(out)} in {out['seconds']}s")


def save(out):
    dst = os.path.join(HERE, "multiscale.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return dst


# ------------------------------------------------------------------ low-memory path (python multiscale.py ms)
FIT_NEG = 150_000
META = ["file", "time", "ttf_s", "label", "error_active", "terminal_any", "terminal_idx", "non_welding", "warmup",
        "class", "gun_norm", "weld_duty_10min_mean"]


def lean_load(split, cols):
    path = os.path.join(H.B2, f"w60_{split}.parquet")
    m = pd.read_parquet(path, columns=[c for c in META if c != "weld_duty_10min_mean"] + cols)
    m = m.sort_values(["file", "time"]).reset_index(drop=True)
    X = m[cols].to_numpy(np.float32)
    m = m.drop(columns=cols).assign(weld_duty_10min_mean=X[:, cols.index("weld_duty_10min_mean")])
    for c in ("file", "class", "gun_norm"):
        m[c] = m[c].astype("category")
    cal = pd.read_parquet(os.path.join(H.B2, f"calib60_{split}.parquet"))
    cal = cal[[c for c in ("file", "time", "error_active", "non_welding", "warmup") if c in cal.columns] + cols]
    return m, X, cal.sort_values(["file", "time"]).reset_index(drop=True)


def lookback(m, Xs, variant):
    """Trailing features of Xs (rows sorted by file, time) per (gun, warm-up part) -> float32 (n, k)."""
    k = Xs.shape[1] * (3 if variant == "ms_mean" else 2)
    out = np.empty((len(m), k), np.float32)
    for _, idx in m.groupby(["file", "warmup"], sort=False, observed=True).indices.items():
        g = pd.DataFrame(Xs[idx].astype(np.float64), index=pd.DatetimeIndex(m["time"].to_numpy()[idx]))
        if variant == "ms_mean":
            blocks = [g.rolling(span, min_periods=1).mean().to_numpy() for span in SCALES.values()]
        else:
            blocks = [g.to_numpy() - g.rolling(SCALES["6h"], min_periods=1).mean().to_numpy(),
                      g.rolling(SCALES["1h"], min_periods=2).std().fillna(0.0).to_numpy()]
        out[idx] = np.hstack(blocks)
    return out


def lean_leak(m, E, names):
    post = m["warmup"].to_numpy() == 0
    res = {}
    groups = m[post].groupby("file", observed=True).indices
    base = np.flatnonzero(post)
    for j, n in enumerate(names):
        rho = []
        for _, idx in groups.items():
            ii = base[idx]
            x = E[ii, j]
            if np.unique(x).size > 2:
                rho.append(abs(spearmanr(x, -m["ttf_s"].to_numpy()[ii]).statistic))
        res[n] = (float(np.nanmedian(rho)), float(np.nanmax(rho))) if rho else (0.0, 0.0)
    return res


def lean_fit(X, m, idx, cols, seed=H.SEED):
    lab, err, nw = (m[c].to_numpy()[idx] for c in ("label", "error_active", "non_welding"))
    ni = idx[(lab == 0) & (err == 0) & (nw == 0)]
    if len(ni) > FIT_NEG:
        ni = np.sort(np.random.default_rng(seed).choice(ni, FIT_NEG, replace=False))
    pi = idx[(lab == 1) & (nw == 0)]
    files = m["file"].astype(str).to_numpy()
    model = T.build_model("ensemble", seed, cols)
    model.fit(X[ni], X[pi], files[ni], files[pi], m["ttf_s"].to_numpy()[ni])
    return model, float(np.quantile(model.fit_scores_, H.Q))


def lean_score(model, X, idx, chunk=100_000):
    return np.concatenate([H.score_parts(model, X[idx[i:i + chunk]])["ens"] for i in range(0, len(idx), chunk)])


def lean_judge(model, thr, m, idx, s, cal, cols):
    v = m.iloc[idx].reset_index(drop=True).copy()
    v["file"] = v["file"].astype(str)
    v["score_ens"] = s
    c = cal.assign(file=cal["file"].astype(str))
    v["alarm"] = H.alarms(v, model, cols, thr, c)
    gun_thr = T.gun_thresholds(model, cols, c, thr, H.Q, H.GATE)
    ev = T.evaluate(v, s, thr, 3, H.GATE, gun_thr, warmup_alarms=False)
    keys = ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal", "recall_pre_failure")
    return v, {k: ev[k] for k in keys}


def lean_variant(tag, tr, te, cols):
    (m, X, cal), (mt, Xt, calt) = tr, te
    t0 = time.time()
    files = m["file"].astype(str).to_numpy()
    folds = T.cv_folds(sorted(set(files)), H.K, H.SEED)
    parts, fm = [], []
    for va in folds:
        te_mask = np.isin(files, va)
        model, thr = lean_fit(X, m, np.flatnonzero(~te_mask), cols)
        vi = np.flatnonzero(te_mask)
        v, mm = lean_judge(model, thr, m, vi, lean_score(model, X, vi), cal[cal["file"].astype(str).isin(va)], cols)
        fm.append(mm | {"threshold": thr})
        parts.append(v)
        del model
        log(f"  {tag} fold {len(fm)}/{H.K}: pre-rule {mm['auroc_pre_rule']:.3f}  {time.time() - t0:.0f}s")
    cv = pd.concat(parts, ignore_index=True)
    model, thr = lean_fit(X, m, np.arange(len(m)), cols)
    ti = np.arange(len(mt))
    t, mtm = lean_judge(model, thr, mt, ti, lean_score(model, Xt, ti), calt, cols)
    del model
    res = {"n_features": len(cols), "fit_normal_windows": FIT_NEG, "folds": fm,
           "cv_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in fm[0]},
           "cv_sd": {k: float(np.nanstd([f[k] for f in fm])) for k in fm[0]}, "test_metrics": mtm}
    report(res, cv, t, tag, time.time() - t0)
    return res


def lean_main():
    t0 = time.time()
    meta = json.load(open(os.path.join(H.B2, "meta60.json"), encoding="utf-8"))
    cols = T.model_input_columns([c for c in meta["feature_cols"] if c != "c19"])
    tr, te = lean_load("train", cols), lean_load("test", cols)
    log(f"loaded train {tr[1].shape} test {te[1].shape} in {time.time() - t0:.0f}s")
    dst = os.path.join(HERE, "multiscale.json")
    out = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {"variants": {}}
    out["variants"]["base_sub"] = lean_variant("base_sub", tr, te, cols)
    save(out)
    src = base_means(cols)
    si = [cols.index(c) for c in src]
    for variant in ("ms_mean", "ms_shift"):
        names = ([f"{c}_r{t}" for t in SCALES for c in src] if variant == "ms_mean"
                 else [f"{c}_d6h" for c in src] + [f"{c}_s1h" for c in src])
        sets = []
        for m, X, cal in (tr, te):
            E = lookback(m, X[:, si], variant)
            Ec = lookback(cal.assign(ttf_s=0), cal[src].to_numpy(np.float32), variant)
            cal2 = pd.concat([cal, pd.DataFrame(Ec, columns=names)], axis=1)
            sets.append((m, np.hstack([X, E]), cal2, E))
        leak = lean_leak(pd.concat([sets[0][0], sets[1][0]], ignore_index=True), np.vstack([sets[0][3], sets[1][3]]),
                         names)
        worst = sorted(leak.items(), key=lambda kv: -kv[1][0])[:3]
        log(f"\n{variant}: {len(names)} new features; elapsed-time |rho| median over guns, top: "
            + ", ".join(f"{k} {v[0]:.2f} (max {v[1]:.2f})" for k, v in worst))
        res = lean_variant(variant, sets[0][:3], sets[1][:3], cols + names)
        res["leak_top"] = dict(worst)
        out["variants"][variant] = res
        save(out)
        del sets
    log(f"saved {dst} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    lean_main() if sys.argv[1:] == ["ms"] else main()
