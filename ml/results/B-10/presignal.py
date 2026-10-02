"""
B-10 (2026-10-01): is there a pre-failure signal in the data at all - and how early?

Every model so far (B-2 .. B-9) generalises ACROSS guns and tops out at rule-free AUROC ~0.6-0.7. That says
"no portable signal", not "no signal": (a) nothing there vs (b) a signal whose shape differs per gun. This
script measures it directly, gun by gun, against a placebo null (history.md §22).

    c2st     per gun: classify the 10-60 min pre-failure windows vs matched windows 24-72 h earlier
             (same time of day +-1.5 h, same plate-thickness setpoint c16). Null = the same test anchored at
             random times 12-90 h before the failure (100 placebos per gun). Also the effect-direction
             agreement between guns (real vs placebo).
    epoch    superposed-epoch profiles: per feature, per gun robust z vs the gun's own earlier windows, binned by
             time to failure; across-gun median vs a placebo band (random anchors).
    raw      1 Hz distribution features per 60 s window (quantiles, ranges, press intervals) -> cache/raw60_*.parquet
    sanity   calibration of the c2st test: a placebo scored as "real" (should be p < 0.05 for ~5% of guns) and a
             0.5 sd injected shift (should be caught)

    python results/B-10/presignal.py c2st [--design day|near] [--features base|phys|raw|all] [--clf lr|lgbm] [--n-placebo 100]
    python results/B-10/presignal.py epoch
    python results/B-10/presignal.py raw          # ~10 min, once
    python results/B-10/presignal.py sanity [--design day|near]

Inputs: results/B-2/cache/w60_*.parquet (46 production features, built 2026-09-29), results/B-7/cache/phys60_*
(state-conditioned physics, built 2026-09-28 before the weld_duty fix - it does not use weld_duty). All 72 guns
(64 train + 8 test, test classes inferred). Never writes outside results/B-10/.
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import train as T  # noqa: E402

warnings.filterwarnings("ignore")
CACHE = os.path.join(HERE, "cache")
B2 = os.path.join(ROOT, "results", "B-2", "cache")
B7 = os.path.join(ROOT, "results", "B-7", "cache")
SEED = 42
H = 3600
TARGET = (600, 3600)        # 10-60 min before the failure: the part the terminal-code rule does not cover
# control designs: "day" = 24-72 h before the anchor at the same time of day (+-HOUR_TOL h); "near" = 2-6 h before
# (less drift between target and control, no time-of-day match). Placebo anchors keep a + control inside the file.
DESIGNS = {"day": {"ctrl": (24 * H, 72 * H), "tod": True, "placebo": (12 * H, 90 * H)},
           "near": {"ctrl": (2 * H, 6 * H), "tod": False, "placebo": (12 * H, 150 * H)}}
DESIGN = {**DESIGNS["day"], "name": "day"}
HOUR_TOL = 1.5
MIN_POS, MIN_NEG = 20, 50
FOLDS = 5
# never inputs: time counters, time of day, gun constants, labels / states
EXCLUDE = {"c19_mean", "c19_std", "hour_sin_mean", "hour_sin_std", "hour_cos_mean", "hour_cos_std",
           "c7_mean", "c7_std", "c8_mean", "c8_std", "c9_mean", "c9_std"}
META = {"error_active", "terminal_code", "terminal_any", "terminal_idx", "non_welding", "label", "warmup", "ttf_s",
        "n", "file", "class", "gun", "gun_norm"}
EPOCH_BINS = [(0, 600), (600, 1800), (1800, 3600), (3600, 2 * H), (2 * H, 6 * H), (6 * H, 24 * H)]
EPOCH_LABELS = ["0-10m", "10-30m", "30-60m", "1-2h", "2-6h", "6-24h"]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ data
KEEP = ["time", "file", "class", "ttf_s", "error_active", "terminal_any", "non_welding", "warmup"]


def read_lean(path, drop=()):
    """Cache frame with only what this script uses, floats as float32 (the machine has ~1 GB free)."""
    import pyarrow.parquet as pq
    names = [c for c in pq.read_schema(path).names if c not in drop]
    x = pd.read_parquet(path, columns=names)
    f64 = x.select_dtypes("float64").columns
    x[f64] = x[f64].astype(np.float32)
    return x


def load(features):
    drop = (META | EXCLUDE) - set(KEEP)
    w = pd.concat([read_lean(os.path.join(B2, f"w60_{s}.parquet"), drop) for s in ("train", "test")],
                  ignore_index=True)
    ok = ((w["error_active"] == 0) & (w["terminal_any"] == 0) & (w["non_welding"] == 0) & (w["warmup"] == 0))
    w = w[ok.to_numpy()].reset_index(drop=True)
    base = [c for c in w.columns if c not in KEEP]
    cols = {"base": base}
    for name, tag, path in (("phys", "phys", B7), ("raw", "raw", CACHE)):
        if features in (name, "all"):
            x = pd.concat([read_lean(os.path.join(path, f"{tag}60_{s}.parquet")) for s in ("train", "test")],
                          ignore_index=True)
            cols[name] = [c for c in x.columns if c not in ("file", "time")]
            w = w.merge(x, on=["file", "time"], how="left")
            del x
    use = {"base": cols["base"], "phys": cols["base"] + cols.get("phys", []), "raw": cols.get("raw", []),
           "all": sum(cols.values(), [])}[features]
    w["tod"] = w["time"].dt.hour + w["time"].dt.minute / 60.0
    w["c16k"] = w["c16_mean"].round(1)
    w[use] = w.groupby("file", sort=False)[use].ffill().fillna(0.0)
    log(f"{len(w):,} eligible windows, {w['file'].nunique()} guns, {len(use)} features ({features})")
    return w, use


def tod_dist(a, b):
    d = np.abs(a - b) % 24
    return np.minimum(d, 24 - d)


def make_sets(g, a):
    """Target windows (a + 10 min, a + 60 min] before the anchor and matched controls 24-72 h before it."""
    tt = g["ttf_s"].to_numpy()
    pos = (tt > a + TARGET[0]) & (tt <= a + TARGET[1])
    if pos.sum() < MIN_POS:
        return None
    tod_c = np.angle(np.exp(1j * g["tod"].to_numpy()[pos] / 24 * 2 * np.pi).mean()) / (2 * np.pi) * 24 % 24
    lo, hi = DESIGN["ctrl"]
    neg = (tt > a + lo) & (tt <= a + hi) & g["c16k"].isin(set(g["c16k"].to_numpy()[pos])).to_numpy()
    if DESIGN["tod"]:
        neg &= tod_dist(g["tod"].to_numpy(), tod_c) <= HOUR_TOL
    if neg.sum() < MIN_NEG:
        return None
    return pos, neg


def block_folds(n, k):
    """Contiguous chunks (rows are in time order) - neighbouring minutes never straddle train / test."""
    return np.minimum(np.arange(n) * k // max(n, 1), k - 1)


def fit_clf(kind):
    if kind == "lr":
        return LogisticRegression(C=0.1, max_iter=500)
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=7, min_child_samples=5,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=SEED)


def c2st_auc(X, pos, neg, kind):
    """Blocked k-fold AUROC of a target-vs-control classifier (rows already in time order)."""
    Xp, Xn = X[pos], X[neg]
    fp, fn = block_folds(len(Xp), FOLDS), block_folds(len(Xn), FOLDS)
    yt, st = [], []
    for k in range(FOLDS):
        Xtr = np.vstack([Xp[fp != k], Xn[fn != k]])
        ytr = np.r_[np.ones((fp != k).sum()), np.zeros((fn != k).sum())]
        Xte = np.vstack([Xp[fp == k], Xn[fn == k]])
        sc = StandardScaler().fit(Xtr)
        m = fit_clf(kind).fit(sc.transform(Xtr), ytr)
        st.append(m.predict_proba(sc.transform(Xte))[:, 1])
        yt.append(np.r_[np.ones((fp == k).sum()), np.zeros((fn == k).sum())])
    return float(roc_auc_score(np.concatenate(yt), np.concatenate(st)))


def effect_vec(X, pos, neg):
    sd = X[neg].std(0) + 1e-6
    return (X[pos].mean(0) - X[neg].mean(0)) / sd


def gun_run(g, cols, kind, n_placebo, rng, inject=0.0):
    g = g.sort_values("time")
    X = g[cols].to_numpy(dtype=np.float64)
    real = make_sets(g, 0)
    if real is None:
        return None
    if inject:
        X = X.copy()
        X[real[0]] += inject * X[real[1]].std(0)
    auc = c2st_auc(X, *real, kind)
    eff = effect_vec(X, *real)
    pl_auc, pl_eff = [], []
    tries = 0
    while len(pl_auc) < n_placebo and tries < n_placebo * 5:
        tries += 1
        s = make_sets(g, rng.uniform(*DESIGN["placebo"]))
        if s is None:
            continue
        pl_auc.append(c2st_auc(X, *s, kind))
        pl_eff.append(effect_vec(X, *s))
    if len(pl_auc) < 20:
        return None
    pl_auc = np.array(pl_auc)
    return {"auc": auc, "p": float((1 + (pl_auc >= auc).sum()) / (1 + len(pl_auc))),
            "null_q95": float(np.quantile(pl_auc, 0.95)), "null_median": float(np.median(pl_auc)),
            "n_pos": int(real[0].sum()), "n_neg": int(real[1].sum()), "n_placebo": len(pl_auc),
            "eff": eff, "pl_eff": np.array(pl_eff), "pl_auc": pl_auc}


def agreement(vecs):
    """Mean pairwise cosine between guns' effect vectors."""
    V = np.array([v / (np.linalg.norm(v) + 1e-9) for v in vecs])
    C = V @ V.T
    return float(C[np.triu_indices(len(V), 1)].mean())


def binom_sf(k, n, p=0.05):
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p))


def c2st(features, kind, n_placebo, inject=0.0, tag=None):
    t0 = time.time()
    w, cols = load(features)
    rng = np.random.default_rng(SEED)
    res = {}
    for i, (f, g) in enumerate(w.groupby("file", sort=True), 1):
        r = gun_run(g, cols, kind, n_placebo, rng, inject)
        if r is None:
            log(f"[{i}] {f}: skipped (too few target / control windows)")
            continue
        r["class"] = g["class"].iloc[0]
        res[f] = r
        log(f"[{i}] {f} {r['class']}: AUROC {r['auc']:.3f}  null median {r['null_median']:.3f} q95 "
            f"{r['null_q95']:.3f}  p {r['p']:.3f}  ({r['n_pos']}/{r['n_neg']})  {time.time() - t0:.0f}s")
    guns = sorted(res)
    n, sig = len(guns), sum(res[f]["p"] < 0.05 for f in guns)
    # pooled null: draw one placebo per gun, mean AUROC -> distribution
    draws = np.array([[rng.choice(res[f]["pl_auc"]) for f in guns] for _ in range(2000)]).mean(1)
    mean_real = float(np.mean([res[f]["auc"] for f in guns]))
    agree_real = agreement([res[f]["eff"] for f in guns])
    agree_null = np.array([agreement([res[f]["pl_eff"][rng.integers(len(res[f]["pl_eff"]))] for f in guns])
                           for _ in range(200)])
    # which features carry the real effects, and do they leak elapsed time?
    E = np.array([res[f]["eff"] for f in guns])
    top = np.argsort(-np.abs(np.median(E, 0)))[:10]
    per_class = {}
    for c in sorted({res[f]["class"] for f in guns}):
        fs = [f for f in guns if res[f]["class"] == c]
        per_class[c] = {"n": len(fs), "significant": int(sum(res[f]["p"] < 0.05 for f in fs)),
                        "mean_auc": float(np.mean([res[f]["auc"] for f in fs])),
                        "mean_null_median": float(np.mean([res[f]["null_median"] for f in fs])),
                        "agreement": agreement([res[f]["eff"] for f in fs]) if len(fs) > 1 else None}
    out = {
        "features": features, "clf": kind, "n_placebo": n_placebo, "inject_sd": inject, "n_features": len(cols),
        "design": {"name": DESIGN["name"], "target_ttf_s": TARGET, "control_before_anchor_s": DESIGN["ctrl"],
                   "tod_match_h": HOUR_TOL if DESIGN["tod"] else None, "placebo_anchor_s": DESIGN["placebo"], "folds": FOLDS, "eligible": "no error state, no terminal code, "
                   "welding only, after warm-up"},
        "n_guns": n, "significant_p05": int(sig), "expected_by_chance": 0.05 * n,
        "binomial_p": binom_sf(sig, n),
        "mean_auc_real": mean_real, "mean_auc_null_median": float(np.median(draws)),
        "mean_auc_null_q95": float(np.quantile(draws, 0.95)),
        "pooled_p": float((1 + (draws >= mean_real).sum()) / (1 + len(draws))),
        "agreement_real": agree_real, "agreement_null_median": float(np.median(agree_null)),
        "agreement_null_q95": float(np.quantile(agree_null, 0.95)),
        "top_effect_features": [{"feature": cols[j], "median_d": float(np.median(E[:, j])),
                                 "share_same_sign": float((np.sign(E[:, j]) == np.sign(np.median(E[:, j]))).mean())}
                                for j in top],
        "per_class": per_class,
        "per_gun": {f: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[f].items()
                        if k not in ("eff", "pl_eff", "pl_auc")} for f in guns},
        "seconds": round(time.time() - t0),
    }
    name = tag or f"c2st_{DESIGN['name']}_{features}_{kind}"
    os.makedirs(HERE, exist_ok=True)
    json.dump(out, open(os.path.join(HERE, f"{name}.json"), "w", encoding="utf-8"), indent=2)
    log(f"\n{name}: {sig}/{n} guns p<0.05 (chance {0.05 * n:.1f}, binomial p {out['binomial_p']:.2g}); "
        f"mean AUROC {mean_real:.3f} vs null {out['mean_auc_null_median']:.3f} (q95 {out['mean_auc_null_q95']:.3f}, "
        f"p {out['pooled_p']:.3g}); direction agreement {agree_real:.3f} vs null {out['agreement_null_median']:.3f} "
        f"(q95 {out['agreement_null_q95']:.3f})")
    return out


# ------------------------------------------------------------------ superposed epochs
def gun_profile(g, cols, a):
    tt = g["ttf_s"].to_numpy()
    basem = tt > a + 24 * H
    if basem.sum() < 200:
        return None
    X = g[cols].to_numpy(dtype=np.float64)
    med = np.median(X[basem], 0)
    mad = 1.4826 * np.median(np.abs(X[basem] - med), 0)
    sd = np.where(mad > 1e-6, mad, X[basem].std(0) + 1e-6)
    Z = (X - med) / sd
    prof = np.full((len(EPOCH_BINS), len(cols)), np.nan)
    for b, (lo, hi) in enumerate(EPOCH_BINS):
        m = (tt > a + lo) & (tt <= a + hi)
        if m.sum() >= 5:
            prof[b] = np.median(Z[m], 0)
    return prof


def epoch(features, n_placebo=200):
    t0 = time.time()
    w, cols = load(features)
    rng = np.random.default_rng(SEED)
    guns = {f: g.sort_values("time") for f, g in w.groupby("file", sort=True)}
    cls = {f: g["class"].iloc[0] for f, g in guns.items()}
    real = {f: p for f, g in guns.items() if (p := gun_profile(g, cols, 0)) is not None}
    fs = sorted(real)
    R = np.array([real[f] for f in fs])                              # gun x bin x feature
    med = np.nanmedian(R, 0)
    boot = np.array([np.nanmedian(R[rng.integers(len(fs), size=len(fs))], 0) for _ in range(1000)])
    lo, hi = np.nanquantile(boot, 0.025, 0), np.nanquantile(boot, 0.975, 0)
    # placebo: each draw anchors every gun at its own random time 30-90 h before the failure
    pl = []
    for k in range(n_placebo):
        ps = [gun_profile(guns[f], cols, rng.uniform(30 * H, 90 * H)) for f in fs]
        pl.append(np.nanmedian(np.array([p for p in ps if p is not None]), 0))
        if k % 50 == 0:
            log(f"placebo {k}/{n_placebo}  {time.time() - t0:.0f}s")
    pl = np.array(pl)
    band_lo, band_hi = np.nanquantile(pl, 0.025, 0), np.nanquantile(pl, 0.975, 0)
    out = {"features": features, "bins": EPOCH_LABELS, "n_guns": len(fs), "n_placebo": n_placebo, "pooled": {},
           "per_class": {}}
    for j, c in enumerate(cols):
        outside = [(med[b, j] > band_hi[b, j]) or (med[b, j] < band_lo[b, j]) for b in range(len(EPOCH_BINS))]
        out["pooled"][c] = {"median_z": med[:, j].round(3).tolist(), "ci_lo": lo[:, j].round(3).tolist(),
                            "ci_hi": hi[:, j].round(3).tolist(), "placebo_lo": band_lo[:, j].round(3).tolist(),
                            "placebo_hi": band_hi[:, j].round(3).tolist(),
                            "outside_placebo": [bool(o) for o in outside]}
    for c in sorted(set(cls.values())):
        idx = [i for i, f in enumerate(fs) if cls[f] == c]
        out["per_class"][c] = {"n": len(idx), "median_z": {col: np.nanmedian(R[idx, :, j], 0).round(3).tolist()
                                                          for j, col in enumerate(cols)}}
    # summary: per bin, features outside the placebo band, ranked by distance from the band
    summ = {}
    for b, lab in enumerate(EPOCH_LABELS):
        rows = []
        for j, c in enumerate(cols):
            d = med[b, j] - band_hi[b, j] if med[b, j] > band_hi[b, j] else (
                band_lo[b, j] - med[b, j] if med[b, j] < band_lo[b, j] else 0.0)
            if d > 0:
                rows.append((c, round(float(med[b, j]), 3), round(float(d), 3)))
        summ[lab] = sorted(rows, key=lambda r: -r[2])[:10]
        log(f"{lab:>7}: {len(rows):2d} features outside the placebo band  "
            + ", ".join(f"{c} {m:+.2f}" for c, m, _ in summ[lab][:5]))
    out["outside_by_bin"] = summ
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(os.path.join(HERE, f"epoch_{features}.json"), "w", encoding="utf-8"), indent=2)
    return out


# ------------------------------------------------------------------ 1 Hz distribution features
RAW_SIG = ["c1", "c2", "c3", "c5", "c6", "c10", "c13", "c14", "c18"]


def raw_window(df):
    sys.path.insert(0, os.path.join(ROOT, ".py"))
    import physics as P
    welding = (df["non_welding"].to_numpy() == 0) & (df["error_active"].to_numpy() == 0)
    d = df.loc[welding, RAW_SIG + ["segment_id"]].copy()
    key = [d["segment_id"], pd.Grouper(freq="60s")]
    g = d.groupby(key)[RAW_SIG]
    parts = [g.quantile(q).add_suffix(f"_q{int(q * 100)}") for q in (0.1, 0.5, 0.9)]
    rng_ = (g.max() - g.min()).add_suffix("_range")
    out = pd.concat(parts + [rng_], axis=1)
    # press timing: seconds between press starts, how regular the cycle is
    thr = P.press_threshold(df[P.PRESS_COL].to_numpy())
    pressed = (df[P.PRESS_COL].to_numpy() > thr) & welding
    seg = df["segment_id"].to_numpy()
    start = pressed & ~np.concatenate([[False], pressed[:-1]]) & np.concatenate([[False], seg[1:] == seg[:-1]])
    st = pd.Series(df.index[start], index=df.index[start])
    gap = st.diff().dt.total_seconds()
    gap[st.index.to_series().diff().dt.total_seconds() > 600] = np.nan
    ps = pd.DataFrame({"gap": gap.to_numpy(), "seg": seg[start]}, index=st.index)
    pg = ps.groupby([ps["seg"], pd.Grouper(freq="60s")])["gap"]
    timing = pd.DataFrame({"press_gap_mean": pg.mean(), "press_gap_sd": pg.std(), "press_gap_max": pg.max()})
    out = out.join(timing, how="left")
    out = out.droplevel(0)
    out = out[~out.index.duplicated(keep="last")]
    out["file"] = df["file"].iloc[0]
    return out


def build_raw():
    os.makedirs(CACHE, exist_ok=True)
    data = os.path.join(ROOT, "preprocessed")
    for split, fs in (("train", sorted(glob.glob(os.path.join(data, "E0*.parquet")))),
                      ("test", T.test_files_in(os.path.join(data, "test")))):
        parts = []
        for i, f in enumerate(fs, 1):
            df = pd.read_parquet(f, columns=["time", "file", "segment_id", "non_welding", "error_active"] + RAW_SIG)
            df = df.set_index("time") if "time" in df.columns else df
            parts.append(raw_window(df))
            log(f"[{split} {i}/{len(fs)}] {os.path.basename(f)}")
        x = pd.concat(parts)
        x.index.name = "time"
        x.reset_index().to_parquet(os.path.join(CACHE, f"raw60_{split}.parquet"))
    log(f"cached raw distribution features in {CACHE}")


# ------------------------------------------------------------------ calibration
def sanity(kind="lr", n_placebo=50):
    """(1) a placebo anchor scored as if real -> p < 0.05 should hit ~5% of guns;
    (2) +0.5 sd shift injected into the real target windows -> should be caught in most guns."""
    w, cols = load("base")
    rng = np.random.default_rng(SEED + 1)
    hits, n = 0, 0
    for f, g in w.groupby("file", sort=True):
        g = g.sort_values("time")
        X = g[cols].to_numpy(dtype=np.float64)
        aucs, tries = [], 0
        while len(aucs) < n_placebo + 1 and tries < (n_placebo + 1) * 5:
            tries += 1
            s = make_sets(g, rng.uniform(*DESIGN["placebo"]))
            if s is not None:
                aucs.append(c2st_auc(X, *s, kind))
        if len(aucs) < 21:
            continue
        p = (1 + (np.array(aucs[1:]) >= aucs[0]).sum()) / (1 + n_placebo)
        hits += p < 0.05
        n += 1
    log(f"placebo-as-real: {hits}/{n} guns p<0.05 (expected {0.05 * n:.1f})")
    inj = c2st("base", kind, n_placebo, inject=0.5, tag=f"sanity_{DESIGN['name']}_inject05_{kind}")
    json.dump({"placebo_as_real_significant": int(hits), "n_guns": n, "expected": 0.05 * n,
               "inject_0.5sd_significant": inj["significant_p05"]},
              open(os.path.join(HERE, f"sanity_{DESIGN['name']}_{kind}.json"), "w", encoding="utf-8"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("c2st")
    c.add_argument("--features", default="base", choices=["base", "phys", "raw", "all"])
    c.add_argument("--clf", default="lr", choices=["lr", "lgbm"])
    c.add_argument("--n-placebo", type=int, default=100)
    e = sub.add_parser("epoch")
    e.add_argument("--features", default="phys", choices=["base", "phys", "raw", "all"])
    sub.add_parser("raw")
    s = sub.add_parser("sanity")
    s.add_argument("--clf", default="lr", choices=["lr", "lgbm"])
    for p in (c, s):
        p.add_argument("--design", default="day", choices=sorted(DESIGNS))
    a = ap.parse_args()
    global DESIGN
    DESIGN = {**DESIGNS[getattr(a, "design", "day")], "name": getattr(a, "design", "day")}
    if a.cmd == "c2st":
        c2st(a.features, a.clf, a.n_placebo)
    elif a.cmd == "epoch":
        epoch(a.features)
    elif a.cmd == "raw":
        build_raw()
    else:
        sanity(a.clf)


if __name__ == "__main__":
    main()
