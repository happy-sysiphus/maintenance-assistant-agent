"""
B-12 (2026-10-03): does a longer aggregation window see the failure earlier? Window length 1 min (production),
10 min, 1 h, 3 h, 6 h - same 23 continuous features (mean + std = 46), same per-gun centring (6 h warm-up), same
IsolationForest + LightGBM recipe, gun-level 4-fold over the 64 train guns + a fit on all 64 scored on the 8 test guns.

    windows   non-overlapping, clock-aligned (as train.window_features). The 1 min windows are the production ones
              (results/B-2/cache, split at gaps > 60 s). Longer windows ignore the segment split: the gap rows are
              deleted, not filled, so a mean / std over hours across a short gap is still a statistic of measured
              rows - splitting would drop most 3-6 h windows (a file has ~16 gaps > 60 s). >= half the samples.
    decision  a window is judged when it ends: its lead time is the time to failure at its END (ttf_s = min).
              A long window therefore also means a decision only every W.
    metrics   (a) train.evaluate per fold (auroc, auroc_pre_rule = windows that do not reach the last 10 min,
              normal alarm rate), (b) the B-11 horizon table by window end + placebo (pseudo failures 36-120 h
              earlier) - real - placebo is the failure-specific part.

    python results/B-12/window_size.py build   # ~10 min once: results/B-12/cache/w{600,3600,10800,21600}_*.parquet
    python results/B-12/window_size.py run     # ~10 min: results/B-12/window_size.json
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
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, os.path.join(ROOT, "results", "B-11"))
import horizon as H  # noqa: E402
import train as T  # noqa: E402

CACHE = os.path.join(HERE, "cache")
B2 = os.path.join(ROOT, "results", "B-2", "cache")
SIZES = [60, 600, 3600, 3 * 3600, 6 * 3600]
NAMES = {60: "1m", 600: "10m", 3600: "1h", 10800: "3h", 21600: "6h"}


def log(msg):
    print(msg, flush=True)


def paths(w, split):
    if w == 60:
        return os.path.join(B2, f"w60_{split}.parquet"), os.path.join(B2, f"calib60_{split}.parquet")
    return os.path.join(CACHE, f"w{w}_{split}.parquet"), os.path.join(CACHE, f"calib{w}_{split}.parquet")


def build():
    os.makedirs(CACHE, exist_ok=True)
    meta = json.load(open(os.path.join(B2, "meta60.json"), encoding="utf-8"))
    feat = [c for c in meta["feature_cols"] if c != "c19"]
    gn = meta["gun_norm"]
    cols = [c for c in gn["columns"] if c != "c19"]
    pre = os.path.join(ROOT, "preprocessed")
    for split, files in (("train", sorted(glob.glob(os.path.join(pre, "E0*.parquet")))),
                         ("test", sorted(glob.glob(os.path.join(pre, "test", "test_*.parquet"))))):
        out = {w: ([], []) for w in SIZES if w != 60}
        t0 = time.time()
        for i, f in enumerate(files, 1):
            df = pd.read_parquet(f).assign(segment_id=0)
            for w, (ws, cs) in out.items():
                win, calib = T.window_file(df, w, feat, gn, cols)
                ws.append(win.reset_index())
                if calib is not None:
                    cs.append(calib.reset_index())
            log(f"[{split} {i}/{len(files)}] {os.path.basename(f)} {time.time() - t0:.0f}s")
        for w, (ws, cs) in out.items():
            wp, cp = paths(w, split)
            pd.concat(ws, ignore_index=True).to_parquet(wp)
            pd.concat(cs, ignore_index=True).to_parquet(cp)
            log(f"{split} {NAMES[w]}: {sum(len(x) for x in ws):,} windows")


def load(w):
    d = {}
    for split in ("train", "test"):
        wp, cp = paths(w, split)
        x, c = pd.read_parquet(wp), pd.read_parquet(cp)
        for y in (x, c):
            y.drop(columns=[k for k in y.columns if k.startswith("c19_")], inplace=True)
        d[split] = (x, c)
    return d


def fold_metrics(v, thr, cal, model, cols):
    gun_thr = T.gun_thresholds(model, cols, cal, thr, H.Q, H.GATE)
    m = T.evaluate(v, v["score_ens"].to_numpy(), thr, 3, H.GATE, gun_thr, warmup_alarms=False)
    return {k: m[k] for k in ("auroc", "auroc_pre_rule", "auroc_gun_mean", "alarm_rate_normal", "recall_pre_failure",
                              "n_windows", "n_pre_failure", "n_normal")}


def run_size(w):
    d = load(w)
    tr, cal_tr = d["train"]
    te, cal_te = d["test"]
    cols = T.model_input_columns([c[:-5] for c in tr.columns if c.endswith("_mean")])
    files = sorted(tr["file"].unique())
    folds = T.cv_folds(files, H.K, H.SEED)
    parts, fm = [], []
    for k, va in enumerate(folds):
        model, thr = H.fit_production(tr[~tr["file"].isin(va)], cols)
        v = tr[tr["file"].isin(va)].copy()
        for name, s in H.score_parts(model, v[cols].to_numpy(np.float32)).items():
            v[f"score_{name}"] = s
        c = cal_tr[cal_tr["file"].isin(va)]
        v["alarm"] = H.alarms(v, model, cols, thr, c)
        fm.append(fold_metrics(v, thr, c, model, cols))
        parts.append(v)
    cv = pd.concat(parts, ignore_index=True)
    model, thr = H.fit_production(tr, cols)
    t = te.copy()
    for name, s in H.score_parts(model, t[cols].to_numpy(np.float32)).items():
        t[f"score_{name}"] = s
    t["alarm"] = H.alarms(t, model, cols, thr, cal_te)
    test_m = fold_metrics(t, thr, cal_te, model, cols)

    res = {"window_s": w, "n_windows_train": len(tr), "windows_per_gun": len(tr) / len(files), "folds": fm,
           "cv_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in fm[0]},
           "cv_sd": {k: float(np.nanstd([f[k] for f in fm])) for k in fm[0]}, "test_metrics": test_m}
    for tag, x in (("cv", cv), ("test", t)):
        rows, neg_alarm = H.horizon_table(x, per_gun=False)
        null = {a: H.horizon_table(x, a * 3600, per_gun=False)[0] for a in H.PLACEBO_H}
        for i, r in enumerate(rows):
            pv = np.array([null[a][i]["auc_ens"] for a in H.PLACEBO_H], float)
            r["placebo_median"] = float(np.nanmedian(pv)) if np.isfinite(pv).any() else np.nan
            r["placebo_q95"] = float(np.nanquantile(pv, .95)) if np.isfinite(pv).any() else np.nan
        res[tag] = {"rows": rows, "neg_alarm_rate": neg_alarm}
    return res


def fmt(x):
    return "   -  " if x != x else f"{x:.3f} "


def main():
    out = {"bins": H.LABELS, "sizes": {}}
    for w in SIZES:
        t0 = time.time()
        r = run_size(w)
        out["sizes"][NAMES[w]] = r
        m, s = r["cv_mean"], r["cv_sd"]
        log(f"\n== {NAMES[w]} windows ({r['windows_per_gun']:.0f}/gun, {time.time() - t0:.0f}s) ==")
        log(f"4-fold: AUROC {m['auroc']:.3f}+-{s['auroc']:.3f}  pre-rule {m['auroc_pre_rule']:.3f}+-{s['auroc_pre_rule']:.3f}"
            f"  per gun {m['auroc_gun_mean']:.3f}  alarm@normal {m['alarm_rate_normal']:.4f}  recall {m['recall_pre_failure']:.3f}"
            f"  | test AUROC {r['test_metrics']['auroc']:.3f} pre-rule {r['test_metrics']['auroc_pre_rule']:.3f}")
        for tag in ("cv", "test"):
            rows = r[tag]["rows"]
            log(f"  {tag:4s} n_pos   " + " ".join(f"{x['n_pos']:6d}" for x in rows))
            log(f"  {tag:4s} AUROC   " + " ".join(fmt(x["auc_ens"]) for x in rows))
            log(f"  {tag:4s} clean   " + " ".join(fmt(x["auc_ens_clean"]) for x in rows))
            log(f"  {tag:4s} placebo " + " ".join(fmt(x["placebo_median"]) for x in rows))
            log(f"  {tag:4s} plc q95 " + " ".join(fmt(x["placebo_q95"]) for x in rows))
            log(f"  {tag:4s} alarm   " + " ".join(fmt(x["alarm_rate"]) for x in rows)
                + f"  (negatives {r[tag]['neg_alarm_rate']:.4f})")
    dst = os.path.join(HERE, "window_size.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    log(f"saved {dst}")


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else main()
