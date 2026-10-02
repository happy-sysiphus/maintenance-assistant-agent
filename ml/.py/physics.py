"""
B-7 (2026-09-28): state-conditioned "physics" window features, from the given data only.

c18 (setpoint of force build-up), c13 (setpoint of counterbalance pressure) and c4 are two-level switches:
the gun is either PRESSED (c18 high: the electrode closes on the sheet) or RELEASED. A 60 s mean / std of
c2 / c3 / c5 therefore mostly measures the share of pressed seconds (= the weld program), which is why the
`_std` features track program changes (MEMORY.md §2). The four faults are defined on the continuous signals
in ONE of the two states (readme 1.5):

    E01 counterbalance timeout  balance pressure c5 does not reach its setpoint        -> c5 per state, after release
    E02 electrode broke         electrode position c3 below the zero of the reference  -> c3 when pressed (wear)
    E03 unwanted movement       position c3 leaves the stroke tolerance                -> c3 spread per state
    E04 drift                   locked cylinder moves >= 5 mm/min                      -> c3 movement while released

So every continuous sensor is summarised separately per state, plus the transitions (press count, press
length, the first second after a release). Everything is in the preprocessed (global z) units and is
per-gun centred like the production features (warm-up median of the gun's normal welding windows).

Experiment code only (not reproducible online as written): the press threshold uses whole-file quantiles and the
gun centring is a plain warm-up median without the normal-welding filter of train.gun_norm_stats (history.md §14).

    python .py/physics.py cache        # per-window physics features for the 64 + 8 files -> results/B-7/cache (~5 min)
    python .py/physics.py effect       # effect sizes pre-failure (10-60 min) vs normal (> 24 h), per class
    python .py/physics.py run          # gun-level 4-fold (IF + LightGBM), base vs base + physics -> results/B-7/
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
OUT = os.path.join(T.PROJECT_ROOT, "results", "B-7")
CACHE = os.path.join(OUT, "cache")
WINDOW = 60
PRESS_COL = "c18"  # two-level: pressed when above the midpoint of its two levels
SIGNALS = ["c1", "c2", "c3", "c5", "c6"]
WARMUP_S = T.GUN_NORM_DEFAULT["warmup_s"]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ features
def press_threshold(x):
    lo, hi = np.nanquantile(x, 0.01), np.nanquantile(x, 0.99)
    return (lo + hi) / 2 if hi - lo > 0.5 else np.inf  # a file that never presses has no pressed state


def physics_rows(df, thr):
    """1 Hz preprocessed rows -> per-row state-conditioned signals (NaN where the state does not hold)."""
    welding = (df["non_welding"].to_numpy() == 0) & (df["error_active"].to_numpy() == 0)
    pressed = df[PRESS_COL].to_numpy() > thr
    seg = df["segment_id"].to_numpy()
    same = np.concatenate([[False], seg[1:] == seg[:-1]])
    prev_p = np.concatenate([[False], pressed[:-1]]) & same
    rel = welding & ~pressed
    out = {}
    for c in SIGNALS:
        v = df[c].to_numpy(dtype="float64")
        out[f"{c}_P"] = np.where(welding & pressed, v, np.nan)
        out[f"{c}_R"] = np.where(rel, v, np.nan)
    c3, c5 = df["c3"].to_numpy(dtype="float64"), df["c5"].to_numpy(dtype="float64")
    d3 = np.abs(np.diff(c3, prepend=np.nan))
    d5 = np.diff(c5, prepend=np.nan)
    steady = rel & ~prev_p & same  # released and was released a second ago: the cylinder should not move
    out["c3_move_R"] = np.where(steady, d3, np.nan)
    out["c5_step_R"] = np.where(steady, d5, np.nan)
    after = rel & prev_p  # first second after a release: how far c5 / c3 are from their steady values
    out["c5_after"] = np.where(after, c5, np.nan)
    out["c3_after"] = np.where(after, c3, np.nan)
    out["press_start"] = (welding & pressed & ~prev_p).astype("float32")
    out["pressed"] = (welding & pressed).astype("float32")
    out["force_track_P"] = np.where(welding & pressed, df["c2"].to_numpy() - df["c14"].to_numpy(), np.nan)
    return pd.DataFrame(out, index=df.index)


def window_physics(df):
    thr = press_threshold(df[PRESS_COL].to_numpy())
    r = physics_rows(df, thr)
    key = [df["segment_id"], pd.Grouper(freq=f"{WINDOW}s")]
    g = r.groupby(key)
    agg = g[[f"{c}_{s}" for c in SIGNALS for s in "PR"] + ["c5_after", "c3_after", "force_track_P"]].mean()
    agg = agg.join(g[["c3_P", "c3_R", "c5_R", "c2_P"]].std().add_suffix("_sd"))
    agg["c3_move_R_sum"] = g["c3_move_R"].sum(min_count=1)
    agg["c3_move_R_max"] = g["c3_move_R"].max()
    agg["c5_drift_R"] = g["c5_step_R"].mean()
    agg["c2_P_min"] = g["c2_P"].min()
    agg["c5_P_min"] = g["c5_P"].min()
    agg["press_n"] = g["press_start"].sum()
    agg["press_s"] = g["pressed"].sum()
    agg["press_len"] = agg["press_s"] / agg["press_n"].where(agg["press_n"] > 0)
    agg = agg.droplevel(0)
    agg = agg[~agg.index.duplicated(keep="last")]
    agg["file"] = df["file"].iloc[0]
    return agg


PHYS_COLS = None  # filled from the cache


def gun_centre(w, cols):
    """Subtract the median of the gun's first-6 h windows (online reproducible, like --gun-norm center);
    level-free columns (counts, movement, drift) are left as they are."""
    level = [c for c in cols if c.endswith(("_P", "_R", "_after")) and "move" not in c]
    out = []
    for f, part in w.groupby("file", sort=False):
        part = part.copy()
        wu = part[part.index < part.index[0] + pd.Timedelta(seconds=WARMUP_S)]
        med = wu[level].median()
        part[level] = part[level] - med
        out.append(part)
    return pd.concat(out)


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    data = os.path.join(T.PROJECT_ROOT, "preprocessed")
    for split, fs in (("train", sorted(glob.glob(os.path.join(data, "E0*.parquet")))),
                      ("test", T.test_files_in(os.path.join(data, "test")))):
        parts = []
        for i, f in enumerate(fs, 1):
            df = pd.read_parquet(f, columns=["file", "segment_id", "non_welding", "error_active", PRESS_COL, "c14"] + SIGNALS)
            parts.append(window_physics(df))
            log(f"[{split} {i}/{len(fs)}] {os.path.basename(f)}")
        x = pd.concat(parts)
        cols = [c for c in x.columns if c != "file"]
        x = gun_centre(x, cols)
        x.index.name = "time"
        x.reset_index().to_parquet(os.path.join(CACHE, f"phys{WINDOW}_{split}.parquet"))
    log(f"cached physics features in {CACHE}")


def phys_cols(x):
    return [c for c in x.columns if c not in ("file", "time")]


def join_physics(w, phys):
    """Windows (index time) + physics features on (file, time); missing state -> filled by the per-file last value,
    then 0 (= the gun's warm-up level)."""
    cols = phys_cols(phys)
    out = w.reset_index().merge(phys, on=["file", "time"], how="left").sort_values(["file", "time"])
    out[cols] = out.groupby("file", sort=False)[cols].ffill().fillna(0.0)
    return out.set_index("time"), cols


# ------------------------------------------------------------------ analysis
def effect():
    import experiments as X

    D = X.load(WINDOW)
    ph = pd.read_parquet(os.path.join(CACHE, f"phys{WINDOW}_train.parquet"))
    w, cols = join_physics(D["all"], ph)
    pre = (w["ttf_s"] > T.RULE_LEAD_S) & (w["ttf_s"] <= 3600) & (w["error_active"] == 0)
    nor = (w["ttf_s"] > 24 * 3600) & (w["error_active"] == 0) & (w["non_welding"] == 0)
    rows = []
    for k in T.CLASSES + ["all"]:
        m = w["class"] == k if k != "all" else np.ones(len(w), bool)
        a, b = w.loc[pre & m, cols], w.loc[nor & m, cols]
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
        rows.append(d.rename(k))
    base = [c for c in D["model_cols"] if c not in ("c19_mean", "c19_std")]
    a, b = w.loc[pre, base], w.loc[nor, base]
    db = ((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)).abs().sort_values(ascending=False)
    t = pd.concat(rows, axis=1)
    t = t.reindex(t["all"].abs().sort_values(ascending=False).index)
    print(t.round(3).to_string())
    print("\nbase features, |d| top 5 (all classes):", db.head(5).round(3).to_dict())


# ------------------------------------------------------------------ CV
def run(kinds):
    import experiments as X

    D = X.load(WINDOW)
    ptr = pd.read_parquet(os.path.join(CACHE, f"phys{WINDOW}_train.parquet"))
    pte = pd.read_parquet(os.path.join(CACHE, f"phys{WINDOW}_test.parquet"))
    D2 = dict(D)
    for key, ph in (("all", ptr), ("cal_all", ptr), ("te", pte), ("cte", pte)):
        D2[key], cols = join_physics(D[key], ph)
    base = [c for c in D["model_cols"] if c not in ("c19_mean", "c19_std")]
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for kind in kinds:
        for name, cc in ((f"p5_base_{kind}", base), (f"p5_phys_{kind}", base + cols), (f"p5_physonly_{kind}", cols)):
            r = X.cv_run(name, D2, cc, kind, note="B-7 physics features, gun-level 4-fold")
            os.replace(os.path.join(X.OUT3, f"{name}.json"), os.path.join(OUT, f"{name}.json"))
            res[name] = r
    summary()
    return res


def summary():
    for f in sorted(glob.glob(os.path.join(OUT, "p5_*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        v, t = r["val_folds"], r["test_over_folds"]
        print(f"{r['name']:<22} {r['n_features']:>3} | cv AUROC {v['mean']['auroc']:.3f}+-{v['sd']['auroc']:.3f} "
              f"pre-rule {v['mean']['auroc_pre_rule']:.3f}+-{v['sd']['auroc_pre_rule']:.3f} per-gun "
              f"{v['mean']['auroc_gun_mean']:.3f} alarm {v['mean']['alarm_rate_normal']:.3f} | test AUROC "
              f"{t['mean']['auroc']:.3f} pre-rule {t['mean']['auroc_pre_rule']:.3f} per-gun {t['mean']['auroc_gun_mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    sub.add_parser("effect")
    r = sub.add_parser("run")
    r.add_argument("--kinds", nargs="+", default=["iforest", "lgbm"])
    sub.add_parser("summary")
    a = ap.parse_args()
    {"cache": build_cache, "effect": effect, "summary": summary}.get(a.cmd, lambda: run(a.kinds))()


if __name__ == "__main__":
    main()
