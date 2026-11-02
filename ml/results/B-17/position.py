"""
B-17 (2026-11-01): does the DataPrep report's class-specific position deviation (c3 - c15; E02 from 24 h, E03 from
6 h before the failure, `datapret 결과/EDA 보고서 (DataPrep).pdf`) survive the controls that removed every earlier
"hours ahead" signal? B-14 pooled the four classes and asked for a majority of ALL guns to agree, so a signal that
lives in one class (18 guns) could have slipped through - this is the one gap the report points at.

c15 is the open-stroke setpoint, not a position target during the weld: open, c3 ~ c15; pressing (c4 = 1), c3 is the
closed position (~ -2 mm). A 1 Hz point of c3 - c15 is therefore ~0 or ~ -c15 depending on the gun state, and its mean
is roughly -(pressing share) x (stroke setpoint). The cache splits it into those parts plus the two position quantities
that mean something physically:

    open_dev   median(c3 - c15) while open and settled (c4 = 0, c18 = 0, |c3 - c15| < 10)   - stroke tracking error
    c3_closed  median(c3) while force is reached (c4 = 1)                                    - closed position / wear

Sub-commands (run one at a time, ~1 GB free RAM):

    cache      per-minute features of the 72 train + 8 test CSVs -> cache/min_<file>.parquet, plus DataPrep's own
               60 s point sample (c16 <= 6 filter, iloc[::60]) -> cache/dp_<file>.parquet, cache/meta.csv
    reproduce  DataPrep's per-class bins of dev_pos vs the gun's 48 h+ mean, and the same bins of pressing share and
               c15 on the same sample (does the composition explain it?)               -> reproduce.json
    profile    PRE-REGISTERED test (criteria below), real anchor vs placebo anchors     -> profile.json

profile - pre-registered before looking at any profile result:
    clean minutes   no error code, >= 30 welding rows (c16 > 0); kept files only (missing <= 40%)
    anchor          real = fault onset (first own terminal code in the last 15 min, else end - 10 min);
                    placebo = real - k x 24 h, k = 1, 2, 3 (whole days: same clock time). Only minutes before the anchor
    baseline        48-96 h before the anchor (same width for real and placebo, so linear drift is matched)
    effect          per gun d = (mean in bin - baseline mean) / baseline sd, raw and setpoint-conditioned
                    (residual vs the gun's baseline mean of the same (c15, c16) program, >= 30 baseline minutes)
    bins            1-6 h, 6-24 h, 24-48 h before the anchor (0-1 h is the alarm horizon B-11 already explains)
    primary         E02 x {6-24 h, 1-6 h}, E03 x {1-6 h} on open_dev and c3_closed (the report's claims)
    survives if     (1) majority of the class's kept guns share the sign of d_real, (2) |median d_real| exceeds
                    |median d_placebo| at every k, (3) Mann-Whitney real vs pooled placebo p < 0.05;
                    test files of the class must not point the other way (direction check only, 1-2 files)

    python results/B-17/position.py cache|reproduce|profile

Never writes outside results/B-17/.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(HERE, "cache")
TERM = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
COLS = ["time", "c1", "c2", "c3", "c4", "c5", "c6", "c10", "c11", "c13", "c14", "c15", "c16", "c18", "error"]
HR = 3600


def log(msg):
    print(msg, flush=True)


def save(name, out):
    dst = os.path.join(HERE, f"{name}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst}")


def files():
    """(name, path, class, dropped) for the 72 train + 8 test files; test class from the test manifest."""
    tr = pd.read_csv(os.path.join(ROOT, "preprocessed", "manifest.csv"))
    te = pd.read_csv(os.path.join(ROOT, "preprocessed", "test", "manifest.csv"))
    out = [(r.file, os.path.join(ROOT, "train", f"{r.file}.csv"), r["class"], isinstance(r.dropped, str))
           for _, r in tr.iterrows()]
    out += [(r.file, os.path.join(ROOT, "test", f"{r.file}.csv"), r["class"], False) for _, r in te.iterrows()]
    return out


def read(path):
    num = {c: "float32" for c in COLS if c not in ("time", "c10", "error")}
    df = pd.read_csv(path, usecols=COLS, dtype={**num, "c10": str, "error": str}, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    return df.sort_values("time").reset_index(drop=True)


def fault_time(df, cls):
    end = df["time"].iloc[-1]
    hit = df.loc[(df["error"] == TERM[cls]) & (df["time"] >= end - pd.Timedelta(minutes=15)), "time"]
    return (hit.iloc[0], "terminal code") if len(hit) else (end - pd.Timedelta(minutes=10), "end - 10 min")


def minute_features(df, cls):
    weld = df["c16"] > 0
    press = weld & (df["c4"] == 1)
    dev = df["c3"] - df["c15"]
    open_ = weld & (df["c4"] == 0) & (df["c18"] == 0) & (dev.abs() < 10)
    err = df["error"].fillna("0")
    w = pd.DataFrame({
        "minute": df["time"].dt.floor("min"),
        "n": 1,
        "n_weld": weld.astype("int32"),
        "n_press": press.astype("int32"),
        "n_open": open_.astype("int32"),
        "c18_on": (weld & (df["c18"] == 1)).astype("float32"),
        "c3_closed": df["c3"].where(press),
        "c2_press": df["c2"].where(press),
        "force_err": (df["c2"] - df["c14"]).where(press),
        "c5_press": df["c5"].where(press),
        "open_dev": dev.where(open_),
        "c1": df["c1"].where(weld),
        "c6": df["c6"].where(weld),
        "c15": df["c15"].where(weld),
        "c16": df["c16"].where(weld),
        "dp_dev": dev.where(df["c16"] <= 6),
        "c10_on": (df["c10"] == "on").astype("float32"),
        "err_any": (err != "0").astype("float32"),
        "err_term": err.isin(TERM.values()).astype("float32"),
        "err_other": (~err.isin(["0", TERM[cls]])).astype("float32"),
        "c11": df["c11"],
    })
    g = w.groupby("minute")
    out = g[["n", "n_weld", "n_press", "n_open"]].sum()
    out = out.join(g[["c3_closed", "c2_press", "force_err", "c5_press", "open_dev", "c1", "c6"]].median())
    out = out.join(g[["c18_on", "c15", "c16", "dp_dev", "c10_on", "err_any", "err_term", "err_other"]].mean())
    welds = (g["c11"].max() - g["c11"].min()).astype("float32")
    out["welds"] = welds.where((welds >= 0) & (welds < 120))
    out["press_share"] = (out["n_press"] / out["n_weld"]).where(out["n_weld"] > 0)
    out["c18_on"] = out["c18_on"] * out["n"] / out["n_weld"].where(out["n_weld"] > 0)
    for c, n in (("c3_closed", "n_press"), ("c2_press", "n_press"), ("force_err", "n_press"),
                 ("c5_press", "n_press"), ("open_dev", "n_open")):
        out.loc[out[n] < 3, c] = np.nan
    return out.reset_index()


def dataprep_sample(df, cls, ft):
    """eda_dataprep.load_one as written: rows before the fault, c16 <= 6, every 60th row."""
    d = df[df["time"] < ft]
    d = d[d["c16"] <= 6].iloc[::60]
    err = d["error"].fillna("0")
    return pd.DataFrame({
        "time": d["time"].values,
        "h": ((ft - d["time"]).dt.total_seconds() / HR).values,
        "dev_pos": (d["c3"] - d["c15"]).values,
        "press": (d["c4"] == 1).astype("float32").values,
        "c15": d["c15"].values,
        "c16": d["c16"].values,
        "other_error": (~err.isin(["0", TERM[cls]])).astype("float32").values,
        "c10_on": (d["c10"] == "on").astype("float32").values,
    })


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    meta = []
    for i, (name, path, cls, dropped) in enumerate(files(), 1):
        t0 = time.time()
        df = read(path)
        ft, src = fault_time(df, cls)
        m = minute_features(df, cls)
        m["h"] = ((ft - (m["minute"] + pd.Timedelta(seconds=60))).dt.total_seconds() / HR).astype("float32")
        m.to_parquet(os.path.join(CACHE, f"min_{name}.parquet"), index=False)
        dataprep_sample(df, cls, ft).to_parquet(os.path.join(CACHE, f"dp_{name}.parquet"), index=False)
        meta.append({"file": name, "class": cls, "split": "test" if name.startswith("test") else "train",
                     "dropped": dropped, "fault_time": ft, "fault_source": src, "start": df["time"].iloc[0],
                     "end": df["time"].iloc[-1], "rows": len(df), "minutes": len(m)})
        log(f"[{i}/80] {name} {cls} rows {len(df):,} minutes {len(m):,} fault {src} ({time.time() - t0:.0f}s)")
        del df, m
    pd.DataFrame(meta).to_csv(os.path.join(CACHE, "meta.csv"), index=False)
    log("cache done")


DP_BINS = [(96, 168, "4-7d"), (48, 96, "2-4d"), (24, 48, "1-2d"), (6, 24, "6-24h"), (1, 6, "1-6h"), (0, 1, "<1h")]


def load_meta():
    meta = pd.read_csv(os.path.join(CACHE, "meta.csv"), parse_dates=["fault_time", "start", "end"])
    return meta[meta["file"] != "E02_2"]  # 10-minute demo file


def dp_bins(d, cols):
    """Per class: mean over rows of (value - the gun's 48 h+ mean), by DataPrep's time-to-fault bins."""
    d = d.reset_index(drop=True)
    for c in cols:
        d[c] = d[c] - d[c].where(d["h"] > 48).groupby(d["file"]).transform("mean")
    out = {}
    for lo, hi, lab in DP_BINS:
        b = d[(d["h"] > lo) & (d["h"] <= hi)]
        out[lab] = {c: round(float(b[c].mean()), 3) for c in cols} | {"rows": len(b)}
    return out


def reproduce():
    meta = load_meta()
    tr = meta[meta["split"] == "train"]
    d = pd.concat([pd.read_parquet(os.path.join(CACHE, f"dp_{f}.parquet")).assign(file=f, cls=c)
                   for f, c in zip(tr["file"], tr["class"])], ignore_index=True)
    d["composition"] = -d["press"] * d["c15"]          # what dev_pos would be if c3 = 0 when pressing, c15 when open
    d["residual"] = d["dev_pos"] - d["composition"]
    d["dev_open"] = d["dev_pos"].where(d["press"] == 0)
    log(f"DataPrep sample: {len(d):,} rows, {d['file'].nunique()} guns (paper: 623,201 / 71)")
    cols = ["dev_pos", "press", "c15", "composition", "residual", "dev_open", "other_error", "c10_on"]
    kept = set(tr.loc[~tr["dropped"], "file"])
    out = {"rows": len(d), "guns": int(d["file"].nunique()),
           "corr_dev_pos_vs_composition": round(float(d[["dev_pos", "composition"]].corr().iloc[0, 1]), 3),
           "all_files": {}, "kept_files": {}, "per_gun_6_24h_minus_48h": {}}
    for cls, g in d.groupby("cls"):
        out["all_files"][cls] = dp_bins(g, cols)
        out["kept_files"][cls] = dp_bins(g[g["file"].isin(kept)], cols)
        log(f"\n{cls} (all files)  " + "  ".join(f"{k}: dev {v['dev_pos']:+.1f} comp {v['composition']:+.1f} "
                                                 f"press {v['press']:+.3f}" for k, v in out["all_files"][cls].items()))
        log(f"{cls} (kept files) " + "  ".join(f"{k}: dev {v['dev_pos']:+.1f} comp {v['composition']:+.1f} "
                                               f"press {v['press']:+.3f}" for k, v in out["kept_files"][cls].items()))
    for f, g in d[d["cls"].isin(["E02", "E03"])].groupby("file"):
        base, near = g[g["h"] > 48], g[(g["h"] > 6) & (g["h"] <= 24)]
        out["per_gun_6_24h_minus_48h"][f] = {
            c: round(float(near[c].mean() - base[c].mean()), 3) for c in ["dev_pos", "composition", "residual", "press"]
        } | {"kept": f in kept}
    save("reproduce", out)


PRIMARY = {("E02", "6-24h"), ("E02", "1-6h"), ("E03", "1-6h")}
PRIMARY_FEATS = ("open_dev", "c3_closed")
SENSOR_FEATS = ["open_dev", "c3_closed", "dp_dev", "press_share", "force_err", "c2_press", "c5_press", "c1", "c6"]
STATE_FEATS = ["welds", "err_any", "err_other"]       # gun state itself: all minutes, not only clean ones
COND_FEATS = ["open_dev", "c3_closed", "dp_dev", "press_share", "force_err", "c2_press", "c5_press"]
PROFILE_BINS = [(1, 6, "1-6h"), (6, 24, "6-24h"), (24, 48, "24-48h")]
BASE = (48, 96)
PLACEBO_K = (1, 2, 3)


def gun_deltas(m, shift_h):
    """Per feature x version x bin: mean in the bin - baseline mean, hours counted back from fault - shift_h."""
    m = m.assign(ha=m["h"] - shift_h)
    m = m[m["ha"] > 0]
    base_all = m[(m["ha"] > BASE[0]) & (m["ha"] <= BASE[1])]
    clean = (m["err_any"] == 0) & (m["n_weld"] >= 30)
    mc, bc = m[clean], base_all[clean.loc[base_all.index]]
    prog = mc["c15"].round(0).astype(str) + "|" + mc["c16"].round(1).astype(str)
    bprog = prog.loc[bc.index]
    out = {}
    for f in SENSOR_FEATS + STATE_FEATS:
        src, b = (mc, bc) if f in SENSOR_FEATS else (m, base_all)
        versions = {"raw": src[f]}
        if f in COND_FEATS:
            stats = bc.groupby(bprog)[f].agg(["mean", "count"])
            ref = prog.map(stats["mean"].where(stats["count"] >= 30))
            versions["cond"] = src[f] - ref
        for v, s in versions.items():
            sb = s.loc[b.index]
            if sb.count() < 30:
                continue
            for lo, hi, lab in PROFILE_BINS + [(0, 1, "0-1h")]:
                sel = s[(src["ha"] > lo) & (src["ha"] <= hi)]
                if sel.count() >= 10:
                    sd = sb.std()
                    out[(f, v, lab)] = (float(sel.mean() - sb.mean()), float((sel.mean() - sb.mean()) / sd) if sd > 0
                                        else np.nan)
    return out


def profile():
    from scipy.stats import mannwhitneyu
    meta = load_meta()
    meta = meta[~meta["dropped"]]
    rec = []                                           # (file, class, split, anchor k, feature, version, bin, delta, d)
    for _, r in meta.iterrows():
        m = pd.read_parquet(os.path.join(CACHE, f"min_{r.file}.parquet"))
        for k in (0,) + PLACEBO_K:
            if r.split == "test" and k:
                continue
            for (f, v, lab), (dl, d) in gun_deltas(m, 24 * k).items():
                rec.append((r.file, r["class"], r.split, k, f, v, lab, dl, d))
    df = pd.DataFrame(rec, columns=["file", "cls", "split", "k", "feat", "ver", "bin", "delta", "d"])
    df.to_csv(os.path.join(HERE, "profile_deltas.csv"), index=False)
    res = []
    for (cls, f, v, lab), g in df[df["split"] == "train"].groupby(["cls", "feat", "ver", "bin"]):
        real = g.loc[g["k"] == 0, "delta"].dropna()
        plc = g.loc[g["k"] > 0, "delta"].dropna()
        if len(real) < 5 or len(plc) < 5:
            continue
        med = float(real.median())
        sign = np.sign(med)
        agree = float((np.sign(real) == sign).mean()) if sign else 0.0
        pk = {f"k{k}": float(g.loc[g["k"] == k, "delta"].median()) for k in PLACEBO_K}
        p = float(mannwhitneyu(real, plc, alternative="two-sided").pvalue)
        te = df[(df["split"] == "test") & (df["cls"] == cls) & (df["feat"] == f) & (df["ver"] == v)
                & (df["bin"] == lab)].set_index("file")["delta"].round(4).to_dict()
        ok = lab != "0-1h" and agree > 0.5 and all(abs(med) > abs(x) for x in pk.values()) and p < 0.05
        test_ok = all(np.sign(x) != -sign for x in te.values())
        res.append({"cls": cls, "feat": f, "ver": v, "bin": lab, "guns": len(real), "median_real": med,
                    "median_d_real": float(g.loc[g["k"] == 0, "d"].median()), "sign_agree": agree,
                    "placebo_medians": pk, "mwu_p": p, "test": te, "passes_cv": ok, "test_not_opposite": test_ok,
                    "primary": (cls, lab) in PRIMARY and f in PRIMARY_FEATS})
    res = pd.DataFrame(res)
    pre = res["bin"] != "0-1h"                          # BH over the pre-registered horizons only
    pv = res.loc[pre, "mwu_p"].sort_values()
    q = (pv * len(pv) / np.arange(1, len(pv) + 1))[::-1].cummin()[::-1].clip(upper=1)
    res["bh_q"] = q.reindex(res.index)
    keep = ["cls", "feat", "ver", "bin", "guns", "median_real", "sign_agree", "placebo_medians", "mwu_p", "bh_q",
            "test", "passes_cv", "test_not_opposite"]
    log("PRIMARY (report's claims):")
    log(res[res["primary"]][keep].to_string())
    log("\nanything passing the CV criteria (1-48 h):")
    log(res[res["passes_cv"]][keep].to_string() or "none")
    log("\n0-1h (descriptive, not a hypothesis):")
    log(res[(res["bin"] == "0-1h") & (res["ver"] == "raw")][keep].to_string())
    save("profile", {"criteria": __doc__.split("profile - ")[1].split("python results")[0].strip(),
                     "n_tests_1_48h": int(pre.sum()), "results": res.to_dict(orient="records")})


def power():
    """Detection limit of the primary test: add s x the gun's baseline sd to the real-anchor delta only."""
    from scipy.stats import mannwhitneyu
    df = pd.read_csv(os.path.join(HERE, "profile_deltas.csv"))
    df = df[(df["split"] == "train") & (df["ver"] == "raw")]
    df["sd"] = (df["delta"] / df["d"]).abs()
    out = {}
    for cls, lab in sorted(PRIMARY):
        for f in PRIMARY_FEATS:
            g = df[(df["cls"] == cls) & (df["bin"] == lab) & (df["feat"] == f)]
            plc = g[g["k"] > 0]
            pk = plc.groupby("k")["delta"].median()
            row = {}
            for s in (0.25, 0.5, 1.0, 2.0):
                for sign in (1, -1):
                    real = g[g["k"] == 0]
                    inj = (real["delta"] + sign * s * real["sd"]).dropna()
                    med = inj.median()
                    ok = ((np.sign(inj) == np.sign(med)).mean() > 0.5 and all(abs(med) > abs(pk))
                          and mannwhitneyu(inj, plc["delta"].dropna()).pvalue < 0.05)
                    row[f"{'+' if sign > 0 else '-'}{s} sd"] = bool(ok)
            out[f"{cls} {lab} {f} ({len(g[g['k'] == 0])} guns)"] = row
            log(f"{cls} {lab} {f:10s} guns {len(g[g['k'] == 0]):2d}  " + "  ".join(f"{k}: {v}" for k, v in row.items()))
    save("power", out)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "cache":
        build_cache()
    elif cmd == "reproduce":
        reproduce()
    elif cmd == "profile":
        profile()
    elif cmd == "power":
        power()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
