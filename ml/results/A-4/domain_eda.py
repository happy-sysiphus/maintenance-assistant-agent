"""
A-4 (2026-11-01): domain EDA of the raw 1 Hz files - what the gun and the line are DOING, not how columns correlate.
Times are reported in KST (data are naive UTC, MEMORY.md §2; KST = UTC + 9 h).

What one raw row says (checked on E03_1 before writing this):
    weld        c18 = 1 (close gun) and c4 = 1 (force reached) for 1-2 s, c11 (weld count) +1, c12 (position count) +2;
                c13 / c14 / c16 carry that weld point's program, c13 falls back to 14.99 between welds
    standby     c13 = c14 = c15 = c16 = 0 (no program), gun closed at c3 ~ -2 mm, often c10 = off
    cap wear    c1 (cap offset) steps up 0.1-0.3 mm after a burst of low-force presses (dressing + measuring), and drops
                by ~3 mm after a c16 = 32 sequence (cap change)
    gaps        two kinds: counters frozen (machine stopped) or counters advanced at the normal rate (logging dropout)

    python results/A-4/domain_eda.py events     # one pass over the 80 raw files -> cache/<table>_<file>.parquet (~3 min)
    python results/A-4/domain_eda.py report     # all analyses -> domain_eda.json + figures fig_*.png (+ .log via shell)

Never writes outside results/A-4/.
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
KST = pd.Timedelta(hours=9)
COLOR = {"E01": "#2a78d6", "E02": "#eb6834", "E03": "#1baf7a", "E04": "#eda100"}


def log(msg):
    print(msg, flush=True)


def files():
    tr = pd.read_csv(os.path.join(ROOT, "preprocessed", "manifest.csv"))
    te = pd.read_csv(os.path.join(ROOT, "preprocessed", "test", "manifest.csv"))
    out = [(r.file, os.path.join(ROOT, "train", f"{r.file}.csv"), r["class"], isinstance(r.dropped, str))
           for _, r in tr.iterrows() if r.file != "E02_2"]
    out += [(r.file, os.path.join(ROOT, "test", f"{r.file}.csv"), r["class"], False) for _, r in te.iterrows()]
    return out


def read(path):
    df = pd.read_csv(path, dtype={"error": str, "c10": str}, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df["error"] = df["error"].fillna("0")
    return df.sort_values("time").reset_index(drop=True)


def runs(mask, t):
    """Start / end time of each run of True in mask (consecutive rows; a gap does not break a run)."""
    m = np.asarray(mask, bool)
    d = np.diff(np.r_[0, m.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return s, e


def events_one(df, name, cls):
    t = df["time"]
    out = {}
    # welds: rows where the weld counter advanced (dc11 > 0 and < 50 - larger jumps are gaps / resets)
    dc11 = df["c11"].diff()
    w = df[(dc11 > 0) & (dc11 < 50)]
    out["welds"] = pd.DataFrame({"time": w["time"], "dc11": dc11[w.index].astype("int16"),
                                 "c1": w["c1"], "c2": w["c2"], "c3": w["c3"], "c4": w["c4"], "c5": w["c5"],
                                 "c13": w["c13"], "c14": w["c14"], "c15": w["c15"], "c16": w["c16"],
                                 "c18": w["c18"], "error": w["error"]})
    # c1 (cap offset) changes
    d1 = df["c1"].diff()
    c = df[d1.fillna(0) != 0]
    out["c1"] = pd.DataFrame({"time": c["time"], "old": df["c1"].shift()[c.index], "new": c["c1"],
                              "c11": c["c11"], "c16": c["c16"], "c10": c["c10"]})
    # programs active in the 60 s before each c1 change (dressing / cap-change sequences)
    seq = []
    for i in c.index:
        win = df[(t >= t[i] - pd.Timedelta(seconds=60)) & (t <= t[i])]
        p = win[win["c4"] == 1]
        seq.append({"time": t[i], "presses": int((win["c4"].diff() == 1).sum()),
                    "press_c2_median": float(p["c2"].median()) if len(p) else np.nan,
                    "c16_values": ",".join(sorted({f"{v:g}" for v in win["c16"].unique()})),
                    "dc11": int(win["c11"].iloc[-1] - win["c11"].iloc[0])})
    out["c1ctx"] = pd.DataFrame(seq)
    # gaps > 60 s with counters across them
    dt = t.diff().dt.total_seconds()
    g = df.index[dt > 60]
    out["gaps"] = pd.DataFrame({
        "start": t.shift()[g], "dur_s": dt[g], "dc11": dc11[g], "dc12": df["c12"].diff()[g],
        "c10_b": df["c10"].shift()[g], "c10_a": df["c10"][g], "c16_b": df["c16"].shift()[g], "c16_a": df["c16"][g],
        "err_b": df["error"].shift()[g], "err_a": df["error"][g], "c1_b": df["c1"].shift()[g], "c1_a": df["c1"][g]})
    # state runs: c10 off, standby (no program), error episodes
    rows = []
    for state, mask in (("c10_off", df["c10"] == "off"), ("standby", df["c16"] <= 0)):
        s, e = runs(mask, t)
        for a, b in zip(s, e):
            rows.append({"state": state, "code": "", "start": t[a], "end": t[b],
                         "dur_s": (t[b] - t[a]).total_seconds() + 1, "rows": b - a + 1,
                         "dc11": df["c11"][b] - df["c11"][a], "c10_off_share": float((df["c10"][a:b + 1] == "off").mean()),
                         "standby_share": float((df["c16"][a:b + 1] <= 0).mean())})
    err = df["error"]
    s = df.index[(err != "0") & (err != err.shift())]
    for a in s:
        code = err[a]
        b = a
        while b + 1 < len(df) and err[b + 1] == code:
            b += 1
        rows.append({"state": "error", "code": code, "start": t[a], "end": t[b],
                     "dur_s": (t[b] - t[a]).total_seconds() + 1, "rows": b - a + 1,
                     "dc11": df["c11"][b] - df["c11"][a], "c10_off_share": float((df["c10"][a:b + 1] == "off").mean()),
                     "standby_share": float((df["c16"][a:b + 1] <= 0).mean())})
    out["runs"] = pd.DataFrame(rows)
    out["static"] = pd.DataFrame([{
        "file": name, "class": cls, "start": t.iloc[0], "end": t.iloc[-1], "rows": len(df),
        "c7": df["c7"].mode()[0], "c8": df["c8"].mode()[0], "c9": df["c9"].mode()[0], "c17": df["c17"].mode()[0],
        "c7_n": df["c7"].nunique(), "c8_n": df["c8"].nunique(), "c9_n": df["c9"].nunique(),
        "c11_first": df["c11"].iloc[0], "c11_last": df["c11"].iloc[-1],
        "c12_first": df["c12"].iloc[0], "c12_last": df["c12"].iloc[-1],
        "c1_min": df["c1"].min(), "c1_max": df["c1"].max()}])
    return out


def build_events():
    os.makedirs(CACHE, exist_ok=True)
    for i, (name, path, cls, dropped) in enumerate(files(), 1):
        t0 = time.time()
        df = read(path)
        ev = events_one(df, name, cls)
        for k, v in ev.items():
            v.assign(file=name).to_parquet(os.path.join(CACHE, f"{k}_{name}.parquet"), index=False)
        log(f"[{i}/79] {name} welds {len(ev['welds']):,} c1 changes {len(ev['c1'])} gaps {len(ev['gaps'])} "
            f"runs {len(ev['runs'])} ({time.time() - t0:.0f}s)")
        del df, ev


def table(kind, kept_only=True):
    parts = [pd.read_parquet(f) for f in sorted(_glob(kind))]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return d[d["file"].isin(KEPT)] if kept_only else d


def _glob(kind):
    import glob
    return glob.glob(os.path.join(CACHE, f"{kind}_*.parquet"))


DROPPED = set(pd.read_csv(os.path.join(ROOT, "preprocessed", "manifest.csv")).query("dropped.notna()")["file"])
KEPT = {n for n, _, _, d in files()} - DROPPED


def weld_rate(w):
    h = w.set_index("time")["dc11"].resample("1h").sum()
    return h[h > 0].median() / 3600 if len(h[h > 0]) else np.nan


def gaps():
    g = table("gaps", kept_only=False)
    rate = {f: weld_rate(w) for f, w in table("welds", kept_only=False).groupby("file")}
    g["ratio"] = g["dc11"] / g["dur_s"] / g["file"].map(rate)
    g["kind"] = np.select([g["dc11"] <= 0, g["ratio"] >= 0.3], ["frozen", "logging_dropout"], "partial")
    g["hour"] = (g["start"] + KST).dt.hour
    out = {}
    for lab, sub in (("kept", g[g["file"].isin(KEPT)]), ("high_missing_e02", g[g["file"].isin(DROPPED)])):
        out[lab] = {k: {"n": len(s), "dur_median_min": round(s["dur_s"].median() / 60, 1),
                        "dur_max_h": round(s["dur_s"].max() / 3600, 1),
                        "c10_off_before": round((s["c10_b"] == "off").mean(), 2),
                        "c10_off_after": round((s["c10_a"] == "off").mean(), 2),
                        "rate_ratio_median": round(s["ratio"].median(), 2),
                        "hour_kst": s["hour"].value_counts().sort_index().to_dict()} for k, s in sub.groupby("kind")}
    return out


def rhythm():
    meta = pd.read_csv(os.path.join(ROOT, "results", "B-17", "cache", "meta.csv"))
    m = pd.concat([pd.read_parquet(os.path.join(ROOT, "results", "B-17", "cache", f"min_{f}.parquet"),
                                   columns=["minute", "n", "n_weld", "welds", "c10_on", "err_any", "err_term", "h"])
                   .assign(file=f) for f in meta["file"] if f in KEPT])
    m = m[m["h"] > 0.25]
    k = m["minute"] + KST
    m = m.assign(active=(m["welds"] > 0).astype(float), off=1 - m["c10_on"],
                 nonterm=((m["err_any"] > 0) & (m["err_term"] == 0)).astype(float),
                 slot=k.dt.hour * 60 + k.dt.minute // 10 * 10, wd=k.dt.dayofweek)
    slot = m.groupby("slot")[["active", "off", "nonterm"]].mean()
    wd = m.groupby("wd")[["active", "off", "nonterm"]].mean()
    w = table("welds")
    w["day"] = (w["time"] + KST).dt.date
    st = table("static")
    cover = [(r.file, d.date()) for r in st.itertuples()
             for d in pd.date_range((r.start + KST).normalize(), (r.end + KST).normalize())]
    d = pd.DataFrame(cover, columns=["file", "day"]).merge(
        w.groupby(["file", "day"])["dc11"].sum().rename("welds").reset_index(), how="left").fillna({"welds": 0})
    day = d.groupby("day").agg(guns=("file", "size"), welds_median=("welds", "median"),
                               idle_guns=("welds", lambda x: int((x < 100).sum())))
    return slot, wd, day


def cap_cycle():
    c1 = table("c1").merge(table("c1ctx"), on=["time", "file"])
    c1["d"] = (c1["new"] - c1["old"]).round(2)
    c1["kind"] = np.select([c1["d"].between(0.05, 0.6), c1["d"] <= -1.0], ["dress", "cap_change"], "other")
    per = []
    for f, g in c1.groupby("file"):
        g = g.sort_values("time")
        d, cc = g[g["kind"] == "dress"], g[g["kind"] == "cap_change"]
        cc = cc[cc["time"].diff().dt.total_seconds().fillna(1e9) > 120]
        per.append({"file": f, "dress_n": len(d), "cap_n": len(cc), "dress_every_welds": d["c11"].diff().median(),
                    "dress_every_min": d["time"].diff().dt.total_seconds().median() / 60,
                    "cap_every_welds": cc["c11"].diff().median(),
                    "cap_every_h": cc["time"].diff().dt.total_seconds().median() / 3600,
                    "wear_per_dress": d["d"].median(), "c1_before_cap": cc["old"].median() if len(cc) else np.nan})
    per = pd.DataFrame(per)
    # short no-program pauses and short power-offs vs c1 changes
    r = table("runs")
    near = {}
    for st in ("standby", "c10_off"):
        hits = []
        for f, x in r[(r["state"] == st) & (r["dur_s"] <= 60)].groupby("file"):
            ev = np.sort(c1.loc[c1["file"] == f, "time"].to_numpy())
            if not len(ev):
                continue
            s = x["start"].to_numpy()
            i = np.searchsorted(ev, s)
            nx = np.abs((ev[np.minimum(i, len(ev) - 1)] - s) / np.timedelta64(1, "s"))
            pv = np.abs((s - ev[np.maximum(i - 1, 0)]) / np.timedelta64(1, "s"))
            hits.append(np.minimum(nx, pv) <= 120)
        h = np.concatenate(hits)
        near[st] = {"short_episodes": int(len(h)), "within_2min_of_c1_change": round(float(h.mean()), 2)}
    return c1, per, near, {
        "steps": {k: int(v) for k, v in c1["kind"].value_counts().items()},
        "cap_change_after_c16_32": round(float(c1.loc[c1["kind"] == "cap_change", "c16_values"]
                                               .str.contains(r"(?:^|,)32(?:,|$)").mean()), 2),
        "dress_after_c16_8": round(float(c1.loc[c1["kind"] == "dress", "c16_values"]
                                         .str.contains(r"(?:^|,)8(?:,|$)").mean()), 2)}


def takt():
    rows = []
    for f, w in table("welds").groupby("file"):
        w = w[(w["dc11"] == 1) & (w["c16"] > 0)]
        if len(w) < 2000:
            rows.append({"file": f, "welds": len(w)})
            continue
        prog = (w["c14"].round(-2).astype(int).astype(str) + "|" + w["c16"].round(1).astype(str)).to_numpy()
        c = pd.factorize(prog)[0][:20000]
        match = np.array([(c[:-p] == c[p:]).mean() for p in range(1, 301)])
        p = int(np.flatnonzero(match >= match.max() - 0.05)[0] + 1)
        gap = w["time"].diff().dt.total_seconds().to_numpy()
        rows.append({"file": f, "welds": len(w), "programs": len(set(prog)), "period_welds": p,
                     "period_match": round(float(match[p - 1]), 2),
                     "interweld_median_s": float(np.median(gap[(gap > 0) & (gap < 60)])),
                     "cycle_s": float(np.median(w["time"].diff(p).dt.total_seconds().to_numpy()[p:]))})
    return pd.DataFrame(rows)


def error_codes():
    r = table("runs")
    e = r[r["state"] == "error"].copy()
    k = e["start"] + KST
    e["hour"], e["wd"] = k.dt.hour, k.dt.dayofweek
    out = {}
    for code, g in e.groupby("code"):
        out[code] = {"episodes": len(g), "guns": int(g["file"].nunique()), "dur_median_min": round(g["dur_s"].median() / 60, 2),
                     "dur_max_h": round(g["dur_s"].max() / 3600, 2), "no_weld_during": round((g["dc11"] <= 0).mean(), 2),
                     "c10_off_share": round(g["c10_off_share"].mean(), 2),
                     "no_program_share": round(g["standby_share"].mean(), 2),
                     "start_kst_03_07": round(g["hour"].between(3, 6).mean(), 2),
                     "start_sunday": round((g["wd"] == 6).mean(), 2),
                     "hour_kst": g["hour"].value_counts().sort_index().to_dict()}
    off = {}
    for st in ("c10_off", "standby"):
        x = r[r["state"] == st]
        off[st] = {"episodes": len(x), "dur_quantiles_s": x["dur_s"].quantile([.25, .5, .75, .9, .99]).round(0).to_dict(),
                   "short_le_60s": round((x["dur_s"] <= 60).mean(), 2),
                   "long_gt_10min_start_hour_kst": (x.loc[x["dur_s"] > 600, "start"] + KST).dt.hour
                   .value_counts().sort_index().to_dict()}
    return out, off


def gun_identity():
    st = table("static", kept_only=False)
    st["key"] = st[["c7", "c8", "c9"]].round(4).astype(str).agg("|".join, axis=1)
    groups = []
    for key, g in st.groupby("key"):
        if len(g) < 2:
            continue
        g = g.sort_values("start")
        a, b = g.iloc[0], g.iloc[1]
        groups.append({"files": g["file"].tolist(), "classes": g["class"].tolist(),
                       "start_kst": (g["start"] + KST).dt.strftime("%Y-%m-%d %H:%M").tolist(),
                       "end_kst": (g["end"] + KST).dt.strftime("%Y-%m-%d %H:%M").tolist(),
                       "c11_first": g["c11_first"].astype(int).tolist(), "c11_last": g["c11_last"].astype(int).tolist(),
                       "overlap_h": round(max(0.0, (min(a.end, b.end) - max(a.start, b.start)).total_seconds() / 3600), 1)})
    return {"files": len(st), "distinct_c7_c8_c9": int(st["key"].nunique()), "groups": groups}


def failure_context():
    meta = pd.read_csv(os.path.join(ROOT, "results", "B-17", "cache", "meta.csv"), parse_dates=["fault_time"])
    meta = meta[meta["file"].isin(KEPT)]
    paths = {n: p for n, p, _, _ in files()}
    rows = []
    for r in meta.itertuples():
        df = read(paths[r.file])
        t = r.fault_time
        b = df[(df["time"] >= t - pd.Timedelta(minutes=5)) & (df["time"] < t)]
        at = df[df["time"] >= t].iloc[0]
        progs = set(b["c16"].round(1))
        rows.append({"file": r.file, "class": r._2, "time_kst": t + KST, "weekday": (t + KST).day_name()[:3],
                     "welds_5min": int(b["c11"].iloc[-1] - b["c11"].iloc[0]) if len(b) else -1,
                     "c10_off_5min": round(float((b["c10"] == "off").mean()), 2) if len(b) else np.nan,
                     "c10_at": at["c10"], "dress_or_capchange_5min": bool(progs & {8.0, 32.0}),
                     "codes_5min": ",".join(sorted(set(b["error"]) - {"0"}))})
        del df
    x = pd.DataFrame(rows).sort_values("time_kst")
    # failures of different guns within 15 min of each other
    x["group"] = (x["time_kst"].diff().dt.total_seconds().fillna(1e9) > 900).cumsum()
    x["simultaneous"] = x.groupby("group")["file"].transform("size") > 1
    x.to_csv(os.path.join(HERE, "failure_context.csv"), index=False)
    return x


def figures(slot, c1, x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#9a9a96", "xtick.color": "#55554f", "ytick.color": "#55554f",
                         "axes.labelcolor": "#33332f", "axes.titlesize": 11, "axes.titleweight": "bold"})
    ink = "#2a78d6"
    # 1. daily rhythm: three single-series panels, one shared time-of-day axis
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    h = slot.index / 60
    for a, col, title in zip(ax, ["active", "off", "nonterm"],
                             ["용접 중인 건 비율", "전원(c10) 꺼진 건 비율", "비종료 에러 코드(E003·E011 등)가 뜬 건 비율"]):
        a.fill_between(h, slot[col] * 100, step="post", color=ink, alpha=0.18, linewidth=0)
        a.step(h, slot[col] * 100, where="post", color=ink, linewidth=2)
        a.set_title(title, loc="left")
        a.set_ylabel("%")
        a.set_ylim(0, max(5, slot[col].max() * 115))
        for lo, hi in ((3.67, 7.0), (11.5, 12.33), (16.0, 16.33), (20.5, 21.33)):
            a.axvspan(lo, hi, color="#55554f", alpha=0.07, linewidth=0)
    for lo, lab in ((3.7, "야간 정지"), (11.55, "점심"), (16.02, "교대"), (20.55, "저녁")):
        ax[0].text(lo, ax[0].get_ylim()[1] * 0.92, lab, fontsize=9, color="#55554f", va="top")
    ax[-1].set_xticks(range(0, 25, 2))
    ax[-1].set_xlim(0, 24)
    ax[-1].set_xlabel("시각 (KST)")
    fig.suptitle("하루 가동 리듬 — 72건, 고장 15분 전까지의 모든 분 (10분 단위)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_daily_rhythm.png"), dpi=130)
    plt.close(fig)
    # 2. cap wear sawtooth of one gun over 24 h
    g = c1[c1["file"] == "E03_1"].sort_values("time")
    t0 = pd.Timestamp("2021-08-30 21:00")
    g = g[(g["time"] >= t0) & (g["time"] < t0 + pd.Timedelta(hours=24))]
    w = pd.read_parquet(os.path.join(CACHE, "welds_E03_1.parquet"), columns=["time", "dc11"])
    w = w[(w["time"] >= t0) & (w["time"] < t0 + pd.Timedelta(hours=24))]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    tk = g["time"] + KST
    ax[0].step(tk, g["new"], where="post", color=ink, linewidth=2)
    dr, cc = g[g["kind"] == "dress"], g[g["kind"] == "cap_change"]
    ax[0].scatter(dr["time"] + KST, dr["new"], s=22, color=ink, zorder=3, label="캡 드레싱 (c16 = 8 직후 +0.2~0.3 mm)",
                  edgecolor="white", linewidth=1)
    ax[0].scatter(cc["time"] + KST, cc["new"], s=60, marker="v", color="#eb6834", zorder=3,
                  label="캡 교체 (c16 = 32 직후 ~3 mm 감소)", edgecolor="white", linewidth=1)
    ax[0].set_title("c1(캡 오프셋) = 캡 마모 보정값 — E03_1, 24시간", loc="left")
    ax[0].set_ylabel("c1 (mm)")
    ax[0].legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax[0].set_ylim(0, g["new"].max() * 1.35)
    per10 = w.set_index("time")["dc11"].resample("10min").sum()
    ax[1].bar(per10.index + KST + pd.Timedelta(minutes=5), per10.values, width=pd.Timedelta(minutes=8), color=ink,
              alpha=0.8)
    ax[1].set_title("10분당 용접 수", loc="left")
    ax[1].set_xlabel("시각 (KST)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_cap_cycle.png"), dpi=130)
    plt.close(fig)
    # 3. failure calendar (2021 only; 2 E02 files from 2019-20 are listed in the caption)
    y = x[x["time_kst"] >= "2021-08-01"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for lo, hi, lab in ((6.4, 7.0, "아침 기동 직전"), (11.0, 12.0, "점심 정지 전후")):
        ax.axhspan(lo, hi, color="#55554f", alpha=0.07, linewidth=0)
        ax.text(pd.Timestamp("2021-08-16"), hi - 0.05, lab, fontsize=9, color="#55554f", va="top")
    for cls, mk in zip(["E01", "E02", "E03", "E04"], ["o", "s", "^", "D"]):
        s = y[y["class"] == cls]
        hod = s["time_kst"].dt.hour + s["time_kst"].dt.minute / 60
        ax.scatter(s["time_kst"].dt.normalize(), hod, s=46, marker=mk, color=COLOR[cls], edgecolor="white",
                   linewidth=1.2, label=f"{cls} ({TERM[cls]})", zorder=3)
    for _, s in y[y["simultaneous"]].groupby("group"):
        t = s["time_kst"]
        ax.annotate(f"{len(s)}건 동시 ({t.min():%H:%M}–{t.max():%H:%M})",
                    xy=(t.iloc[0].normalize(), t.dt.hour.mean() + t.dt.minute.mean() / 60), xytext=(9, 7),
                    textcoords="offset points", fontsize=8.5, color="#33332f",
                    arrowprops={"arrowstyle": "-", "color": "#9a9a96", "linewidth": 0.8})
    ax.axvspan(pd.Timestamp("2021-09-25"), pd.Timestamp("2021-10-13"), color="#55554f", alpha=0.05, linewidth=0)
    ax.text(pd.Timestamp("2021-09-25"), 23.6, "대부분의 건 거의 무가동 (09-25~10-12)", fontsize=9, color="#55554f", va="top")
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel("시각 (KST)")
    ax.set_title("고장(종료 코드) 시각 — 날짜 × 하루 중 시각, 2021년 70건", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_failure_calendar.png"), dpi=130)
    plt.close(fig)


def report():
    t0 = time.time()
    out = {"gaps": gaps()}
    log("gaps: " + json.dumps({k: {kk: vv["n"] for kk, vv in v.items()} for k, v in out["gaps"].items()}))
    slot, wd, day = rhythm()
    out["rhythm"] = {"by_10min_kst": slot.round(3).to_dict(orient="index"), "by_weekday": wd.round(3).to_dict(orient="index"),
                     "by_day": {str(k): v for k, v in day.to_dict(orient="index").items()}}
    log("low-activity 10-min slots (<35% welding): "
        + ", ".join(f"{s // 60:02d}:{s % 60:02d}" for s in slot.index[slot["active"] < 0.35]))
    c1, per, near, steps = cap_cycle()
    out["cap_cycle"] = {"steps": steps, "short_pauses_vs_c1_change": near,
                        "per_gun_summary": per.describe().round(2).to_dict(), "per_gun": per.round(2).to_dict(orient="records")}
    log(f"cap cycle: {steps} {near}\n{per.describe().round(2).T.to_string()}")
    tk = takt()
    out["takt"] = tk.to_dict(orient="records")
    ok = tk[tk["period_match"] >= 0.6]
    log(f"takt: {len(ok)} guns with a repeating program sequence; period {ok['period_welds'].median()} welds, cycle "
        f"{ok['cycle_s'].median()} s (IQR {ok['cycle_s'].quantile(.25)}-{ok['cycle_s'].quantile(.75)})")
    out["error_codes"], out["off_and_standby"] = error_codes()
    log("error codes: " + json.dumps({k: {kk: v[kk] for kk in ("episodes", "no_weld_during", "c10_off_share",
                                                                "start_kst_03_07")} for k, v in out["error_codes"].items()}))
    out["gun_identity"] = gun_identity()
    log(f"guns: {out['gun_identity']['distinct_c7_c8_c9']} distinct (c7, c8, c9) in {out['gun_identity']['files']} files")
    for g in out["gun_identity"]["groups"]:
        log(f"  {g['files']} {g['classes']} overlap {g['overlap_h']} h, c11 {g['c11_first']} - {g['c11_last']}")
    x = failure_context()
    hod = x["time_kst"].dt.hour
    out["failures"] = {
        "hour_kst": hod.value_counts().sort_index().to_dict(),
        "simultaneous_groups": [g["file"].tolist() for _, g in x[x["simultaneous"]].groupby("group")],
        "by_class": {c: {"n": len(g), "kst_06": int(g["time_kst"].dt.hour.eq(6).sum()),
                         "kst_10_11": int(g["time_kst"].dt.hour.between(10, 11).sum()),
                         "welding_in_5min": int((g["welds_5min"] > 0).sum()),
                         "c10_off_at_onset": int((g["c10_at"] == "off").sum()),
                         "dress_or_capchange_in_5min": int(g["dress_or_capchange_5min"].sum()),
                         "stop_code_in_5min": int(g["codes_5min"].str.contains("E003|E011").sum())}
                     for c, g in x.groupby("class")}}
    log("failures by class: " + json.dumps(out["failures"]["by_class"]))
    log(f"simultaneous: {out['failures']['simultaneous_groups']}")
    figures(slot, c1, x)
    dst = os.path.join(HERE, "domain_eda.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False,
                  default=lambda o: None if isinstance(o, float) and o != o else str(o))
    log(f"saved {dst} + fig_*.png ({time.time() - t0:.0f}s)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "events":
        build_events()
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
