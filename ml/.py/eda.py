"""
RSW welding-gun fault prediction dataset — train set EDA.

Follows the paper's own data description (Wang et al., Sci Data 2024) and the
open questions in readme.md:

  1. file summary        rows, span, collection period, missing rate (paper: >40 % -> discard),
                          outlier rate (c16 sheet thickness), welding activity, final error code
  2. gaps ("block out")  gap-length distribution per class            (paper Fig. 6)
  3. outliers            c16 distribution and outlier share per file   (paper Fig. 7)
  4. column stats        per-class distribution of c1..c19, zero share, c10 on/off
  5. error column        minor-code frequency, which codes cluster at the failure end,
                          minor-error rate vs. hours-to-failure        (readme: precursor feature)
  6. target drift        electrode force (E02) / balance pressure (E01,E03,E04) vs. hours-to-failure
  7. correlation (SCM)   sliding-window Pearson matrix, averaged        (paper "Filtering")
  8. welding pattern     welds per hour by hour-of-day, duty cycle     (paper Fig. 8)
  9. timeline            collection period of every file               (readme: E02 is 2019-20)
 10. per-file overview   c2/c5/c11 at 1-min resolution with gaps shaded (paper Fig. 5)

Usage:
    python eda.py                        # all of ./train -> ./eda_out
    python eda.py --pattern "E02_*"      # subset
    python eda.py --max-files 4 --no-file-plots   # quick run
"""
import argparse
import glob
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .py/ -> repo root
SENSOR_COLS = [f"c{i}" for i in range(1, 20)]
NUM_COLS = [c for c in SENSOR_COLS if c != "c10"]  # c10 (US2) is on/off
SENSOR_NAME = {
    "c1": "Electrode cap offset", "c2": "Electrode force", "c3": "Electrode position",
    "c4": "Force build-up", "c5": "Balance pressure", "c6": "Friction", "c7": "Maximum aperture",
    "c8": "Maximum electrode force", "c9": "Start friction", "c10": "US2",
    "c11": "Welding point count", "c12": "Position count", "c13": "Setpoint counterbalance pressure",
    "c14": "Setpoint electrode force", "c15": "Setpoint electrode position",
    "c16": "Setpoint sheet thickness", "c17": "Setpoint velocity", "c18": "Setpoint force build-up",
    "c19": "Offset value in robot",
}
CLASSES = ["E01", "E02", "E03", "E04"]
CLASS_NAME = {"E01": "Counterbalance timeout", "E02": "Electrode broke",
              "E03": "Unwanted movement", "E04": "Drift"}
# paper: target = electrode force for E02, balance pressure for the other three
TARGET_OF = {"E01": "c5", "E02": "c2", "E03": "c5", "E04": "c5"}
MAX_MISSING_RATE = 0.40  # paper
OUTLIER_HI, OUTLIER_LO = 5, 0  # Benchmark.py rule (paper text says "> 6")
# hours-to-failure buckets for the error-code analysis
TTF_BUCKETS = [(0, 1 / 6, "last 10 min"), (1 / 6, 1, "10 min-1 h"), (1, 6, "1-6 h"),
               (6, 24, "6-24 h"), (24, np.inf, "> 24 h")]

# fixed categorical order (never cycled); chrome colours
COLOR = {"E01": "#2a78d6", "E02": "#eb6834", "E03": "#1baf7a", "E04": "#eda100"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#0d366b"])
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of all-NaN / constant columns

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "lines.linewidth": 1.5,
    "legend.frameon": False,
})


# ------------------------------------------------------------------- loading
def load_file(path):
    """CSV -> (1 Hz grid DataFrame, present-row mask, n_raw_rows)."""
    df = pd.read_csv(path, low_memory=False, dtype={"c10": "string", "error": "string"})
    n_raw = len(df)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").astype("float32")
    raw_index = df.index
    df = df.resample("s").asfreq()
    present = df.index.isin(raw_index)  # whole rows drop out ("block out")
    return df, present, n_raw


def run_lengths(mask):
    """Lengths of consecutive True runs in a boolean array."""
    m = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges[1::2] - edges[0::2]


def error_episodes(err, present, n_grid):
    """Runs of one non-'0' code -> DataFrame(code, start_pos, end_pos, duration_s, ttf_start_h, ttf_end_h).
    The error column is a *state* that can persist for hours, so episodes, not samples, are the events."""
    # gaps inside a code run must not split the episode -> carry the code across gaps
    codes = err.where(present).ffill().fillna("<gap>").astype(str).to_numpy()
    nz = (codes != "0") & (codes != "<gap>")
    change = np.concatenate([[True], codes[1:] != codes[:-1]])
    run_id = np.cumsum(change)
    pos = np.flatnonzero(nz)
    if not len(pos):
        return pd.DataFrame(columns=["code", "start_pos", "end_pos", "duration_s", "ttf_start_h", "ttf_end_h"])
    g = pd.DataFrame({"run": run_id[pos], "code": codes[pos], "pos": pos}).groupby("run")
    ep = pd.DataFrame({"code": g["code"].first(), "start_pos": g["pos"].min(), "end_pos": g["pos"].max()})
    ep["duration_s"] = ep["end_pos"] - ep["start_pos"] + 1
    ep["ttf_start_h"] = (n_grid - 1 - ep["start_pos"]) / 3600
    ep["ttf_end_h"] = (n_grid - 1 - ep["end_pos"]) / 3600
    return ep.reset_index(drop=True)


def clean(df):
    """Paper pre-processing: outliers -> blank, then forward fill. Returns numeric frame."""
    x = df[SENSOR_COLS].copy()
    x["c10"] = (x["c10"] == "on").astype("float32").where(x["c10"].notna())
    out = (x["c16"] > OUTLIER_HI) | (x["c16"] <= OUTLIER_LO)
    x.loc[out, :] = np.nan
    return x.ffill()


# ------------------------------------------------------------- per-file pass
def analyse_file(path, args):
    name = os.path.splitext(os.path.basename(path))[0]
    cls, gun = name.split("_")
    df, present, n_raw = load_file(path)
    n_grid = len(df)
    t_end = df.index[-1]
    hours_to_failure = (t_end - df.index).total_seconds().values / 3600.0

    # --- gaps
    gap_len = run_lengths(~present)
    # --- outliers (c16)
    c16 = df["c16"]
    outlier = ((c16 > OUTLIER_HI) | (c16 <= OUTLIER_LO)) & present
    # --- error column (state codes -> episodes)
    err = df["error"]
    ep = error_episodes(err, present, n_grid)
    ep.insert(0, "file", name)
    ep.insert(1, "class", cls)
    in_state = np.zeros(n_grid, dtype=bool)
    for a, b in zip(ep["start_pos"], ep["end_pos"]):
        in_state[a:b + 1] = True
    last600 = ep[ep["ttf_end_h"] <= 600 / 3600]
    last_ep = ep.iloc[-1] if len(ep) else None
    # --- activity
    c11, c12 = df["c11"].dropna(), df["c12"].dropna()
    welds_total = float(c11.iloc[-1] - c11.iloc[0]) if len(c11) else np.nan
    span_days = (t_end - df.index[0]).total_seconds() / 86400

    summary = {
        "file": name, "class": cls, "gun": int(gun), "n_raw_rows": n_raw, "n_grid_1hz": n_grid,
        "start": df.index[0], "end": t_end, "span_days": round(span_days, 2),
        "missing_rate": 1 - present.mean(), "n_gaps": len(gap_len),
        "longest_gap_s": int(gap_len.max()) if len(gap_len) else 0,
        "gap_over_1h": int((gap_len > 3600).sum()),
        "outlier_rate": outlier.sum() / present.sum(),
        "c10_on_share": (df["c10"] == "on").sum() / present.sum(),
        "welds_total": welds_total, "welds_per_hour": welds_total / (span_days * 24),
        "position_count_delta": float(c12.iloc[-1] - c12.iloc[0]) if len(c12) else np.nan,
        "c2_zero_share": float((df["c2"] == 0).sum() / present.sum()),
        "c5_zero_share": float((df["c5"] == 0).sum() / present.sum()),
        "n_error_episodes": int(len(ep)), "error_state_share": float(in_state[present].mean()),
        "n_error_codes": int(ep["code"].nunique()),
        "top_code_by_time": ep.groupby("code")["duration_s"].sum().idxmax() if len(ep) else "",
        "last_code": last_ep["code"] if last_ep is not None else "",
        "last_code_end_s_before_failure": int(n_grid - 1 - last_ep["end_pos"]) if last_ep is not None else np.nan,
        "last_code_duration_s": int(last_ep["duration_s"]) if last_ep is not None else np.nan,
        "last10min_codes": ",".join(sorted(last600["code"].unique())),
    }
    summary["discard_by_paper"] = summary["missing_rate"] > MAX_MISSING_RATE

    # --- error codes by hours-to-failure bucket: episode starts + share of time in state
    code_bucket = []
    err_codes = err.where(present).fillna("<gap>").astype(str).to_numpy()
    for lo, hi, label in TTF_BUCKETS:
        sel = (hours_to_failure >= lo) & (hours_to_failure < hi)
        hours = present[sel].sum() / 3600
        started = ep[(ep["ttf_start_h"] >= lo) & (ep["ttf_start_h"] < hi)]["code"].value_counts()
        in_bucket = pd.Series(err_codes[sel & present & in_state]).value_counts()
        for code in set(started.index) | set(in_bucket.index):
            code_bucket.append({"file": name, "class": cls, "bucket": label, "code": code,
                                "episodes_started": int(started.get(code, 0)),
                                "seconds_in_state": int(in_bucket.get(code, 0)), "hours": hours})
        if not (len(started) or len(in_bucket)):
            code_bucket.append({"file": name, "class": cls, "bucket": label, "code": "",
                                "episodes_started": 0, "seconds_in_state": 0, "hours": hours})

    # --- hourly profile vs. hours-to-failure (minor errors, target sensor)
    hbin = np.floor(hours_to_failure).astype(int)
    is_err = in_state & present
    tcol = TARGET_OF[cls]
    tv = df[tcol].where(present)
    if tcol == "c2":
        tv = tv.where(tv > 0)  # electrode force is 0 between welds; keep the welding samples
    hourly = pd.DataFrame({"hbin": hbin, "err": is_err, "present": present, "target": tv.values})
    hourly = hourly.groupby("hbin").agg(err_s=("err", "sum"), present_h=("present", lambda s: s.sum() / 3600),
                                        target_mean=("target", "mean"), target_std=("target", "std"))
    hourly["error_state_share"] = hourly["err_s"] / (hourly["present_h"] * 3600).replace(0, np.nan)
    hourly["file"], hourly["class"] = name, cls

    # --- welding pattern: welds per hour-of-day
    c11_h = df["c11"].resample("h").agg(["first", "last"])
    welds_h = (c11_h["last"] - c11_h["first"]).clip(lower=0)
    pattern = pd.DataFrame({"hour": welds_h.index.hour, "welds": welds_h.values, "file": name, "class": cls})

    # --- correlation: sliding-window Pearson (paper SCM, width = step = window)
    x = clean(df)
    mats = []
    for s in range(0, len(x) - args.window + 1, args.window):
        w = x.iloc[s:s + args.window].dropna()
        if len(w) < args.window * 0.5:
            continue
        v = w.values
        sd = v.std(axis=0)
        ok = sd > 0
        if ok.sum() < 2:
            continue
        m = np.full((len(SENSOR_COLS), len(SENSOR_COLS)), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.corrcoef(v[:, ok], rowvar=False)
        m[np.ix_(ok, ok)] = sub
        mats.append(m)
    scm = np.nanmean(np.stack(mats), axis=0) if mats else np.full((19, 19), np.nan)

    # --- columns constant within this file (candidate static covariates: gun id-like attributes)
    nunique = df.loc[present, NUM_COLS].nunique()
    summary.update({f"nunique_{c}": int(nunique[c]) for c in NUM_COLS})

    # --- pooled sample of raw present rows for distributions
    sample = df.loc[present].iloc[::args.stride][SENSOR_COLS].copy()
    sample["class"], sample["file"] = cls, name

    if args.file_plots:
        plot_file_overview(name, cls, df, present, outlier, args.out_dir)

    return summary, code_bucket, hourly, pattern, scm, sample, gap_len, c16[present].values, ep


# -------------------------------------------------------------------- plots
def shade_runs(ax, index, mask, color, alpha):
    """Shade contiguous True runs of `mask` (aligned to `index`) as background bands."""
    m = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    for a, b in zip(edges[0::2], edges[1::2]):
        ax.axvspan(index[a], index[min(b, len(index) - 1)], color=color, alpha=alpha, lw=0)


def plot_file_overview(name, cls, df, present, outlier, out_dir):
    d = os.path.join(out_dir, "files")
    os.makedirs(d, exist_ok=True)
    cols = ["c2", "c5", "c3", "c11"]
    m = df[cols].resample("min").mean()
    gap_min = (~pd.Series(present, index=df.index)).resample("min").mean().values > 0.5
    out_min = pd.Series(outlier, index=df.index).resample("min").mean().values > 0
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 8), sharex=True)
    for ax, c in zip(axes, cols):
        ax.plot(m.index, m[c], color=COLOR[cls], lw=0.8)
        shade_runs(ax, m.index, gap_min, "#898781", 0.25)
        shade_runs(ax, m.index, out_min, "#d03b3b", 0.15)
        ax.set_ylabel(f"{c}\n{SENSOR_NAME[c]}", fontsize=8)
    axes[0].set_title(f"{name} — {CLASS_NAME[cls]} (1-min mean; grey = gap, red = c16 outlier)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{name}.png"), dpi=120)
    plt.close(fig)


def save(fig, out_dir, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def plot_missing_and_outliers(summary, out_dir):
    s = summary.sort_values(["class", "gun"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, col, title in [(axes[0], "missing_rate", "Missing rate (paper discards > 40 %)"),
                           (axes[1], "outlier_rate", f"c16 outlier share (c16 > {OUTLIER_HI} or <= {OUTLIER_LO})")]:
        ax.bar(s["file"], s[col], color=[COLOR[c] for c in s["class"]], width=0.7)
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].axhline(MAX_MISSING_RATE, color="#d03b3b", lw=1, ls="--")
    axes[0].text(0, MAX_MISSING_RATE, " 40 % threshold", color="#d03b3b", va="bottom", fontsize=8)
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in CLASSES]
    fig.legend(handles, [f"{c} {CLASS_NAME[c]}" for c in CLASSES], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "01_missing_outlier_by_file.png"), dpi=150)
    plt.close(fig)


def plot_gap_distribution(gaps, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(0, 6, 40)
    for c in CLASSES:
        g = gaps[gaps["class"] == c]["gap_s"].values
        if len(g):
            ax.hist(g, bins=bins, histtype="step", color=COLOR[c], lw=1.5, label=f"{c} (n={len(g)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap length (s)")
    ax.set_ylabel("number of gaps")
    ax.set_title("Gap (block-out) length distribution per class", loc="left")
    ax.legend()
    save(fig, out_dir, "02_gap_length_distribution.png")


def plot_c16(c16_values, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    v = c16_values[np.isfinite(c16_values)]
    ax.hist(v, bins=np.arange(-0.25, max(10, v.max()) + 0.5, 0.25), color=COLOR["E01"])
    ax.axvline(OUTLIER_HI, color="#d03b3b", ls="--", lw=1)
    ax.axvline(OUTLIER_LO, color="#d03b3b", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("c16 setpoint sheet thickness")
    ax.set_ylabel("samples (log)")
    ax.set_title("c16 distribution — outside dashed lines = non-welding operation (outlier)", loc="left")
    save(fig, out_dir, "03_c16_distribution.png")


def plot_timeline(summary, out_dir):
    s = summary.sort_values("start").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, r in s.iterrows():
        ax.barh(i, (r["end"] - r["start"]).total_seconds() / 86400, left=r["start"], color=COLOR[r["class"]], height=0.7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["file"], fontsize=6)
    ax.set_title("Collection period of every training file (bar = start..failure)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in CLASSES]
    ax.legend(handles, CLASSES, loc="lower right")
    save(fig, out_dir, "04_collection_timeline.png")


def plot_code_heatmap(share, starts, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.4 * len(share))))
    for ax, tab, title, fmt in [(axes[0], share, "share of present time in this code", "{:.1%}"),
                                (axes[1], starts, "episode starts per present hour", "{:.2g}")]:
        im = ax.imshow(tab.values, cmap=SEQ_CMAP, aspect="auto")
        ax.set_xticks(range(tab.shape[1]))
        ax.set_xticklabels(tab.columns, rotation=30, ha="right")
        ax.set_yticks(range(tab.shape[0]))
        ax.set_yticklabels(tab.index)
        ax.grid(False)
        vmax = np.nanmax(tab.values) if np.isfinite(tab.values).any() else 1
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.values[i, j]
                if np.isfinite(v) and v > 0:
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if v > vmax * 0.6 else INK)
        ax.set_title(title, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("error-column codes by hours-to-failure bucket (all classes pooled)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "05_error_code_by_ttf.png"), dpi=150)
    plt.close(fig)


def plot_ttf_profiles(hourly, out_dir):
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    top = fig.add_subplot(gs[0, :])
    for c in CLASSES:
        h = hourly[hourly["class"] == c]
        if h.empty:
            continue
        e = h.groupby("hbin")["error_state_share"].mean()
        top.plot(e.index, e.values, color=COLOR[c], label=f"{c} (files={h['file'].nunique()})")
    top.set_title("Share of time the error column is non-zero, per hour before failure (mean over files)", loc="left")
    top.set_ylabel("share of samples")
    top.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    top.invert_xaxis()
    top.legend(loc="upper left")
    for k, c in enumerate(CLASSES):
        ax = fig.add_subplot(gs[1 + k // 2, k % 2], sharex=top)
        h = hourly[hourly["class"] == c]
        tcol = TARGET_OF[c]
        if not h.empty:
            for f, hf in h.groupby("file"):  # individual files, faint
                ax.plot(hf["hbin"], hf["target_mean"], color=COLOR[c], lw=0.5, alpha=0.25)
            g = h.groupby("hbin")["target_mean"]
            m, sd = g.mean(), g.std()
            ax.fill_between(m.index, m - sd, m + sd, color=COLOR[c], alpha=0.15, lw=0)
            ax.plot(m.index, m.values, color=COLOR[c], lw=2, label="mean +/- 1 sd over files")
            ax.legend(loc="lower left")
        note = " (c2 > 0 only)" if tcol == "c2" else ""
        ax.set_title(f"{c} {CLASS_NAME[c]} - hourly mean of {tcol} {SENSOR_NAME[tcol]}{note}", loc="left", fontsize=9)
        if k >= 2:
            ax.set_xlabel("hours to failure")
    save(fig, out_dir, "06_ttf_profiles.png")


def plot_scm(scm_df, out_dir, fname, title):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(scm_df.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(19))
    ax.set_xticklabels(scm_df.columns, rotation=90)
    ax.set_yticks(range(19))
    ax.set_yticklabels([f"{c} {SENSOR_NAME[c]}" for c in scm_df.index], fontsize=7)
    ax.grid(False)
    for i in range(19):
        for j in range(19):
            v = scm_df.values[i, j]
            if np.isfinite(v) and abs(v) >= 0.3 and i != j:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, label="mean window Pearson r")
    ax.set_title(title, loc="left")
    save(fig, out_dir, fname)


def plot_pattern(pattern, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in CLASSES:
        p = pattern[pattern["class"] == c].groupby("hour")["welds"].mean()
        if len(p):
            ax.plot(p.index, p.values, color=COLOR[c], marker="o", ms=3, label=c)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("welds / hour (Δ c11)")
    ax.set_title("Welding activity by hour of day (mean over files)", loc="left")
    ax.legend()
    save(fig, out_dir, "08_welding_pattern_hour_of_day.png")


def plot_class_distributions(sample, out_dir):
    cols = [c for c in NUM_COLS]
    fig, axes = plt.subplots(6, 3, figsize=(14, 16))
    for ax, c in zip(axes.ravel(), cols):
        data = [sample.loc[sample["class"] == k, c].dropna().values for k in CLASSES]
        parts = ax.boxplot(data, tick_labels=CLASSES, showfliers=False, patch_artist=True, widths=0.6)
        for patch, k in zip(parts["boxes"], CLASSES):
            patch.set(facecolor=COLOR[k], alpha=0.6, edgecolor=INK2)
        for key in ("medians",):
            plt.setp(parts[key], color=INK)
        ax.set_title(f"{c} {SENSOR_NAME[c]}", loc="left", fontsize=8)
    fig.suptitle("Sensor distributions per class (raw present rows, subsampled; whiskers = 1.5 IQR, no fliers)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(out_dir, "09_sensor_distributions_by_class.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- report
def md_table(df, floatfmt="{:.4g}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    # defaults are anchored to the project root (this file's parent directory), not the
    # working directory, so running from an IDE or from inside .py/ finds the data too
    ap.add_argument("--train-dir", default=os.path.join(PROJECT_ROOT, "train"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "eda_out"))
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--window", type=int, default=3600, help="SCM window length l in seconds")
    ap.add_argument("--stride", type=int, default=20, help="subsampling stride for pooled distributions")
    ap.add_argument("--no-file-plots", dest="file_plots", action="store_false")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.train_dir, args.pattern)))[: args.max_files]
    if not paths:
        raise SystemExit(f"no files matching {args.pattern!r} in {os.path.abspath(args.train_dir)} "
                         f"(cwd: {os.getcwd()}) — pass --train-dir")
    summaries, code_rows, hourly_l, pattern_l, scm_d, samples, gap_rows, c16_all, episodes = [], [], [], [], {}, [], [], [], []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", flush=True)
        s, cb, h, pt, scm, sm, gl, c16, ep = analyse_file(p, args)
        episodes.append(ep)
        summaries.append(s)
        code_rows += cb
        hourly_l.append(h)
        pattern_l.append(pt)
        scm_d[s["file"]] = scm
        samples.append(sm)
        gap_rows.append(pd.DataFrame({"file": s["file"], "class": s["class"], "gap_s": gl}))
        c16_all.append(c16)
    print(f"loaded {len(paths)} files in {time.time() - t0:.0f}s")

    # ---------- tables
    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(args.out_dir, "file_summary.csv"), index=False)

    class_summary = summary.groupby("class").agg(
        files=("file", "count"), span_days=("span_days", "mean"), missing_rate=("missing_rate", "mean"),
        missing_max=("missing_rate", "max"), discard_by_paper=("discard_by_paper", "sum"),
        outlier_rate=("outlier_rate", "mean"), welds_per_hour=("welds_per_hour", "mean"),
        c2_zero_share=("c2_zero_share", "mean"), c10_on_share=("c10_on_share", "mean"),
        error_episodes=("n_error_episodes", "mean"), error_state_share=("error_state_share", "mean"), start_min=("start", "min"), end_max=("end", "max"),
    ).reset_index()
    class_summary.to_csv(os.path.join(args.out_dir, "class_summary.csv"), index=False)

    gaps = pd.concat(gap_rows, ignore_index=True)
    gap_stats = gaps.groupby("class")["gap_s"].describe(percentiles=[0.5, 0.9, 0.99]).reset_index()
    gap_stats.to_csv(os.path.join(args.out_dir, "gap_stats.csv"), index=False)

    sample = pd.concat(samples, ignore_index=True)
    col_stats = []
    for c in NUM_COLS:
        for k, g in sample.groupby("class"):
            v = g[c].dropna()
            col_stats.append({"column": c, "name": SENSOR_NAME[c], "class": k, "n": len(v),
                              "zero_share": float((v == 0).mean()), "nunique": v.nunique(),
                              "mean": v.mean(), "std": v.std(), "min": v.min(), "p01": v.quantile(0.01),
                              "p50": v.median(), "p99": v.quantile(0.99), "max": v.max()})
    col_stats = pd.DataFrame(col_stats)
    col_stats.to_csv(os.path.join(args.out_dir, "column_stats_by_class.csv"), index=False)

    episodes = pd.concat(episodes, ignore_index=True)
    episodes.to_csv(os.path.join(args.out_dir, "error_episodes.csv"), index=False)
    ep_stats = episodes.groupby(["class", "code"]).agg(
        files=("file", "nunique"), episodes=("code", "size"), total_hours=("duration_s", lambda x: x.sum() / 3600),
        median_duration_s=("duration_s", "median"), max_duration_s=("duration_s", "max"),
        median_ttf_end_h=("ttf_end_h", "median"), min_ttf_end_h=("ttf_end_h", "min")).reset_index()
    ep_stats.to_csv(os.path.join(args.out_dir, "error_code_stats_by_class.csv"), index=False)

    codes = pd.DataFrame(code_rows)
    hours_per_bucket = codes.drop_duplicates(["file", "bucket"]).groupby("bucket")["hours"].sum()
    codes = codes[codes["code"] != ""]
    order = [b[2] for b in TTF_BUCKETS]
    share = codes.groupby(["code", "bucket"])["seconds_in_state"].sum().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    share = share.div(hours_per_bucket.reindex(order) * 3600, axis=1)
    starts = codes.groupby(["code", "bucket"])["episodes_started"].sum().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    starts = starts.div(hours_per_bucket.reindex(order), axis=1)
    code_order = share.sum(axis=1).sort_values(ascending=False).index
    share, starts = share.loc[code_order], starts.loc[code_order]
    share.to_csv(os.path.join(args.out_dir, "error_code_time_share_by_ttf.csv"))
    starts.to_csv(os.path.join(args.out_dir, "error_code_episode_rate_by_ttf.csv"))
    # readme hypothesis: a specific code marks the failure moment -> codes present in the last 10 min, per class
    last10 = (summary.assign(code=summary["last10min_codes"].str.split(",")).explode("code"))
    last10 = last10[last10["code"].astype(str) != ""].groupby(["class", "code"])["file"].nunique().unstack(fill_value=0)
    last10.to_csv(os.path.join(args.out_dir, "codes_in_last10min_by_class.csv"))
    last_code = summary.groupby(["class", "last_code"]).size().unstack(fill_value=0)
    last_code.to_csv(os.path.join(args.out_dir, "last_code_by_class.csv"))

    # columns constant within a file (gun-level attributes -> static covariates, not time series)
    nun = summary[[f"nunique_{c}" for c in NUM_COLS]]
    static = pd.DataFrame({"column": NUM_COLS, "name": [SENSOR_NAME[c] for c in NUM_COLS],
                           "files_constant": (nun.values == 1).sum(axis=0), "files": len(summary),
                           "median_nunique": np.median(nun.values, axis=0)})
    static.to_csv(os.path.join(args.out_dir, "column_constancy.csv"), index=False)

    hourly = pd.concat(hourly_l).reset_index()
    hourly.to_csv(os.path.join(args.out_dir, "hourly_ttf_profile_by_file.csv"), index=False)
    pattern = pd.concat(pattern_l, ignore_index=True)

    scm_all = pd.DataFrame(np.nanmean(np.stack(list(scm_d.values())), axis=0), index=SENSOR_COLS, columns=SENSOR_COLS)
    scm_all.to_csv(os.path.join(args.out_dir, "scm_overall.csv"))
    scm_by_class = {}
    for k in CLASSES:
        mats = [scm_d[f] for f in summary.loc[summary["class"] == k, "file"]]
        if mats:
            scm_by_class[k] = pd.DataFrame(np.nanmean(np.stack(mats), axis=0), index=SENSOR_COLS, columns=SENSOR_COLS)
            scm_by_class[k].to_csv(os.path.join(args.out_dir, f"scm_{k}.csv"))
    target_corr = pd.DataFrame({t: scm_all[t].drop(t).sort_values(key=np.abs, ascending=False) for t in ["c2", "c5"]})
    target_corr.to_csv(os.path.join(args.out_dir, "target_correlations.csv"))

    # ---------- plots
    plot_missing_and_outliers(summary, args.out_dir)
    plot_gap_distribution(gaps, args.out_dir)
    plot_c16(np.concatenate(c16_all), args.out_dir)
    plot_timeline(summary, args.out_dir)
    plot_code_heatmap(share.head(20), starts.head(20), args.out_dir)
    plot_ttf_profiles(hourly, args.out_dir)
    plot_scm(scm_all, args.out_dir, "07_scm_overall.png", f"Series correlation matrix (window {args.window}s, mean over windows & files)")
    for k, m in scm_by_class.items():
        plot_scm(m, args.out_dir, f"07_scm_{k}.png", f"SCM — {k} {CLASS_NAME[k]}")
    plot_pattern(pattern, args.out_dir)
    plot_class_distributions(sample, args.out_dir)

    # ---------- report
    disc = summary.loc[summary["discard_by_paper"], ["file", "missing_rate"]]
    top_c2 = scm_all["c2"].drop("c2").abs().sort_values(ascending=False).head(5)
    top_c5 = scm_all["c5"].drop("c5").abs().sort_values(ascending=False).head(5)
    rep = [
        "# RSW train set — EDA report", "",
        f"files: {len(summary)}  |  grid rows (1 Hz): {summary['n_grid_1hz'].sum():,}  |  raw rows: {summary['n_raw_rows'].sum():,}", "",
        "## 1. Per-class summary", md_table(class_summary), "",
        "## 2. Missing data (paper: discard > 40 %)",
        f"files over threshold: {len(disc)} -> " + (", ".join(f"{r.file} ({r.missing_rate:.0%})" for r in disc.itertuples()) or "none"), "",
        md_table(gap_stats), "",
        "## 3. Outliers (c16)",
        md_table(summary.groupby("class")["outlier_rate"].describe()[["mean", "min", "max"]].reset_index()), "",
        "## 4. Error column (state codes; an episode = one uninterrupted run of a code)",
        md_table(ep_stats), "",
        "### codes present in the last 10 min before failure (number of files)", md_table(last10.reset_index()), "",
        "### last non-zero code per file", md_table(last_code.reset_index()), "",
        "### share of present time in each code, by hours-to-failure", md_table(share.head(15).reset_index()), "",
        "### episode starts per present hour, by hours-to-failure", md_table(starts.head(15).reset_index()), "",
        "## 4b. Columns constant within a file (candidate static covariates)",
        md_table(static[static["files_constant"] > 0]), "",
        "## 5. Correlation (SCM) — strongest |r| with the paper targets",
        "c2 electrode force: " + ", ".join(f"{i} ({scm_all.loc[i, 'c2']:+.2f})" for i in top_c2.index),
        "c5 balance pressure: " + ", ".join(f"{i} ({scm_all.loc[i, 'c5']:+.2f})" for i in top_c5.index), "",
        "## 6. Files", md_table(summary[["file", "class", "start", "end", "span_days", "missing_rate", "outlier_rate",
                                         "welds_per_hour", "n_error_episodes", "error_state_share", "last_code",
                                         "last_code_end_s_before_failure", "last10min_codes", "discard_by_paper"]]), "",
        "## Outputs", "tables: file_summary, class_summary, gap_stats, column_stats_by_class, column_constancy, error_episodes,",
        "error_code_stats_by_class, error_code_time_share_by_ttf, error_code_episode_rate_by_ttf, codes_in_last10min_by_class,",
        "last_code_by_class, hourly_ttf_profile_by_file, scm_overall / scm_E0x, target_correlations",
        "plots: 01-09 *.png, files/<file>.png",
    ]
    with open(os.path.join(args.out_dir, "eda_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))
    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
