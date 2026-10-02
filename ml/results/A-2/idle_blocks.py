"""A-2 Q1 follow-up (history.md §21): is a long non-welding block itself a precursor?

For every file: per-minute non-welding share, blocks = runs of minutes with share >= 0.9 lasting >= L minutes.
Precursor hit = a block that STARTS in (10 min, 60 min] before the failure (before the terminal code - the rule
covers the last ~10 min). Base rate = blocks starting > 24 h before the failure, per day of that period.

    python results/A-2/idle_blocks.py    # -> results/A-2/idle_blocks.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(ROOT, "preprocessed")


def blocks(nw_min, min_len):
    """(start index, length) of runs of minutes with non-welding share >= 0.9."""
    on = np.r_[False, nw_min.to_numpy() >= 0.9, False]
    e = np.flatnonzero(on[1:] != on[:-1])
    return [(a, b - a) for a, b in zip(e[0::2], e[1::2]) if b - a >= min_len]


out = {}
paths = sorted(glob.glob(os.path.join(PRE, "E0*.parquet"))) + sorted(glob.glob(os.path.join(PRE, "test", "test_*.parquet")))
frames = []
for p in paths:
    d = pd.read_parquet(p, columns=["non_welding", "ttf_s"])
    g = d.groupby(d.index.floor("60s")).agg(nw=("non_welding", "mean"), ttf=("ttf_s", "min"))
    frames.append((os.path.splitext(os.path.basename(p))[0], g))
for L in (10, 20, 30, 45):
    hits, per_day = 0, []
    for name, g in frames:
        bl = blocks(g["nw"], L)
        starts = g["ttf"].to_numpy()[[a for a, _ in bl]] if bl else np.array([])
        hits += int(((starts > 600) & (starts <= 3600)).any())
        days = max((g["ttf"].max() - 86400) / 86400, 1e-9)
        per_day.append(float((starts > 86400).sum() / days))
    pd_ = np.array(per_day)
    out[f"min_{L}"] = {"files": len(frames), "precursor_files": hits, "precursor_share": round(hits / len(frames), 3),
                       "normal_blocks_per_day_mean": round(float(pd_.mean()), 2),
                       "normal_blocks_per_day_median": round(float(np.median(pd_)), 2),
                       "files_over_1_per_day": int((pd_ > 1).sum())}
    # chance of a block starting in any given 50-min stretch at the normal rate (Poisson)
    out[f"min_{L}"]["chance_in_50min_at_normal_rate"] = round(float(1 - np.exp(-pd_.mean() * 50 / 1440)), 3)
    print(L, out[f"min_{L}"], flush=True)
json.dump(out, open(os.path.join(HERE, "idle_blocks.json"), "w"), indent=1)
