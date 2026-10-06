"""
C-6 (2026-10-06): the terminal-code rule at ROW level, exactly as main.Detector.rule_check fires it (shared cooldown vs
per-code cooldown, 24 h repeat check, optional profile = the file's class), over the 72 preprocessed guns (64 train +
8 test, test class = the class inferred by preprocess.py). The window-level evaluation in train.evaluate is an
approximation of this (one value per minute); this script is the serving behaviour.

    python results/C-6/rule_rowlevel.py     -> results/C-6/rule_rowlevel.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, ".py"))
import train as T  # noqa: E402

CODES = {c: k for k, c in enumerate(T.TERMINAL_CODES, 1)}  # E012 -> 1 ...


def triggers(code, t, per_code):
    """code: 1 Hz terminal code id (0 = none), t: seconds; -> list of (row, code, repeat) like rule_check."""
    prev = np.r_[0, code[:-1]]
    if per_code:
        onset = (code > 0) & (code != prev)
    else:
        onset = (code > 0) & (prev == 0)
    out, last_any, last_by = [], None, {}
    for i in np.flatnonzero(onset):
        c = code[i]
        last = last_by.get(c) if per_code else last_any
        if last is None or t[i] - last >= T.RULE_COOLDOWN_S:
            repeat = c in last_by and t[i] - last_by[c] < T.RULE_REPEAT_S
            out.append((i, int(c), bool(repeat)))
            last_any, last_by[c] = t[i], t[i]
    return out


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "preprocessed", "E0*.parquet"))) + \
        sorted(glob.glob(os.path.join(ROOT, "preprocessed", "test", "test_*.parquet")))
    res = {}
    for mode in ("shared", "per_code"):
        for prof in (False, True):
            hits = false = days = 0
            guns_hit, leads, out_prof = 0, [], 0
            for f in files:
                d = pd.read_parquet(f, columns=["class", "error_code", "ttf_s"])
                code = d["error_code"].map(CODES).fillna(0).astype(int).to_numpy()
                t = (d.index - d.index[0]).total_seconds().to_numpy()
                ttf = d["ttf_s"].to_numpy()
                own = T.CLASSES.index(str(d["class"].iloc[0])) + 1
                ev = triggers(code, t, mode == "per_code")
                crit = [(i, c) for i, c, r in ev if not r and (not prof or c == own)]
                out_prof += sum(1 for i, c, r in ev if not r and prof and c != own)
                pre = [i for i, c in crit if ttf[i] <= T.RULE_EVAL_WINDOW_S]
                hits += len(pre)
                false += len(crit) - len(pre)
                days += ttf.max() / 86400
                if pre:
                    guns_hit += 1
                    leads.append(ttf[pre[0]] / 60)
            key = f"{mode}{'+profile' if prof else ''}"
            res[key] = {"guns_hit": guns_hit, "guns": len(files), "lead_min_median": float(np.median(leads)),
                        "critical_hits": hits, "critical_false": false, "false_per_gun_day": false / days,
                        "out_of_profile": out_prof}
            print(f"{key:18s} guns hit {guns_hit}/{len(files)}, lead median {np.median(leads):.1f} min, critical false "
                  f"{false} ({false / days:.3f}/gun/day), out of profile {out_prof}", flush=True)
    with open(os.path.join(HERE, "rule_rowlevel.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
