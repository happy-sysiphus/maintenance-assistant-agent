"""D-7 check (history.md 35): does symptom P1's "c4 high" condition hold before E01 failures?
Run 2026-10-12 under handoff v1.1; v1.2 later dropped every symptom rule (history.md 33), so this is now the record of
what c4 does before an E01 failure.

    python results/D-7/c4_check.py        (raw train/ CSVs, ~5 min; writes results/D-7/c4_check.json)


c4 = Festo "Force reached" bit, c18 = robot "Close gun" bit (manual A.2 p.100). Pre-registered decision (written before
running): over the 72 train files, compare the 60 min before the terminal-code onset with the first 6 h (warm-up),
welding rows only, per-minute metrics, z = (pre mean - base mean) / base minute sd.
  keep "c4 high"   if >= 2/3 of E01 files have c4_share z > +0.5 and E01 median z - others median >= +0.5
  switch "c4 low"  if >= 2/3 of E01 files have c4_share z < -0.5 and E01 median z - others median <= -0.5
  else drop c4 from P1.
Validity note found while running: before an E01 terminal code the gun is often idle (3/18 E01 files have no welding
row in that hour; build_share, which needs Close-gun seconds, exists for 5/18), so "c4_share" there partly measures
activity rather than force build-up. The outcome (drop c4) does not depend on it - build_share shows no E01 shift either.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

ML = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().with_suffix(".json")
TERMINAL = {"E01": "E012", "E02": "E016", "E03": "E028", "E04": "E029"}
COLS = ["time", "c2", "c4", "c5", "c16", "c18", "error"]


def minute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df.c16 > 0]  # welding rows only (MEMORY §3: non-welding = c16 <= 0)
    if w.empty:
        return pd.DataFrame(columns=["c4_share", "build_share", "c5_mean"])
    g = w.groupby(w.time.dt.floor("min"))
    building = ((w.c18 == 1) & (w.c4 == 0)).groupby(w.time.dt.floor("min")).sum()
    closed = (w.c18 == 1).groupby(w.time.dt.floor("min")).sum()
    out = pd.DataFrame({"c4_share": g.c4.mean(), "c5_mean": g.c5.mean()})
    out["build_share"] = (building / closed).where(closed > 0)
    return out


rows = []
for path in sorted((ML / "train").glob("E0*_*.csv")):
    cls = path.stem.split("_")[0]
    df = pd.read_csv(path, usecols=COLS, dtype={"error": str})
    df["time"] = pd.to_datetime(df.time, utc=True).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    end = df.time.iloc[-1]
    tail = df[(df.time >= end - pd.Timedelta(minutes=20)) & (df.error.astype(str) == TERMINAL[cls])]
    onset = tail.time.iloc[0] if len(tail) else end - pd.Timedelta(minutes=10)
    base = minute_metrics(df[df.time < df.time.iloc[0] + pd.Timedelta(hours=6)])
    pre = minute_metrics(df[(df.time >= onset - pd.Timedelta(minutes=60)) & (df.time < onset)])
    fault = minute_metrics(df[df.time >= onset])
    row = {"file": path.stem, "class": cls, "onset_found": bool(len(tail)), "pre_minutes": int(len(pre)),
           "fault_minutes": int(len(fault))}
    for m in ("c4_share", "build_share", "c5_mean"):
        sd = base[m].std()
        row[f"{m}_z_pre"] = float((pre[m].mean() - base[m].mean()) / sd) if sd and len(pre) else None
        row[f"{m}_z_fault"] = float((fault[m].mean() - base[m].mean()) / sd) if sd and len(fault) else None
    rows.append(row)
    print(path.stem, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k.endswith("pre")},
          flush=True)

res = pd.DataFrame(rows)
summary = {}
for m in ("c4_share", "build_share", "c5_mean"):
    for phase in ("pre", "fault"):
        col = f"{m}_z_{phase}"
        e01 = res.loc[res["class"] == "E01", col].dropna()
        oth = res.loc[res["class"] != "E01", col].dropna()
        summary[col] = {
            "E01_n": int(len(e01)), "E01_median": float(e01.median()), "others_median": float(oth.median()),
            "E01_up_gt_0.5": int((e01 > 0.5).sum()), "E01_down_lt_-0.5": int((e01 < -0.5).sum()),
            "by_class_median": {c: float(res.loc[res["class"] == c, col].dropna().median()) for c in TERMINAL},
            "mannwhitney_p_E01_vs_others": float(mannwhitneyu(e01, oth).pvalue) if len(e01) and len(oth) else None,
        }
s = summary["c4_share_z_pre"]
n = s["E01_n"]
diff = s["E01_median"] - s["others_median"]
if s["E01_up_gt_0.5"] >= 2 * n / 3 and diff >= 0.5:
    decision = "keep c4 high"
elif s["E01_down_lt_-0.5"] >= 2 * n / 3 and diff <= -0.5:
    decision = "switch to c4 low"
else:
    decision = "drop c4 from P1"
OUT.write_text(json.dumps({"decision": decision, "summary": summary, "files": rows}, indent=2), encoding="utf-8")
print("DECISION:", decision)
print(json.dumps(summary, indent=1))
