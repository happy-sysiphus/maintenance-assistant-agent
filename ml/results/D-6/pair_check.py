"""
D-6: are the actual / setpoint pairs (by name) really the same quantity? Check them in the raw CSVs.

    python results/D-6/pair_check.py                       # test/test_*.csv (8 files, ~1-3 min)
    python results/D-6/pair_check.py --files train/E01_1.csv test/test_3.csv

Pairs (README table, names only so far): c2<->c14 force, c3<->c15 position, c5<->c13 balance pressure,
c4<->c18 force build-up.

Rows are split by state with the weld counter c11:
    weld = for every weld (c11 goes up at second t) the one row in t-1 .. t+2 where the electrode force c2 is highest -
           the moment the gun presses (the counter often ticks a second before the force is up)
    idle = rows more than 2 s away from any weld
For each pair: weld rows compare actual and setpoint on the same row; idle rows also with the actual taken 1 / 2 s
after the setpoint (a moving axis lags its setpoint):
    corr        Pearson correlation of actual vs setpoint
    ratio_med   median of actual / setpoint (setpoint != 0) - 1.0 means "the actual equals the setpoint"
    within_10   share of rows with |actual - setpoint| <= 10 % of |setpoint|
and, per setpoint value (the 6 most frequent in weld rows), the median actual: if the pair is the same quantity the
actual follows each setpoint value.

How to read (a rough guide, not a test): same quantity = corr near 1 AND ratio_med near 1 AND within_10 high in the
state where the setpoint is in force; related = high |corr| but ratio far from 1 (other unit / offset / opposite);
no evidence = |corr| small. Output: console + results/D-6/pair_check.md / pair_check.csv.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAIRS = [("c2", "c14", "전극 힘"), ("c3", "c15", "전극 위치"), ("c5", "c13", "보정(밸런스) 압력"),
         ("c4", "c18", "힘 형성")]
LAGS = (0, 1, 2)


def load(path: str) -> pd.DataFrame:
    cols = ["time", "c11"] + sorted({c for p in PAIRS for c in p[:2]})
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    t = df["time"].astype("int64") // 10**9
    ticks = np.flatnonzero(df["c11"].diff().fillna(0).to_numpy() > 0)
    press = np.zeros(len(df), bool)
    near = np.zeros(len(df), bool)
    c2 = df["c2"].to_numpy(dtype=float)
    for i in ticks:
        lo, hi = max(i - 1, 0), min(i + 3, len(df))
        idx = np.arange(lo, hi)
        idx = idx[np.abs(t.iloc[idx].to_numpy() - t.iloc[i]) <= 2]  # same 4-s neighbourhood, no gap
        near[idx] = True
        if np.isfinite(c2[idx]).any():
            press[idx[np.nanargmax(c2[idx])]] = True
    df["state"] = np.where(press, "weld", np.where(near, "edge", "idle"))
    df["file"] = os.path.splitext(os.path.basename(path))[0]
    return df


def stats(act: pd.Series, sp: pd.Series) -> dict:
    ok = act.notna() & sp.notna()
    a, s = act[ok].astype(float), sp[ok].astype(float)
    nz = s != 0
    corr = float(np.corrcoef(a, s)[0, 1]) if len(a) > 2 and a.std() > 0 and s.std() > 0 else float("nan")
    ratio = float((a[nz] / s[nz]).median()) if nz.any() else float("nan")
    within = float(((a[nz] - s[nz]).abs() <= 0.1 * s[nz].abs()).mean()) if nz.any() else float("nan")
    return {"n": len(a), "corr": corr, "ratio_med": ratio, "within_10": within}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help="CSV files relative to ml (default test/test_*.csv)")
    ap.add_argument("--out", default=os.path.join(ML, "results", "D-6"))
    a = ap.parse_args()
    files = [os.path.join(ML, f) for f in a.files] if a.files else sorted(glob.glob(os.path.join(ML, "test", "test_*.csv")))
    if not files:
        raise SystemExit("CSV가 없습니다: ml/test/ 에 test_*.csv 를 넣거나 --files 로 지정하세요")
    rows, levels = [], []
    for path in files:
        print(f"읽는 중 {os.path.basename(path)} ...", flush=True)
        df = load(path)
        for act, sp, name in PAIRS:
            for state in ("weld", "idle"):
                for lag in (LAGS if state == "idle" else (0,)):
                    shifted = df[act].shift(-lag)  # actual `lag` seconds after the setpoint row
                    m = df["state"] == state
                    rows.append({"file": df["file"].iloc[0], "pair": f"{act}<->{sp}", "name": name, "state": state,
                                 "lag_s": lag, **stats(shifted[m], df.loc[m, sp])})
            w = df[df["state"] == "weld"]
            for v, g in w.groupby(w[sp].round(3)):
                levels.append({"file": df["file"].iloc[0], "pair": f"{act}<->{sp}", "setpoint": v, "n": len(g),
                               "actual_med": float(g[act].median())})
    res = pd.DataFrame(rows)
    # pooled over files: median of the per-file numbers (each gun counts once)
    agg = (res.groupby(["pair", "name", "state", "lag_s"], sort=False)
           .agg(files=("file", "nunique"), n=("n", "sum"), corr=("corr", "median"), ratio_med=("ratio_med", "median"),
                within_10=("within_10", "median")).reset_index())
    lv = pd.DataFrame(levels)
    lv = (lv.groupby(["pair", "setpoint"]).agg(n=("n", "sum"), actual_med=("actual_med", "median")).reset_index()
          .sort_values(["pair", "n"], ascending=[True, False]).groupby("pair").head(6).sort_values(["pair", "setpoint"]))

    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "pair_check.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    def f(spec: str):
        return lambda v: "-" if pd.isna(v) else spec.format(v)  # "-" = one side constant / no non-zero setpoint
    fmt = {"corr": f("{:+.2f}"), "ratio_med": f("{:.2f}"), "within_10": f("{:.0%}")}
    lines = [f"# 실측-설정 짝 검증 ({len(files)}개 파일, 파일별 값의 중앙값)\n",
             "weld = 용접할 때마다(c11 증가) 전극 힘이 가장 큰 1초(누르는 순간), idle = 용접에서 2초 넘게 떨어진 줄. lag = 실측을 설정보다 몇 초 뒤에서 읽었나\n",
             "| 짝 | 이름 | 상태 | lag(초) | 파일 수 | 줄 수 | 상관 corr | 실측/설정 중앙값 | ±10% 안 비율 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        lines.append(f"| {r.pair} | {r.name} | {r.state} | {r.lag_s} | {r.files} | {r.n:,} | {fmt['corr'](r.corr)} | "
                     f"{fmt['ratio_med'](r.ratio_med)} | {fmt['within_10'](r.within_10)} |")
    lines += ["", "## 누르는 순간, 설정값별 실측 중앙값 (가장 많이 쓰인 설정값 6개)", "",
              "| 짝 | 설정값 | 용접 수 | 실측 중앙값 |", "|---|---|---|---|"]
    for r in lv.itertuples():
        lines.append(f"| {r.pair} | {r.setpoint:g} | {r.n:,} | {r.actual_med:g} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "pair_check.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"저장: {os.path.join(a.out, 'pair_check.md')} (파일별 상세: pair_check.csv)")


if __name__ == "__main__":
    main()
