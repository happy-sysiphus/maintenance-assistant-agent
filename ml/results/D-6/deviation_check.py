"""
D-6: does "actual vs setpoint" grow before a failure? (option 라 for the per-minute RAG document)

Uses only the pairs pair_check.py confirmed:
    force     c2 vs c14 at the moment the gun presses (per weld: the row of max c2 in t-1 .. t+2 of the c11 tick)
    position  c3 vs c15 while idle (rows more than 2 s from any weld)

Per minute and file:
    force_miss  share of the minute's presses with |c2 - c14| > 10 % of c14       (가압 미달·초과 비율)
    force_dev   median (c2 - c14) / c14 of the minute's presses, signed            (평균 차이, + = 더 셈)
    pos_off     share of idle rows with |c3 - c15| > 10 % of |c15|                (위치 이탈 비율)
Time to failure = file end - minute end (the dataset's failure = the file end, readme 1.5).

For every file, "normal" = its minutes more than 24 h before the end. A minute is UNUSUAL when its value is above the
file's own normal 95th percentile (force_dev: |value| above the 95th percentile of |normal|). Chance level = 5 %:
if a time band before the failure shows far more than 5 % unusual minutes, the deviation grows before failures.
The last 10 min hold the terminal code (E012 ...), so they show the failure itself, not a warning.

    python results/D-6/deviation_check.py                         # test/*.csv (+ train/*.csv if present)
    python results/D-6/deviation_check.py --files test/test_3.csv "train/E01_*.csv"
Output: console + results/D-6/deviation_check.md (table) / deviation_check.csv (per file and band).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANDS = [("0-10분", 0, 10), ("10-60분", 10, 60), ("1-6시간", 60, 360), ("6-24시간", 360, 1440)]
NORMAL_MIN = 1440  # minutes before the end that count as normal
METRICS = [("force_miss", "가압 미달·초과 비율 (c2 vs c14)"), ("force_dev", "가압 차이 중앙값 (c2-c14)/c14"),
           ("pos_off", "위치 이탈 비율 (c3 vs c15)")]
TERMINAL = {"E012": "E01", "E016": "E02", "E028": "E03", "E029": "E04"}


def per_minute(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, usecols=["time", "c2", "c3", "c11", "c14", "c15", "error"], dtype={"error": str},
                     low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    t = (df["time"].astype("int64") // 10**9).to_numpy()
    c2 = df["c2"].to_numpy(float)
    n = len(df)
    # press moment of every weld, vectorised: candidates i-1 .. i+2 of each c11 tick, same 4-s neighbourhood
    ticks = np.flatnonzero(np.diff(df["c11"].to_numpy(float), prepend=np.nan) > 0)
    cand = np.clip(ticks[:, None] + np.arange(-1, 3)[None, :], 0, n - 1)
    ok = np.abs(t[cand] - t[ticks][:, None]) <= 2
    val = np.where(ok, c2[cand], -np.inf)
    val = np.where(np.isnan(val), -np.inf, val)
    press = np.zeros(n, bool)
    press[cand[np.arange(len(ticks)), val.argmax(axis=1)]] = True
    near = np.zeros(n, bool)
    near[cand[ok]] = True
    end = df["time"].iloc[-1]
    df["minute"] = df["time"].dt.floor("min")
    # class: from the file name (train) or the terminal code of the last 10 min (test, readme 5.5)
    name = os.path.splitext(os.path.basename(path))[0]
    cls = name.split("_")[0] if name[:2] == "E0" else next(
        (TERMINAL[c] for c in reversed(df.loc[df["time"] > end - pd.Timedelta(minutes=10), "error"].tolist())
         if c in TERMINAL), "?")

    p = df[press & (df["c14"] > 0)]
    rel = (p["c2"] - p["c14"]) / p["c14"]
    force = pd.DataFrame({"minute": p["minute"], "miss": (rel.abs() > 0.1).astype(float), "dev": rel})
    force = force.groupby("minute").agg(force_miss=("miss", "mean"), force_dev=("dev", "median"))
    i = df[~near & (df["c15"] != 0)]
    pos = ((i["c3"] - i["c15"]).abs() > 0.1 * i["c15"].abs()).astype(float).groupby(i["minute"]).mean()
    out = force.join(pos.rename("pos_off"), how="outer")
    out["ttf_min"] = (end - out.index).total_seconds() / 60 - 1  # minute end to the file end
    out["file"], out["cls"] = name, cls
    return out.reset_index(), cls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help='CSV files or patterns relative to ml (default "test/*.csv" "train/*.csv")')
    ap.add_argument("--out", default=os.path.join(ML, "results", "D-6"))
    a = ap.parse_args()
    pats = a.files or ["test/*.csv", "train/*.csv"]
    files = sorted({f for p in pats for f in glob.glob(os.path.join(ML, p))})
    if not files:
        raise SystemExit("CSV가 없습니다: ml/test/ 또는 ml/train/ 에 CSV를 넣거나 --files 로 지정하세요")

    rows = []
    for k, path in enumerate(files, 1):
        m, cls = per_minute(path)
        normal = m[m["ttf_min"] > NORMAL_MIN]
        print(f"[{k}/{len(files)}] {os.path.basename(path)} ({cls}) 평소 {len(normal)}분", flush=True)
        if len(normal) < 60:
            continue  # less than an hour of normal minutes: no reference
        for metric, _ in METRICS:
            ref = normal[metric].dropna()
            if len(ref) < 60:
                continue
            absolute = metric == "force_dev"
            thr = float(np.quantile(ref.abs() if absolute else ref, 0.95))
            for band, lo, hi in BANDS + [("평소(24시간+)", NORMAL_MIN, np.inf)]:
                x = m.loc[(m["ttf_min"] >= lo) & (m["ttf_min"] < hi), metric].dropna()
                if x.empty:
                    continue
                v = x.abs() if absolute else x
                rows.append({"file": m["file"].iloc[0], "cls": cls, "metric": metric, "band": band, "minutes": len(x),
                             "median": float(x.median()), "unusual": float((v > thr).mean()), "threshold": thr})
    if not rows:
        raise SystemExit("비교할 수 있는 파일이 없습니다 (파일마다 고장 24시간 전보다 앞선 데이터가 1시간 이상 필요)")
    res = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "deviation_check.csv"), index=False, encoding="utf-8-sig")

    order = [b for b, *_ in BANDS] + ["평소(24시간+)"]
    lines = [f"# 설정 대비 실측 차이가 고장 전에 커지는가 ({res['file'].nunique()}개 파일)\n",
             "평범하지 않은 분 = 그 건의 평소(고장 24시간 전보다 앞) 상위 5%를 넘은 분. 우연이면 5% 안팎.",
             "값 = 파일별 값의 중앙값. 0-10분은 종료 코드(E012 등)가 이미 뜬 구간이라 '예고'가 아니라 고장 자체.\n"]
    for metric, title in METRICS:
        r = res[res["metric"] == metric]
        if r.empty:
            continue
        lines += [f"## {title}", "", "| 고장까지 | 파일 수 | 값 중앙값 | 평범하지 않은 분 비율 | 그 비율이 10% 넘은 파일 |",
                  "|---|---|---|---|---|"]
        for band in order:
            b = r[r["band"] == band]
            if b.empty:
                continue
            lines.append(f"| {band} | {len(b)} | {b['median'].median():+.3f} | {b['unusual'].median():.0%} | "
                         f"{int((b['unusual'] > 0.10).sum())}/{len(b)} |")
        cls_rows = r[r["band"].isin(["0-10분", "10-60분", "1-6시간"])]
        if cls_rows["cls"].nunique() > 1:
            lines += ["", "고장 유형별 평범하지 않은 분 비율 (중앙값):", "",
                      "| 유형 | 파일 수 | 0-10분 | 10-60분 | 1-6시간 |", "|---|---|---|---|---|"]
            for cls, g in cls_rows.groupby("cls"):
                cell = {bd: g.loc[g["band"] == bd, "unusual"].median() for bd in ("0-10분", "10-60분", "1-6시간")}
                lines.append(f"| {cls} | {g['file'].nunique()} | " + " | ".join(
                    "-" if pd.isna(cell[bd]) else f"{cell[bd]:.0%}" for bd in cell) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "deviation_check.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n" + text)
    print(f"저장: {os.path.join(a.out, 'deviation_check.md')} (파일·구간별 상세: deviation_check.csv)")


if __name__ == "__main__":
    main()
