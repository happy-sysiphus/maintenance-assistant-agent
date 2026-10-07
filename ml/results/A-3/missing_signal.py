"""
A-3 (1/2) 결측이 고장 신호인가?

ml 폴더에서 실행:  python results/A-3/missing_signal.py
입력: train/*.csv (원본 72건, time 열만 읽음)
출력: results/A-3/missing_by_ttf.csv, per_file_test.csv, gap_rate_by_ttf.csv, missing_by_ttf.png, missing_signal.log

질문 1. 고장에 가까울수록 결측률이 올라가나?       -> 고장까지 남은 시간 구간별 결측률 (클래스별)
질문 2. 고장 직전 결측이 '그 건의 평소 변동'을 넘나? -> 건마다 고장 직전 결측률이 평소 창들 중 상위 5% 안에 드는지
질문 3. 끊김이 고장 직전에 더 자주 생기나?        -> 구간별 끊김(gap) 시작 횟수 / 시간

판정 기준(미리 정해 둔 것):
  - 질문 2에서 상위 5%를 넘는 건이 우연 수준(약 5%)이면 -> 결측은 고장 신호가 아님(통신 문제)
  - 우연보다 훨씬 많고(예: 25% 이상), 질문 1·3에서도 고장 직전 구간만 튀면 -> 고장 신호 후보
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # results/A-3 -> ml
TRAIN = os.path.join(ROOT, "train")
OUT = os.path.join(ROOT, "results", "A-3")
MAX_MISSING = 0.40  # 논문·전처리 규칙: 결측 40% 초과 건은 폐기

BINS = [(0, 600, "0~10분"), (600, 1800, "10~30분"), (1800, 3600, "30~60분"), (3600, 3 * 3600, "1~3시간"),
        (3 * 3600, 6 * 3600, "3~6시간"), (6 * 3600, 24 * 3600, "6~24시간"), (86400, 3 * 86400, "1~3일"),
        (3 * 86400, 8 * 86400, "3~7일")]


def load_present(path):
    """원본 CSV -> 1초 격자의 '데이터 있음' 마스크와 고장까지 남은 초(ttf). 고장 시점 = 파일 마지막 행(전처리와 동일)."""
    t = pd.read_csv(path, usecols=["time"])["time"]
    t = pd.to_datetime(t, utc=True).dt.tz_localize(None).dt.floor("s").drop_duplicates().sort_values()
    grid = pd.date_range(t.iloc[0], t.iloc[-1], freq="s")
    present = grid.isin(t.values)
    ttf = (grid[-1] - grid).total_seconds().astype(np.int64)
    return present, np.asarray(ttf)


def window_rates(present, ttf, length, min_ttf):
    """고장 min_ttf초보다 이전 구간을 length초 창으로 잘라 창마다 결측률."""
    idx = np.where(ttf >= min_ttf)[0]
    if len(idx) < length:
        return np.array([])
    miss = ~present[idx]
    n = len(miss) // length
    return miss[: n * length].reshape(n, length).mean(axis=1)


def main():
    files = sorted(glob.glob(os.path.join(TRAIN, "*.csv")))
    if not files:
        sys.exit(f"train/*.csv 가 없습니다: {TRAIN}\n원본 데이터를 ml/train 폴더에 넣어 주세요.")
    os.makedirs(OUT, exist_ok=True)
    rows, tests, gaps = [], [], []
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        cls = name.split("_")[0]
        present, ttf = load_present(f)
        overall = 1 - present.mean()
        kept = overall <= MAX_MISSING
        gap_start = np.r_[False, present[:-1] & ~present[1:]]
        for lo, hi, label in BINS:
            m = (ttf >= lo) & (ttf < hi)
            if m.sum() == 0:
                continue
            rows.append({"file": name, "class": cls, "kept": kept, "bin": label,
                         "seconds": int(m.sum()), "missing_rate": float(1 - present[m].mean())})
            gaps.append({"file": name, "class": cls, "kept": kept, "bin": label,
                         "gaps_per_hour": float(gap_start[m].sum() / (m.sum() / 3600))})
        # 질문 2: 고장 직전 10분·60분 결측률이 그 건의 평소(고장 24시간 이전) 같은 길이 창들 중 몇 % 위치인가
        for length, label in [(600, "직전 10분"), (3600, "직전 60분")]:
            base = window_rates(present, ttf, length, 86400)
            near = 1 - present[ttf < length].mean()
            if len(base) < 20:
                continue
            rank = float((base < near).mean() + 0.5 * (base == near).mean())  # 동점은 절반
            tests.append({"file": name, "class": cls, "kept": kept, "window": label, "near_rate": float(near),
                          "base_median": float(np.median(base)), "base_q95": float(np.quantile(base, 0.95)),
                          "percentile": rank, "above_q95": bool(near > np.quantile(base, 0.95))})
        print(f"[{i}/{len(files)}] {name}: 전체 결측 {overall:.1%}{'' if kept else ' (40% 초과, 참고만)'}", flush=True)

    df, tf, gf = pd.DataFrame(rows), pd.DataFrame(tests), pd.DataFrame(gaps)
    df.to_csv(os.path.join(OUT, "missing_by_ttf.csv"), index=False, encoding="utf-8-sig")
    tf.to_csv(os.path.join(OUT, "per_file_test.csv"), index=False, encoding="utf-8-sig")
    gf.to_csv(os.path.join(OUT, "gap_rate_by_ttf.csv"), index=False, encoding="utf-8-sig")
    order = [b[2] for b in BINS if b[2] in set(df["bin"])]  # 짧은 파일만 있으면 없는 구간은 뺀다
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    k = df[df.kept]
    out("\n=== 질문 1. 고장까지 남은 시간 구간별 결측률 (건별 평균, 결측 40% 이하 건만) ===")
    t1 = k.pivot_table(index="class", columns="bin", values="missing_rate", aggfunc="mean")[order]
    t1.loc["전체"] = k.groupby("bin")["missing_rate"].mean()[order]
    out((t1 * 100).round(2).to_string())
    out("(단위 %. 고장 직전 칸만 뚜렷이 높으면 신호 후보)")

    out("\n=== 질문 2. 고장 직전 결측이 그 건의 평소 변동을 넘나? (평소 = 고장 24시간 이전의 같은 길이 창들) ===")
    kt = tf[tf.kept]
    for w, g in kt.groupby("window", sort=False):
        out(f"{w}: 평소 상위 5%를 넘은 건 {int(g.above_q95.sum())}/{len(g)} ({g.above_q95.mean():.0%}),"
            f" 직전 결측률이 평소 창들 중 위치(중앙값) {g.percentile.median():.0%}")
        for c, gc in g.groupby("class"):
            out(f"    {c}: {int(gc.above_q95.sum())}/{len(gc)}")
    out("(우연이면 약 5%. 이보다 훨씬 많아야 신호 후보)")

    out("\n=== 질문 3. 끊김(gap) 시작 횟수 / 시간 (건별 평균, 결측 40% 이하 건만) ===")
    kg = gf[gf.kept]
    t3 = kg.pivot_table(index="class", columns="bin", values="gaps_per_hour", aggfunc="mean")[order]
    t3.loc["전체"] = kg.groupby("bin")["gaps_per_hour"].mean()[order]
    out(t3.round(1).to_string())

    dropped = sorted(df[~df.kept].file.unique())
    out(f"\n참고: 결측 40% 초과로 위 표에서 뺀 건 {len(dropped)}개: {', '.join(dropped) or '없음'}")
    out("주의: E02는 수집 시기(2019~20년)에 따라 결측이 많아 클래스 차이가 고장 탓이 아닐 수 있다 (README 5.3.1).")
    with open(os.path.join(OUT, "missing_signal.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    try:
        import logging
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 글꼴 경고 숨김
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for c in t1.index:
            ax.plot(order, t1.loc[c] * 100, marker="o", label=c, linewidth=2.5 if c == "전체" else 1.2)
        ax.set_xlabel("고장까지 남은 시간")
        ax.set_ylabel("결측률 (%)")
        ax.set_title("고장까지 남은 시간별 결측률")
        ax.legend()
        ax.invert_xaxis()  # 오른쪽 끝 = 고장 직전
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "missing_by_ttf.png"), dpi=120)
        print(f"\n그래프: {os.path.join(OUT, 'missing_by_ttf.png')}")
    except Exception as e:  # matplotlib 없으면 표만
        print(f"(그래프 생략: {e})")


if __name__ == "__main__":
    main()
