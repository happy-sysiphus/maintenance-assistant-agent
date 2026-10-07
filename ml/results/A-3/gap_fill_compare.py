"""
A-3 (2/2) 결측을 채우는 방식 vs 그대로 두는 방식 비교

ml 폴더에서 실행:  python results/A-3/gap_fill_compare.py           (운영 모델 = 앙상블, 약 1시간)
                   python results/A-3/gap_fill_compare.py --quick   (IsolationForest만, 약 25분, 먼저 돌려 보기용)
입력: train/*.csv, test/*.csv (원본)
출력: results/A-3/gap_fill/<방식>/ (전처리 결과·모델 — 운영 폴더 preprocessed/·models/는 건드리지 않음)
      results/A-3/gap_fill_compare.csv, gap_fill_compare.log

비교하는 세 방식 (나머지 설정은 운영과 동일):
  keep      그대로 두기   : 빈 초는 채우지 않고, 그 자리에서 데이터를 끊는다
  fill60    짧게만 채우기 : 60초 이하 끊김만 직전 값으로 채우고, 긴 끊김에서 끊는다 (지금 운영 방식)
  fill_all  전부 채우기   : 끊김 길이와 상관없이 직전 값으로 채운다 (논문 전처리 방식)

보는 숫자:
  남은 행·세그먼트  데이터가 얼마나 남고 몇 조각으로 나뉘었나
  CV AUROC 등       학습 64건을 4묶음으로 돌려 본 평균 (믿을 만한 비교의 기준)
  테스트 ...        처음 보는 8건 결과 (건수가 적어 참고용)
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "results", "A-3")
RUNNER = os.path.join(HERE, "_preprocess_run.py")
TRAIN_PY = os.path.join(ROOT, ".py", "train.py")
CONFIGS = [("keep", 0, "그대로 두기"), ("fill60", 60, "짧게만 채우기(현재)"), ("fill_all", 10_000_000, "전부 채우기(논문)")]


def run(cmd, log):
    print("  $ " + " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd), flush=True)
    with open(log, "a", encoding="utf-8") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"실패했습니다. 로그를 확인하세요: {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="IsolationForest만 (빠른 확인용)")
    ap.add_argument("--only", nargs="*", default=None, help="일부 방식만, 예: --only keep fill60")
    ap.add_argument("--skip-existing", action="store_true", help="이미 결과가 있는 방식은 건너뜀")
    args = ap.parse_args()
    for sub in ("train", "test"):
        if not os.path.isdir(os.path.join(ROOT, sub)):
            sys.exit(f"{sub}/ 폴더가 없습니다. 원본 CSV를 ml/{sub}/ 에 넣어 주세요.")
    model = "iforest" if args.quick else "ensemble"
    rows = []
    for name, limit, label in CONFIGS:
        if args.only and name not in args.only:
            continue
        base = os.path.join(HERE, "gap_fill", name)
        pre, models = os.path.join(base, "preprocessed"), os.path.join(base, "models")
        mfile = os.path.join(models, f"baseline_{model}_metrics.json")
        log = os.path.join(base, "run.log")
        os.makedirs(base, exist_ok=True)
        t0 = time.time()
        if not (args.skip_existing and os.path.exists(mfile)):
            print(f"\n[{label}] gap-fill-limit={limit} — 전처리(train)", flush=True)
            run([sys.executable, RUNNER, "--split", "train", "--gap-fill-limit", str(limit), "--out-dir", pre], log)
            print(f"[{label}] 전처리(test)", flush=True)
            run([sys.executable, RUNNER, "--split", "test", "--train-out-dir", pre,
                 "--out-dir", os.path.join(pre, "test")], log)
            print(f"[{label}] 학습 + 4-fold 교차검증 + 테스트 ({model})", flush=True)
            run([sys.executable, TRAIN_PY, "--exclude-non-welding", "--cv", "4", "--model", model,
                 "--data-dir", pre, "--model-dir", models], log)
        m = json.load(open(mfile, encoding="utf-8"))["metrics"]
        man = pd.read_csv(os.path.join(pre, "manifest.csv"))
        cv, te = m["cv"]["mean"], m.get("test", {})
        rows.append({"방식": label, "gap_fill_limit": limit,
                     "남은 행(백만)": round(man["rows"].sum() / 1e6, 2), "세그먼트": int(man["segments"].sum()),
                     "CV AUROC": round(cv["auroc"], 3), "CV AUROC ±": round(m["cv"]["sd"]["auroc"], 3),
                     "CV 규칙 전 AUROC": round(cv["auroc_pre_rule"], 3),
                     "CV 정상 알람률": round(cv["alarm_rate_normal"], 4),
                     "테스트 AUROC": round(te.get("auroc", float("nan")), 3),
                     "테스트 규칙 전 AUROC": round(te.get("auroc_pre_rule", float("nan")), 3),
                     "테스트 정상 알람률": round(te.get("alarm_rate_normal", float("nan")), 4),
                     "걸린 시간(분)": round((time.time() - t0) / 60, 1)})
        print(f"[{label}] 완료", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "gap_fill_compare.csv"), index=False, encoding="utf-8-sig")
    text = df.set_index("방식").T.to_string()
    note = ("\n읽는 법: CV 값 차이가 'CV AUROC ±'(폴드 편차)보다 작으면 사실상 같은 성능이다.\n"
            "남은 행·세그먼트가 크게 다르면, 성능이 같아도 쓸 수 있는 데이터 양이 다르다는 뜻이다.")
    print("\n=== 결측 처리 방식 비교 (모델: " + model + ") ===\n" + text + note)
    with open(os.path.join(HERE, "gap_fill_compare.log"), "w", encoding="utf-8") as fh:
        fh.write(f"model: {model}\n{text}\n{note}\n")


if __name__ == "__main__":
    main()
