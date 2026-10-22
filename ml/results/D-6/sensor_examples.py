"""
D-6: make example sensor-features-v4 documents (what the RAG side receives every minute) from real CSV files.

You choose WHICH FILE and WHICH MINUTE; the script replays the file through the real API code (no server needed)
from far enough before that minute for the gun warm-up to finish, and keeps the document of that minute.

Run from the ml folder (needs models/baseline_ensemble.joblib and the CSVs, README 0):

    python results/D-6/sensor_examples.py
        default sweep = 10 examples: test_0, test_3 x (5, 30, 180, 1440 min before the file end + the minute a
        terminal code first shows in the last 15 min)
    python results/D-6/sensor_examples.py --files test/test_1.csv train/E01_1.csv --before-end 5 60 600
        files x minutes-before-end (the parameter sweep)
    python results/D-6/sensor_examples.py --at test/test_0.csv=2021-09-11T02:55:59
        one exact minute (UTC, the window end :59 or any second inside the minute)
    python results/D-6/sensor_examples.py --plan my_plan.csv
        a CSV with columns file,at ; at = minutes before the end (e.g. 30), an ISO time, or "code"

Output (results/D-6/examples/ unless --out):
    <gun>_<label>.json   the document exactly as the RAG side gets it (checked against docs/sensor_features.schema.json)
    summary.csv / summary.md   one row per example: minute, code, gun_norm, samples, the 4 measurements with the
                               largest |z| (z / raw), state on-shares, setpoint mean ± std (raw)

Time: the replay covers lead (warm-up, ~6.5 h) + the span of the chosen minutes per file; 1440 min before the end
means ~31 h of stream, a few minutes per file at --chunk-s 600.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict

import pandas as pd

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, ".py"))

DEFAULT_FILES = ["test/test_0.csv", "test/test_3.csv"]
DEFAULT_BEFORE_END = [5, 30, 180, 1440]
TERMINAL = ("E012", "E016", "E028", "E029")


def file_span(path: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    df = pd.read_csv(path, usecols=["time", "error"], dtype={"error": str})
    t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    return t.min(), t.max(), pd.DataFrame({"t": t, "error": df["error"].fillna("0")})


def code_minute(codes: pd.DataFrame, t_end: pd.Timestamp) -> pd.Timestamp | None:
    """The first second a terminal code shows in the last 15 min (readme 5.3.4: codes start ~10 min before the end)."""
    last = codes[(codes["t"] > t_end - pd.Timedelta(minutes=15)) & codes["error"].isin(TERMINAL)]
    return None if last.empty else last["t"].min()


def resolve(at: str, t0: pd.Timestamp, t_end: pd.Timestamp, codes: pd.DataFrame) -> tuple[str, pd.Timestamp] | None:
    at = str(at).strip()
    if at == "code":
        t = code_minute(codes, t_end)
        return None if t is None else ("code", t)
    try:
        n = float(at)
        return f"end-{int(n)}m", t_end - pd.Timedelta(minutes=n)
    except ValueError:
        t = pd.Timestamp(at)
        t = t.tz_convert(None) if t.tzinfo else t
        return f"{t:%Y%m%dT%H%M}", t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help=f"CSV files relative to ml (default {DEFAULT_FILES})")
    ap.add_argument("--before-end", nargs="+", type=float, help=f"minutes before the file end (default {DEFAULT_BEFORE_END})")
    ap.add_argument("--no-code", action="store_true", help="do not add the minute a terminal code first shows")
    ap.add_argument("--at", nargs="+", default=[], help="FILE=ISO_TIME, exact minutes (added to the sweep)")
    ap.add_argument("--plan", help="CSV with columns file,at (replaces --files / --before-end)")
    ap.add_argument("--model", default=os.path.join(ML, "models", "baseline_ensemble.joblib"))
    ap.add_argument("--lead-hours", type=float, help="replay start before the first minute (default: warm-up + 0.5 h)")
    ap.add_argument("--chunk-s", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ML, "results", "D-6", "examples"))
    a = ap.parse_args()

    plan: dict[str, list[str]] = defaultdict(list)
    if a.plan:
        for r in csv.DictReader(open(a.plan, encoding="utf-8-sig")):
            plan[r["file"].strip()].append(r["at"].strip())
    elif a.files or a.before_end or not a.at:
        for f in a.files or DEFAULT_FILES:
            plan[f] += [str(m) for m in (a.before_end or DEFAULT_BEFORE_END)] + ([] if a.no_code else ["code"])
    for x in a.at:
        f, t = x.split("=", 1)
        plan[f].append(t)

    missing = [p for p in [a.model] + [os.path.join(ML, f) for f in plan] if not os.path.exists(p)]
    if missing:
        sys.exit("없음:\n  " + "\n  ".join(missing) + "\n-> 모델은 README 0절 드라이브, CSV는 train/ test/ 폴더")

    os.environ["RSW_MODEL_PATH"] = a.model
    os.environ["RSW_STATE_DIR"] = tempfile.mkdtemp(prefix="rsw_d6_")  # never touches the serving state
    for k in ("RSW_RAG_URL", "RSW_RAG_SENSOR_URL"):
        os.environ.pop(k, None)  # nothing is sent anywhere
    import jsonschema
    from fastapi.testclient import TestClient
    from replay import replay_file

    import main as api

    schema = json.load(open(os.path.join(ML, "docs", "sensor_features.schema.json"), encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    with TestClient(api.app) as client:
        d = api.app.state.detector
        warm_h = ((d.gun_norm or {}).get("warmup_s", 0) or 0) / 3600
        lead = a.lead_hours if a.lead_hours is not None else warm_h + 0.5
        for f, ats in plan.items():
            path = os.path.join(ML, f)
            gun = os.path.splitext(os.path.basename(f))[0]
            t0, t_end, codes = file_span(path)
            targets = [x for x in (resolve(at, t0, t_end, codes) for at in ats) if x is not None]
            if len(targets) < len(ats):
                print(f"{gun}: no terminal code in the last 15 min - 'code' skipped")
            if not targets:
                continue
            first = min(t for _, t in targets)
            start = max(t0, first - pd.Timedelta(hours=lead))
            hours = (max(t for _, t in targets) - start).total_seconds() / 3600 + 2 / 60
            print(f"{gun}: replay {start} .. +{hours:.1f} h for {len(targets)} minutes", flush=True)
            docs: dict[str, dict] = {}
            replay_file(client, path, gun, chunk_s=a.chunk_s, start_hours=(start - t0).total_seconds() / 3600,
                        hours=hours, on_result=lambda res, ev, docs=docs: docs.update(
                            {x["observed_at"]: x for x in res["sensor_features"]}))
            ends = sorted(docs)
            for label, t in targets:
                minute = f"{t:%Y-%m-%dT%H:%M}"  # the clock minute containing t; its document ends at :59
                cand = [e for e in ends if e[:16] >= minute][:1]  # that minute, else the next scored one (gap)
                if not cand:
                    print(f"  {label}: no scored minute at or after {minute}")
                    continue
                if cand[0][:16] != minute:
                    print(f"  {label}: {minute} was not scored (gap / no welding) - using {cand[0]}")
                doc = docs[cand[0]]
                jsonschema.validate(doc, schema)
                name = f"{gun}_{label}.json"
                json.dump(doc, open(os.path.join(a.out, name), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                top = "  ".join(f"{m['feature']} {m['z']:+.2f} ({m['value']:g})" for m in doc["measurements"][:4]
                                if m["z"] is not None)
                states = " ".join(f"{x['signal']} {x['on_share']:.0%}" for x in doc["states"] if x["on_share"] is not None)
                still = all(not x["on_share"] for x in doc["states"])
                sets = "  ".join(f"{x['setpoint']} {x['mean']:g}±{x['std']:.3g}" for x in doc["setpoints"]
                                 if x["mean"] is not None)
                rows.append({"file": gun, "label": label, "observed_at": doc["observed_at"],
                             "error_code": doc["error_code"] or "", "fault_class": doc["fault_class"] or "",
                             "gun_norm": doc["gun_norm"], "n_samples": doc["n_samples"], "top4": top,
                             "states": states + (" (정지)" if still else ""), "setpoints": sets, "json": name})
                print(f"  {label:>10} {doc['observed_at']} code={doc['error_code']} {doc['gun_norm']:<10} {top} | {states}",
                      flush=True)
    if not rows:
        sys.exit("예시가 하나도 만들어지지 않았습니다")
    with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("| 파일 | 시점 | 분(UTC) | 에러코드 | 고장유형 | gun_norm | 초 | 측정값 상위 4 (z, 원래 값) | 상태 켜짐 비율 "
                 "| 설정값 평균±흔들림 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['file']} | {r['label']} | {r['observed_at']} | {r['error_code'] or '-'} | "
                     f"{r['fault_class'] or '-'} | {r['gun_norm']} | {r['n_samples']} | {r['top4']} | {r['states']} | "
                     f"{r['setpoints']} |\n")
    print(f"\n{len(rows)}개 예시 -> {a.out} (summary.md, summary.csv, *.json; 모두 스키마 통과)")


if __name__ == "__main__":
    main()
