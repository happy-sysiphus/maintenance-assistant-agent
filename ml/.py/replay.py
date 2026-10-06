"""
Replay raw RSW CSV files into the running inference API (main.py), chunk by chunk, as a live gun would.

Each file becomes one gun stream (gun_id = file stem unless --gun-id). Rows are sent in chunks of
--chunk-s seconds (<= main.MAX_CHUNK = 1200) to POST /predict; every completed window in a chunk is
scored by the API, so the chunk size changes the number of requests, not the verdict. Critical results
(a sustained model alarm or a terminal-code rule trigger) are printed as they happen, and a summary per
file is printed at the end. --out-dir writes every response as JSON lines (<out-dir>/<gun_id>.jsonl).

    uvicorn main:app                                         # in another terminal
    python .py/replay.py test/test_0.csv                     # whole file, as fast as possible
    python .py/replay.py test/test_*.csv --chunk-s 600       # 8 guns, 10-min chunks (faster)
    python .py/replay.py test/test_3.csv --start-hours 160 --hours 8 --out-dir results/replay
    python .py/replay.py test/test_0.csv --speed 60          # 60x real time

`replay_file()` takes any client with httpx's .post/.delete (httpx.Client or fastapi's TestClient),
which is how tests/test_pipeline.py drives it without a server.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_CHUNK = 1200  # main.MAX_CHUNK (BUFFER_S - 10 min); not imported to keep the client free of the model stack


def load_rows(path: str, start_hours: float = 0.0, hours: float | None = None) -> pd.DataFrame:
    """Raw CSV rows (time, c1..c19, error) in time order, optionally a slice [start, start + hours)."""
    df = pd.read_csv(path, dtype={"c10": str, "error": str}, low_memory=False)
    df["error"] = df["error"].fillna("0")
    t = pd.to_datetime(df["time"], utc=True)
    df = df.assign(_t=t).sort_values("_t")
    t0 = df["_t"].iloc[0] + pd.Timedelta(hours=start_hours)
    sel = df["_t"] >= t0
    if hours is not None:
        sel &= df["_t"] < t0 + pd.Timedelta(hours=hours)
    return df[sel].drop(columns="_t").reset_index(drop=True)


def chunks(df: pd.DataFrame, chunk_s: int):
    """Consecutive chunks of at most chunk_s seconds of stream time (rows are 1 Hz with gaps)."""
    t = pd.to_datetime(df["time"], utc=True)
    key = ((t - t.iloc[0]).dt.total_seconds() // chunk_s).astype(int)
    for _, part in df.groupby(key, sort=True):
        for i in range(0, len(part), MAX_CHUNK):  # a chunk never exceeds the API limit
            yield part.iloc[i: i + MAX_CHUNK]


def replay_file(client: Any, path: str, gun_id: str | None = None, chunk_s: int = 60, start_hours: float = 0.0,
                hours: float | None = None, speed: float = 0.0, reset: bool = True,
                on_result: Callable[[dict[str, Any], dict[str, Any] | None], None] | None = None) -> dict[str, Any]:
    """Stream one CSV into /predict. Returns a summary: requests, windows scored, critical events.
    on_result(response, event) is called per scored response; event is the new critical event or None."""
    if not 1 <= chunk_s <= MAX_CHUNK:
        raise ValueError(f"chunk_s must be in 1..{MAX_CHUNK}")
    gun_id = gun_id or os.path.splitext(os.path.basename(path))[0]
    df = load_rows(path, start_hours, hours)
    if reset:
        client.delete(f"/guns/{gun_id}")  # 404 for an unknown gun is fine
    summary: dict[str, Any] = {"gun_id": gun_id, "file": os.path.basename(path), "rows": len(df), "requests": 0,
                               "warming_up": 0, "windows_scored": 0, "critical_events": [], "rule_triggers": 0, "rule_repeats": 0, "rule_out_of_profile": 0,
                               "max_score": None, "last": None}
    was_critical = False
    for part in chunks(df, chunk_s):
        r = client.post("/predict", json={"gun_id": gun_id, "readings": part.to_dict("records")})
        summary["requests"] += 1
        if r.status_code == 202:
            summary["warming_up"] += 1
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{gun_id}: HTTP {r.status_code} at {part['time'].iloc[0]}: {r.text[:500]}")
        res = r.json()
        summary["windows_scored"] += res["windows_scored"]
        summary["rule_triggers"] += int(res["rule_triggered"])
        rule_critical = (res["rule_triggered"] and not res.get("rule_repeat", False)
                         and not res.get("rule_out_of_profile", False))
        summary["rule_repeats"] += int(res["rule_triggered"] and res.get("rule_repeat", False))
        summary["rule_out_of_profile"] += int(res["rule_triggered"] and not res.get("rule_repeat", False)
                                              and res.get("rule_out_of_profile", False))
        summary["max_score"] = res["anomaly_score"] if summary["max_score"] is None else max(summary["max_score"], res["anomaly_score"])
        summary["last"] = res
        critical, event = res["critical_in_request"], None
        # one event per critical episode (a sustained alarm stays critical over many requests); a rule trigger
        # is always its own event, unless it is a repeat (not critical)
        if (critical and not was_critical) or rule_critical:
            event = {
                "window_end": res["window_end"],
                # source of the request's criticality (severity_source describes the latest window only)
                "severity_source": res.get("critical_source") or "+".join(
                    k for k, v in (("model", res["sustained_in_request"]), ("rule", rule_critical)) if v),
                "rule_trigger_time": res["rule_trigger_time"],
                # the code that fired the rule (it has often cleared by window_end), else the latest one
                "error_code": res["rule_code"] or res["context"]["latest_error_code"],
                "class_hint": res["rule_class_hint"] or res["context"]["known_code_class_hint"],
                "anomaly_score": res["anomaly_score"], "threshold": res["threshold"],
                "top_features": [c["feature"] for c in res["contributing_features"][:3]],
                # ML -> RAG handoff attached by the API to the same event (None on an older server)
                "handoff_event_id": (res.get("handoff") or {}).get("event_id"),
                "summary_ko": (res.get("handoff") or {}).get("summary_ko"),
                "situation_ids": (res.get("handoff") or {}).get("situation_ids")}
            summary["critical_events"].append(event)
        was_critical = critical
        if on_result:
            on_result(res, event)
        if speed > 0:
            time.sleep(len(part) / speed)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="raw CSV files (globs allowed), e.g. test/test_0.csv")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--gun-id", default=None, help="gun id (single file only); default: file stem")
    ap.add_argument("--chunk-s", type=int, default=60, help=f"seconds of stream per request, 1..{MAX_CHUNK}")
    ap.add_argument("--start-hours", type=float, default=0.0, help="skip this many hours from the file start")
    ap.add_argument("--hours", type=float, default=None, help="replay only this many hours")
    ap.add_argument("--speed", type=float, default=0.0, help="x real time (0 = as fast as possible)")
    ap.add_argument("--no-reset", action="store_true", help="keep the gun's server state (default: DELETE it first)")
    ap.add_argument("--out-dir", default=None, help="write every response to <out-dir>/<gun_id>.jsonl")
    args = ap.parse_args()

    import httpx  # only the CLI needs it; replay_file() accepts any client

    paths = sorted({p for f in args.files for p in (glob.glob(f) or [f])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"not found: {missing} (cwd {os.getcwd()}, project root {PROJECT_ROOT})")
    if args.gun_id and len(paths) > 1:
        raise SystemExit("--gun-id needs a single file")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    with httpx.Client(base_url=args.url, timeout=60) as client:
        try:
            health = client.get("/health").json()
        except httpx.HTTPError as e:
            raise SystemExit(f"API not reachable at {args.url} ({e}) - start it with: uvicorn main:app")
        print(f"API {args.url}: model {health['model']['name']} (window {health['model']['window_s']} s)")
        for p in paths:
            gun = args.gun_id or os.path.splitext(os.path.basename(p))[0]
            sink = open(os.path.join(args.out_dir, f"{gun}.jsonl"), "w", encoding="utf-8") if args.out_dir else None

            def on_result(res: dict[str, Any], event: dict[str, Any] | None, gun: str = gun, sink: Any = sink) -> None:
                if sink:
                    sink.write(json.dumps(res) + "\n")
                if event:
                    print(f"  [{gun}] CRITICAL {event['window_end']} source={event['severity_source']} "
                          f"code={event['error_code']} hint={event['class_hint']} "
                          f"score={event['anomaly_score']:.3f}/{event['threshold']:.3f} top={event['top_features']} "
                          f"situations={event['situation_ids']} | {event['summary_ko']}",
                          flush=True)

            t0 = time.time()
            print(f"{gun}: replaying {p}", flush=True)
            try:
                s = replay_file(client, p, gun, args.chunk_s, args.start_hours, args.hours, args.speed,
                                reset=not args.no_reset, on_result=on_result)
            finally:
                if sink:
                    sink.close()
            last = s["last"] or {}
            print(f"{gun}: {s['rows']:,} rows, {s['requests']:,} requests, {s['windows_scored']:,} windows, "
                  f"{len(s['critical_events'])} critical events ({s['rule_triggers']} rule triggers, {s['rule_repeats']} repeats), "
                  f"max score {s['max_score'] or float('nan'):.3f}, final gun_norm {last.get('gun_norm')}, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
