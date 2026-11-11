"""
D-9 offline check (docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md, 10.2): the feedback loop on real data,
in foreground steps that each finish within 10 minutes on the 8 GB PC. Run (1) uses SIMULATED labels: it shows the
loop turning, it is not performance.

    python results/D-9/offline_check.py replay test/test_0.csv [...] [--start-hours H] [--hours H] [--no-reset]
        test files through the serving app (TestClient, no server; .py/replay.py replay_file, 600-s chunks: 60-s
        chunks took ~28 min per file on the 8 GB PC; the window log is one line per window at any chunk size) with
        RSW_WINDOW_LOG=1 and RSW_EARLY_WARNING_HANDOFFS=1 into results/D-9/runs/state-test8 (RSW_RAG_URL unset)
    python results/D-9/offline_check.py simulate
        runs/history_sim.json, a /api/history-shaped file (11.1 fields, "simulated": true on every item):
        critical - resolved if its rule fired within 15 min before the file's end (meta.csv end, B-18's T), else a
        false alarm (reason 기타, note "시뮬레이션: 파일 끝 고장 아님"); early warning - closed as a false alarm when the
        automatic label says so (reason 설비 정지 · 전원 꺼짐 if the onset window's non-welding share > 0.5, else
        정상 작업 중 변동), otherwise left open (not in the history). One case per handoff, closed 10 min after it
    python results/D-9/offline_check.py loop1             run (1): state-test8 + history_sim.json + the B-19 lab
    python results/D-9/offline_check.py loop2 [--tag T]    run (2): the 10-29 demo's real labels
                                                           (state/d8-demo1029 + ../api/state/d8-demo1029.db)
    python results/D-9/offline_check.py scores            window-log scores vs models/scores/<gun>_ensemble.csv

Writes only under results/D-9/runs/ (git-ignored: case numbers and field notes stay out of the repo).
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
STATE = os.path.join(RUNS, "state-test8")
HISTORY = os.path.join(RUNS, "history_sim.json")
META = os.path.join(ROOT, "results", "B-17", "cache", "meta.csv")
LAB = os.path.join(ROOT, "results", "B-19", "cache")
CHUNK_S = 600  # replay.py's own "faster" example; scores and the window log do not depend on the chunk size
sys.path.insert(0, os.path.join(ROOT, ".py"))
sys.path.insert(0, ROOT)


def replay(files, start_hours, hours, reset):
    os.environ.update({"RSW_STATE_DIR": STATE, "RSW_WINDOW_LOG": "1", "RSW_EARLY_WARNING_HANDOFFS": "1"})
    os.environ.pop("RSW_RAG_URL", None)
    import main  # noqa: E402  (reads the environment in its lifespan)
    from fastapi.testclient import TestClient
    from replay import replay_file

    with TestClient(main.app) as client:
        for path in files:
            t0 = time.time()
            s = replay_file(client, path, chunk_s=CHUNK_S, start_hours=start_hours, hours=hours, reset=reset)
            print(f"{s['gun_id']}: {s['requests']} requests, {s['windows_scored']} windows, "
                  f"{len(s['critical_events'])} critical, {s['early_warnings']} early-warning starts "
                  f"({s['early_warning_handoffs']} handoffs), {time.time() - t0:.0f} s", flush=True)
    return 0


def _event(h):
    tr = h.get("trigger") or {}
    fc = h.get("fault_class")
    return {"event_id": h["event_id"], "gun_id": h["gun_id"], "detected_at": h["detected_at"],
            "event_type": h.get("event_type") or "critical",
            "trigger": {k: tr.get(k) for k in ("source", "rule_code", "rule_trigger_time")},
            "fault_class": {"code": fc["code"], "name_ko": fc["name_ko"]} if fc else None,
            "summary_ko": h.get("summary_ko")}


def simulate():
    import feedback_eval as fe
    import feedback_loop as fl

    windows, _, notes = fl.read_windows(STATE)
    outbox, more = fl.read_outbox(STATE)
    ends = pd.read_csv(META, parse_dates=["end"]).set_index("file")["end"]
    items, resolved = [], []

    def item(h, outcome, reason=None, note=None):
        closure = {"outcome": outcome,
                   "at": (fl.utc_naive(h["detected_at"]) + pd.Timedelta(minutes=10)).isoformat() + "Z"}
        if reason:
            closure["reason"] = reason
        if note:
            closure["note"] = note
        ev = _event(h)
        return {"case_id": None, "status": outcome, "severity": "warning" if ev["event_type"] == "early_warning"
                else "critical", "assignee": None, "event": ev, "repeat_count": 0, "closure": closure,
                "started_at": None, "cause": None, "action_kind": None, "worker": None, "repeat_event_ids": [],
                "simulated": True}

    hs = sorted(outbox, key=lambda h: (h["detected_at"], h["event_id"]))
    for h in hs:
        if h.get("event_type") == "early_warning":
            continue
        rt, end = fl.utc_naive((h.get("trigger") or {}).get("rule_trigger_time")), ends[h["gun_id"]]
        if rt is not None and end - pd.Timedelta(minutes=15) <= rt <= end:
            items.append(item(h, "resolved"))
            resolved.append(h)
        else:
            items.append(item(h, "false_alarm", "기타", "시뮬레이션: 파일 끝 고장 아님"))
    fails = pd.DataFrame([(h["gun_id"], fl.utc_naive(h["trigger"]["rule_trigger_time"])) for h in resolved
                          if h["trigger"].get("rule_code")], columns=["gun", "T"])
    fails["T"] = pd.to_datetime(fails["T"])
    fr = fe.build_frame(windows, fails)
    rr = fe.evaluate(fr, fr.w["early_warning"].to_numpy(bool))
    n_ew = n_fa = 0
    for h in hs:
        if h.get("event_type") != "early_warning":
            continue
        n_ew += 1
        i = fe.onset_index(fr, h["gun_id"], fl.utc_naive(h["early_warning"]["onset_at"]).floor("60s"))
        if i is None or fe.onset_label(fr, rr, i) != "false_alarm":
            continue
        n_fa += 1
        stop = float(fr.w.loc[i, "non_welding_share"]) > fe.GATE_NON_WELDING
        items.append(item(h, "false_alarm", "설비 정지 · 전원 꺼짐" if stop else "정상 작업 중 변동"))
    items.sort(key=lambda x: (x["event"]["detected_at"], x["event"]["event_id"]))
    for k, x in enumerate(items, 1):
        x["case_id"] = f"SIM-{k:04d}"
    os.makedirs(RUNS, exist_ok=True)
    with open(HISTORY, "w", encoding="utf-8") as fh:
        json.dump({"items": items}, fh, ensure_ascii=False, indent=1)
    crit = [x for x in items if x["event"]["event_type"] != "early_warning"]
    print(f"history_sim.json: {len(items)} cases - critical {len(crit)} (resolved {len(resolved)}, false alarm "
          f"{len(crit) - len(resolved)}), early warning closed as false alarm {n_fa} of {n_ew}; "
          f"windows {len(windows)}, warnings {notes + more}")
    return 0


def scores():
    import feedback_loop as fl

    windows, _, _ = fl.read_windows(STATE)
    out = {}
    for gun, grp in windows.groupby("gun"):
        path = os.path.join(ROOT, "models", "scores", f"{gun}_ensemble.csv")
        if not os.path.exists(path):
            continue
        off = pd.read_csv(path, usecols=["time", "score", "threshold"], parse_dates=["time"])
        j = grp.merge(off, left_on="window_start", right_on="time", suffixes=("", "_off"))
        diff = (j["score"] - j["score_off"]).abs()
        out[gun] = {"windows": int(len(grp)), "matched": int(len(j)), "max_abs_diff": float(diff.max()),
                    "within_1e-4": int((diff <= 1e-4).sum()), "csv_threshold": float(off["threshold"].iloc[0])}
        print(f"{gun}: {out[gun]}")
    with open(os.path.join(RUNS, "scores.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    ok = [g for g, v in out.items() if v["matched"] and v["within_1e-4"] == v["matched"]]
    print(f"files with every window within 1e-4: {ok or 'none'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("files", nargs="+")
    r.add_argument("--start-hours", type=float, default=0.0)
    r.add_argument("--hours", type=float, default=None)
    r.add_argument("--no-reset", action="store_true")
    sub.add_parser("simulate")
    sub.add_parser("loop1")
    lp2 = sub.add_parser("loop2")
    lp2.add_argument("--tag", default="offline-2")
    sub.add_parser("scores")
    a = ap.parse_args()
    if a.cmd == "replay":
        return replay(a.files, a.start_hours, a.hours, not a.no_reset)
    if a.cmd == "simulate":
        return simulate()
    if a.cmd == "scores":
        return scores()
    import feedback_loop as fl
    if a.cmd == "loop1":
        return fl.main(["--state", STATE, "--history", HISTORY, "--lab", LAB, "--tag", "offline-1", "--out", RUNS])
    return fl.main(["--state", os.path.join(ROOT, "state", "d8-demo1029"),
                    "--cases", os.path.join(os.path.dirname(ROOT), "api", "state", "d8-demo1029.db"),
                    "--tag", a.tag, "--out", RUNS])


if __name__ == "__main__":
    sys.exit(main())
