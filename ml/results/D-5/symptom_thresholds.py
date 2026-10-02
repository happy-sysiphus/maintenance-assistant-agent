"""D-5 (history.md §19): calibrate rag_mapping.MIN_SHARE / MIN_DEV on real events.

    python results/D-5/symptom_thresholds.py collect   # last 12.2 h of the 72 train + 8 test CSVs -> symptom_events.jsonl
    python results/D-5/symptom_thresholds.py sweep     # findings per event over a (MIN_SHARE, MIN_DEV) grid -> symptom_sweep.json

Rule fixed before looking (no labels exist, the aim is neither over- nor under-reporting): among grid points with a
median of 1-3 findings per event and at most 25 % of events without any finding, keep the current values if they
qualify, else take the qualifying point closest to them (relative distance). The last 12.2 h of every file hold its
failure (rule trigger, often a model episode) and a warm-up of their own; false triggers days earlier are not in it.
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = os.path.join(HERE, "symptom_events.jsonl")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".py"))
import rag_mapping as R  # noqa: E402

SHARES = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
DEVS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MAX_ZERO = 0.25


def collect():
    os.environ["RSW_STATE_DIR"] = tempfile.mkdtemp(prefix="rsw_ho5_")
    os.environ.pop("RSW_RAG_URL", None)
    import main
    from fastapi.testclient import TestClient
    from replay import replay_file

    files = sorted(glob.glob(os.path.join(ROOT, "train", "*.csv"))) + sorted(glob.glob(os.path.join(ROOT, "test", "*.csv")))
    done = set()
    if os.path.exists(EVENTS):
        done = {json.loads(line)["file"] for line in open(EVENTS, encoding="utf-8")}
    with TestClient(main.app) as c, open(EVENTS, "a", encoding="utf-8") as out:
        for i, path in enumerate(files, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in done:
                continue
            n_h = sum(1 for _ in open(path, encoding="utf-8")) / 3600
            events = []

            def keep(res, event, events=events):
                if event is not None:
                    events.append({"window_end": res["window_end"], "source": event["severity_source"],
                                   "rule_code": res["rule_code"], "class_hint": event["class_hint"],
                                   "contributing_features": res["contributing_features"]})
            replay_file(c, path, name, chunk_s=1200, start_hours=max(n_h - 12.2, 0), on_result=keep)
            c.delete(f"/guns/{name}")
            for e in events:
                out.write(json.dumps({"file": name, **e}, ensure_ascii=False) + "\n")
            if not events:  # remember the file was done
                out.write(json.dumps({"file": name, "none": True}) + "\n")
            out.flush()
            print(f"[{i}/{len(files)}] {name}: {len(events)} events", flush=True)


def sweep():
    ev = [e for e in map(json.loads, open(EVENTS, encoding="utf-8")) if not e.get("none")]
    rows = []
    for s in SHARES:
        for d in DEVS:
            n = np.array([len(R.sensor_findings(e["contributing_features"], s, d)) for e in ev])
            rows.append({"min_share": s, "min_dev": d, "median": float(np.median(n)), "mean": round(float(n.mean()), 2),
                         "zero": round(float((n == 0).mean()), 3), "at_cap": round(float((n >= R.MAX_FINDINGS).mean()), 3),
                         "ok": bool(1 <= np.median(n) <= 3 and (n == 0).mean() <= MAX_ZERO)})
    cur = next(r for r in rows if r["min_share"] == R.MIN_SHARE and r["min_dev"] == R.MIN_DEV)

    def dist(r):
        return abs(np.log(r["min_share"] / R.MIN_SHARE)) + abs(np.log(r["min_dev"] / R.MIN_DEV))
    ok = [r for r in rows if r["ok"]]
    pick = cur if cur["ok"] else (min(ok, key=dist) if ok else None)
    by_source = {}
    for src in sorted({e["source"] for e in ev}):
        n = [len(R.sensor_findings(e["contributing_features"], (pick or cur)["min_share"], (pick or cur)["min_dev"]))
             for e in ev if e["source"] == src]
        by_source[src] = {"events": len(n), "median": float(np.median(n)), "zero": round(float(np.mean(np.array(n) == 0)), 3)}
    res = {"events": len(ev), "files": len({e["file"] for e in ev}), "current": cur, "pick": pick,
           "by_source_at_pick": by_source, "grid": rows}
    json.dump(res, open(os.path.join(HERE, "symptom_sweep.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(ev)} events from {res['files']} files; MAX_FINDINGS {R.MAX_FINDINGS}")
    print("current", cur)
    print("pick   ", pick)
    print("by source", by_source)
    for r in rows:
        print(f"  share {r['min_share']:.2f} dev {r['min_dev']:.2f}: median {r['median']:.1f} mean {r['mean']:.2f} "
              f"zero {r['zero']:.0%} cap {r['at_cap']:.0%} {'ok' if r['ok'] else ''}")


if __name__ == "__main__":
    {"collect": collect, "sweep": sweep}[sys.argv[1]]()
