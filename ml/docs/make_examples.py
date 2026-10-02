"""Regenerate the example documents in docs/ from real replays through the serving model (main.MODEL_PATH).

    python docs/make_examples.py

anomaly_result.example.json + rag_handoff.example.json: the last handoff of test_3's last 12.2 h (its failure event);
trace.example.json: GET /guns/test_2/trace after test_2's last 12.2 h. Rerun after a model, mapping or schema change.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".py"))
os.environ["RSW_STATE_DIR"] = tempfile.mkdtemp(prefix="rsw_examples_")
os.environ.pop("RSW_RAG_URL", None)
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from replay import replay_file  # noqa: E402


def tail_replay(client, name, hours=12.2):
    path = os.path.join(ROOT, "test", f"{name}.csv")
    n_h = sum(1 for _ in open(path, encoding="utf-8")) / 3600
    events = []
    replay_file(client, path, name, chunk_s=600, start_hours=max(n_h - hours, 0),
                on_result=lambda res, ev: events.append(res) if res.get("handoff") else None)
    return events


def dump(obj, name, indent=2):
    with open(os.path.join(DOCS, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


with TestClient(main.app) as c:
    ev = tail_replay(c, "test_3")
    last = ev[-1]
    handoff = last.pop("handoff")
    dump(last, "anomaly_result.example.json")
    dump(handoff, "rag_handoff.example.json")
    print("test_3 handoffs:", [(e["window_end"], e["critical_source"], e["rule_code"]) for e in ev],
          "situations", handoff["situation_ids"], "findings", [f["text_ko"] for f in handoff["sensor_findings"]])
    ev2 = tail_replay(c, "test_2")
    tr = c.get("/guns/test_2/trace", params={"minutes": 360}).json()
    dump(tr, "trace.example.json", indent=1)
    print("test_2 handoffs:", [(e["window_end"], e["critical_source"], e["rule_code"]) for e in ev2],
          tr["n_points"], [f["feature"] for f in tr["features"]])
