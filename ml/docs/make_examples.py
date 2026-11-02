"""Regenerate the example documents in docs/ from real replays through the serving model (main.MODEL_PATH).

    python docs/make_examples.py

anomaly_result.example.json + rag_handoff.example.json: the last handoff of test_3's last 12.2 h (its failure event);
trace.example.json: GET /guns/test_2/trace after test_2's last 12.2 h; sensor_features.example.json: the per-minute
sensor document (D-6) of test_3's failure-event minute. Rerun after a model, mapping or schema change.
rag_handoff_early_warning.example.json (C-8): the first early-warning handoff of test_5's last 12.2 h, served with
RSW_EARLY_WARNING_HANDOFFS=1 by a model with an early warning - the serving model, or RSW_EW_EXAMPLE_MODEL (e.g. a
candidate bundle before it replaces the serving one); skipped when neither has one.
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
os.environ.pop("RSW_EARLY_WARNING_HANDOFFS", None)  # the critical examples come from the default (off)
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from replay import replay_file  # noqa: E402


def tail_replay(client, name, hours=12.2):
    path = os.path.join(ROOT, "test", f"{name}.csv")
    n_h = sum(1 for _ in open(path, encoding="utf-8")) / 3600
    events = []
    replay_file(client, path, name, chunk_s=60, start_hours=max(n_h - hours, 0),  # a 1-min client, as in operation
                on_result=lambda res, ev: events.append(res) if res.get("handoff") else None)
    return events


def dump(obj, name, indent=2):
    with open(os.path.join(DOCS, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)


with TestClient(main.app) as c:
    ev = tail_replay(c, "test_3")
    last = ev[-1]
    handoff = last.pop("handoff")
    dump(last["sensor_features"][-1], "sensor_features.example.json")
    dump(last, "anomaly_result.example.json")
    dump(handoff, "rag_handoff.example.json")
    print("test_3 handoffs:", [(e["window_end"], e["critical_source"], e["rule_code"]) for e in ev],
          "fault_class", (handoff["fault_class"] or {}).get("code"), "findings", [f["text_ko"] for f in handoff["sensor_findings"]])
    ev2 = tail_replay(c, "test_2")
    tr = c.get("/guns/test_2/trace", params={"minutes": 360}).json()
    dump(tr, "trace.example.json", indent=1)
    print("test_2 handoffs:", [(e["window_end"], e["critical_source"], e["rule_code"]) for e in ev2],
          tr["n_points"], [f["feature"] for f in tr["features"]])

if os.environ.get("RSW_EW_EXAMPLE_MODEL"):
    os.environ["RSW_MODEL_PATH"] = os.environ["RSW_EW_EXAMPLE_MODEL"]
os.environ["RSW_EARLY_WARNING_HANDOFFS"] = "1"
with TestClient(main.app) as c:  # a new app start: the model and the flag are read again
    if c.app.state.detector.early_warning is None:
        print("no early warning in", c.app.state.detector.path, "- rag_handoff_early_warning.example.json not written")
    else:
        ev = [e for e in tail_replay(c, "test_5") if e["handoff"]["event_type"] == "early_warning"]
        dump(ev[0]["handoff"], "rag_handoff_early_warning.example.json")
        print("test_5 early-warning handoffs:", len(ev), "from", c.app.state.detector.path)
