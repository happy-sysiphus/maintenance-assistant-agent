"""
Stand-in receiver for ML's two push paths (tests). The real handoff receiver is FE's api (POST /api/ml/handoffs,
which acknowledges with {"result": "new" | "repeat" | "duplicate"}); the per-minute document is not sent in
operation - RAG declined it and ML does not send it (history.md 37).

  POST /diagnose         critical-event handoff (schema rag_handoff v1.2 - main.RagHandoff), answered with an old
                         report shape that no real receiver uses (docs/rag_response.example.json)
  POST /sensor-features  the per-minute sensor document (sensor-features-v4 - .py/sensor_features.py), 204
  GET  /received         handoffs received, newest last          GET /received-sensors  sensor documents, newest last

Every arrival is printed as one console line and appended to results/rag_received/<gun_id>.jsonl (handoffs) or
results/rag_received/<gun_id>_sensors.jsonl (sensor documents), so you can see exactly what a receiver gets.

    uvicorn mock_rag:app --app-dir .py --port 8001                       # from the ml folder
    set RSW_RAG_URL=http://127.0.0.1:8001/diagnose                       # Windows cmd (bash: export ...)
    set RSW_RAG_SENSOR_URL=http://127.0.0.1:8001/sensor-features
    uvicorn main:app                                                     # another window, ml folder
    python .py/replay.py test/test_0.csv --start-hours 150 --chunk-s 60 --speed 60

A real receiver has to drop a known event_id / (gun_id, observed_at) it already has (main.py retries; handoffs
carry Idempotency-Key: <event_id>, sensor documents Idempotency-Key: <gun_id>|<observed_at>).
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

from fastapi import FastAPI, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "rag_received")  # RSW_MOCK_RAG_OUT overrides (tests write to a temp folder)
app = FastAPI(title="mock RAG (RSW)", version="0.3.0")
RECEIVED: deque[dict[str, Any]] = deque(maxlen=200)
SENSORS: deque[dict[str, Any]] = deque(maxlen=2000)
ANSWERED: dict[str, dict[str, Any]] = {}  # event_id -> response (duplicates are not processed twice)
SEEN_SENSORS: set[tuple[str, str]] = set()
TODO = "[mock] RAG가 매뉴얼에서 채울 항목"


def _log(name: str, doc: dict[str, Any]) -> None:
    out = os.environ.get("RSW_MOCK_RAG_OUT") or OUT
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def report(h: dict[str, Any]) -> dict[str, Any]:
    fc = h.get("fault_class")
    return {
        "detected_anomaly": (f"{h['gun_id']} {h['detected_at']} - {h.get('summary_ko', '')} "
                             f"(트리거: {h['trigger']['source']})"),
        "fault_class": fc and {"code": fc["code"], "terminal_code": fc["terminal_code"]},
        "cause_candidates": [TODO],
        "manual_references": [],
        "recommended_actions": [TODO],
        "confidence_and_limits": ([f"고장 유형 근거: {fc['basis']}"] if fc else []) + h.get("caveats", []),
    }


@app.post("/diagnose")
async def diagnose(handoff: dict[str, Any]) -> dict[str, Any]:
    """A known event_id is answered again but not processed again - the real service must do the same."""
    eid = handoff.get("event_id")
    if eid in ANSWERED:
        return {**ANSWERED[eid], "duplicate": True}
    RECEIVED.append(handoff)
    fc = handoff.get("fault_class") or {}
    print(f"[handoff] {handoff.get('gun_id')} {handoff.get('detected_at')} class={fc.get('code')} "
          f"code={fc.get('terminal_code')} | {handoff.get('summary_ko')}", flush=True)
    _log(str(handoff.get("gun_id")), handoff)
    ANSWERED[eid] = {"event_id": eid, "status": "ok", "generator": "mock", "report": report(handoff)}
    return ANSWERED[eid]


@app.post("/sensor-features", status_code=204)
async def sensor_features(doc: dict[str, Any]) -> Response:
    key = (str(doc.get("gun_id")), str(doc.get("observed_at")))
    if key not in SEEN_SENSORS:
        SEEN_SENSORS.add(key)
        SENSORS.append(doc)
        top = " ".join(f"{m['feature']}={m['z']:+.2f}" for m in doc.get("measurements", [])[:4] if m["z"] is not None)
        states = " ".join(f"{s['signal']}={s['on_share']:.0%}" for s in doc.get("states", []) if s["on_share"] is not None)
        print(f"[sensor] {key[0]} {key[1]} code={doc.get('error_code')} norm={doc.get('gun_norm')} {top} | {states}",
              flush=True)
        _log(f"{key[0]}_sensors", doc)
    return Response(status_code=204)


@app.get("/received")
async def received(limit: int = 20, full: bool = False) -> list[dict[str, Any]]:
    docs = list(RECEIVED)[-limit:]
    if full:
        return docs
    return [{"event_id": h.get("event_id"), "gun_id": h.get("gun_id"), "detected_at": h.get("detected_at"),
             "summary_ko": h.get("summary_ko"), "fault_class": (h.get("fault_class") or {}).get("code")} for h in docs]


@app.get("/received-sensors")
async def received_sensors(limit: int = 20) -> list[dict[str, Any]]:
    return list(SENSORS)[-limit:]
