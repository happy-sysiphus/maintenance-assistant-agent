"""
Stand-in receiver for ML's handoff push (tests). The real receiver is FE's api (POST /api/ml/handoffs, which
acknowledges with {"result": "new" | "repeat" | "duplicate"}). The per-minute sensor document is pushed nowhere -
RAG declined it and the push was removed (history.md 37).

  POST /diagnose         critical-event handoff (schema rag_handoff v1.2 - main.RagHandoff), answered with an old
                         report shape that no real receiver uses (docs/rag_response.example.json)
  GET  /received         handoffs received, newest last

Every arrival is printed as one console line and appended to results/rag_received/<gun_id>.jsonl, so you can see
exactly what a receiver gets.

    uvicorn mock_rag:app --app-dir .py --port 8001                       # from the ml folder
    set RSW_RAG_URL=http://127.0.0.1:8001/diagnose                       # Windows cmd (bash: export ...)
    uvicorn main:app                                                     # another window, ml folder
    python .py/replay.py test/test_0.csv --start-hours 150 --chunk-s 60 --speed 60

A real receiver has to drop a known event_id it already has (main.py retries; handoffs carry
Idempotency-Key: <event_id>).
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

from fastapi import FastAPI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "rag_received")  # RSW_MOCK_RAG_OUT overrides (tests write to a temp folder)
app = FastAPI(title="mock RAG (RSW)", version="0.3.0")
RECEIVED: deque[dict[str, Any]] = deque(maxlen=200)
ANSWERED: dict[str, dict[str, Any]] = {}  # event_id -> response (duplicates are not processed twice)
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


@app.get("/received")
async def received(limit: int = 20, full: bool = False) -> list[dict[str, Any]]:
    docs = list(RECEIVED)[-limit:]
    if full:
        return docs
    return [{"event_id": h.get("event_id"), "gun_id": h.get("gun_id"), "detected_at": h.get("detected_at"),
             "summary_ko": h.get("summary_ko"), "fault_class": (h.get("fault_class") or {}).get("code")} for h in docs]
