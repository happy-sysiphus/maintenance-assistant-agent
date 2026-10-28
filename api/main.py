"""RSW 정비 지원 api 서버: 화면(ui) ↔ ML 이상 감지 서버 ↔ RAG 매뉴얼 근거를 잇는다.

기존 저장 주소를 유지하고 E 코드 기반 RAG·다문서 인용을 연결한다 (ui/docs/09 답신).
- ML: 고장 알림(핸드오프)을 받아 케이스로 만든다. ML이 push하거나(POST /api/ml/handoffs), api가 ML /handoffs를 당겨 온다.
- RAG: 알림에서는 E 코드·고정 점검을 준비하고 상세 조회에서 한국어 JSON 안내를 생성한다.
- 사람 기록(점검 · 판단 · 조치 · 일지 · 도움 요청 · 사진)은 SQLite에 저장한다 (store.py).

    uvicorn main:app --port 9000       # ML 서버는 ML_URL (기본 http://localhost:8000)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool


import assistant
import guidance
from rag.documents import manuals, manual_path
import store

ML_URL = os.environ.get("ML_URL", "http://localhost:8000").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
MANUALS = {doc: manual_path(doc) for doc in manuals()}
MANUAL_CAUSE = "MANUAL"
TRACE_MINUTES = 30

app = FastAPI(title="RSW 정비 지원 api")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ml_get(path: str) -> Any:
    try:
        r = httpx.get(f"{ML_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"ML 서버에 연결하지 못했습니다: {e}") from e


# ---- ML 고장 알림 → 케이스 ----


def score_trace(gun_id: str, detected_at: str) -> list[dict[str, Any]]:
    """알림 직전 30분의 점수 (ML trace는 최근 6시간만 들고 있어서, 늦게 받으면 빈 목록)."""
    try:
        t = ml_get(f"/guns/{gun_id}/trace?minutes=360")
    except HTTPException:
        return []
    end = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
    out = []
    for p in t.get("points", []):
        ws = datetime.fromisoformat(p["window_start"]).replace(tzinfo=timezone.utc)
        if end - timedelta(minutes=TRACE_MINUTES) < ws <= end:
            out.append(
                {
                    "t": ws.isoformat().replace("+00:00", "Z"),
                    "score": round(p["score"], 4),
                    "threshold": round(p["threshold"], 4),
                    "non_welding_share": p.get("non_welding_share", 0.0),
                    "error_codes": p.get("error_codes", []),
                }
            )
    return out


def ingest(handoff: dict[str, Any]) -> str:
    """알림 하나를 케이스로. 같은 설비 · 같은 코드의 열린 케이스가 있으면 새 줄 대신 '반복'으로 묶는다."""
    eid = handoff["event_id"]
    if store.seen(eid):
        return "duplicate"
    code = handoff["trigger"].get("rule_code")
    same = store.open_case(handoff["gun_id"], code)
    if same:
        same.setdefault("repeats", []).append({"event_id": eid, "detected_at": handoff["detected_at"]})
        store.put(same)
        store.mark_seen(eid, same["case_id"])
        return "repeat"
    d = {
        "case_id": store.next_id(),
        "status": "new",
        # 알림은 ML이 critical일 때만 보낸다 (ml/README 5.10.2)
        "severity": "critical",
        "assignee": None,
        "event": handoff,
        "score_trace": score_trace(handoff["gun_id"], handoff["detected_at"]),
        "guidance": guidance.guidance_for(handoff),
        "records": {"checks": {}, "judgments": [], "actions": [], "results": [], "field": [], "handovers": []},
        "repeats": [],
    }
    store.put(d)
    store.mark_seen(eid, d["case_id"])
    return "new"


_last_sync = 0.0


def sync_from_ml(force: bool = False) -> dict[str, int]:
    """ML /handoffs에서 아직 안 받은 알림을 당겨 온다 (push를 안 켰을 때). 5초에 한 번만."""
    global _last_sync
    if not force and time.time() - _last_sync < 5:
        return {}
    _last_sync = time.time()
    try:
        records = ml_get("/handoffs")
    except HTTPException:
        return {}  # ML이 꺼져 있어도 저장된 케이스는 보여준다
    counts: dict[str, int] = {}
    for r in sorted(records, key=lambda r: r["handoff"]["detected_at"]):
        k = ingest(r["handoff"])
        counts[k] = counts.get(k, 0) + 1
    return counts


@app.post("/api/ml/handoffs")
def receive_handoff(handoff: dict[str, Any]):
    """ML push 받는 곳: ML 서버를 RSW_RAG_URL=http://localhost:9000/api/ml/handoffs 로 켜면 된다."""
    return {"result": ingest(handoff)}


@app.post("/api/ml/sync")
def force_sync():
    return sync_from_ml(force=True)


# ---- 케이스 ----


def summary(d: dict[str, Any]) -> dict[str, Any]:
    e = d["event"]
    fc = e.get("fault_class")
    return {
        "case_id": d["case_id"],
        "status": d["status"],
        "severity": d["severity"],
        "assignee": d["assignee"],
        "event": {
            "event_id": e["event_id"],
            "gun_id": e["gun_id"],
            "detected_at": e["detected_at"],
            "trigger": {k: e["trigger"].get(k) for k in ("source", "rule_code", "rule_trigger_time")},
            "fault_class": {"code": fc["code"], "name_ko": fc["name_ko"]} if fc else None,
            "summary_ko": e.get("summary_ko"),
        },
        "repeat_count": len(d.get("repeats", [])),
    }


def load(case_id: str) -> dict[str, Any]:
    d = store.get(case_id)
    if d is None:
        raise HTTPException(404, "not found")
    return guidance.attach_criteria(d)


def save(d: dict[str, Any]) -> dict[str, Any]:
    store.put(d)
    return d


def touch(d: dict[str, Any]) -> None:
    """사람이 처음 무언가를 저장하면 '점검 중'으로."""
    if d["status"] == "new":
        d["status"] = "in_progress"
    d["records"].setdefault("started_at", now())


@app.get("/api/health")
def health():
    try:
        ml = ml_get("/health")["status"]
    except HTTPException:
        ml = "down"
    return {"status": "ok", "source": "api", "ml": ml, "rag": "e_code_hybrid"}


@app.get("/api/cases")
def list_cases():
    sync_from_ml()
    cases = [summary(d) for d in store.all_cases() if d["status"] not in ("resolved", "false_alarm")]
    cases.sort(key=lambda c: c["event"]["detected_at"], reverse=True)
    return {"cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    enriched = guidance.enrich(load(case_id))
    latest = load(case_id)  # Preserve records saved while generation was running.
    latest["guidance"] = enriched["guidance"]
    return save(latest)


@app.put("/api/cases/{case_id}/checks")
def put_checks(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    touch(d)
    d["records"]["checks"].update(body["checks"])
    return save(d)


def _append(case_id: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    d = load(case_id)
    touch(d)
    d["records"][key].append({**body, "at": now()})
    return d


@app.post("/api/cases/{case_id}/judgments")
def post_judgment(case_id: str, body: dict[str, Any]):
    return save(_append(case_id, "judgments", body))


@app.post("/api/cases/{case_id}/actions")
def post_action(case_id: str, body: dict[str, Any]):
    return save(_append(case_id, "actions", body))


@app.post("/api/cases/{case_id}/results")
def post_result(case_id: str, body: dict[str, Any]):
    d = _append(case_id, "results", body)
    if body["outcome"] == "resolved":
        d["status"] = "log_approved" if (d["records"].get("log") or {}).get("approved_at") else "logging"
    return save(d)


@app.put("/api/cases/{case_id}/log")
def put_log(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    if d["status"] == "resolved":
        raise HTTPException(409, "closed")
    approved = body.pop("approved", False)
    prev = d["records"].get("log") or {}
    t = now()
    d["records"]["log"] = {**body, "saved_at": t, "approved_at": (prev.get("approved_at") or t) if approved else None}
    d["status"] = "log_approved" if approved else "logging"
    return save(d)


@app.post("/api/cases/{case_id}/close")
def close(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    if body["outcome"] == "resolved" and not (d["records"].get("log") or {}).get("approved_at"):
        raise HTTPException(409, "log not approved")
    d["records"]["closure"] = {"outcome": body["outcome"], "at": now()}
    d["status"] = body["outcome"]
    return save(d)


@app.put("/api/cases/{case_id}/manual")
def put_manual(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    touch(d)
    d["records"]["manual"] = {**body, "saved_at": now()}
    return save(d)


@app.post("/api/cases/{case_id}/field")
def post_field(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    touch(d)
    d["records"]["checks"].update(body["checks"])
    d["records"]["field"].append(
        {"by": body["by"], "check_ids": list(body["checks"]), "photos": body.get("photos", []), "at": now()}
    )
    return save(d)


@app.post("/api/cases/{case_id}/handover")
def post_handover(case_id: str, body: dict[str, Any]):
    d = load(case_id)
    t = now()
    d["records"]["handovers"].append({**body, "at": t})
    d["records"]["closure"] = {"outcome": "handed_over", "at": t}
    d["status"] = "handed_over"
    return save(d)


def cause_name(d: dict[str, Any]) -> str | None:
    """맞다고 판단한 원인 이름 (ui/src/lib/caseFlow.ts resolvedCauseName과 같은 규칙)."""
    r = d["records"]
    manual = (r.get("manual") or {}).get("cause") or None
    last = next((x for x in reversed(r["results"]) if x["outcome"] == "resolved"), None)
    if last and last["situation_id"] == MANUAL_CAUSE:
        return manual
    sid = (last or {}).get("situation_id") or next(
        (j["situation_id"] for j in reversed(r["judgments"]) if j["verdict"] == "yes"), None
    )
    c = next((c for c in d["guidance"]["candidates"] if c["situation_id"] == sid), None)
    return c["name"] if c else manual


@app.get("/api/history")
def history():
    items = []
    for d in store.all_cases():
        r = d["records"]
        if not r.get("closure"):
            continue
        action = r["actions"][-1] if r["actions"] else {}
        items.append(
            {
                **summary(d),
                "closure": r["closure"],
                "started_at": r.get("started_at"),
                "cause": cause_name(d),
                "action_kind": action.get("kind"),
                "worker": action.get("worker"),
            }
        )
    items.sort(key=lambda i: i["closure"]["at"], reverse=True)
    return {"items": items}


# ---- 설비 · 매뉴얼 · 챗봇 ----


@app.get("/api/guns")
def guns():
    open_by_gun: dict[str, list[dict[str, Any]]] = {}
    for d in store.all_cases():
        if d["status"] not in ("resolved", "false_alarm"):
            open_by_gun.setdefault(d["event"]["gun_id"], []).append(summary(d))
    out = []
    for g in ml_get("/guns"):
        cases = sorted(open_by_gun.get(g["gun_id"], []), key=lambda c: c["event"]["detected_at"], reverse=True)
        out.append({**g, "open_cases": cases})
    return {"guns": out}


@app.get("/api/manuals/{doc}.pdf")
def manual_pdf(doc: str):
    path = MANUALS.get(doc)
    if path is None or not path.exists():
        raise HTTPException(404, "manual not found")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/manuals/festo/search")
def manual_search(q: str = Query("")):
    return {"items": guidance.search_manuals(q.strip()) if q.strip() else []}



@app.get("/api/situations")
def situations():
    return {"situations": guidance.all_situations()}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class AssistantAskInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "question": "E029 오류의 의미와 점검 방법을 알려주세요.",
        "photos": [],
        "history": [],
    }})
    question: str = Field(min_length=1, max_length=2000)
    photos: list[dict[str, Any]] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list)


@app.post("/api/assistant/ask")
async def ask(body: AssistantAskInput):
    try:
        return await run_in_threadpool(assistant.answer,body.question,body.photos,
                                       [message.model_dump() for message in body.history])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/assistant/feedback")
def feedback(body: dict[str, Any]):
    return {"ok": True}
