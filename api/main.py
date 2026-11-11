"""RSW 정비 지원 api 서버: 화면(ui) ↔ ML 이상 감지 서버 ↔ RAG 매뉴얼 근거를 잇는다.

기존 저장 주소를 유지하고 E 코드 기반 RAG·다문서 인용을 연결한다 (ui/docs/09 답신).
- ML: 고장 알림(핸드오프)을 받아 케이스로 만든다. ML이 push하거나(POST /api/ml/handoffs), api가 ML /handoffs를 당겨 온다.
- RAG: CRITICAL은 E 코드 안내를 유지한다. WARNING은 센서 소견을 조회하고 별도 POST로 후보·수동 검색·해결 기록을 연결한다.
- 사람 기록(점검 · 판단 · 조치 · 일지 · 도움 요청 · 사진)은 SQLite에 저장한다 (store.py).

    uvicorn main:app --port 9000       # ML 서버는 ML_URL (기본 http://localhost:8000)
"""

from __future__ import annotations

import functools
import os
import threading
import time
from contextlib import nullcontext, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Header
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.routing import APIRoute
from starlette.routing import Match


import assistant
import field_memory
import guidance
from rag.documents import manuals, manual_path
import store
import warning_bridge

ML_URL = os.environ.get("ML_URL", "http://localhost:8000").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
MANUALS = {doc: manual_path(doc) for doc in manuals()}
MANUAL_CAUSE = "MANUAL"
TRACE_MINUTES = 30
# 현장 기억을 켜면 ingest 전체(seen → put)를 한 번에 하나씩 (store._lock을 다시 잡으면 교착)
_INGEST_LOCK = threading.Lock()
# 케이스를 고치는 곳(읽기 → 고치기 → 저장)은 한 번에 하나씩 (11/04 FE, ml/docs/hitl-loop.md ④ 요청 9 · 10).
# 겹치면 나중 저장이 먼저 저장한 기록(예: 닫기)을 옛 내용으로 덮는다. 상세 조회의 RAG 생성은 이 잠금 밖에서 한다.
# store._lock과는 다른 잠금이고, store 함수 안에서는 잡지 않으므로 교착이 없다
_WRITE_LOCK = threading.RLock()


def locked(fn):
    """케이스를 고치는 주소에 붙인다: 함수 전체를 _WRITE_LOCK 안에서 돌린다."""

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with _WRITE_LOCK:
            return fn(*args, **kwargs)

    return run

@asynccontextmanager
async def lifespan(app):
    yield
    warning_bridge.shutdown()


app = FastAPI(title="RSW 정비 지원 api", lifespan=lifespan)


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
    # 반복을 붙이는 저장이 같은 케이스의 닫기 · 기록과 겹치지 않게 _WRITE_LOCK도 잡는다 (플래그와 상관없이)
    with _INGEST_LOCK if field_memory.enabled() else nullcontext(), _WRITE_LOCK:
        eid = handoff["event_id"]
        if store.seen(eid):
            return "duplicate"
        same = store.open_case(handoff["gun_id"], store.group_key(handoff))
        if same:
            same.setdefault("repeats", []).append({"event_id": eid, "detected_at": handoff["detected_at"], "handoff": handoff})
            store.put(same)
            store.mark_seen(eid, same["case_id"])
            return "repeat"
        card = guidance.guidance_for(handoff)
        # R2(현장 기억): 닫힌 케이스의 '이상' 점검 기록으로 이 케이스의 점검 순서를 한 번 정한다 (field_memory.order_for)
        order = None
        if field_memory.enabled():
            order = field_memory.order_for(guidance.event_code(handoff), store.all_cases())
        d = {
            "case_id": store.next_id(),
            "status": "new",
            # 조기 경보 핸드오프(event_type early_warning, ML RSW_EARLY_WARNING_HANDOFFS=1)는 주의, 나머지는 긴급
            "severity": "warning" if handoff.get("event_type") == "early_warning" else "critical",
            "assignee": None,
            "event": handoff,
            "score_trace": score_trace(handoff["gun_id"], handoff["detected_at"]),
            "guidance": field_memory.apply_order(card, order["lead"]) if order else card,
            "records": {"checks": {}, "judgments": [], "actions": [], "results": [], "field": [], "handovers": []},
            "repeats": [],
        }
        if order:
            d["field_order"] = order
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
            # v1.2까지의 핸드오프에는 event_type이 없다: 모두 critical
            "event_type": e.get("event_type") or "critical",
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


CLOSED = ("resolved", "false_alarm")


def load_open(case_id: str) -> dict[str, Any]:
    """기록을 바꾸는 주소용 load. 해결 종료 · 오알람으로 닫힌 케이스는 409 (읽기 전용, 11/04 FE).
    다시 닫으면 마지막 closure가 현장 기억 라벨을 덮어쓰기 때문이다 (ml/docs/hitl-loop.md ⑥).
    미해결 · 도움 요청은 이어서 하므로 막지 않는다."""
    d = load(case_id)
    if d["status"] in CLOSED:
        raise HTTPException(409, "closed")
    return d


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
    out = {"status": "ok", "source": "api", "ml": ml, "rag": "e_code_hybrid"}
    if field_memory.enabled():
        out["memory"] = "on"
    return out


@app.get("/api/cases")
def list_cases():
    sync_from_ml()
    cases = [summary(d) for d in store.all_cases() if d["status"] not in ("resolved", "false_alarm")]
    cases.sort(key=lambda c: c["event"]["detected_at"], reverse=True)
    return {"cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    current = load(case_id)
    if current['event'].get('event_type') == 'early_warning':
        try:
            result = warning_bridge.inspect(current)
        except HTTPException as exc:
            result = {'status': 'invalid_input', 'event_id': current['event']['event_id'],
                      'observations': [], 'automatic_card': None, 'detail': exc.detail,
                      'feedback_enabled': warning_bridge.feedback_enabled()}
        # Persist automatic cards as server-verified choices. Never overwrite concurrent records.
        with _WRITE_LOCK:
            latest = load(case_id)
            if not store.warning_resolution_for_case(case_id) and result.get('status') != 'invalid_input':
                warning_bridge.snapshot(latest, result['event_id'], result, 'inspection')
                store.put(latest)
        return {**latest, 'warning_rag': result}
    # R2로 순서를 정한 케이스에 판단 · 조치 · 결과가 있으면 카드를 다시 만들지 않는다 (원인 번호가 카드 위치라서)
    enriched = current if field_memory.order_pinned(current) else guidance.enrich(current)  # RAG 생성(수 초)은 잠금 밖에서
    with _WRITE_LOCK:
        latest = load(case_id)  # Preserve records saved while generation was running.
        card = enriched["guidance"]
        order = latest.get("field_order")
        if order:
            # 새 RAG 카드에도 같은 순서를 다시 적용한다. 생성 중에 기록이 들어왔으면 저장된 카드를 지킨다
            pinned = field_memory.order_pinned(latest)
            card = latest["guidance"] if pinned else field_memory.apply_order(card, order["lead"])
        # 안내가 바뀌었을 때만 저장한다 (11/04): 그냥 열 때마다 저장하면 같은 순간의 닫기 · 기록을 덮을 수 있다
        if latest["guidance"] != card:
            latest["guidance"] = card
            save(latest)
    out = latest
    # 현장 기억은 볼 때마다 새로 세어 사본에만 붙인다 (field_memory는 DB에 저장하지 않는다. 순서는 ingest가 정한다, R2)
    if not field_memory.enabled():
        return out
    cases = store.all_cases()
    out = field_memory.annotate(out, cases)
    out["field_memory"]["recent"] = recent_closures(out, cases)
    return out


RECENT_MAX = 3


def recent_closures(d: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """같은 설비 · 코드(field_memory.alarm_key)로 오알람 · 해결 종료된 케이스, 최근 닫은 순 RECENT_MAX건 (UI 이전 기록).

    field_memory.alarm의 건수와 같은 범위(라벨 있는 닫힘만, 자기 자신 제외)다.
    해결이면 맞다고 판단한 원인 이름(cause), 오알람이면 닫을 때 남긴 사유(reason)를 준다.
    """
    key = field_memory.alarm_key(d["event"])
    rows = []
    for c in cases:
        if c["case_id"] == d["case_id"] or field_memory.alarm_key(c["event"]) != key:
            continue
        if field_memory.alarm_label(c) is None:
            continue
        closure = c["records"]["closure"]
        rows.append(
            {
                "case_id": c["case_id"],
                "outcome": closure["outcome"],
                "closed_at": closure["at"],
                "cause": cause_name(c) if closure["outcome"] == "resolved" else None,
                "reason": closure.get("reason"),
            }
        )
    rows.sort(key=lambda r: r["closed_at"], reverse=True)
    return rows[:RECENT_MAX]


@app.put("/api/cases/{case_id}/checks")
@locked
def put_checks(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    touch(d)
    d["records"]["checks"].update(body["checks"])
    return save(d)


def _append(case_id: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
    d = load_open(case_id)
    touch(d)
    d["records"][key].append({**body, "at": now()})
    return d


@app.post("/api/cases/{case_id}/judgments")
@locked
def post_judgment(case_id: str, body: dict[str, Any]):
    return save(_append(case_id, "judgments", body))


@app.post("/api/cases/{case_id}/actions")
@locked
def post_action(case_id: str, body: dict[str, Any]):
    return save(_append(case_id, "actions", body))


@app.post("/api/cases/{case_id}/results")
@locked
def post_result(case_id: str, body: dict[str, Any]):
    d = _append(case_id, "results", body)
    if body["outcome"] == "resolved":
        d["status"] = "log_approved" if (d["records"].get("log") or {}).get("approved_at") else "logging"
    return save(d)


@app.put("/api/cases/{case_id}/log")
@locked
def put_log(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    if d["status"] == "resolved":
        raise HTTPException(409, "closed")
    approved = body.pop("approved", False)
    prev = d["records"].get("log") or {}
    t = now()
    d["records"]["log"] = {**body, "saved_at": t, "approved_at": (prev.get("approved_at") or t) if approved else None}
    d["status"] = "log_approved" if approved else "logging"
    return save(d)


@app.post("/api/cases/{case_id}/close")
@locked
def close(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    if body["outcome"] == "resolved" and not (d["records"].get("log") or {}).get("approved_at"):
        raise HTTPException(409, "log not approved")
    closure = {"outcome": body["outcome"], "at": now()}
    if field_memory.enabled():
        # 오알람 이유 · 메모 (빈 문자열은 남기지 않는다)
        for k in ("reason", "note"):
            v = body.get(k)
            if isinstance(v, str) and v.strip():
                closure[k] = v
    d["records"]["closure"] = closure
    if d['event'].get('event_type') == 'early_warning' and body['outcome'] == 'resolved':
        # Existing UI posts /close. Save the human record without inventing a verified RAG selection.
        # /warning/resolve remains available when the UI supplies event-specific verified evidence.
        d['warning_resolution_status'] = 'not_linked'
    d["status"] = body["outcome"]
    return save(d)


@app.put("/api/cases/{case_id}/manual")
@locked
def put_manual(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    touch(d)
    d["records"]["manual"] = {**body, "saved_at": now()}
    return save(d)


@app.post("/api/cases/{case_id}/field")
@locked
def post_field(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    touch(d)
    d["records"]["checks"].update(body["checks"])
    d["records"]["field"].append(
        {"by": body["by"], "check_ids": list(body["checks"]), "photos": body.get("photos", []), "at": now()}
    )
    return save(d)


@app.post("/api/cases/{case_id}/handover")
@locked
def post_handover(case_id: str, body: dict[str, Any]):
    d = load_open(case_id)
    t = now()
    d["records"]["handovers"].append({**body, "at": t})
    d["records"]["closure"] = {"outcome": "handed_over", "at": t}
    d["status"] = "handed_over"
    return save(d)


def cause_name(d: dict[str, Any]) -> str | None:
    """맞다고 판단한 원인 이름 (ui/src/lib/caseFlow.ts resolvedCauseName과 같은 규칙)."""
    r = d["records"]
    if d['event'].get('event_type') == 'early_warning' and r.get('closure', {}).get('outcome') == 'resolved':
        return (r.get('log') or {}).get('cause')
    manual = (r.get("manual") or {}).get("cause") or None
    last = next((x for x in reversed(r["results"]) if x["outcome"] == "resolved"), None)
    if last and last["situation_id"] == MANUAL_CAUSE:
        return manual
    sid = (last or {}).get("situation_id") or next(
        (j["situation_id"] for j in reversed(r["judgments"]) if j["verdict"] == "yes"), None
    )
    c = next((c for c in d["guidance"]["candidates"] if c["situation_id"] == sid), None)
    if c:
        return c["name"]
    # 화면은 RAG 원인 목록(guidance.causes)을 원인 카드로 풀어 "E016-2"(코드-원인 번호)로 판단을 남긴다 (ui/src/lib/guidance.ts)
    code, _, n = (sid or "").rpartition("-")
    causes = d["guidance"].get("causes") or []
    if code and n.isdigit() and 1 <= int(n) <= len(causes):
        return causes[int(n) - 1]["name"]
    return manual


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
                # 반복으로 묶인 알림 (ml/.py/feedback_loop.py가 케이스와 ML 알림을 잇는다)
                "repeat_event_ids": [r["event_id"] for r in d.get("repeats", [])],
            }
        )
    items.sort(key=lambda i: i["closure"]["at"], reverse=True)
    return {"items": items}


class _MemoryRoute(APIRoute):
    """FIELD_MEMORY를 끄면 라우터가 이 주소를 못 본다 (다른 메서드 405 · 끝 슬래시 307 없이, 없는 주소처럼 404)."""

    def matches(self, scope):
        if not field_memory.enabled():
            return Match.NONE, {}
        return super().matches(scope)


def memory_summary():
    """현장 기억 요약 (FIELD_MEMORY=1일 때만, 끄면 없는 주소처럼 404)."""
    if not field_memory.enabled():
        raise HTTPException(404, "Not Found")
    return {"enabled": True, **field_memory.snapshot(store.all_cases())}


app.router.add_api_route("/api/memory", memory_summary, methods=["GET"], include_in_schema=False,
                         route_class_override=_MemoryRoute)


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
    # 현장 기억(FIELD_MEMORY=1)이면 요약을 memory=로 넘긴다. 끄면 인자는 그대로 셋
    extra: dict[str, Any] = {}
    if field_memory.enabled():
        extra["memory"] = await run_in_threadpool(lambda: field_memory.snapshot(store.all_cases()))
    try:
        return await run_in_threadpool(assistant.answer,body.question,body.photos,
                                       [message.model_dump() for message in body.history],**extra)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


FEEDBACK_KEYS = ("conversation_id", "message_index", "helpful", "question", "answer", "status", "situation_ids")


@app.post("/api/assistant/feedback")
def feedback(body: dict[str, Any]):
    """챗봇 답 👍/👎 (11/04 FE, ml/docs/hitl-loop.md ④ 요청 5). SQLite feedback 표에 남긴다.
    question · answer가 없는 옛 화면 본문도 받는다(빈 값으로 저장). 현장 기억 · RAG에는 아직 쓰지 않는다."""
    if not isinstance(body.get("conversation_id"), str) or not isinstance(body.get("message_index"), int):
        raise HTTPException(422, "conversation_id · message_index가 필요합니다")
    row = {k: body.get(k) for k in FEEDBACK_KEYS}
    row["helpful"] = bool(row["helpful"])
    row["at"] = now()
    store.put_feedback(row)
    return {"ok": True}


@app.get("/api/assistant/feedback")
def feedback_list():
    """모인 평가 (최근 순). RAG 담당이 👎 받은 질문 · 답을 볼 때 쓴다."""
    return {"items": store.all_feedback()}

# ---- Early warning: full raw event → RAG → verified choice → atomic SQLite close ----
class WarningEventInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str = Field(min_length=1, max_length=200)


class WarningSearchInput(WarningEventInput):
    question: str = Field(min_length=1, max_length=2000)
    explain: bool = False


class WarningSelectInput(WarningEventInput):
    manual_case_id: str | None = None
    evidence_id: str | None = None


class WarningResolveInput(WarningEventInput):
    actual_action: str = Field(min_length=1, max_length=3000)
    worker: str = Field(min_length=1, max_length=200)
    worker_confirmed: bool
    log_approved: bool
    checks: dict[str, Literal['normal', 'abnormal', 'unchecked']] = Field(default_factory=dict)


def warning_result(case_id, event_id, operation, **kwargs):
    event = warning_bridge.event_for(load_open(case_id), event_id)
    try:
        result = getattr(warning_bridge.service(), operation)(event, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        # Do not leak credentials/provider error bodies through HTTP.
        raise HTTPException(503, 'RAG connection unavailable; retry or record a manual check')
    with _WRITE_LOCK:
        latest = load_open(case_id)
        warning_bridge.snapshot(latest, event_id, result, operation)
        store.put(latest)
    return result


@app.get('/api/cases/{case_id}/warning')
def warning_inspect(case_id: str, event_id: str | None = None):
    current = load(case_id)
    result = warning_bridge.inspect(current, event_id)
    with _WRITE_LOCK:
        latest = load(case_id)
        if latest['status'] not in CLOSED:
            warning_bridge.snapshot(latest, result['event_id'], result, 'inspection')
            store.put(latest)
    return result


@app.post('/api/cases/{case_id}/warning/candidates')
def warning_candidates(case_id: str, body: WarningEventInput):
    return warning_result(case_id, body.event_id, 'generate_candidates')


@app.post('/api/cases/{case_id}/warning/manual')
def warning_manual(case_id: str, body: WarningSearchInput):
    return warning_result(case_id, body.event_id, 'search_manuals', question=body.question, explain=body.explain)


@app.post('/api/cases/{case_id}/warning/select')
@locked
def warning_select(case_id: str, body: WarningSelectInput):
    d = load_open(case_id)
    warning_bridge.event_for(d, body.event_id)
    if bool(body.manual_case_id) == bool(body.evidence_id):
        raise HTTPException(422, 'select exactly one case or manual source')
    state = d.get('warning_events', {}).get(body.event_id, {})
    if body.manual_case_id:
        if body.manual_case_id not in state.get('shown_cards', {}):
            raise HTTPException(422, 'case was not displayed for this event')
        selection = {'manual_case_id': body.manual_case_id}
    else:
        source = state.get('shown_sources', {}).get(body.evidence_id)
        if not source:
            raise HTTPException(422, 'source was not returned by this event search')
        selection = {'manual_source': source}
    if state.get('selection') != selection and state.get('draft'):
        state['draft']['checks'] = {}
        state['draft']['selection'] = selection
    state['selection'] = selection
    store.put(d)
    return selection


@app.post('/api/cases/{case_id}/warning/reject')
@locked
def warning_reject(case_id: str, body: WarningEventInput):
    d = load_open(case_id)
    warning_bridge.event_for(d, body.event_id)
    state = d.get('warning_events', {}).get(body.event_id, {})
    card = state.get('inspection', {}).get('automatic_card')
    if not card:
        raise HTTPException(409, 'no displayed automatic card')
    state['rejected_auto_case_id'] = card['manual_case_id']
    if state.get('selection', {}).get('manual_case_id') == card['manual_case_id']:
        state.pop('selection', None)
    store.put(d)
    return {'status': 'rejected'}


@app.post('/api/cases/{case_id}/warning/resolve')
@locked
def warning_resolve(case_id: str, body: WarningResolveInput):
    d = load(case_id)
    event = warning_bridge.event_for(d, body.event_id)
    previous = store.warning_resolution_for_case(case_id)
    if previous:
        if (previous['event_id'] == body.event_id and previous['actual_action'] == body.actual_action.strip()
                and d.get('warning_close_worker') == body.worker.strip()
                and d.get('warning_events', {}).get(body.event_id, {}).get('draft', {}).get('checks', {}) == body.checks
                and body.worker_confirmed and body.log_approved):
            return d
        raise HTTPException(409, 'resolution already saved with different content')
    d = load_open(case_id)
    if not body.worker_confirmed or not body.log_approved or not body.actual_action.strip() or not body.worker.strip():
        raise HTTPException(422, 'worker, actual action, resolution confirmation and log approval required')
    state = d.get('warning_events', {}).get(body.event_id, {})
    selection = state.get('selection')
    if not selection:
        raise HTTPException(422, 'select a displayed case or manual source first')
    t = now()
    service = warning_bridge.service()
    service.trusted_sources = state.get('shown_sources', {})
    card = state.get('shown_cards', {}).get(selection.get('manual_case_id'), {})
    allowed = {step.get('id', str(i)) for i, step in enumerate(card.get('checks', []))}
    if set(body.checks) - allowed:
        raise HTTPException(422, 'check does not belong to the selected card')
    rejected = state.get('rejected_auto_case_id')
    if rejected and rejected == selection.get('manual_case_id'):
        raise HTTPException(422, 'choose another case after rejecting the automatic card')
    try:
        result = service.record_resolution(event, case_id=case_id, actual_action=body.actual_action.strip(),
            worker_confirmed=True, closed_at=t, rejected_auto_case_id=rejected, **selection)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception:
        raise HTTPException(503, 'manual verification unavailable; records were not closed, retry later')
    title = (state.get('shown_cards', {}).get(selection.get('manual_case_id')) or {}).get('title', '수동 매뉴얼 근거 적용')
    state['draft'] = {'worker': body.worker.strip(), 'actual_action': body.actual_action.strip(),
                      'checks': body.checks, 'selection': selection, 'saved_at': t}
    r = d['records']
    sid = selection.get('manual_case_id', 'MANUAL')
    r['judgments'].append({'situation_id': sid, 'verdict': 'yes', 'event_id': body.event_id, 'at': t})
    r['actions'].append({'situation_id': sid, 'kind': '기타', 'did': body.actual_action.strip(), 'parts': [],
                         'worker': body.worker.strip(), 'started_at': t, 'ended_at': t, 'at': t})
    r['results'].append({'situation_id': sid, 'outcome': 'resolved', 'event_id': body.event_id, 'at': t})
    r['log'] = {'date': datetime.fromisoformat(t.replace('Z', '+00:00')).astimezone(timezone(timedelta(hours=9))).date().isoformat(), 'work_time': '', 'worker_gun': body.worker.strip(), 'problem': '조기 경보 · 원인 미확정',
                'cause': title, 'action': body.actual_action.strip(), 'missed_checks': '', 'recurrence': '모름',
                'saved_at': t, 'approved_at': t}
    r['closure'] = {'outcome': 'resolved', 'at': t}
    d['status'] = 'resolved'
    d['warning_close_worker'] = body.worker.strip()
    d['warning_resolution_status'] = result['status']
    try:
        store.put(d, warning_resolution=service.memory.staged)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return d



class WarningDraftInput(WarningEventInput):
    worker: str = Field(default='', max_length=200)
    actual_action: str = Field(default='', max_length=3000)
    checks: dict[str, Literal['normal', 'abnormal', 'unchecked']] = Field(default_factory=dict)


@app.put('/api/cases/{case_id}/warning/draft')
@locked
def warning_draft(case_id: str, body: WarningDraftInput):
    d = load_open(case_id)
    warning_bridge.event_for(d, body.event_id)
    state = d.setdefault('warning_events', {}).setdefault(body.event_id, {})
    selection = state.get('selection', {})
    card = state.get('shown_cards', {}).get(selection.get('manual_case_id'), {})
    allowed = {step.get('id', str(i)) for i, step in enumerate(card.get('checks', []))}
    if set(body.checks) - allowed:
        raise HTTPException(422, 'check does not belong to the selected card')
    state['draft'] = {**body.model_dump(exclude={'event_id'}), 'selection': selection, 'saved_at': now()}
    store.put(d)
    return state['draft']

class WarningReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    existing_case_id: str | None = None
    title: str | None = None
    equipment_scope: str | None = None
    checks: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


def require_warning_reviewer(token):
    import secrets
    expected = os.getenv('RAG_WARNING_REVIEW_TOKEN', '')
    if not expected:
        raise HTTPException(404, 'review API disabled')
    if not token or not secrets.compare_digest(expected, token):
        raise HTTPException(403, 'review token required')


@app.post('/api/warning/resolutions/{event_id}/review')
@locked
def warning_review(event_id: str, body: WarningReviewInput,
                   x_warning_review_token: str | None = Header(default=None)):
    require_warning_reviewer(x_warning_review_token)
    kwargs = {'existing_case_id': body.existing_case_id} if body.existing_case_id else {
        'title': body.title, 'equipment_scope': body.equipment_scope, 'checks': body.checks, 'actions': body.actions}
    try:
        return warning_bridge.service().review_resolution(event_id, **kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post('/api/warning/resolutions/{event_id}/acknowledge')
@locked
def warning_acknowledge(event_id: str, x_warning_review_token: str | None = Header(default=None)):
    require_warning_reviewer(x_warning_review_token)
    try:
        warning_bridge.service().memory.acknowledge_counterexample(event_id)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {'status': 'reviewed'}
