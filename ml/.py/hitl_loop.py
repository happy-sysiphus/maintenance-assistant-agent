"""
D-8 HITL 피드백 루프 드라이버: 시험 CSV를 재생하다 알림이 오면 멈추고, 사람이 화면에서 케이스를 닫으면 이어 간다.

Feeder 스레드가 replay.replay_file()로 파일을 시작 시각 순으로 재생하고(gun = 파일 이름), 새 critical 이벤트마다
("alert", seq, gun, event)를 큐에 넣은 뒤 release(seq)가 올 때까지 기다린다. 큐를 읽는 쪽(LangGraph 그래프)은
api 케이스가 닫히면 release()로 재생을 풀어 준다. 두 쪽은 HTTP로만 연결되고, 이 파일은 api 모듈을 import하지 않는다.

    cd ml && RSW_STATE_DIR=state/d8-<run> uvicorn main:app --port 8000   # ML (RSW_RAG_URL 없이, 새 폴더)
    cd api && FIELD_MEMORY=1 API_DB=state/d8-<run>.db ../.venv/Scripts/python.exe -m uvicorn main:app --port 9000
    py -3.11 .py/hitl_loop.py test/test_0.csv              # ml/에서: 알림마다 멈춤, 화면에서 닫으면 이어 감
    py -3.11 .py/hitl_loop.py test/test_0.csv --dry-run    # api 없이 끝까지 재생하고 gun별 핸드오프 건수만

api는 rag를 import하므로 RAG 의존성(neo4j 등)이 있는 인터프리터로 켠다: 저장소 .venv, 또는
`pip install -r api/requirements.txt` 뒤의 python. 현장 기억(FIELD_MEMORY=1)과 새 DB(API_DB)로 켠다.
드라이버는 py -3.11(langgraph, pandas)로 돌린다. rag/.env가 없으면 안내 카드는 매핑에서 오고 RAG 글은 오류로
보이지만, 루프(멈춤 → 사람이 닫음 → 다음 알림의 안내 문장)는 그대로 돈다.
안내 카드(E 코드 하나)는 api가 정한 그대로 보여 주고, 현장 기억은 GET 케이스의 field_memory(문장)로만 읽는다.
GET /api/cases/{id}는 RAG를 부를 수 있어(키가 있으면 수 초) 닫힘은 GET /api/history로 기다린다.

산출물은 results/D-8/runs/<시각>/: loop_view.html(2초마다 새로고침되는 현황), loop_log.jsonl, summary.md, workflow.mmd.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import queue
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, TypedDict

import httpx
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay  # noqa: E402

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    LANGGRAPH_ERROR: ImportError | None = None
except ImportError as e:  # main()이 langgraph 설치 명령을 알려 주고 끝낸다
    LANGGRAPH_ERROR = e

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 요청 하나에 실을 스트림 길이(초). 60초면 test_5에서 06:14:59에 코드 없는 핸드오프가 하나 더 생긴다.
CHUNK_S = 600
# train.py:114 RULE_EVAL_WINDOW_S의 사본: 규칙 trigger를 고장 전 이 시간 안이면 적중으로 센다.
# 모델 스택을 피하려고 train을 import하지 않는다(replay.MAX_CHUNK처럼).
RULE_EVAL_WINDOW_S = 3600
# api 케이스가 닫혔는지 GET /api/history로 다시 조회하는 간격(초)
POLL_S = 2.0
# 알림 대기 중 stop()을 확인하는 간격(초)
_WAIT_S = 0.5
# POST /api/ml/sync가 {}일 때(새 핸드오프가 있는데도) 다시 시도하는 횟수와 간격(초)
_SYNC_RETRIES = 3
_SYNC_RETRY_S = 0.5


def file_span(path: str) -> tuple[str, str]:
    """CSV의 time 열만 읽어 (최소, 최대)를 ISO `...Z` 문자열로 돌려준다."""
    t = pd.to_datetime(pd.read_csv(path, usecols=["time"])["time"], utc=True)
    return t.min().strftime("%Y-%m-%dT%H:%M:%SZ"), t.max().strftime("%Y-%m-%dT%H:%M:%SZ")


class _Stopped(Exception):
    """stop() 뒤(알림 대기 중 포함) replay_fn을 빠져나오는 예외. run()이 잡고, 큐에 error 항목을 남기지 않는다."""


class Feeder(threading.Thread):
    """파일을 차례로 재생해 큐에 넣는 스레드. 큐 항목:
    ("file", gun, path, start_iso, end_iso), ("alert", seq, gun, event), ("end",), ("error", traceback_str).
    pause=True면 알림마다 release(seq)까지 재생을 멈추고, pause=False(드라이런)면 기다리지 않는다."""

    def __init__(self, paths: list[str], make_client: Callable[[], Any], chunk_s: int = CHUNK_S, pause: bool = True,
                 replay_fn: Callable[..., dict] = replay.replay_file) -> None:
        super().__init__(daemon=True, name="hitl-feeder")
        spans = {p: file_span(p) for p in paths}
        self.files = sorted(spans.items(), key=lambda kv: kv[1][0])  # 시작 시각 순
        self.make_client = make_client
        self.chunk_s = chunk_s
        self.pause = pause
        self.replay_fn = replay_fn
        self._q: queue.Queue[tuple] = queue.Queue()
        self._halt = threading.Event()  # Thread._stop과 이름이 겹치지 않게
        self._gates: dict[int, threading.Event] = {}  # 알림 seq -> release 신호
        self._seq = 0
        self._gun = ""

    def next_item(self, timeout: float = 0.5) -> tuple | None:
        """다음 큐 항목, timeout 안에 없으면 None."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self, seq: int) -> None:
        """알림 seq에서 멈춘 재생을 이어 간다(모르는 seq는 무시)."""
        gate = self._gates.get(seq)
        if gate is not None:
            gate.set()

    def stop(self) -> None:
        """재생을 멈춘다. 알림 대기 중이면 _WAIT_S 안에 빠져나온다."""
        self._halt.set()

    def _on_result(self, res: dict[str, Any], event: dict[str, Any] | None) -> None:
        if self._halt.is_set():
            raise _Stopped
        if event is None:  # 새 critical 이벤트일 때만 멈춘다
            return
        self._seq += 1
        if not self.pause:
            self._q.put(("alert", self._seq, self._gun, event))
            return
        seq, gate = self._seq, threading.Event()
        self._gates[seq] = gate  # 큐에 넣기 전에 등록해 둬야 바로 온 release()를 놓치지 않는다
        self._q.put(("alert", seq, self._gun, event))
        while not gate.wait(_WAIT_S):
            if self._halt.is_set():
                raise _Stopped
        del self._gates[seq]

    def run(self) -> None:
        client = None
        try:
            client = self.make_client()  # 이 스레드가 자기 client를 가진다
            for path, (start, end) in self.files:
                if self._halt.is_set():
                    raise _Stopped
                self._gun = os.path.splitext(os.path.basename(path))[0]
                self._q.put(("file", self._gun, path, start, end))
                self.replay_fn(client, path, self._gun, self.chunk_s, reset=True, on_result=self._on_result)
            self._q.put(("end",))
        except _Stopped:
            pass
        except Exception:
            self._q.put(("error", traceback.format_exc()))
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


# ---- 그래프: advance → ingest → guide → operator(⏸) → learn → evaluate → … → report ----


class LoopState(TypedDict, total=False):
    """그래프 state. 체크포인트에 그대로 저장되므로 JSON 값만 둔다(Feeder·client는 Deps 쪽)."""
    files: dict[str, dict]  # gun -> {"path", "start", "end"}
    seen: list[str]  # 이미 행으로 만든 ML outbox event_id
    queue: list[str]  # 사람 확인을 기다리는 행의 event_id(앞에서부터)
    alerts: list[dict]  # 핸드오프 하나당 한 행(_row 키)
    memory: dict | None  # 직전 GET /api/memory
    seq: int | None  # 재생을 멈춰 둔 Feeder 알림 seq
    resume: dict | None  # wait_for_closure() 결과
    done: bool
    error: str | None


@dataclass
class Deps:
    """그래프 밖에 두는 것들: 노드가 클로저로 잡는다."""
    feeder: Feeder
    ml: Any  # httpx.Client 같은 것(base_url = ML 서버)
    api: Any  # httpx.Client 같은 것(base_url = api 서버)
    ui_url: str


def _json(resp: Any) -> Any:
    resp.raise_for_status()
    return resp.json()


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _row(h: dict[str, Any], status: str, case_id: str | None = None, folded_into: str | None = None) -> dict:
    """핸드오프 하나의 알림 행. 사람·데이터 쪽 칸은 guide / learn / evaluate가 채운다."""
    trig = h.get("trigger") or {}
    return {"event_id": h["event_id"], "case_id": case_id, "gun_id": h["gun_id"], "code": trig.get("rule_code"),
            "source": trig.get("source"), "data_time": trig.get("rule_trigger_time") or h["detected_at"],
            "status": status, "folded_into": folded_into, "order": None, "applied": None,
            "note_ko": None, "closure": None, "human_label": None, "data_label": None, "ttf_s": None, "agree": None,
            "memory_delta": None, "warn": None}


def _current(state: LoopState) -> tuple[list[dict], dict]:
    """alerts 사본과 그 안의 지금 행(queue 맨 앞). 노드는 사본을 고쳐 돌려준다."""
    alerts = [dict(r) for r in state["alerts"]]
    eid = state["queue"][0]
    return alerts, next(r for r in alerts if r["event_id"] == eid)


def _changed(new: list[dict], old: list[dict]) -> list[dict]:
    return [x for x in new if x not in old]


def build_graph(deps: Deps):
    """D-8 루프 그래프를 InMemorySaver로 컴파일한다. operator 노드에서 interrupt()로 멈춘다."""
    feeder, ml, api = deps.feeder, deps.ml, deps.api

    def advance(state: LoopState) -> dict:
        # 앞 알림의 재생을 풀어 주고, 다음 알림·끝·오류까지 Feeder 큐를 읽는다
        if state.get("seq") is not None:
            feeder.release(state["seq"])
        files = dict(state["files"])
        while True:
            item = feeder.next_item()
            if item is None:
                continue
            if item[0] == "file":
                _, gun, path, start, end = item
                files[gun] = {"path": path, "start": start, "end": end}
            elif item[0] == "alert":
                return {"files": files, "seq": item[1]}
            elif item[0] == "end":
                return {"files": files, "seq": None, "done": True}
            elif item[0] == "error":
                return {"files": files, "seq": None, "done": True, "error": item[1]}

    def ingest(state: LoopState) -> dict:
        # 새 핸드오프(ML outbox − seen)마다 api 케이스를 찾아 행을 만든다
        seen = set(state["seen"])
        new = [r["handoff"] for r in _json(ml.get("/handoffs")) if r["handoff"]["event_id"] not in seen]
        counts = _json(api.post("/api/ml/sync"))
        for _ in range(_SYNC_RETRIES):
            if counts or not new:
                break
            time.sleep(_SYNC_RETRY_S)
            counts = _json(api.post("/api/ml/sync"))
        cases = _json(api.get("/api/cases"))["cases"]
        found = {c["event"]["event_id"]: c for c in cases + _json(api.get("/api/history"))["items"]}
        details: dict[str, dict] = {}
        alerts, queue_ = list(state["alerts"]), list(state["queue"])
        for h in new:
            eid = h["event_id"]
            if eid in found:
                alerts.append(_row(h, "waiting", case_id=found[eid]["case_id"]))
                queue_.append(eid)
                continue
            # api가 같은 설비의 열린 케이스에 '반복'으로 묶었는지 본다
            folded = None
            for c in cases:
                if c["event"]["gun_id"] != h["gun_id"]:
                    continue
                cid = c["case_id"]
                if cid not in details:
                    details[cid] = _json(api.get(f"/api/cases/{cid}"))
                if any(r["event_id"] == eid for r in details[cid].get("repeats", [])):
                    folded = cid
                    break
            row = _row(h, "folded", folded_into=folded) if folded else _row(h, "missing")
            if folded and details[folded].get("status") == "handed_over":  # 도움 요청 케이스는 열린 채 계속 묶는다
                row["warn"] = (f"{folded}가 도움 요청 상태라 이 알림이 반복으로 묶였습니다(멈춤·R1 없음). "
                               f"작업함에서 {folded}를 열어 '오알람으로 닫기'로 닫으면 다음 알림부터 새 케이스가 됩니다.")
            alerts.append(row)
        return {"seen": state["seen"] + [h["event_id"] for h in new], "alerts": alerts, "queue": queue_}

    def guide(state: LoopState) -> dict:
        # 안내 카드(guidance.candidates, api가 정한 그대로)와 현장 기억 문장(최상위 field_memory, 플래그가 꺼져 있으면 없음).
        # 이 GET이 케이스의 첫 상세 조회라 api가 여기서 RAG 답을 만든다(수 초 걸릴 수 있음)
        alerts, row = _current(state)
        case = _json(api.get(f"/api/cases/{row['case_id']}"))
        g = case.get("guidance") or {}
        fm = case.get("field_memory") or {}
        row["order"] = [c["situation_id"] for c in g.get("candidates", [])]
        row["applied"] = list(fm.get("applied") or [])
        row["note_ko"] = fm.get("note_ko")
        return {"alerts": alerts}

    def operator(state: LoopState) -> dict:
        # 이어 갈 때 이 노드는 처음부터 다시 돌므로 payload만 만들고 멈춘다
        _, row = _current(state)
        payload = {"case_id": row["case_id"], "url": f"{deps.ui_url}/cases/{row['case_id']}", "gun_id": row["gun_id"],
                   "code": row["code"], "order": row["order"], "note_ko": row["note_ko"]}
        return {"resume": interrupt(payload)}

    def learn(state: LoopState) -> dict:
        # 닫힌 뒤 현장 기억을 다시 읽어 사람 라벨과 바뀐 항목을 남긴다
        alerts, row = _current(state)
        mem = _json(api.get("/api/memory"))
        old = state.get("memory") or {}
        row["status"] = "closed"
        row["closure"] = state["resume"]["closure"]
        row["human_label"] = mem.get("labels", {}).get(row["case_id"])
        row["memory_delta"] = {k: _changed(mem.get(k, []), old.get(k, [])) for k in ("alarms", "checks")}
        if (row["closure"] or {}).get("outcome") == "handed_over":  # 라벨 없이 열린 채 남는다
            row["warn"] = (f"{row['case_id']}를 도움 요청으로 넘겼습니다. 이 케이스는 열린 채 남아 같은 설비·코드의 "
                           f"다음 알림을 반복으로 묶습니다. 다음 알림 전에 작업함에서 열어 '오알람으로 닫기'로 닫으세요.")
        return {"alerts": alerts, "memory": mem}

    def evaluate(state: LoopState) -> dict:
        # 데이터 기준 라벨은 닫힌 뒤 여기서만 정한다(사람에게 답을 미리 보이지 않게)
        alerts, row = _current(state)
        f = state["files"].get(row["gun_id"])
        if f is not None:
            row["ttf_s"] = (_ts(f["end"]) - _ts(row["data_time"])).total_seconds()
            row["data_label"] = "pre_failure" if row["ttf_s"] <= RULE_EVAL_WINDOW_S else "far"
            if row["human_label"] is not None:
                row["agree"] = (row["human_label"] == "false_alarm") == (row["data_label"] == "far")
        return {"alerts": alerts, "queue": state["queue"][1:]}

    def report(state: LoopState) -> dict:
        return {"done": True}

    def after_advance(state: LoopState) -> Literal["ingest", "report"]:
        return "report" if state.get("done") else "ingest"

    def after_ingest(state: LoopState) -> Literal["guide", "advance"]:
        return "guide" if state["queue"] else "advance"

    def after_evaluate(state: LoopState) -> Literal["guide", "advance"]:
        return "guide" if state["queue"] else "advance"

    g = StateGraph(LoopState)
    for name, fn in (("advance", advance), ("ingest", ingest), ("guide", guide), ("operator", operator),
                     ("learn", learn), ("evaluate", evaluate), ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "advance")
    g.add_conditional_edges("advance", after_advance)
    g.add_conditional_edges("ingest", after_ingest)
    g.add_edge("guide", "operator")
    g.add_edge("operator", "learn")
    g.add_edge("learn", "evaluate")
    g.add_conditional_edges("evaluate", after_evaluate)
    g.add_edge("report", END)
    return g.compile(checkpointer=InMemorySaver())


# ---- 실행기 ----


_ML_FRESH = "새 RSW_STATE_DIR 폴더(예: state/d8-<새 이름>)로 ML 서버를 다시 켜세요"
_API_FRESH = "새 API_DB 파일(예: state/d8-<새 이름>.db)로 api 서버를 다시 켜세요"


def _unreachable(name: str, client: Any, e: Exception) -> str:
    addr = getattr(client, "base_url", None)
    return f"{name} 서버{f'({addr})' if addr else ''}에 연결하지 못했습니다({type(e).__name__}): 먼저 그 서버를 켜세요"


def startup_check(ml: Any, api: Any) -> list[str]:
    """두 서버가 새로 켠 깨끗한 상태인지 본다. 문제 목록을 돌려주고, 비면 정상.
    ML이 깨끗할 때만 /api/cases를 부른다: 이 조회가 api에 ML outbox를 당겨 오기 때문이다(api/main.py sync_from_ml)."""
    problems: list[str] = []
    try:
        h = _json(ml.get("/health"))
        if h.get("guns_tracked"):
            problems.append(f"ML guns_tracked={h['guns_tracked']}: {_ML_FRESH}")
        if any((h.get("outbox") or {}).values()):
            problems.append(f"ML outbox가 비어 있지 않습니다 {h.get('outbox')}: {_ML_FRESH}")
        if h.get("rag_url") is not None:
            problems.append(f"ML rag_url={h['rag_url']}: RSW_RAG_URL 없이(pull only) 켜야 합니다")
    except httpx.HTTPError as e:
        problems.append(_unreachable("ML", ml, e))
    ml_clean = not problems
    try:
        ah = _json(api.get("/api/health"))
        if ah.get("memory") != "on":
            problems.append("api memory가 꺼져 있습니다: FIELD_MEMORY=1로 api 서버를 다시 켜세요")
        if ah.get("ml") != "ok":
            problems.append(f"api가 ML에 닿지 못합니다(ml={ah.get('ml')}): api 서버의 ML_URL이 ML 서버 주소인지 확인하세요")
        if ml_clean:
            if _json(api.get("/api/cases"))["cases"]:
                problems.append(f"api cases가 비어 있지 않습니다: {_API_FRESH}")
            if _json(api.get("/api/history"))["items"]:
                problems.append(f"api history가 비어 있지 않습니다: {_API_FRESH}")
    except httpx.HTTPError as e:
        problems.append(_unreachable("api", api, e))
    return problems


def _get_or_none(api: Any, path: str) -> Any:
    """GET path의 JSON. api가 잠깐 안 닿거나(타임아웃 포함) 5xx면 None, 4xx는 그대로 올린다."""
    try:
        return _json(api.get(path))
    except httpx.TransportError:
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code < 500:
            raise
        return None


def wait_for_closure(api: Any, case_id: str, poll_s: float, stop: threading.Event | None = None) -> dict:
    """케이스가 닫힐 때까지 GET /api/history를 poll_s 간격으로 조회하고, 닫힌 뒤 케이스 상세를 한 번 읽는다
    (judgments·results). 상세 GET은 RAG를 부를 수 있어 닫힘 확인에 쓰지 않는다.
    stop이 서면 그때까지 본 closure(없으면 None)로 돌아온다. 안 닿거나 5xx면 다음 조회까지 기다린다."""
    closure = None
    while True:
        if closure is None:
            h = _get_or_none(api, "/api/history")
            if h is not None:
                closure = next((i["closure"] for i in h.get("items", [])
                                if i.get("case_id") == case_id and i.get("closure")), None)
        if closure is not None:
            d = _get_or_none(api, f"/api/cases/{case_id}")
            if d is not None:
                r = d.get("records") or {}
                return {"case_id": case_id, "closure": closure, "judgments": r.get("judgments", []),
                        "results": r.get("results", [])}
        if stop is not None and stop.is_set():
            return {"case_id": case_id, "closure": closure, "judgments": [], "results": []}
        if stop is None:
            time.sleep(poll_s)
        else:
            stop.wait(poll_s)


def run_loop(deps: Deps, on_update: Callable[[str, dict, dict], None], poll_s: float = POLL_S,
             thread_id: str = "run") -> dict:
    """Feeder를 켜고 그래프를 끝까지 돌린다. 중단마다 케이스가 닫히길 기다렸다가 Command(resume=…)로 이어 간다.
    on_update(node, update, state)는 청크마다(중단은 node "__interrupt__", update = payload) 부른다.
    다만 이어 갈 때 다시 도는 operator의 {"resume": …} 청크는 넘기지 않는다(닫힌 결과는 state["resume"]에 있다)."""
    graph = build_graph(deps)
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}
    inp: Any = {"files": {}, "seen": [], "queue": [], "alerts": [], "memory": None, "seq": None, "resume": None,
                "done": False, "error": None}
    deps.feeder.start()
    try:
        while True:
            # durability="sync": 청크를 받을 때 체크포인트가 이미 저장돼 있어 get_state가 이 갱신까지 보인다
            for chunk in graph.stream(inp, cfg, stream_mode="updates", durability="sync"):
                for node, update in chunk.items():
                    if node == "operator":  # 이어 간 중단 노드의 재실행: 순서는 __interrupt__ 다음 learn
                        continue
                    if node == "__interrupt__":
                        update = update[0].value
                    on_update(node, update, graph.get_state(cfg).values)
            snap = graph.get_state(cfg)
            if not snap.next or not snap.interrupts:
                return snap.values
            payload = snap.interrupts[0].value
            inp = Command(resume=wait_for_closure(deps.api, payload["case_id"], poll_s))
    finally:
        deps.feeder.stop()


# ---- 현황·요약 출력 ----

# 노드 순서와 화면 이름
NODES = [("advance", "재생 진행"), ("ingest", "케이스 확인"), ("guide", "안내 확인"), ("operator", "사람 확인(⏸)"),
         ("learn", "기억 갱신"), ("evaluate", "기록"), ("report", "정리")]
FOOTER = "재생 데이터 시연 · 현장 아님 · 데이터 기준 라벨은 답을 아는 것이라 성능 아님"
# UI 규칙: 파랑·빨강·주황 + 회색만 (보라 없음)
BLUE, RED, AMBER, GRAY, LIGHT = "#2563eb", "#dc2626", "#d97706", "#6b7280", "#e5e7eb"
_STATUS_KO = {"waiting": "확인 대기", "folded": "반복으로 묶임", "missing": "케이스 없음", "closed": "닫힘"}
# 화면 낱말은 api·UI·문서와 같게 (ui/src/pages/case/parts.tsx CLOSURE). loop_log.jsonl은 원래 토큰을 그대로 둔다
_CLOSURE_KO = {"false_alarm": "오알람으로 닫음", "resolved": "해결 종료", "unresolved": "미해결로 저장",
               "handed_over": "도움 요청"}
_HUMAN_KO = {"false_alarm": "오알람", "real": "고장 확인"}
_DATA_KO = {"pre_failure": "고장 직전(1시간 안)", "far": "먼 알림"}
# 상태 칸은 5번째(_html_table의 글자색)
_COLS = ["데이터 시각", "설비", "코드", "출처", "상태", "케이스", "안내 카드", "기억 반영", "닫힘",
         "사람 라벨", "데이터 라벨", "일치", "안내 문장", "기억 변화"]
_MEM_COLS = ["설비|코드", "오알람", "고장 확인", "케이스"]
_CHECK_COLS = ["점검 항목", "이상", "정상"]


def running_node(node: str, state: dict) -> str | None:
    """on_update(node, …)는 node가 끝난 뒤 불리므로, 지금 도는 노드는 그다음 노드다(build_graph의 router와 같은 규칙).
    중단("__interrupt__")이면 operator, report 뒤에는 없음."""
    if node == "__interrupt__":
        return "operator"
    if node == "advance":
        return "report" if state.get("done") else "ingest"
    if node in ("ingest", "evaluate"):
        return "guide" if state.get("queue") else "advance"
    return {"guide": "operator", "operator": "learn", "learn": "evaluate"}.get(node)


def write_atomic(path: str, text: str, retries: int = 5) -> None:
    """utf-8로 tmp에 쓰고 os.replace로 바꾼다. 브라우저가 파일을 잡고 있으면(PermissionError) 0.2초 간격으로 다시 한다."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                os.remove(tmp)
                raise
            time.sleep(0.2)


def _delta_text(delta: dict | None) -> str:
    """learn이 남긴 기억 변화(memory_delta)를 한 줄로. 바뀐 것이 없으면 "—"."""
    delta = delta or {}
    parts = [f"{a['key']} 오알람 {a.get('false_alarm', 0)} · 고장 확인 {a.get('real', 0)}"
             for a in delta.get("alarms", [])]
    parts += [f"{c['check_id']} 이상 {c.get('abnormal', 0)} · 정상 {c.get('normal', 0)}"
              for c in delta.get("checks", [])]
    return "; ".join(parts) or "—"


def _cells(r: dict) -> list[str]:
    """알림 행 하나의 칸(escape 전 문자열). 데이터 라벨·일치는 닫히기 전 "—"."""
    closed = r["status"] == "closed"
    status = _STATUS_KO.get(r["status"], r["status"])
    if r["folded_into"]:
        status += f" ({r['folded_into']})"
    agree = {True: "일치", False: "다름"}.get(r["agree"], "—") if closed else "—"
    outcome = (r["closure"] or {}).get("outcome")
    data = _DATA_KO.get(r["data_label"], r["data_label"]) if closed and r["data_label"] else "—"
    return [r["data_time"] or "—", r["gun_id"], r["code"] or "—", r["source"] or "—", status, r["case_id"] or "—",
            " → ".join(r["order"] or []) or "—", ", ".join(r["applied"] or []) or "—", _CLOSURE_KO.get(outcome, outcome) if outcome else "—",
            _HUMAN_KO.get(r["human_label"], r["human_label"]) if r["human_label"] else "—", data, agree,
            r["note_ko"] or "—", _delta_text(r["memory_delta"])]


def _memory_rows(mem: dict | None) -> tuple[list[list[str]], list[list[str]]]:
    """기억 표 두 개의 행: 알람(설비|코드, 오알람, 고장 확인, 케이스)과 점검 기록(점검 항목, 이상, 정상)."""
    mem = mem or {}
    alarms = [[a["key"], str(a.get("false_alarm", 0)), str(a.get("real", 0)), ", ".join(a.get("case_ids", []))]
              for a in mem.get("alarms", [])]
    checks = [[c["check_id"], str(c.get("abnormal", 0)), str(c.get("normal", 0))] for c in mem.get("checks", [])]
    return alarms, checks


def _warns(state: dict) -> list[str]:
    return [r["warn"] for r in state.get("alerts", []) if r.get("warn")]


def _html_table(head: list[str], rows: list[list[str]], marks: list[str | None] | None = None) -> str:
    """escape한 표. marks[i]가 있으면 i행 상태 칸(5번째)의 글자색."""
    e = html.escape
    out = ["<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>"]
    for i, row in enumerate(rows):
        color = marks[i] if marks else None
        tds = [f'<td style="color:{color}">{e(c)}</td>' if color and j == 4 else f"<td>{e(c)}</td>"
               for j, c in enumerate(row)]
        out.append("<tr>" + "".join(tds) + "</tr>")
    if not rows:
        out.append(f'<tr><td colspan="{len(head)}">—</td></tr>')
    return "".join(out) + "</table>"


def render_view(state: dict, current: str | None, waiting: dict | None, ui_url: str, data_time: str | None, *,
                ended: str | None = None) -> str:
    """loop_view.html: 노드 사슬, 상태 줄(+⚠ 경고), 알림 표, 기억 표. 2초마다 새로고침하고 외부 CSS/JS는 쓰지 않는다.
    ended가 있으면 실행이 끝난 화면이다: 상태 줄에 그 문장(중단됨…은 빨강, 그 밖은 회색)만 둔다."""
    e = html.escape
    boxes = "".join(
        f'<span class="node" style="background:{BLUE if name == current else LIGHT};'
        f'color:{"#fff" if name == current else GRAY}">{e(label)}</span>'
        for name, label in NODES)
    if ended:
        status = f'<p class="status" style="color:{RED if ended.startswith("중단됨") else GRAY}">{e(ended)}</p>'
    elif waiting:
        cid = waiting["case_id"]
        # 링크는 이름 붙은 탭 하나(d8-ui)를 다시 쓴다: 현황 탭은 그대로 열려 있다
        status = (f'<p class="status" style="color:{AMBER}">⏸ 사람 확인 대기 · '
                  f'<a href="{e(f"{ui_url}/cases/{cid}")}" target="d8-ui">{e(cid)}</a>'
                  f' · 안내 카드 {e(" → ".join(waiting.get("order") or []) or "—")}</p>')
        if waiting.get("note_ko"):
            status += f'<p class="note">{e(waiting["note_ko"])}</p>'
    else:
        status = f'<p class="status">재생 중 · 마지막 알림 데이터 시각 {e(data_time or "—")}</p>'
    status += "".join(f'<p style="color:{AMBER}">⚠ {e(w)}</p>' for w in _warns(state))
    if state.get("error"):
        status += f'<pre style="color:{RED}">{e(state["error"])}</pre>'
    alerts = state.get("alerts", [])
    marks = [AMBER if r.get("warn") else {"waiting": AMBER, "missing": RED}.get(r["status"]) for r in alerts]
    alarms, checks = _memory_rows(state.get("memory"))
    return f"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>D-8 HITL 루프</title>
<style>
body {{ font-family: sans-serif; margin: 16px; color: #111827; background: #fff; }}
.node {{ display: inline-block; padding: 4px 10px; margin: 0 4px 4px 0; border-radius: 4px; }}
.status {{ font-size: 18px; }}
.note {{ font-size: 15px; }}
table {{ border-collapse: collapse; margin-bottom: 16px; }}
th, td {{ border: 1px solid {LIGHT}; padding: 3px 8px; font-size: 13px; text-align: left; }}
th {{ color: {GRAY}; }}
.footer {{ color: {GRAY}; font-size: 12px; }}
</style></head><body>
<div>{boxes}</div>
{status}
<h3>알림</h3>
{_html_table(_COLS, [_cells(r) for r in alerts], marks)}
<h3>현장 기억</h3>
{_html_table(_MEM_COLS, alarms)}
{_html_table(_CHECK_COLS, checks)}
<p class="footer">{e(FOOTER)}</p>
</body></html>
"""


def _md_table(head: list[str], rows: list[list[str]]) -> str:
    """markdown 표. 칸 안의 |는 \\|로 escape하고, 행이 없으면 — 한 줄."""
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"
    return "\n".join([line(head), "|" + "---|" * len(head)] + [line(r) for r in rows or [["—"] * len(head)]])


def render_summary(state: dict) -> str:
    """summary.md: 알림 표, 경고(있을 때만), 현장 기억, 고정 문구."""
    alarms, checks = _memory_rows(state.get("memory"))
    parts = ["# D-8 HITL 루프 요약", ""]
    if state.get("error"):
        parts += ["재생이 오류로 끝났습니다:", "", "```", state["error"].rstrip(), "```", ""]
    parts += ["## 알림", "", _md_table(_COLS, [_cells(r) for r in state.get("alerts", [])]), ""]
    if _warns(state):
        parts += ["## 경고", ""] + [f"- ⚠ {w}" for w in _warns(state)] + [""]
    parts += ["## 현장 기억", "", _md_table(_MEM_COLS, alarms), "",
              _md_table(_CHECK_COLS, checks), "", f"_{FOOTER}_", ""]
    return "\n".join(parts)


# ---- CLI ----


def _dry_run(paths: list[str], ml_url: str) -> None:
    """api 없이 멈추지 않고 끝까지 재생한 뒤, ML outbox의 gun별 핸드오프 건수를 (출처, 코드)별로 출력한다."""
    feeder = Feeder(paths, make_client=lambda: httpx.Client(base_url=ml_url, timeout=60), pause=False)
    feeder.start()
    guns: list[str] = []
    while True:
        item = feeder.next_item()
        if item is None:
            continue
        if item[0] == "file":
            guns.append(item[1])
            print(f"{item[1]}: {item[2]} 재생", flush=True)
        elif item[0] == "alert":
            ev = item[3]
            print(f"  [{item[2]}] 알림 {item[1]} {ev.get('window_end')} code={ev.get('error_code')}", flush=True)
        elif item[0] == "end":
            break
        elif item[0] == "error":
            raise SystemExit(f"재생 오류:\n{item[1]}")
    feeder.join(timeout=5.0)
    total = 0
    with httpx.Client(base_url=ml_url, timeout=60) as ml:
        for gun in guns:
            # limit 기본값은 50이라 ML OUTBOX_MAX(1000)까지 받는다
            hs = [r["handoff"] for r in _json(ml.get("/handoffs", params={"gun_id": gun, "limit": 1000}))]
            counts = Counter(((h.get("trigger") or {}).get("source"), (h.get("trigger") or {}).get("rule_code"))
                             for h in hs)
            detail = ", ".join(f"{src} {code or '-'} x{n}" for (src, code), n in counts.most_common())
            print(f"{gun}: {len(hs)} ({detail})")
            total += len(hs)
    print(f"합계: {total}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 콘솔이 cp949라도 ⏸·한글이 깨지지 않게
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # SystemExit 문장도 한글
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("files", nargs="+", help="시험 CSV (glob 가능), 예: test/test_*.csv")
    ap.add_argument("--api", default="http://127.0.0.1:9000", help="api 서버 주소")
    ap.add_argument("--ml", default="http://127.0.0.1:8000", help="ML 서버 주소")
    ap.add_argument("--ui", default="http://localhost:5173", help="UI 주소 (케이스 링크용)")
    ap.add_argument("--out", default=None, help="산출물 폴더, 기본 results/D-8/runs/<YYYYmmdd-HHMMSS>")
    ap.add_argument("--dry-run", action="store_true", help="api 없이 멈추지 않고 재생해 gun별 핸드오프 건수만 출력")
    args = ap.parse_args()

    paths = sorted({p for f in args.files for p in (glob.glob(f) or [f])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"없는 파일: {missing} (cwd {os.getcwd()}, project root {PROJECT_ROOT})")
    if args.dry_run:
        _dry_run(paths, args.ml)
        return
    if LANGGRAPH_ERROR is not None:
        raise SystemExit('langgraph가 필요합니다: py -3.11 -m pip install "langgraph>=1.0"')
    ui = args.ui.rstrip("/")

    # api 첫 상세 조회는 RAG 답을 만든다(OpenAI 최대 90초 + Neo4j 연결 15초). 그래서 api만 120초로 기다린다
    with httpx.Client(base_url=args.ml, timeout=60) as ml, httpx.Client(base_url=args.api, timeout=120) as api:
        # 점검이 끝나기 전에는 산출물 폴더를 만들지 않는다
        problems = startup_check(ml, api)
        if problems:
            raise SystemExit("시작 점검 실패:\n" + "\n".join(f"- {p}" for p in problems))
        out = args.out or os.path.join(PROJECT_ROOT, "results", "D-8", "runs",
                                       datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(out, exist_ok=True)
        view_path, log_path = os.path.join(out, "loop_view.html"), os.path.join(out, "loop_log.jsonl")
        open(log_path, "w", encoding="utf-8").close()  # 같은 --out을 다시 써도 옛 기록에 이어 붙이지 않는다
        print(f"산출물 {out}\n현황 {view_path}\nUI {ui}", flush=True)
        feeder = Feeder(paths, make_client=lambda: httpx.Client(base_url=args.ml, timeout=60))
        deps = Deps(feeder=feeder, ml=ml, api=api, ui_url=ui)
        last: dict[str, Any] = {"state": {}}
        printed: set[str] = set()
        warned: set[tuple[str, str]] = set()

        def on_update(node: str, update: dict, state: dict) -> None:
            last["state"] = state
            waiting = update if node == "__interrupt__" else None
            for r in state.get("alerts", []):  # 알림 행마다 콘솔 한 줄, 경고는 한 번씩
                if r["event_id"] not in printed:
                    printed.add(r["event_id"])
                    print(f"  [{r['gun_id']}] {r['data_time']} {r['source']} {r['code']} "
                          f"{_STATUS_KO.get(r['status'], r['status'])} {r['case_id'] or r['folded_into'] or '-'}",
                          flush=True)
                if r.get("warn") and (r["event_id"], r["warn"]) not in warned:
                    warned.add((r["event_id"], r["warn"]))
                    print(f"  ⚠ {r['warn']}", flush=True)
            if waiting:
                order = " → ".join(waiting.get("order") or []) or "—"
                print(f"  ⏸ {waiting['case_id']} 사람 확인 대기 · 안내 카드 {order}: {waiting['url']}", flush=True)
                if waiting.get("note_ko"):
                    print(f"      {waiting['note_ko']}", flush=True)
            times = [r["data_time"] for r in state.get("alerts", []) if r["data_time"]]
            current = running_node(node, state)
            try:
                write_atomic(view_path, render_view(state, current, waiting, ui, max(times) if times else None))
            except OSError as e:  # 다음 갱신이 다시 쓴다
                print(f"  ⚠ 현황 파일을 쓰지 못했습니다({e}): 다음 갱신에서 다시 씁니다", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "node": node,
                                    "update": update}, ensure_ascii=False, default=str) + "\n")

        ended = "중단됨: 알 수 없는 이유"
        try:
            last["state"] = run_loop(deps, on_update)
            n = len(last["state"].get("alerts", []))
            ended = "중단됨: 재생 오류(아래)" if last["state"].get("error") else f"끝남 · 알림 {n}건 · 요약 summary.md"
        except BaseException as e:  # 노드의 HTTP 오류, Ctrl+C: 요약과 현황에 남기고 다시 올린다
            if isinstance(e, KeyboardInterrupt):
                ended, note = "중단됨: 사용자가 멈춤(Ctrl+C)", "사용자가 멈춤(Ctrl+C)"
            else:
                ended, note = f"중단됨: {traceback.format_exception_only(e)[-1].strip()}", traceback.format_exc()
            prev = last["state"].get("error")
            last["state"] = {**last["state"], "error": f"{prev}\n{note}" if prev else note}
            raise
        finally:
            times = [r["data_time"] for r in last["state"].get("alerts", []) if r["data_time"]]
            for path, text in (
                    (view_path, lambda: render_view(last["state"], None, None, ui, max(times) if times else None,
                                                    ended=ended)),
                    (os.path.join(out, "summary.md"), lambda: render_summary(last["state"])),
                    (os.path.join(out, "workflow.mmd"), lambda: build_graph(deps).get_graph().draw_mermaid())):
                try:
                    write_atomic(path, text())
                except OSError as e:  # 원래 예외를 덮지 않는다
                    print(f"  ⚠ {os.path.basename(path)}를 쓰지 못했습니다: {e}", flush=True)
                else:
                    if path.endswith("summary.md"):
                        print(f"요약 {path}", flush=True)


if __name__ == "__main__":
    main()
