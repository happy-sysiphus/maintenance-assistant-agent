"""D-8 드라이버(ml/.py/hitl_loop.py) ↔ 진짜 api 계약: 서버 없이 한 프로세스 안에서 돈다.

api는 TestClient(임시 DB, FIELD_MEMORY=1, RAG는 conftest의 가짜 답)이고 ML은 스크립트 outbox다.
드라이버 그래프를 끝까지 돌리면서, 중단마다 작업자 대역이 UI와 같은 주소 · 본문으로 케이스를 닫는다.
안내 카드는 E 코드 하나 그대로이고, 현장 기억은 참고 문장(field_memory.note_ko)으로만 바뀐다.
소켓도 실제 CSV도 쓰지 않는다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / ".py"))

import hitl_loop as hl  # noqa: E402
import main  # noqa: E402
from fastapi import HTTPException  # noqa: E402

UI = "http://ui"
NOW = "2026-10-28T00:00:00Z"
_DETAIL = re.compile(r"^/api/cases/[^/]+$")


def r1(gun: str, code: str, fa: int, real: int = 0) -> str:
    """field_memory의 R1 문장 (브리프의 글 그대로)."""
    return (
        f"같은 설비·코드({gun} {code})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 → "
        "오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
    )


def _record(h: dict[str, Any]) -> dict[str, Any]:
    """ML GET /handoffs 한 줄 (ml/main.py HandoffRecord 모양, pull only)."""
    return {
        "handoff": h,
        "created_at": h["detected_at"],
        "delivery": "pull_only",
        "delivery_detail": None,
        "rag_response": None,
        "attempts": 0,
        "last_attempt_at": None,
        "next_attempt_at": None,
    }


class ScriptML:
    """ML 서버 대역: 드라이버 쪽 client(/health · /handoffs)와 api 쪽 main.ml_get이 같은 outbox를 본다."""

    def __init__(self) -> None:
        self.outbox: list[dict[str, Any]] = []

    def get(self, path: str, params: dict | None = None) -> httpx.Response:
        req = httpx.Request("GET", f"http://ml{path}")
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "guns_tracked": 0,
                    "rag_url": None,
                    "outbox": {"pending": 0, "delivered": 0, "failed": 0, "pull_only": len(self.outbox)},
                },
                request=req,
            )
        assert path == "/handoffs", path
        return httpx.Response(200, json=[_record(h) for h in self.outbox], request=req)

    def ml_get(self, path: str) -> Any:
        if path == "/health":
            return {"status": "ok"}
        if path == "/handoffs":
            return [_record(h) for h in self.outbox]
        raise HTTPException(502, "ML 서버 없음 (테스트)")


class ScriptFeeder:
    """스레드 없는 Feeder 대역: 알림 항목을 내기 전에 그 알림의 핸드오프를 outbox에 넣는다."""

    def __init__(self, ml: ScriptML, script: list[tuple[tuple, list[dict[str, Any]]]]) -> None:
        self.ml = ml
        self.script = list(script)
        self.released: list[int] = []

    def start(self) -> None:
        pass

    def next_item(self, timeout: float = 0.5) -> tuple:
        assert self.script, "script exhausted"
        item, hs = self.script.pop(0)
        self.ml.outbox.extend(hs)
        return item

    def release(self, seq: int) -> None:
        self.released.append(seq)

    def stop(self) -> None:
        pass


class CountingAPI:
    """드라이버 쪽 api client: 요청 경로를 (wait_for_closure 안인지, 메서드, 경로)로 남긴다. 작업자 대역은 원래 client를 쓴다."""

    def __init__(self, client) -> None:
        self.client = client
        self.in_wait = False
        self.calls: list[tuple[bool, str, str]] = []

    def get(self, path: str, **kw: Any) -> httpx.Response:
        self.calls.append((self.in_wait, "GET", path))
        return self.client.get(path, **kw)

    def post(self, path: str, **kw: Any) -> httpx.Response:
        self.calls.append((self.in_wait, "POST", path))
        return self.client.post(path, **kw)

    def detail_gets(self, in_wait: bool) -> list[str]:
        return [p for w, m, p in self.calls if w == in_wait and m == "GET" and _DETAIL.match(p)]


def ok(r: httpx.Response) -> Any:
    assert r.status_code < 400, r.text
    return r.json()


def close_false_alarm(api, cid: str) -> None:
    """개요의 '오알람으로 닫기'."""
    ok(api.post(f"/api/cases/{cid}/close", json={"outcome": "false_alarm", "reason": "고장과 먼 알림"}))


def resolve(api, cid: str, code: str) -> None:
    """E 코드 카드로 실제 고장 처리: 첫 점검 이상 → 판단(카드 맞아요) → 조치 → 해결됐어요 → 일지 승인 → 해결 종료."""
    d = ok(api.get(f"/api/cases/{cid}"))
    card = next(c for c in d["guidance"]["candidates"] if c["situation_id"] == code)
    first = card["checks"][0]["id"]
    ok(api.put(f"/api/cases/{cid}/checks", json={"checks": {first: {"result": "abnormal"}}}))
    ok(api.post(f"/api/cases/{cid}/judgments", json={"situation_id": code, "verdict": "yes"}))
    ok(
        api.post(
            f"/api/cases/{cid}/actions",
            json={
                "situation_id": code,
                "kind": "조정",
                "did": "시험 조치",
                "parts": [],
                "worker": "시험",
                "started_at": NOW,
                "ended_at": NOW,
            },
        )
    )
    ok(api.post(f"/api/cases/{cid}/results", json={"situation_id": code, "outcome": "resolved"}))
    ok(
        api.put(
            f"/api/cases/{cid}/log",
            json={
                "date": NOW[:10],
                "work_time": "00:10",
                "worker_gun": f"시험 / {d['event']['gun_id']}",
                "problem": "시험",
                "cause": code,
                "action": "시험 조치",
                "missed_checks": "",
                "recurrence": "처음",
                "approved": True,
            },
        )
    )
    ok(api.post(f"/api/cases/{cid}/close", json={"outcome": "resolved"}))


def hand_over(api, cid: str) -> None:
    """상단 '도움 요청' (HandoverInput 본문)."""
    ok(
        api.post(
            f"/api/cases/{cid}/handover",
            json={"to": "보전 담당", "equipment_state": "가동 중", "reasons": ["원인을 못 찾음"], "note": ""},
        )
    )


class Operator:
    """on_update 대역: 중단마다 plan[n번째 중단]으로 케이스를 닫는다 ("false_alarm" · "handover" · E 코드 = 해결)."""

    def __init__(self, api, plan: list[str]) -> None:
        self.api = api
        self.plan = list(plan)
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, node: str, update: dict, state: dict) -> None:
        if node != "__interrupt__":
            return
        self.payloads.append(update)
        act = self.plan[len(self.payloads) - 1]
        cid = update["case_id"]
        if act == "false_alarm":
            close_false_alarm(self.api, cid)
        elif act == "handover":
            hand_over(self.api, cid)
        else:
            resolve(self.api, cid, act)


@pytest.fixture
def run(api, monkeypatch):
    """(script, plan) → (최종 state, Operator, CountingAPI). 드라이버는 진짜 api TestClient에 HTTP 모양 그대로 붙는다."""
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ml = ScriptML()
    monkeypatch.setattr(main, "ml_get", ml.ml_get)
    counted = CountingAPI(api)
    real_wait = hl.wait_for_closure
    waits: list[str] = []

    def _wait(client, case_id, poll_s, stop=None):
        assert client is counted
        waits.append(case_id)
        counted.in_wait = True
        try:
            return real_wait(client, case_id, poll_s, stop)
        finally:
            counted.in_wait = False

    monkeypatch.setattr(hl, "wait_for_closure", _wait)

    def _run(script, plan):
        assert hl.startup_check(ml, api) == []
        feeder = ScriptFeeder(ml, script)
        op = Operator(api, plan)
        final = hl.run_loop(hl.Deps(feeder=feeder, ml=ml, api=counted, ui_url=UI), op, poll_s=0)
        assert final["error"] is None and final["queue"] == []
        # wait_for_closure는 닫힘을 /api/history로만 기다리고, 케이스 상세는 닫힌 뒤 한 번만 읽는다
        assert counted.detail_gets(in_wait=True) == [f"/api/cases/{cid}" for cid in waits]
        in_wait = [(m, p) for w, m, p in counted.calls if w]
        assert all(p == "/api/history" or _DETAIL.match(p) for _, p in in_wait), in_wait
        for cid in waits:  # 상세 GET 바로 앞은 늘 history 조회
            i = in_wait.index(("GET", f"/api/cases/{cid}"))
            assert i > 0 and in_wait[i - 1] == ("GET", "/api/history")
        return final, op, counted

    return _run


def file_item(gun: str, start: str, end: str) -> tuple:
    return ("file", gun, f"test/{gun}.csv", start, end)


def test_scenario_test_0(run, api, make_handoff):
    times = [
        "2021-09-04T16:45:00Z",
        "2021-09-06T19:30:00Z",
        "2021-09-07T19:30:00Z",
        "2021-09-08T19:30:00Z",
        "2021-09-10T19:30:00Z",
    ]
    script = [(file_item("test_0", "2021-09-04T02:46:08Z", "2021-09-11T02:56:08Z"), [])]
    script += [(("alert", i + 1, "test_0", {}), [make_handoff(f"e{i}", t=t)]) for i, t in enumerate(times)]
    script += [
        (("alert", 6, "test_0", {}), [make_handoff("e5", code="E012", t="2021-09-11T02:46:00Z")]),
        (("end",), []),
    ]
    plan = ["false_alarm"] * 5 + ["E012"]
    final, op, counted = run(script, plan)
    rows = final["alerts"]
    # 카드는 E 코드 하나 그대로 (순서를 바꾸지 않는다)
    assert [r["order"] for r in rows] == [["E029"]] * 5 + [["E012"]]
    assert [p["order"] for p in op.payloads] == [r["order"] for r in rows]
    assert [r["applied"] for r in rows] == [[]] + [["R1"]] * 4 + [[]]
    assert rows[0]["note_ko"] is None
    assert [r["note_ko"] for r in rows[1:5]] == [r1("test_0", "E029", n) for n in range(1, 5)]
    assert [p["note_ko"] for p in op.payloads] == [r["note_ko"] for r in rows]
    assert rows[5]["note_ko"] is None  # E012: 다른 코드라 오알람 문장 없음, 지난 점검 기록도 없음
    assert [r["human_label"] for r in rows] == ["false_alarm"] * 5 + ["real"]
    assert [r["closure"]["outcome"] for r in rows] == ["false_alarm"] * 5 + ["resolved"]
    assert [r["data_label"] for r in rows] == ["far"] * 5 + ["pre_failure"]
    assert all(r["agree"] is True for r in rows)
    assert all(r["warn"] is None for r in rows)
    mem = ok(api.get("/api/memory"))
    alarms = {a["key"]: a for a in mem["alarms"]}
    assert set(alarms) == {"test_0|E029", "test_0|E012"}
    assert (alarms["test_0|E029"]["false_alarm"], alarms["test_0|E029"]["real"]) == (5, 0)
    assert (alarms["test_0|E012"]["false_alarm"], alarms["test_0|E012"]["real"]) == (0, 1)
    assert mem["checks"] == [{"check_id": "E012-K1", "abnormal": 1, "normal": 0}]
    assert "past_causes" not in mem
    assert rows[5]["memory_delta"]["checks"] == mem["checks"]
    # 케이스마다 상세 GET은 guide 한 번 + 닫힌 뒤 한 번 (열린 같은 설비 케이스가 없어 반복 확인 GET도 없다)
    assert sorted(counted.detail_gets(in_wait=False)) == sorted(f"/api/cases/{r['case_id']}" for r in rows)


def test_scenario_test_5_then_test_4(run, api, make_handoff):
    script = [
        (file_item("test_5", "2021-09-26T00:00:00Z", "2021-10-03T03:00:00Z"), []),
        (("alert", 1, "test_5", {}), [make_handoff("a1", gun_id="test_5", t="2021-09-28T06:05:00Z")]),
        (("alert", 2, "test_5", {}), [make_handoff("a2", gun_id="test_5", code="E028", t="2021-10-03T02:15:00Z")]),
        (file_item("test_4", "2021-10-11T00:00:00Z", "2021-10-18T12:30:00Z"), []),
        (("alert", 3, "test_4", {}), [make_handoff("b1", gun_id="test_4", t="2021-10-12T10:00:00Z")]),
        (("alert", 4, "test_4", {}), [make_handoff("b2", gun_id="test_4", t="2021-10-16T10:00:00Z")]),
        (("alert", 5, "test_4", {}), [make_handoff("b3", gun_id="test_4", code="E028", t="2021-10-18T12:00:00Z")]),
        (("end",), []),
    ]
    final, op, _ = run(script, ["false_alarm", "E028", "false_alarm", "false_alarm", "E028"])
    rows = final["alerts"]
    assert [r["order"] for r in rows] == [["E029"], ["E028"], ["E029"], ["E029"], ["E028"]]
    assert [r["applied"] for r in rows] == [[], [], [], ["R1"], []]
    assert rows[0]["note_ko"] is None and rows[1]["note_ko"] is None
    assert rows[2]["note_ko"] is None  # t4 첫 E029: 다른 설비라 오알람 문장 없음
    assert rows[3]["note_ko"] == r1("test_4", "E029", 1)
    last = rows[4]["note_ko"]
    # t4 E028: 같은 키의 기록은 없고, test_5에서 남긴 점검 기록만 (설비를 넘어) 참고로 붙는다
    assert "지난 점검 기록(E028): E028-K1 이상 1 · 정상 0" in last
    assert last == "지난 점검 기록(E028): E028-K1 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)."
    assert "같은 설비·코드" not in last and "오알람" not in last
    d = ok(api.get(f"/api/cases/{rows[4]['case_id']}"))
    assert [c["situation_id"] for c in d["guidance"]["candidates"]] == ["E028"]
    assert "field_memory" not in d["guidance"]
    assert d["field_memory"]["checks"] == {"E028-K1": {"abnormal": 1, "normal": 0}}  # 닫힌 뒤에도 자기 기록은 뺀다
    assert [r["human_label"] for r in rows] == ["false_alarm", "real", "false_alarm", "false_alarm", "real"]
    assert [r["data_label"] for r in rows] == ["far", "pre_failure", "far", "far", "pre_failure"]


def test_help_request_absorbs_next_alert(run, api, make_handoff):
    script = [
        (file_item("test_0", "2021-09-04T02:46:08Z", "2021-09-11T02:56:08Z"), []),
        (("alert", 1, "test_0", {}), [make_handoff("e0", t="2021-09-04T16:45:00Z")]),
        (("alert", 2, "test_0", {}), [make_handoff("e1", t="2021-09-06T19:30:00Z")]),
        (("end",), []),
    ]
    final, op, _ = run(script, ["handover"])
    assert len(op.payloads) == 1  # 묶인 알림에서는 멈추지 않는다
    first, second = final["alerts"]
    assert first["order"] == ["E029"]
    assert first["closure"]["outcome"] == "handed_over" and first["human_label"] is None
    assert first["warn"] and "도움 요청" in first["warn"]
    assert second["status"] == "folded" and second["folded_into"] == first["case_id"]
    assert second["warn"] and "도움 요청 상태라" in second["warn"] and "오알람으로 닫기" in second["warn"]
    assert ok(api.get("/api/memory"))["labels"] == {}
