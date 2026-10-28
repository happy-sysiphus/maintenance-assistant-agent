"""D-8 드라이버(ml/.py/hitl_loop.py) ↔ 진짜 api 계약: 서버 없이 한 프로세스 안에서 돈다.

api는 TestClient(임시 DB, FIELD_MEMORY=1)이고 ML은 스크립트 outbox다. 드라이버 그래프를 끝까지 돌리면서,
중단마다 작업자 대역이 UI와 같은 주소 · 본문으로 케이스를 닫는다. 소켓도 실제 CSV도 쓰지 않는다.
"""

from __future__ import annotations

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


def ok(r: httpx.Response) -> Any:
    assert r.status_code < 400, r.text
    return r.json()


def close_false_alarm(api, cid: str) -> None:
    """개요의 '오알람으로 닫기'."""
    ok(api.post(f"/api/cases/{cid}/close", json={"outcome": "false_alarm", "reason": "고장과 먼 알림"}))


def resolve(api, cid: str, steps: list[tuple[str, str]]) -> None:
    """점검 → 판단 → 조치 → 결과 resolved → 일지 승인 → 해결 종료 (UI와 같은 주소 · 본문)."""
    d = ok(api.get(f"/api/cases/{cid}"))
    cands = {c["situation_id"]: c for c in d["guidance"]["candidates"]}
    decisive = None
    for sid, verdict in steps:
        c = cands.get(sid)
        if c and c["checks"]:
            result = "abnormal" if verdict == "yes" else "normal"
            ok(api.put(f"/api/cases/{cid}/checks", json={"checks": {c["checks"][0]["id"]: {"result": result}}}))
        ok(api.post(f"/api/cases/{cid}/judgments", json={"situation_id": sid, "verdict": verdict}))
        if verdict == "yes":
            decisive = sid
            break
    assert decisive is not None
    ok(
        api.post(
            f"/api/cases/{cid}/actions",
            json={
                "situation_id": decisive,
                "kind": "조정",
                "did": "시험 조치",
                "parts": [],
                "worker": "시험",
                "started_at": NOW,
                "ended_at": NOW,
            },
        )
    )
    ok(api.post(f"/api/cases/{cid}/results", json={"situation_id": decisive, "outcome": "resolved"}))
    ok(
        api.put(
            f"/api/cases/{cid}/log",
            json={
                "date": NOW[:10],
                "work_time": "00:10",
                "worker_gun": f"시험 / {d['event']['gun_id']}",
                "problem": "시험",
                "cause": decisive,
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
    """on_update 대역: 중단마다 plan[n번째 중단]으로 케이스를 닫는다."""

    def __init__(self, api, plan: list) -> None:
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
    """(script, plan) → (최종 state, Operator). 드라이버는 진짜 api TestClient에 HTTP 모양 그대로 붙는다."""
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ml = ScriptML()
    monkeypatch.setattr(main, "ml_get", ml.ml_get)

    def _run(script, plan):
        assert hl.startup_check(ml, api) == []
        feeder = ScriptFeeder(ml, script)
        op = Operator(api, plan)
        final = hl.run_loop(hl.Deps(feeder=feeder, ml=ml, api=api, ui_url=UI), op, poll_s=0)
        assert final["error"] is None and final["queue"] == []
        return final, op

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
    plan = ["false_alarm", [("S10", "yes")], "false_alarm", "false_alarm", "false_alarm", [("S01", "yes")]]
    final, op = run(script, plan)
    rows = final["alerts"]
    assert [r["order"] for r in rows] == [["S04", "S10"]] + [["S10", "S04"]] * 4 + [["S01", "S05", "S10"]]
    assert [r["applied"] for r in rows] == [[]] + [["R1"]] * 4 + [[]]
    for n, r in enumerate(rows[1:5], start=1):
        assert f"오알람으로 닫은 기록 {n}건" in r["note_ko"]
    assert [p["order"] for p in op.payloads] == [r["order"] for r in rows]
    # S10 "맞아요"로 해결 종료한 두 번째 E029도 오알람
    assert [r["human_label"] for r in rows] == ["false_alarm"] * 5 + ["real"]
    assert rows[1]["closure"]["outcome"] == "resolved"
    assert [r["data_label"] for r in rows] == ["far"] * 5 + ["pre_failure"]
    assert all(r["warn"] is None for r in rows)
    alarms = {a["key"]: a for a in ok(api.get("/api/memory"))["alarms"]}
    assert (alarms["test_0|E029"]["false_alarm"], alarms["test_0|E029"]["real"]) == (5, 0)
    assert (alarms["test_0|E012"]["false_alarm"], alarms["test_0|E012"]["real"]) == (0, 1)


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
    s08 = [("S03", "no"), ("S06", "no"), ("S08", "yes")]
    final, op = run(script, ["false_alarm", s08, "false_alarm", "false_alarm", s08])
    rows = final["alerts"]
    assert [r["order"] for r in rows] == [
        ["S04", "S10"],
        ["S03", "S06", "S08", "S10"],
        ["S04", "S10"],
        ["S10", "S04"],
        ["S03", "S06", "S08", "S10"],
    ]
    assert [r["applied"] for r in rows] == [[], [], [], ["R1"], []]  # t4 첫 E029: 다른 설비라 R1 없음
    last = rows[4]
    assert "S08" in last["note_ko"] and "1건" in last["note_ko"]
    cands = {c["situation_id"]: c for c in ok(api.get(f"/api/cases/{last['case_id']}"))["guidance"]["candidates"]}
    assert cands["S08"]["evidence_level"] == "field" and cands["S06"]["evidence_level"] == "manual"
    assert [r["human_label"] for r in rows] == ["false_alarm", "real", "false_alarm", "false_alarm", "real"]


def test_help_request_absorbs_next_alert(run, api, make_handoff):
    script = [
        (file_item("test_0", "2021-09-04T02:46:08Z", "2021-09-11T02:56:08Z"), []),
        (("alert", 1, "test_0", {}), [make_handoff("e0", t="2021-09-04T16:45:00Z")]),
        (("alert", 2, "test_0", {}), [make_handoff("e1", t="2021-09-06T19:30:00Z")]),
        (("end",), []),
    ]
    final, op = run(script, ["handover"])
    assert len(op.payloads) == 1  # 묶인 알림에서는 멈추지 않는다
    first, second = final["alerts"]
    assert first["closure"]["outcome"] == "handed_over" and first["human_label"] is None
    assert first["warn"] and "도움 요청" in first["warn"]
    assert second["status"] == "folded" and second["folded_into"] == first["case_id"]
    assert second["warn"] and "도움 요청 상태라" in second["warn"] and "오알람으로 닫기" in second["warn"]
