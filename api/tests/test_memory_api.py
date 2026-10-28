"""현장 기억 api 훅: FIELD_MEMORY=1이면 안내 순서 · 케이스 주석 · /api/memory · 챗봇 한 줄, 끄면 지금과 같다."""

from __future__ import annotations

import assistant
import guidance
import store

Q = "E029 알림 진짜야?"
NOTE_HEAD = "현장 기록(사람 판정, 매뉴얼 근거 아님)"


def _sids(case: dict) -> list[str]:
    return [c["situation_id"] for c in case["guidance"]["candidates"]]


def _false_alarm_first(api, make_handoff, reason: str = "다른 고장 유형 코드") -> None:
    """test_0 E029 #1 → CASE-0001을 오알람으로 닫는다."""
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": reason})
    assert r.status_code == 200


def _note_blocks(blocks: list[dict]) -> list[dict]:
    return [b for b in blocks if b["type"] == "p" and b["parts"][0].get("text", "").startswith(NOTE_HEAD)]


def test_flag_on_r1_after_false_alarm(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    api.post("/api/ml/handoffs", json=make_handoff("e1"))
    first = api.get("/api/cases/CASE-0001").json()
    assert _sids(first) == ["S04", "S10"]
    assert first["guidance"]["memory"]["applied"] == []
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "다른 고장 유형 코드"})
    closure = r.json()["records"]["closure"]
    assert closure["outcome"] == "false_alarm"
    assert closure["reason"] == "다른 고장 유형 코드"
    assert "note" not in closure
    assert api.post("/api/ml/handoffs", json=make_handoff("e2", t="2021-09-06T20:30:09Z")).json() == {"result": "new"}
    second = api.get("/api/cases/CASE-0002").json()
    assert _sids(second) == ["S10", "S04"]
    assert second["guidance"]["memory"]["applied"] == ["R1"]


def test_flag_on_other_gun_and_code_unaffected(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    api.post("/api/ml/handoffs", json=make_handoff("e2", gun_id="test_1"))
    api.post("/api/ml/handoffs", json=make_handoff("e3", code="E012"))
    other_gun = api.get("/api/cases/CASE-0002").json()
    assert _sids(other_gun) == ["S04", "S10"]
    assert other_gun["guidance"]["memory"]["applied"] == []
    other_code = api.get("/api/cases/CASE-0003").json()
    assert _sids(other_code) == ["S01", "S05", "S10"]
    assert other_code["guidance"]["memory"]["applied"] == []


def test_flag_on_get_case_has_field_memory_not_saved(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    api.post("/api/ml/handoffs", json=make_handoff("e2", t="2021-09-06T20:30:09Z"))
    got = api.get("/api/cases/CASE-0002").json()
    assert got["field_memory"]["alarm_key"] == "test_0|E029"
    assert got["field_memory"]["alarm"] == {"false_alarm": 1, "real": 0}
    assert "field_memory" not in store.get("CASE-0002")
    # 닫힌 케이스 자기 자신은 빼고 센다
    assert api.get("/api/cases/CASE-0001").json()["field_memory"]["alarm"] == {"false_alarm": 0, "real": 0}


def test_flag_on_memory_endpoint(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    r = api.get("/api/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["alarms"][0]["key"] == "test_0|E029"
    assert body["alarms"][0]["false_alarm"] == 1
    assert body["labels"] == {"CASE-0001": "false_alarm"}
    assert api.get("/api/health").json()["memory"] == "on"


def test_flag_on_chatbot_note(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    got = api.post("/api/assistant/ask", json={"question": Q}).json()
    monkeypatch.delenv("FIELD_MEMORY")
    plain = api.post("/api/assistant/ask", json={"question": Q}).json()
    notes = _note_blocks(got["blocks"])
    assert len(notes) == 1
    assert got["blocks"][-1] == notes[0]
    assert all("cite" not in p for p in notes[0]["parts"])
    assert got["status"] == "retrieval_only"
    assert got["evidence"] == plain["evidence"]
    assert got["situation_ids"] == plain["situation_ids"]
    assert got["caution"] == plain["caution"]
    assert got["blocks"][:-1] == plain["blocks"]


def test_flag_off_identical(api, make_handoff, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    health = api.get("/api/health").json()
    assert "memory" not in health
    r = api.get("/api/memory")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
    assert "/api/memory" not in api.get("/openapi.json").json()["paths"]

    _false_alarm_first(api, make_handoff)
    assert set(store.get("CASE-0001")["records"]["closure"]) == {"outcome", "at"}
    h2 = make_handoff("e2", t="2021-09-06T20:30:09Z")
    api.post("/api/ml/handoffs", json=h2)
    saved = store.get("CASE-0002")
    assert saved["guidance"] == guidance.guidance_for(h2)
    assert _sids(saved) == ["S04"]
    assert "field_memory" not in api.get("/api/cases/CASE-0002").json()
    assert api.post("/api/assistant/ask", json={"question": Q}).json() == assistant.answer(Q, [], [])
