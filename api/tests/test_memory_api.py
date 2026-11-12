"""이전 기록 api 훅: FIELD_MEMORY=1이면 GET 케이스의 field_memory(alarm_key · alarm · recent)와 닫기 사유 저장.

기록을 보여 주기만 한다. 안내 카드 · 챗봇 답은 켜도 끌 때와 같다. 끄면 api는 RAG 담당 판과 똑같다.
"""

from __future__ import annotations

from unittest.mock import patch

import assistant
import guidance
import main
import store
from conftest import fake_answer_question
from rag.service import DEFAULT_QUESTION

Q = "E029 알림 진짜야?"
NOW = "2021-09-11T03:00:00Z"


def _card(code: str) -> dict:
    """상세 조회 뒤의 카드: 가짜 RAG 답을 guidance.adapt로 바꾼 것 (이전 기록과 상관없음)."""
    return guidance.adapt(fake_answer_question(DEFAULT_QUESTION, error_code=code, use_cache=True))


def _false_alarm_first(api, make_handoff, reason: str = "센서 · 신호 문제") -> dict:
    """test_0 E029 #1 → CASE-0001을 오알람으로 닫는다."""
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": reason})
    assert r.status_code == 200
    return r.json()


def _resolve(api, cid: str, judge: str) -> None:
    """해결 종료까지: 점검 → 판단 맞아요 → 조치 → 해결 → 일지 승인 → 닫기."""
    code = judge.split("-")[0]
    steps = [
        ("put", "/checks", {"checks": {f"{code}-K1": {"result": "abnormal"}}}),
        ("post", "/judgments", {"situation_id": judge, "verdict": "yes"}),
        (
            "post",
            "/actions",
            {"situation_id": judge, "kind": "조정", "did": "시험 조치", "parts": [], "worker": "시험",
             "started_at": NOW, "ended_at": NOW},
        ),
        ("post", "/results", {"situation_id": judge, "outcome": "resolved"}),
        (
            "put",
            "/log",
            {"date": NOW[:10], "work_time": "00:10", "worker_gun": "시험", "problem": "시험", "cause": judge,
             "action": "시험 조치", "missed_checks": "", "recurrence": "처음", "approved": True},
        ),
        ("post", "/close", {"outcome": "resolved"}),
    ]
    for method, path, body in steps:
        r = getattr(api, method)(f"/api/cases/{cid}{path}", json=body)
        assert r.status_code == 200, (path, r.text)


# ---- 켰을 때 ----


def test_flag_on_recent_after_false_alarm(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    first = api.get("/api/cases/CASE-0001").json()
    assert first["field_memory"] == {
        "alarm_key": "test_0|E029",
        "alarm": {"false_alarm": 0, "real": 0},
        "recent": [],
    }
    closed = api.post(
        "/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "센서 · 신호 문제"}
    ).json()
    assert api.post("/api/ml/handoffs", json=make_handoff("e2", t="2021-09-06T20:30:09Z")).json() == {"result": "new"}
    second = api.get("/api/cases/CASE-0002").json()
    assert second["field_memory"] == {
        "alarm_key": "test_0|E029",
        "alarm": {"false_alarm": 1, "real": 0},
        "recent": [
            {
                "case_id": "CASE-0001",
                "outcome": "false_alarm",
                "closed_at": closed["records"]["closure"]["at"],
                "cause": None,
                "reason": "센서 · 신호 문제",
            }
        ],
    }
    # 이전 기록은 안내 카드를 바꾸지 않는다
    assert second["guidance"] == _card("E029")


def test_flag_on_field_memory_not_saved(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    api.post("/api/ml/handoffs", json=make_handoff("e2", t="2021-09-06T20:30:09Z"))
    got = api.get("/api/cases/CASE-0002").json()
    saved = store.get("CASE-0002")
    assert "field_memory" not in saved
    assert {k: v for k, v in got.items() if k != "field_memory"} == saved
    # 닫힌 케이스 자기 자신은 빼고 센다
    assert api.get("/api/cases/CASE-0001").json()["field_memory"]["alarm"] == {"false_alarm": 0, "real": 0}


def test_flag_on_other_gun_and_code_unaffected(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    api.post("/api/ml/handoffs", json=make_handoff("e2", gun_id="test_1"))
    api.post("/api/ml/handoffs", json=make_handoff("e3", code="E012"))
    for cid, code in (("CASE-0002", "E029"), ("CASE-0003", "E012")):
        got = api.get(f"/api/cases/{cid}").json()
        assert got["guidance"] == _card(code)
        assert got["field_memory"]["alarm"] == {"false_alarm": 0, "real": 0}
        assert got["field_memory"]["recent"] == []


def test_flag_on_recent_resolved_cause_newest_first_max_three(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    # 같은 설비 · 코드로 네 건: 오알람 → 해결 → 오알람 → 해결 (닫은 순서대로)
    for i in range(1, 5):
        api.post("/api/ml/handoffs", json=make_handoff(f"e{i}", t=f"2021-09-06T2{i}:30:09Z"))
        cid = f"CASE-000{i}"
        card = api.get(f"/api/cases/{cid}").json()["guidance"]
        if i % 2:
            r = api.post(f"/api/cases/{cid}/close", json={"outcome": "false_alarm", "reason": f"사유 {i}"})
            assert r.status_code == 200
        else:
            _resolve(api, cid, "E029-1")
    api.post("/api/ml/handoffs", json=make_handoff("e5", t="2021-09-07T01:30:09Z"))
    m = api.get("/api/cases/CASE-0005").json()["field_memory"]
    assert m["alarm"] == {"false_alarm": 2, "real": 2}
    assert [r["case_id"] for r in m["recent"]] == ["CASE-0004", "CASE-0003", "CASE-0002"]
    assert [r["outcome"] for r in m["recent"]] == ["resolved", "false_alarm", "resolved"]
    # 해결이면 맞다고 판단한 원인 이름, 오알람이면 닫을 때 남긴 사유
    assert m["recent"][0]["cause"] == card["causes"][0]["name"] and m["recent"][0]["reason"] is None
    assert m["recent"][1]["cause"] is None and m["recent"][1]["reason"] == "사유 3"


def test_flag_on_close_keeps_reason_and_note(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    api.post("/api/ml/handoffs", json=make_handoff("e1"))
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "기타", "note": "메모"})
    assert {k: v for k, v in r.json()["records"]["closure"].items() if k != "at"} == {
        "outcome": "false_alarm",
        "reason": "기타",
        "note": "메모",
    }
    # 빈 문자열은 남기지 않는다
    api.post("/api/ml/handoffs", json=make_handoff("e2", gun_id="test_1"))
    r = api.post("/api/cases/CASE-0002/close", json={"outcome": "false_alarm", "reason": " ", "note": ""})
    assert set(r.json()["records"]["closure"]) == {"outcome", "at"}


def test_flag_on_ingest_repeat_and_duplicate(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "duplicate"}
    assert api.post("/api/ml/handoffs", json=make_handoff("e2")).json() == {"result": "repeat"}


def test_flag_on_health_and_no_memory_route(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    assert api.get("/api/health").json()["memory"] == "on"
    # 요약 주소는 없앴다: 켜도 없는 주소다
    assert api.get("/api/memory").status_code == 404
    assert "/api/memory" not in api.get("/openapi.json").json()["paths"]


def test_flag_on_chatbot_same_as_flag_off(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    on = api.post("/api/assistant/ask", json={"question": Q}).json()
    monkeypatch.delenv("FIELD_MEMORY")
    assert on == api.post("/api/assistant/ask", json={"question": Q}).json()


# ---- 껐을 때: RAG 담당 판과 같다 ----


def test_flag_off_health(api, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    assert api.get("/api/health").json() == {"status": "ok", "source": "api", "ml": "down", "rag": "e_code_hybrid"}
    assert api.get("/api/memory").status_code == 404


def test_flag_off_case_unchanged(api, make_handoff, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    # 닫힌 케이스는 다시 닫을 수 없으므로(409, 11/04) 처음 닫을 때 reason · note를 같이 보내 무시되는지 본다
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "x", "note": "y"})
    assert set(r.json()["records"]["closure"]) == {"outcome", "at"}
    assert api.post("/api/cases/CASE-0001/close", json={"outcome": "resolved"}).status_code == 409
    assert set(store.get("CASE-0001")["records"]["closure"]) == {"outcome", "at"}
    h2 = make_handoff("e2", t="2021-09-06T20:30:09Z")
    api.post("/api/ml/handoffs", json=h2)
    assert store.get("CASE-0002")["guidance"] == guidance.guidance_for(h2)
    got = api.get("/api/cases/CASE-0002").json()
    assert "field_memory" not in got
    assert got == store.get("CASE-0002")
    assert got["guidance"] == _card("E029")


def test_flag_off_chatbot_identical(api, make_handoff, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    _false_alarm_first(api, make_handoff)
    got = api.post("/api/assistant/ask", json={"question": Q}).json()
    assert got == assistant.answer(Q, [], [])
    # RAG 담당 판의 assistant.answer가 만드는 모양을 손으로 다시 짠 기대값
    raw = fake_answer_question(Q, history=[], use_cache=True)
    assert got == {
        **raw,
        "situation_ids": ["E029"],
        "blocks": assistant.blocks_from_text(raw["answer"]),
        "caution": " ".join(raw["review_notes"]) or None,
        "evidence": [{**e, "title": f"{e['doc']} PDF p.{e['page']}"} for e in raw["evidence"]],
    }
    history = [{"role": "user", "text": "E029 설명"}]
    with patch.object(main.assistant, "answer", return_value={"status": "ok"}) as answer:
        api.post("/api/assistant/ask", json={"question": "그럼?", "history": history})
    assert answer.call_args.args == ("그럼?", [], history)
    assert answer.call_args.kwargs == {}
    assert api.post("/api/assistant/ask", json={"question": ""}).status_code == 422
