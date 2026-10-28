"""현장 기억 api 훅: FIELD_MEMORY=1이면 GET 케이스의 field_memory · /api/memory · 챗봇 문단 · 닫기 이유.

RAG 카드(guidance)는 켜도 꺼도 RAG가 만든 그대로다. 끄면 api는 RAG 담당 판(MERGE_HEAD)과 똑같다.
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
NOTE_HEAD = "현장 기록(사람 판정, 매뉴얼 근거 아님)"
R1_1 = (
    "같은 설비·코드(test_0 E029)에서 사람이 오알람으로 닫은 기록 1건 · 고장 확인 0건 → "
    "오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
)


def _card(code: str) -> dict:
    """상세 조회 뒤의 카드: 가짜 RAG 답을 guidance.adapt로 바꾼 것 (현장 기억과 상관없음)."""
    return guidance.adapt(fake_answer_question(DEFAULT_QUESTION, error_code=code, use_cache=True))


def _sids(case: dict) -> list[str]:
    return [c["situation_id"] for c in case["guidance"]["candidates"]]


def _false_alarm_first(api, make_handoff, reason: str = "다른 고장 유형 코드") -> dict:
    """test_0 E029 #1 → CASE-0001을 오알람으로 닫는다."""
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": reason})
    assert r.status_code == 200
    return r.json()


def _note_blocks(blocks: list[dict]) -> list[dict]:
    return [b for b in blocks if b["type"] == "p" and b["parts"][0].get("text", "").startswith(NOTE_HEAD)]


# ---- 켰을 때 ----


def test_flag_on_r1_note_after_false_alarm(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    h1 = make_handoff("e1")
    api.post("/api/ml/handoffs", json=h1)
    assert store.get("CASE-0001")["guidance"] == guidance.guidance_for(h1)
    first = api.get("/api/cases/CASE-0001").json()
    assert _sids(first) == ["E029"]
    assert first["field_memory"]["applied"] == []
    assert first["field_memory"]["note_ko"] is None
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "다른 고장 유형 코드"})
    closure = r.json()["records"]["closure"]
    assert closure["outcome"] == "false_alarm"
    assert closure["reason"] == "다른 고장 유형 코드"
    assert "note" not in closure
    h2 = make_handoff("e2", t="2021-09-06T20:30:09Z")
    assert api.post("/api/ml/handoffs", json=h2).json() == {"result": "new"}
    assert store.get("CASE-0002")["guidance"] == guidance.guidance_for(h2)
    second = api.get("/api/cases/CASE-0002").json()
    assert second["field_memory"]["applied"] == ["R1"]
    assert second["field_memory"]["alarm_key"] == "test_0|E029"
    assert second["field_memory"]["alarm"] == {"false_alarm": 1, "real": 0}
    assert second["field_memory"]["note_ko"] == R1_1
    # 카드는 RAG가 만든 그대로 (순서 · 내용 모두)
    assert second["guidance"] == _card("E029")
    assert _sids(second) == ["E029"]


def test_flag_on_field_memory_not_saved(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    api.post("/api/ml/handoffs", json=make_handoff("e2", t="2021-09-06T20:30:09Z"))
    got = api.get("/api/cases/CASE-0002").json()
    saved = store.get("CASE-0002")
    assert "field_memory" not in saved
    assert saved["guidance"] == got["guidance"]
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
        assert got["field_memory"]["applied"] == []
        assert got["field_memory"]["note_ko"] is None


def test_flag_on_check_history_note(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    api.post("/api/ml/handoffs", json=make_handoff("e1", gun_id="test_5", code="E028"))
    api.put("/api/cases/CASE-0001/checks", json={"checks": {"E028-K1": {"result": "abnormal"}}})
    api.post("/api/cases/CASE-0001/handover", json={"by": "현장", "reason": "테스트"})
    api.post("/api/ml/handoffs", json=make_handoff("e2", gun_id="test_4", code="E028"))
    got = api.get("/api/cases/CASE-0002").json()
    assert got["field_memory"]["checks"] == {"E028-K1": {"abnormal": 1, "normal": 0}}
    assert got["field_memory"]["note_ko"] == (
        "지난 점검 기록(E028): E028-K1 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)."
    )
    assert api.get("/api/memory").json()["checks"] == [{"check_id": "E028-K1", "abnormal": 1, "normal": 0}]


def test_flag_on_memory_endpoint(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    r = api.get("/api/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["labeled"] == 1
    assert body["labels"] == {"CASE-0001": "false_alarm"}
    assert body["alarms"] == [
        {
            "key": "test_0|E029",
            "gun_id": "test_0",
            "code": "E029",
            "false_alarm": 1,
            "real": 0,
            "case_ids": ["CASE-0001"],
        }
    ]
    assert body["checks"] == []
    assert "past_causes" not in body
    assert api.get("/api/health").json()["memory"] == "on"
    assert "/api/memory" not in api.get("/openapi.json").json()["paths"]


def test_flag_on_ingest_repeat_and_duplicate(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "new"}
    assert api.post("/api/ml/handoffs", json=make_handoff("e1")).json() == {"result": "duplicate"}
    assert api.post("/api/ml/handoffs", json=make_handoff("e2")).json() == {"result": "repeat"}


def test_flag_on_chatbot_note(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    got = api.post("/api/assistant/ask", json={"question": Q}).json()
    monkeypatch.delenv("FIELD_MEMORY")
    plain = api.post("/api/assistant/ask", json={"question": Q}).json()
    notes = _note_blocks(got["blocks"])
    assert notes == [
        {"type": "p", "parts": [{"text": f"{NOTE_HEAD}: E029 알림 — test_0 오알람으로 닫음 1 · 고장 확인 0"}]}
    ]
    assert got["blocks"][-1] == notes[0]
    assert got["blocks"][:-1] == plain["blocks"]
    assert {k: v for k, v in got.items() if k != "blocks"} == {k: v for k, v in plain.items() if k != "blocks"}
    assert not any(NOTE_HEAD in str(e) for e in got["evidence"])
    assert NOTE_HEAD not in (got["caution"] or "")


def test_flag_on_chatbot_note_on_rag_error(api, make_handoff, monkeypatch):
    # RAG가 실패해도(키 없음 · 시간 초과) 오류 응답에 error_code가 남아 현장 기록 문단은 붙는다
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    failed = {
        "status": "error",
        "error_code": "E029",
        "answer": None,
        "sections": {"interpretation": "", "summary": ""},
        "causes": [],
        "evidence": [],
        "review_notes": [],
        "error": "APITimeoutError",
    }
    with patch.object(assistant, "answer_question", return_value=failed):
        got = api.post("/api/assistant/ask", json={"question": Q}).json()
    assert got["status"] == "error"
    assert got["blocks"] == [
        {"type": "p", "parts": [{"text": "답변을 생성하지 못했습니다. 다시 시도해 주세요."}]},
        {"type": "p", "parts": [{"text": f"{NOTE_HEAD}: E029 알림 — test_0 오알람으로 닫음 1 · 고장 확인 0"}]},
    ]
    assert got["evidence"] == []
    assert got["caution"] is None


def test_flag_on_chatbot_no_note_without_records(api, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    got = api.post("/api/assistant/ask", json={"question": Q}).json()
    assert _note_blocks(got["blocks"]) == []


def test_flag_on_ask_passes_memory_keyword(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _false_alarm_first(api, make_handoff)
    history = [{"role": "user", "text": "E029 설명"}]
    with patch.object(main.assistant, "answer", return_value={"status": "ok"}) as answer:
        assert api.post("/api/assistant/ask", json={"question": "그럼?", "history": history}).status_code == 200
    assert answer.call_args.args == ("그럼?", [], history)
    assert answer.call_args.kwargs["memory"]["labels"] == {"CASE-0001": "false_alarm"}
    assert api.post("/api/assistant/ask", json={"question": ""}).status_code == 422


# ---- 껐을 때: RAG 담당 판과 같다 ----


def test_flag_off_health_memory_and_openapi(api, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    assert api.get("/api/health").json() == {"status": "ok", "source": "api", "ml": "down", "rag": "e_code_hybrid"}
    r = api.get("/api/memory")
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
    assert "/api/memory" not in api.get("/openapi.json").json()["paths"]


def test_flag_off_case_unchanged(api, make_handoff, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    _false_alarm_first(api, make_handoff)
    r = api.post("/api/cases/CASE-0001/close", json={"outcome": "false_alarm", "reason": "x", "note": "y"})
    assert set(r.json()["records"]["closure"]) == {"outcome", "at"}
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
    # RAG 담당 판(MERGE_HEAD)의 assistant.answer가 만드는 모양을 손으로 다시 짠 기대값
    raw = fake_answer_question(Q, history=[], use_cache=True)
    assert got == {
        **raw,
        "situation_ids": ["E029"],
        "blocks": assistant.blocks_from_text(raw["answer"]),
        "caution": " ".join(raw["review_notes"]) or None,
        "evidence": [{**e, "title": f"{e['doc']} PDF p.{e['page']}"} for e in raw["evidence"]],
    }
    assert _note_blocks(got["blocks"]) == []
    history = [{"role": "user", "text": "E029 설명"}]
    with patch.object(main.assistant, "answer", return_value={"status": "ok"}) as answer:
        api.post("/api/assistant/ask", json={"question": "그럼?", "history": history})
    assert answer.call_args.args == ("그럼?", [], history)
    assert answer.call_args.kwargs == {}
