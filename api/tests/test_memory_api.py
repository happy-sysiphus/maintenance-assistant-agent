"""현장 기억 api 훅: FIELD_MEMORY=1이면 GET 케이스의 field_memory · /api/memory · 챗봇 문단 · 닫기 이유 · R2 카드 순서.

R1은 카드를 바꾸지 않는다. R2는 새 케이스 카드의 점검 순서만 바꾼다. 끄면 api는 RAG 담당 판(MERGE_HEAD)과 똑같다.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import assistant
import guidance
import main
import store
from conftest import fake_answer_question
from rag.documents import mapping
from rag.response import mapped_causes
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
    # R1은 카드를 바꾸지 않는다 (해결 종료가 없어 R2 순서도 없다)
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
    assert api.post("/api/memory").status_code == 405  # 켜면 보통 GET 주소


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


def test_flag_off_memory_like_unknown_route(api, monkeypatch):
    """끄면 /api/memory는 없는 주소와 같다: 다른 메서드는 405가 아니고 끝 슬래시는 307이 아니라 404."""
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    for method in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
        for slash in ("", "/"):
            got = api.request(method, "/api/memory" + slash, follow_redirects=False)
            ref = api.request(method, "/api/no-such-route" + slash, follow_redirects=False)
            assert got.status_code == ref.status_code == 404, (method, slash)
            assert got.content == ref.content, (method, slash)
            assert got.headers.get("content-type") == ref.headers.get("content-type"), (method, slash)
            assert "allow" not in got.headers, (method, slash)


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


# ---- R2: 카드 점검 순서 (ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md 4 · 8절) ----

K1, K2 = "E029-K1", "E029-K2"
N, A = "normal", "abnormal"
NOW = "2021-09-11T03:00:00Z"
# RAG 캐시 카드 모양: E029는 원인 1이 K1 · K2와 재시작 주의를 갖고 원인 2는 비었다(384d49e794ce), E012는 원인 하나(81b481dc2af8)
CACHED = {
    "E029": [("E029-C1", [K1, K2]), ("E029-C2", [])],
    "E012": [("E012-C1", ["E012-K1", "E012-K2"])],
}
# 스펙 8절 대본: (event_id, 설비, 코드, 데이터 시각, 점검 기록, 캐시 카드 판단, 매핑 대체 카드 판단)
SCRIPT = [
    ("t7-1", "test_7", "E029", "2021-09-08T17:55:05Z", {K1: N, K2: A}, "E029-1", "E029-2"),
    ("t0-1", "test_0", "E029", "2021-09-04T16:45:09Z", {K1: N, K2: A}, "E029-1", "E029-2"),
    ("t0-2", "test_0", "E029", "2021-09-06T19:30:09Z", {K1: N, K2: A}, "E029-1", "E029-2"),
    ("t0-3", "test_0", "E029", "2021-09-07T19:30:10Z", {K2: N, K1: A}, "E029-1", "E029-1"),
    ("t0-4", "test_0", "E029", "2021-09-08T19:30:10Z", {K1: N, K2: A}, "E029-1", "E029-2"),
    ("t0-5", "test_0", "E029", "2021-09-10T19:30:10Z", {K2: A}, "E029-1", "E029-2"),
    ("t0-6", "test_0", "E012", "2021-09-11T02:46:08Z", {"E012-K1": A}, "E012-1", "E012-1"),
]


def _regroup(code: str, groups: list[tuple[str, list[str]]]) -> list[dict]:
    """매핑 원인을 groups = [(원인 id, [점검 id…])]로 다시 묶은 RAG 원인 목록."""
    item = mapping()[code]
    mapped = dict(zip([c["id"] for c in item["causes"]], mapped_causes(code, item)))
    checks = {k["id"]: k for c in mapped.values() for k in c["checks"]}
    return [{**copy.deepcopy(mapped[cid]), "checks": [copy.deepcopy(checks[k]) for k in ids]} for cid, ids in groups]


def _cached_rag(monkeypatch) -> dict:
    """RAG 캐시 카드 모양의 가짜 답. ctl["fail"]이면 오류, ctl["hook"]이 있으면 답하기 전에 부른다."""
    ctl: dict = {"calls": [], "fail": False, "hook": None}

    def fake(question, *, error_code=None, history=None, use_cache=False, **kw):
        ctl["calls"].append(error_code)
        if ctl["hook"]:
            ctl["hook"]()
        if ctl["fail"]:
            return {"status": "error", "error": "test", "error_code": error_code}
        raw = fake_answer_question(question, error_code=error_code, history=history, use_cache=use_cache, **kw)
        if error_code in CACHED:
            raw["causes"] = _regroup(error_code, CACHED[error_code])
        return raw

    monkeypatch.setattr(guidance, "answer_question", fake)
    return ctl


def _layout(case: dict) -> list[list[str]]:
    return [[k["id"] for k in c.get("checks") or []] for c in case["guidance"]["causes"]]


def _r2_note(n2: int, n1: int) -> str:
    return (
        "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 "
        f"E029-K2 {n2}건 · E029-K1 {n1}건(설비 2대)이라 E029-K2부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로."
    )


def _orders() -> list[dict | None]:
    def order(n2: int, n1: int) -> dict:
        return {"lead": K2, "counts": {K1: n1, K2: n2}, "guns": 2, "note_ko": _r2_note(n2, n1)}

    return [None, None, None, order(3, 0), None, order(4, 1), None]


def _resolve(api, cid: str, marks: dict[str, str], judge: str) -> dict[str, dict]:
    """해결 종료까지(test_driver_contract.resolve와 같은 순서): 점검 → 판단 맞아요 → 조치 → 해결 → 일지 승인 → 닫기."""
    steps = [
        ("checks", "put", "/checks", {"checks": {k: {"result": r} for k, r in marks.items()}}),
        ("judgments", "post", "/judgments", {"situation_id": judge, "verdict": "yes"}),
        (
            "actions",
            "post",
            "/actions",
            {"situation_id": judge, "kind": "조정", "did": "시험 조치", "parts": [], "worker": "시험",
             "started_at": NOW, "ended_at": NOW},
        ),
        ("results", "post", "/results", {"situation_id": judge, "outcome": "resolved"}),
        (
            "log",
            "put",
            "/log",
            {"date": NOW[:10], "work_time": "00:10", "worker_gun": "시험", "problem": "시험", "cause": judge,
             "action": "시험 조치", "missed_checks": "", "recurrence": "처음", "approved": True},
        ),
        ("close", "post", "/close", {"outcome": "resolved"}),
    ]
    out = {}
    for name, method, path, body in steps:
        r = getattr(api, method)(f"/api/cases/{cid}{path}", json=body)
        assert r.status_code == 200, (name, r.text)
        out[name] = r.json()
    return out


def _run_script(api, make_handoff, judge: int, rows=SCRIPT) -> list[dict]:
    """대본 줄마다 알림 → 상세 조회 → 대본대로 해결 종료. judge는 SCRIPT 줄의 판단 칸(5 캐시, 6 대체)."""
    seen = []
    for row in rows:
        eid, gun, code, t, marks = row[:5]
        h = make_handoff(eid, gun_id=gun, code=code, t=t)
        assert api.post("/api/ml/handoffs", json=h).json() == {"result": "new"}
        cid = store.open_case(gun, code)["case_id"]
        got = api.get(f"/api/cases/{cid}").json()
        m = got.get("field_memory") or {}
        seen.append({"cid": cid, "layout": _layout(got), "applied": m.get("applied"), "note_ko": m.get("note_ko"),
                     "field_order": got.get("field_order")})
        _resolve(api, cid, marks, row[judge])
    return seen


def _new_e029(api, make_handoff, eid: str, gun: str = "test_0") -> str:
    api.post("/api/ml/handoffs", json=make_handoff(eid, gun_id=gun))
    return store.open_case(gun, "E029")["case_id"]


def test_r2_script_cached_card(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ctl = _cached_rag(monkeypatch)
    seen = _run_script(api, make_handoff, judge=5)
    moved, kept = [[K2, K1], []], [[K1, K2], []]
    assert [s["layout"] for s in seen] == [kept, kept, kept, moved, kept, moved, [["E012-K1", "E012-K2"]]]
    assert [s["applied"] for s in seen] == [[], [], [], ["R2"], [], ["R2"], []]
    assert [s["field_order"] for s in seen] == _orders()
    assert _r2_note(3, 0) in seen[3]["note_ko"] and _r2_note(4, 1) in seen[5]["note_ko"]
    assert all("순서를 바꿈" not in (s["note_ko"] or "") for i, s in enumerate(seen) if i not in (3, 5))
    assert len(ctl["calls"]) == 7  # 케이스마다 첫 상세 조회에서 한 번


def test_r2_script_mapping_card(api, make_handoff, monkeypatch):
    # 매핑 대체 카드(RAG 키 · Neo4j 없음)는 재시작 주의가 있는 원인 1을 밀 수 없어 순서가 그대로다(조치 보호)
    monkeypatch.setenv("FIELD_MEMORY", "1")
    seen = _run_script(api, make_handoff, judge=6)
    assert [s["layout"] for s in seen] == [[[K1], [K2]]] * 6 + [[["E012-K1"], ["E012-K2"]]]
    assert [s["applied"] for s in seen] == [[]] * 7
    assert [s["field_order"] for s in seen] == _orders()


def test_r2_write_responses(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "w1")
    api.get(f"/api/cases/{cid}")
    stored = store.get(cid)["guidance"]
    for name, body in _resolve(api, cid, {K2: A}, "E029-1").items():
        assert body["guidance"] == stored, name
        assert _layout(body) == [[K2, K1], []], name
        assert body["field_order"]["lead"] == K2, name


def test_r2_open_case_keeps_card(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _cached_rag(monkeypatch)
    a = _new_e029(api, make_handoff, "a1", gun="test_5")  # 순서가 정해지기 전에 연 케이스
    assert _layout(api.get(f"/api/cases/{a}").json()) == [[K1, K2], []]
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    assert _layout(api.get(f"/api/cases/{a}").json()) == [[K1, K2], []]
    b = _new_e029(api, make_handoff, "b1", gun="test_6")
    assert _layout(api.get(f"/api/cases/{b}").json()) == [[K2, K1], []]
    for eid, gun in (("k1", "test_8"), ("k2", "test_9")):
        _resolve(api, _new_e029(api, make_handoff, eid, gun=gun), {K1: A}, "E029-1")
    got = api.get(f"/api/cases/{b}").json()
    assert _layout(got) == [[K2, K1], []] and got["field_memory"]["applied"] == ["R2"]
    # 차이가 3보다 작아졌으므로 다음 새 케이스는 RAG 순서다
    c = _new_e029(api, make_handoff, "c1", gun="test_4")
    got = api.get(f"/api/cases/{c}").json()
    assert _layout(got) == [[K1, K2], []] and "field_order" not in got


def test_r2_repeat_decides_nothing(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "b1")
    api.get(f"/api/cases/{cid}")
    before = store.get(cid)
    h = make_handoff("b2", t="2021-09-08T19:30:10Z")
    assert api.post("/api/ml/handoffs", json=h).json() == {"result": "repeat"}
    after = store.get(cid)
    assert after["guidance"] == before["guidance"] and after["field_order"] == before["field_order"]
    assert len(after["repeats"]) == 1


def test_r2_rag_error_then_ok(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ctl = _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "x1")
    n = len(ctl["calls"])
    ctl["fail"] = True
    first = api.get(f"/api/cases/{cid}").json()
    assert first["guidance"]["status"] == "error"
    assert _layout(first) == [[K1], [K2]]  # 매핑 카드는 조치 보호로 그대로
    assert first["field_memory"]["applied"] == [] and first["field_order"]["lead"] == K2
    ctl["fail"] = False
    second = api.get(f"/api/cases/{cid}").json()
    assert _layout(second) == [[K2, K1], []] and second["field_memory"]["applied"] == ["R2"]
    assert len(ctl["calls"]) - n == 2  # 오류 한 번, 성공 한 번


def test_r2_pin_after_judgment(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ctl = _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    ctl["fail"] = True
    pinned, loose = _new_e029(api, make_handoff, "p1"), _new_e029(api, make_handoff, "p2", gun="test_5")
    for cid in (pinned, loose):
        api.get(f"/api/cases/{cid}")
    api.post(f"/api/cases/{pinned}/judgments", json={"situation_id": "E029-1", "verdict": "yes"})
    api.put(f"/api/cases/{loose}/checks", json={"checks": {K2: {"result": A}}})
    ctl["fail"] = False
    n = len(ctl["calls"])
    got = api.get(f"/api/cases/{pinned}").json()
    assert _layout(got) == [[K1], [K2]] and got["guidance"]["status"] == "error"
    assert len(ctl["calls"]) == n  # 고정된 케이스는 RAG를 부르지 않는다
    assert _layout(api.get(f"/api/cases/{loose}").json()) == [[K2, K1], []]  # 점검만 있으면 고정하지 않는다
    assert len(ctl["calls"]) == n + 1


def test_r2_pin_when_judgment_lands_during_generation(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ctl = _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "g1")

    def judge_meanwhile():  # RAG가 답을 만드는 사이에 작업자가 판단을 저장한다
        d = store.get(cid)
        d["records"]["judgments"].append({"situation_id": "E029-1", "verdict": "yes", "at": NOW})
        store.put(d)

    ctl["hook"] = judge_meanwhile
    api.get(f"/api/cases/{cid}")
    saved = store.get(cid)
    assert _layout(saved) == [[K1], [K2]] and saved["guidance"]["status"] == "retrieval_only"
    assert len(saved["records"]["judgments"]) == 1


def test_r2_history_cause_name_after_move(api, make_handoff, monkeypatch):
    # 매핑 E016 [[K1,K2],[K3]]: 조치는 원인 2(K3)에만 있어 원인을 옮길 수 있다
    monkeypatch.setenv("FIELD_MEMORY", "1")
    k1, k2, k3 = "E016-K1", "E016-K2", "E016-K3"
    for i, gun in enumerate(("g1", "g2", "g1")):
        api.post("/api/ml/handoffs", json=make_handoff(f"m{i}", gun_id=gun, code="E016"))
        cid = store.open_case(gun, "E016")["case_id"]
        api.get(f"/api/cases/{cid}")
        _resolve(api, cid, {k3: A}, "E016-2")
    api.post("/api/ml/handoffs", json=make_handoff("m9", gun_id="g3", code="E016"))
    cid = store.open_case("g3", "E016")["case_id"]
    got = api.get(f"/api/cases/{cid}").json()
    assert _layout(got) == [[k3], [k1, k2]] and got["field_memory"]["applied"] == ["R2"]
    _resolve(api, cid, {k3: A}, "E016-1")
    item = next(i for i in api.get("/api/history").json()["items"] if i["case_id"] == cid)
    c2 = next(c for c in mapping()["E016"]["causes"] if c["id"] == "E016-C2")
    assert item["cause"] == c2["name"]


def test_flag_off_no_field_order(api, make_handoff, monkeypatch):
    monkeypatch.delenv("FIELD_MEMORY", raising=False)
    _cached_rag(monkeypatch)
    for eid, gun, code, t, marks, judge, _ in SCRIPT[:4]:
        h = make_handoff(eid, gun_id=gun, code=code, t=t)
        api.post("/api/ml/handoffs", json=h)
        cid = store.open_case(gun, code)["case_id"]
        assert store.get(cid)["guidance"] == guidance.guidance_for(h)
        got = api.get(f"/api/cases/{cid}").json()
        assert "field_order" not in got and "field_memory" not in got
        if eid != "t0-3":
            assert all("field_order" not in r for r in _resolve(api, cid, marks, judge).values())
    assert _layout(got) == [[K1, K2], []]
    assert all("field_order" not in d for d in store.all_cases())


def test_flag_on_row_kept_with_flag_off(api, make_handoff, monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "k1")
    assert _layout(api.get(f"/api/cases/{cid}").json()) == [[K2, K1], []]
    monkeypatch.delenv("FIELD_MEMORY")
    got = api.get(f"/api/cases/{cid}").json()
    assert _layout(got) == [[K2, K1], []] and got["field_order"]["lead"] == K2 and "field_memory" not in got
    monkeypatch.setattr(guidance, "PROMPT_VERSION", "test-bump")
    again = api.get(f"/api/cases/{cid}").json()
    assert again["guidance"]["response_version"] == "test-bump"  # 다시 만들어졌다
    assert _layout(again) == [[K2, K1], []]


def test_r2_pin_kept_with_flag_off(api, make_handoff, monkeypatch):
    # 스펙 4.5: 플래그를 켜고 순서를 정한 케이스는 끈 뒤에도 고정을 지킨다(판단 뒤 RAG를 부르지 않는다)
    monkeypatch.setenv("FIELD_MEMORY", "1")
    ctl = _cached_rag(monkeypatch)
    _run_script(api, make_handoff, judge=5, rows=SCRIPT[:3])
    cid = _new_e029(api, make_handoff, "q1")
    assert _layout(api.get(f"/api/cases/{cid}").json()) == [[K2, K1], []]
    api.post(f"/api/cases/{cid}/judgments", json={"situation_id": "E029-1", "verdict": "yes"})
    monkeypatch.delenv("FIELD_MEMORY")
    monkeypatch.setattr(guidance, "PROMPT_VERSION", "test-bump")
    n = len(ctl["calls"])
    got = api.get(f"/api/cases/{cid}").json()
    assert len(ctl["calls"]) == n and got["guidance"]["response_version"] != "test-bump"
    assert _layout(got) == [[K2, K1], []] and got["field_order"]["lead"] == K2
