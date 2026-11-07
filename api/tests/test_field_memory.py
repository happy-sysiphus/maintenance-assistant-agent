"""현장 기억(field_memory): 라벨 규칙 · 집계 · 참고 문장(note_ko) · 주석 · 요약 · 챗봇 문장.

R1은 문장만 붙이고 카드는 바꾸지 않는다. R2(카드 점검 순서)는 위치만 바꾸고 글 · 근거 · 조치는 그대로 둔다.
"""

from __future__ import annotations

import copy
import json

import field_memory as fm
import guidance
import pytest
from rag.response import mapped_causes


def _r1(gun: str, code: str, fa: int, real: int) -> str:
    return (
        f"같은 설비·코드({gun} {code})에서 사람이 오알람으로 닫은 기록 {fa}건 · 고장 확인 {real}건 → "
        "오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
    )


def _history(gun: str, code: str, fa: int, real: int) -> str:
    return f"같은 설비·코드({gun} {code}) 지난 기록: 오알람 {fa}건 · 고장 확인 {real}건 (참고)."


def test_enabled_reads_env_each_call(monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert fm.enabled() is True
    monkeypatch.delenv("FIELD_MEMORY")
    assert fm.enabled() is False


def test_removed_names_are_gone():
    for name in ("links", "guidance_for", "ALARM_TRUTH", "_baseline", "code_of", "_past_causes"):
        assert not hasattr(fm, name), name


def test_alarm_key_uses_rule_code(make_handoff):
    assert fm.alarm_key(make_handoff("e1", code="E029")) == "test_0|E029"
    assert fm.alarm_key(make_handoff("e2", code="E029", source="model")) == "test_0|"


# ---- 라벨 ----


def test_label_false_alarm_close(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1"), closure="false_alarm", judgments=[("E029", "yes")])
    assert fm.alarm_label(c) == "false_alarm"


def test_label_resolved_is_real(make_handoff, make_case):
    c = make_case(
        "C1", make_handoff("e1"), closure="resolved", judgments=[("E029", "yes")], results=[("E029", "resolved")]
    )
    assert fm.alarm_label(c) == "real"
    m = make_case(
        "C2", make_handoff("e2"), closure="resolved", results=[("MANUAL", "resolved")], manual={"cause": "호스 꺾임"}
    )
    assert fm.alarm_label(m) == "real"
    # 결정 원인이 없어도 해결 종료면 고장 확인
    assert fm.alarm_label(make_case("C3", make_handoff("e3"), closure="resolved")) == "real"


def test_label_unlabelled(make_handoff, make_case):
    for closure in ("handed_over", "unresolved", None):
        c = make_case("C1", make_handoff("e1"), closure=closure, judgments=[("E029", "yes")])
        assert fm.alarm_label(c) is None


# ---- 집계 ----


def test_build_counts_alarms_per_key(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029"), closure="false_alarm"),
        make_case("C3", make_handoff("e3", code="E012"), closure="resolved", results=[("E012", "resolved")]),
        make_case("C4", make_handoff("e4", gun_id="test_5", code="E029"), closure="resolved"),
    ]
    m = fm.build(cases)
    assert set(m) == {"alarms", "checks", "manual_causes", "missed_checks", "labels", "labeled"}
    assert m["alarms"]["test_0|E029"] == {
        "gun_id": "test_0",
        "code": "E029",
        "false_alarm": 2,
        "real": 0,
        "case_ids": ["C1", "C2"],
    }
    assert m["alarms"]["test_0|E012"]["real"] == 1
    assert m["alarms"]["test_0|E012"]["false_alarm"] == 0
    assert m["alarms"]["test_5|E029"]["real"] == 1
    assert m["labels"] == {"C1": "false_alarm", "C2": "false_alarm", "C3": "real", "C4": "real"}
    assert m["labeled"] == 4


def test_build_manual_causes(make_handoff, make_case):
    cases = [
        make_case(
            "C1",
            make_handoff("e1", code="E012"),
            closure="resolved",
            results=[("MANUAL", "resolved")],
            manual={"cause": "호스 꺾임"},
        ),
        make_case("C2", make_handoff("e2", code="E028"), closure="resolved", results=[("E028", "resolved")]),
    ]
    m = fm.build(cases)
    assert m["manual_causes"] == [{"case_id": "C1", "code": "E012", "cause": "호스 꺾임"}]


def test_build_checks_and_missed(make_handoff, make_case):
    cases = [
        make_case(
            "C1",
            make_handoff("e1"),
            closure="false_alarm",
            checks={"E029-K1": {"result": "abnormal"}, "E029-K2": {"result": "skipped"}},
            log={"missed_checks": "에어 호스 확인"},
        ),
        make_case(
            "C2",
            make_handoff("e2"),
            closure="handed_over",
            checks={"E029-K1": {"result": "normal"}, "E029-K2": {"result": "normal"}},
            log={"missed_checks": "  "},
        ),
    ]
    m = fm.build(cases)
    assert m["checks"] == {"E029-K1": {"abnormal": 1, "normal": 1}, "E029-K2": {"abnormal": 0, "normal": 1}}
    assert m["missed_checks"] == [{"case_id": "C1", "code": "E029", "text": "에어 호스 확인"}]
    # 인계(handed_over)는 점검은 세지만 라벨은 없다
    assert m["labels"] == {"C1": "false_alarm"}


def test_build_model_only_code_falls_back_to_terminal_code(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E028", source="model"), closure="false_alarm", log={"missed_checks": "x"})
    ]
    m = fm.build(cases)
    assert m["missed_checks"] == [{"case_id": "C1", "code": "E028", "text": "x"}]
    assert m["alarms"]["test_0|"]["code"] is None


def test_build_ignores_malformed_log_and_checks(make_handoff, make_case):
    # PUT /log · PUT /checks 본문이 이상해도 build()가 멈추지 않는다
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm", log={"missed_checks": ["a", "b"]}),
        make_case("C2", make_handoff("e2"), closure="false_alarm", log={"missed_checks": None}),
        make_case("C3", make_handoff("e3"), closure="false_alarm", checks={"E029-K1": "abnormal"}),
    ]
    m = fm.build(cases)
    assert m["missed_checks"] == []
    assert m["checks"] == {}
    assert m["labeled"] == 3


def test_build_ignores_malformed_judgments_and_results(make_handoff, make_case):
    # POST /judgments · /results 본문이 이상해도(키 없음, dict 아님) build() · snapshot()이 멈추지 않는다
    c1 = make_case("C1", make_handoff("e1", code="E012"), closure="resolved")
    c1["records"]["judgments"] = [{"situation_id": "E012"}, "yes", {"verdict": "yes"}]
    c1["records"]["results"] = [{"situation_id": "MANUAL"}, None]
    c2 = make_case("C2", make_handoff("e2", code="E028"), closure="resolved")
    c2["records"]["results"] = [{"outcome": "resolved"}]
    c3 = make_case("C3", make_handoff("e3", code="E016"), closure="resolved")
    del c3["records"]["judgments"]
    del c3["records"]["results"]
    assert fm.decisive(c1) is None
    assert fm.decisive(c2) is None
    assert fm.decisive(c3) is None
    m = fm.build([c1, c2, c3])
    assert m["labels"] == {"C1": "real", "C2": "real", "C3": "real"}
    assert m["manual_causes"] == []
    assert fm.snapshot([c1, c2, c3])["labeled"] == 3


def test_build_exclude_and_open_cases(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm", checks={"E029-K1": {"result": "abnormal"}}),
        make_case(
            "C2",
            make_handoff("e2"),
            judgments=[("E029", "yes")],
            checks={"E029-K1": {"result": "abnormal"}},
            log={"missed_checks": "에어 호스"},
        ),
    ]
    m = fm.build(cases, exclude="C1")
    assert m == {
        "alarms": {},
        "checks": {},
        "manual_causes": [],
        "missed_checks": [],
        "labels": {},
        "labeled": 0,
    }


# ---- 참고 문장 (annotate) ----


def test_annotate_cold_start(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1"))
    out = fm.annotate(c, [c])
    assert out["field_memory"] == {
        "alarm_key": "test_0|E029",
        "alarm": {"false_alarm": 0, "real": 0},
        "applied": [],
        "note_ko": None,
        "checks": {},
        "labeled": 0,
    }


def test_r1_sentence_and_applied(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm"),
        make_case("C2", make_handoff("e2"), closure="false_alarm"),
        make_case("C3", make_handoff("e3"), closure="resolved", results=[("E029", "resolved")]),
    ]
    target = make_case("C4", make_handoff("e4"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["applied"] == ["R1"]
    assert fm_["alarm"] == {"false_alarm": 2, "real": 1}
    assert fm_["note_ko"] == _r1("test_0", "E029", 2, 1)


def test_r1_tie_gives_history_sentence(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm"),
        make_case("C2", make_handoff("e2"), closure="resolved", results=[("E029", "resolved")]),
    ]
    target = make_case("C3", make_handoff("e3"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["applied"] == []
    assert fm_["note_ko"] == _history("test_0", "E029", 1, 1)


def test_history_sentence_when_real_wins(make_handoff, make_case):
    cases = [make_case("C1", make_handoff("e1"), closure="resolved")]
    target = make_case("C2", make_handoff("e2"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["applied"] == []
    assert fm_["note_ko"] == _history("test_0", "E029", 0, 1)


def test_other_gun_and_other_code_unaffected(make_handoff, make_case):
    cases = [make_case("C1", make_handoff("e1"), closure="false_alarm")]
    for target in (
        make_case("C2", make_handoff("e2", gun_id="test_6")),
        make_case("C3", make_handoff("e3", code="E012")),
    ):
        fm_ = fm.annotate(target, [*cases, target])["field_memory"]
        assert fm_["applied"] == []
        assert fm_["alarm"] == {"false_alarm": 0, "real": 0}
        assert fm_["note_ko"] is None


def test_model_only_no_alarm_sentence(make_handoff, make_case):
    # 모델 신호만(규칙 코드 없음): 키 "gun|"을 모든 코드가 같이 쓰므로 R1도 오알람 문장도 없다 (R-8)
    cases = [
        make_case("C1", make_handoff("e1", code="E012", source="model"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029", source="model"), closure="false_alarm"),
    ]
    target = make_case("C3", make_handoff("e3", code="E029", source="model"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["alarm_key"] == "test_0|"
    assert fm_["alarm"] == {"false_alarm": 2, "real": 0}
    assert fm_["applied"] == []
    assert fm_["note_ko"] is None  # 문장에 "None" 글자가 찍히지 않음


def test_model_only_keeps_check_history(make_handoff, make_case):
    cases = [
        make_case(
            "C1",
            make_handoff("e1", code="E028", source="model"),
            closure="false_alarm",
            checks={"E028-K2": {"result": "abnormal"}},
        )
    ]
    target = make_case("C2", make_handoff("e2", code="E028", source="model"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["applied"] == []
    assert fm_["note_ko"] == "지난 점검 기록(E028): E028-K2 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)."


def test_check_history_card_order_only_counted_cross_gun(make_handoff, make_case):
    # 카드 순서 E016-K1, K2, K3. 기록은 K3(다른 설비)가 먼저 저장돼도 카드 순서로, K2(기록 없음)는 빠진다
    cases = [
        make_case(
            "C1",
            make_handoff("e1", gun_id="test_5", code="E016"),
            closure="handed_over",
            checks={"E016-K3": {"result": "abnormal"}, "E029-K1": {"result": "abnormal"}},
        ),
        make_case("C2", make_handoff("e2", code="E016"), closure="handed_over", checks={"E016-K1": {"result": "normal"}}),
    ]
    target = make_case("C3", make_handoff("e3", code="E016"))
    card = [k["id"] for c in target["guidance"]["candidates"] for k in c["checks"]]
    assert card == ["E016-K1", "E016-K2", "E016-K3"]
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert list(fm_["checks"]) == ["E016-K1", "E016-K3"]
    assert fm_["checks"] == {"E016-K1": {"abnormal": 0, "normal": 1}, "E016-K3": {"abnormal": 1, "normal": 0}}
    # 인계만 있으면 라벨이 없어 오알람 문장은 없다
    assert fm_["note_ko"] == (
        "지난 점검 기록(E016): E016-K1 이상 0 · 정상 1, E016-K3 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)."
    )


def test_r1_and_check_history_joined(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm", checks={"E029-K2": {"result": "normal"}}),
    ]
    target = make_case("C2", make_handoff("e2"))
    fm_ = fm.annotate(target, [*cases, target])["field_memory"]
    assert fm_["applied"] == ["R1"]
    assert fm_["note_ko"] == (
        _r1("test_0", "E029", 1, 0)
        + " 지난 점검 기록(E029): E029-K2 이상 0 · 정상 1 (참고, 점검 순서는 매뉴얼 기준)."
    )


def test_annotate_excludes_self_and_keeps_card(make_handoff, make_case):
    h = make_handoff("e1")
    c1 = make_case("C1", h, closure="false_alarm", checks={"E029-K1": {"result": "abnormal"}})
    c2 = make_case(
        "C2",
        make_handoff("e2"),
        closure="false_alarm",
        checks={"E029-K2": {"result": "normal"}, "E012-K1": {"result": "abnormal"}},
    )
    out = fm.annotate(c1, [c1, c2])
    assert "field_memory" not in c1
    assert out["guidance"] == c1["guidance"] == guidance.guidance_for(h)
    assert {k: v for k, v in out.items() if k != "field_memory"} == c1
    assert out["field_memory"]["alarm"] == {"false_alarm": 1, "real": 0}
    assert out["field_memory"]["checks"] == {"E029-K2": {"abnormal": 0, "normal": 1}}
    assert out["field_memory"]["labeled"] == 1
    assert out["field_memory"]["applied"] == ["R1"]


def test_annotate_no_card(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1", code=None, source="model"))
    assert c["guidance"]["candidates"] == []
    out = fm.annotate(c, [c])
    assert out["field_memory"]["checks"] == {}
    assert out["field_memory"]["note_ko"] is None


# ---- 요약 · 챗봇 ----


def test_snapshot_shapes(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", gun_id="test_6", code="E029"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029"), closure="false_alarm"),
        make_case(
            "C3",
            make_handoff("e3", code="E028"),
            closure="resolved",
            results=[("E028", "resolved")],
            checks={"E028-K1": {"result": "abnormal"}},
        ),
    ]
    s = fm.snapshot(cases)
    assert set(s) == {"labeled", "labels", "alarms", "checks", "manual_causes", "missed_checks"}
    assert s["labeled"] == 3
    assert s["labels"] == {"C1": "false_alarm", "C2": "false_alarm", "C3": "real"}
    assert [a["key"] for a in s["alarms"]] == ["test_0|E028", "test_0|E029", "test_6|E029"]
    assert s["alarms"][2] == {
        "key": "test_6|E029",
        "gun_id": "test_6",
        "code": "E029",
        "false_alarm": 1,
        "real": 0,
        "case_ids": ["C1"],
    }
    assert s["checks"] == [{"check_id": "E028-K1", "abnormal": 1, "normal": 0}]
    assert s["manual_causes"] == []
    assert s["missed_checks"] == []


HEAD = "현장 기록(사람 판정, 매뉴얼 근거 아님): "


def test_chat_note_alarms_only(make_handoff, make_case):
    snap = fm.snapshot(
        [
            make_case("C1", make_handoff("e1", gun_id="test_5"), closure="false_alarm"),
            make_case("C2", make_handoff("e2"), closure="resolved"),
        ]
    )
    assert fm.chat_note("E029", snap) == (
        HEAD + "E029 알림 — test_0 오알람으로 닫음 0 · 고장 확인 1, test_5 오알람으로 닫음 1 · 고장 확인 0"
    )
    assert fm.chat_note("E012", snap) is None


def test_chat_note_checks_only(make_handoff, make_case):
    snap = fm.snapshot(
        [
            make_case(
                "C1",
                make_handoff("e1", code="E028"),
                closure="handed_over",
                checks={"E028-K2": {"result": "normal"}, "E028-K1": {"result": "abnormal"}, "E029-K1": {"result": "normal"}},
            )
        ]
    )
    assert fm.chat_note("E028", snap) == HEAD + "E028 알림 — 지난 점검: E028-K1 이상 1 · 정상 0, E028-K2 이상 0 · 정상 1 (참고)"


def test_chat_note_both(make_handoff, make_case):
    snap = fm.snapshot(
        [make_case("C1", make_handoff("e1", code="E028"), closure="resolved", checks={"E028-K1": {"result": "abnormal"}})]
    )
    assert fm.chat_note("E028", snap) == (
        HEAD + "E028 알림 — test_0 오알람으로 닫음 0 · 고장 확인 1 / 지난 점검: E028-K1 이상 1 · 정상 0 (참고)"
    )


def test_chat_note_none(make_handoff, make_case):
    assert fm.chat_note("E029", fm.snapshot([])) is None
    snap = fm.snapshot([make_case("C1", make_handoff("e1"), closure="false_alarm")])
    assert fm.chat_note(None, snap) is None
    assert fm.chat_note("", snap) is None
    # 모델 신호만(코드 자리 None) 기록은 코드별 문장에 들어가지 않는다
    model = fm.snapshot([make_case("C1", make_handoff("e1", source="model"), closure="false_alarm")])
    assert fm.chat_note("E029", model) is None


# ---- R2: 카드 점검 순서 (ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md 3절) ----

K1, K2 = "E029-K1", "E029-K2"
_BASE3 = [("g1", K2), ("g2", K2), ("g1", K2)]
# RAG 캐시 카드(384d49e794ce) 모양: 원인 1이 K1 · K2와 재시작 주의를 갖고, 원인 2는 점검 · 조치가 없다
_CACHED_E029 = [("E029-C1", [K1, K2]), ("E029-C2", [])]


def _resolved(make_case, make_handoff, cid, gun, marks, code="E029", decisive=None, source="rule"):
    """해결 종료 케이스: marks = {점검 id: 결과}. 결정 원인은 decisive, 없으면 '{code}-1'."""
    sid = decisive or f"{code}-1"
    return make_case(
        cid,
        make_handoff(f"e-{cid}", gun_id=gun, code=code, source=source),
        closure="resolved",
        judgments=[(sid, "yes")],
        results=[(sid, "resolved")],
        checks={k: {"result": r} for k, r in marks.items()},
    )


def _stream(make_case, make_handoff, items, code="E029"):
    """(설비, 점검 id) 하나 = 그 점검만 이상인 해결 종료 케이스 하나."""
    return [
        _resolved(make_case, make_handoff, f"S{i}", gun, {check: "abnormal"}, code=code)
        for i, (gun, check) in enumerate(items)
    ]


def _card(make_handoff, code, groups):
    """guidance_for 카드의 원인을 groups = [(매핑 원인 id, [점검 id…])]로 다시 묶는다 (RAG 카드 흉내)."""
    g = guidance.guidance_for(make_handoff(f"card-{code}", code=code))
    item = guidance.mapping()[code]
    mapped = dict(zip([c["id"] for c in item["causes"]], mapped_causes(code, item)))
    checks = {k["id"]: k for c in mapped.values() for k in c["checks"]}
    causes = [{**copy.deepcopy(mapped[cid]), "checks": [copy.deepcopy(checks[k]) for k in ids]} for cid, ids in groups]
    g["causes"] = causes
    g["candidates"][0]["causes"] = copy.deepcopy(causes)
    return g


def _layout(g):
    return [[k["id"] for k in c.get("checks") or []] for c in g["causes"]]


def _r2(lead, counts, guns):
    """R2 문장 (스펙 5.2): lead 먼저, 나머지는 매핑 순서."""
    rest = " · ".join(f"{k} {n}건" for k, n in counts.items() if k != lead)
    return (
        "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 "
        f"{lead} {counts[lead]}건 · {rest}(설비 {guns}대)이라 {lead}부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로."
    )


def test_order_counts_only_resolved_abnormal(make_handoff, make_case):
    counted = [
        _resolved(make_case, make_handoff, "C1", "g1", {K2: "abnormal"}),
        _resolved(make_case, make_handoff, "C2", "g2", {K2: "abnormal", K1: "normal"}),
    ]
    k2 = {"checks": {K2: {"result": "abnormal"}}}
    ignored = [
        make_case("C3", make_handoff("e3", gun_id="g3"), closure="false_alarm", **k2),
        make_case("C4", make_handoff("e4", gun_id="g3"), closure="unresolved", **k2),
        make_case("C5", make_handoff("e5", gun_id="g3"), closure="handed_over", **k2),
        make_case("C6", make_handoff("e6", gun_id="g3"), **k2),  # 열림
        _resolved(make_case, make_handoff, "C7", "g3", {K2: "abnormal"}, decisive="MANUAL"),  # 카드 밖 원인
        _resolved(make_case, make_handoff, "C8", "g3", {K2: "normal"}),
        _resolved(make_case, make_handoff, "C9", "g3", {K2: "skipped"}),
        make_case("C10", make_handoff("e10", gun_id="g3"), closure="resolved", checks={K2: "abnormal"}),  # dict 아님
        _resolved(make_case, make_handoff, "C11", "g3", {"E029-K9": "abnormal"}),  # 매핑에 없는 id
        _resolved(make_case, make_handoff, "C12", "g3", {"E012-K2": "abnormal"}, code="E012"),  # 다른 코드
    ]
    assert fm.order_counts("E029", counted + ignored) == ({K1: 0, K2: 2}, {K1: set(), K2: {"g1", "g2"}})


def test_order_model_only_counts_as_terminal_code(make_handoff, make_case):
    cases = [_resolved(make_case, make_handoff, "C1", "g1", {K2: "abnormal"}, source="model")]
    assert fm.order_counts("E029", cases) == ({K1: 0, K2: 1}, {K1: set(), K2: {"g1"}})


@pytest.mark.parametrize(
    ("items", "lead", "counts"),
    [
        (_BASE3, K2, {K1: 0, K2: 3}),
        (_BASE3[:2], None, None),
        ([*_BASE3, ("g1", K1)], None, None),
        ([*_BASE3, ("g1", K1), ("g2", K2)], K2, {K1: 1, K2: 4}),
        ([("g1", K1), ("g2", K1), ("g1", K1), ("g1", K2), ("g2", K2), ("g1", K2)], None, None),
        ([("g1", K2)] * 3, None, None),
        ([("g1", K1), ("g2", K1)] * 2 + [("g1", K1)], None, None),
    ],
    ids=["3:0", "2:0", "3:1", "4:1", "tie", "one-gun", "first-in-mapping"],
)
def test_order_for_lead_rule(make_handoff, make_case, items, lead, counts):
    order = fm.order_for("E029", _stream(make_case, make_handoff, items))
    if lead is None:
        assert order is None
    else:
        assert (order["lead"], order["counts"], order["guns"]) == (lead, counts, 2)


def test_order_for_e016_three_checks(make_handoff, make_case):
    k2, k3 = "E016-K2", "E016-K3"

    def lead(items):
        order = fm.order_for("E016", _stream(make_case, make_handoff, items, code="E016"))
        return order and order["lead"]

    assert lead([("g1", k3), ("g2", k3), ("g1", k3), ("g2", k3), ("g1", k2)]) == k3
    assert lead([("g1", k2), ("g2", k2), ("g1", k2), ("g1", k3), ("g2", k3), ("g1", k3)]) == k2
    # 가장 많은 K3가 한 대에서만 나왔으면, 조건을 만족하는 K2가 있어도 앞세우지 않는다(가장 많은 항목만 본다)
    assert lead([("g1", k3)] * 5 + [("g1", k2), ("g2", k2), ("g1", k2)]) is None


def test_order_for_note_and_shape(make_handoff, make_case):
    cases = _stream(make_case, make_handoff, _BASE3)
    order = fm.order_for("E029", cases)
    assert order == {"lead": K2, "counts": {K1: 0, K2: 3}, "guns": 2, "note_ko": _r2(K2, {K1: 0, K2: 3}, 2)}
    assert list(order["counts"]) == [K1, K2]
    assert order["note_ko"] == (
        "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 "
        "E029-K2 3건 · E029-K1 0건(설비 2대)이라 E029-K2부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로."
    )
    assert fm.order_for("E999", cases) is None
    assert fm.order_for(None, cases) is None
    assert fm.mapping_checks("E016") == ["E016-K1", "E016-K2", "E016-K3"]
    assert fm.mapping_checks(None) == []


_E016 = [("E016-C1", ["E016-K1", "E016-K2"]), ("E016-C2", ["E016-K3"])]


def _split_e029(make_handoff, c2_actions):
    """매핑대로 나뉜 E029 카드 [[K1], [K2]]의 원인 2에 조치를 둔다 (원인 1은 재시작 주의)."""
    g = _card(make_handoff, "E029", [("E029-C1", [K1]), ("E029-C2", [K2])])
    for causes in (g["causes"], g["candidates"][0]["causes"]):
        causes[1]["actions"] = c2_actions(copy.deepcopy(causes[0]["actions"]))
    return g


@pytest.mark.parametrize(
    ("c2_actions", "moved"),
    [
        (lambda caution: caution, True),  # 재시작 주의가 두 원인에 같은 항목으로 있다
        (lambda caution: [*caution, {"text": "다른 조치"}], True),  # 앞세울 원인이 더 많이 갖는다
        (lambda caution: [{**caution[0], "text": "글이 다른 주의"}], False),  # 같은 항목이 아니다
        (lambda caution: [{"text": "다른 조치"}], False),
    ],
    ids=["same-caution", "superset", "reworded", "other-action"],
)
def test_apply_order_guard_passes_when_lead_cause_carries_actions(make_handoff, c2_actions, moved):
    # 조치 보호(11-08 완화): 앞 원인의 조치를 앞세울 원인이 모두 갖고 있으면 옮긴다. 어느 원인을 골라도 그 조치가 보인다
    g = _split_e029(make_handoff, c2_actions)
    out = fm.apply_order(g, K2)
    if not moved:
        assert out is g
        return
    assert _layout(out) == [[K2], [K1]] and fm.first_check(out) == K2
    assert [c["actions"] for c in out["causes"]] == [g["causes"][1]["actions"], g["causes"][0]["actions"]]
    assert [[k["id"] for k in c["checks"]] for c in out["candidates"][0]["causes"]] == [[K2], [K1]]


@pytest.mark.parametrize(
    ("code", "groups", "lead", "expected"),
    [
        ("E029", _CACHED_E029, K2, [[K2, K1], []]),
        # 조치 보호: 재시작 주의가 있는 원인 1을 조치 없는 원인 뒤로 밀지 않는다
        ("E029", [("E029-C1", [K1]), ("E029-C2", [K2])], K2, None),
        ("E016", _E016, "E016-K3", [["E016-K3"], ["E016-K1", "E016-K2"]]),
        ("E016", _E016, "E016-K2", [["E016-K2", "E016-K1"], ["E016-K3"]]),
        ("E012", [("E012-C1", ["E012-K1"]), ("E012-C2", ["E012-K2"])], "E012-K2", [["E012-K2"], ["E012-K1"]]),
        ("E012", [("E012-C1", ["E012-K1", "E012-K2"])], "E012-K2", [["E012-K2", "E012-K1"]]),
        (
            "E012",
            [("E012-C1", ["E012-K1", "E012-K2"]), ("E012-C2", ["E012-K2"])],
            "E012-K2",
            [["E012-K2", "E012-K1"], ["E012-K2"]],
        ),
        ("E029", [("E029-C1", [K1])], K2, None),  # 원인 목록에 lead가 없다
    ],
    ids=["e029-cached", "e029-split-guard", "e016-k3", "e016-k2", "e012-split", "e012-cached", "e012-dup", "absent"],
)
def test_apply_order_layouts(make_handoff, code, groups, lead, expected):
    g = _card(make_handoff, code, groups)
    out = fm.apply_order(g, lead)
    if expected is None:
        assert out is g
        return
    assert _layout(out) == expected
    assert [[k["id"] for k in c["checks"]] for c in out["candidates"][0]["causes"]] == expected
    assert out["candidates"][0]["checks"][0]["id"] == lead
    assert fm.first_check(out) == lead
    if code == "E029":
        # 조치는 원인을 따라간다: 캐시 카드의 재시작 주의는 원인 1에 남는다
        assert out["causes"][0]["actions"] and not out["causes"][1]["actions"]


def test_apply_order_keeps_content(make_handoff):
    g = _card(make_handoff, "E029", _CACHED_E029)
    before = copy.deepcopy(g)
    once = fm.apply_order(g, K2)
    assert g == before  # 입력은 고치지 않는다
    assert fm.apply_order(once, K2) == once  # 두 번 적용해도 같다

    def key(x):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)

    def bag(gg):
        causes = [{**c, "checks": sorted(c["checks"], key=key)} for c in gg["causes"]]
        return sorted(causes, key=key), sorted(gg["candidates"][0]["checks"], key=key)

    assert bag(once) == bag(g)  # 위치만 다르다
    for k in ("status", "response_version", "answer", "sections", "evidence"):
        assert once.get(k) == g.get(k)
    assert fm.apply_order(g, None) is g


def test_apply_order_empty_causes_and_missing_checks(make_handoff):
    g = _card(make_handoff, "E029", _CACHED_E029)
    g["causes"] = []
    out = fm.apply_order(g, K2)
    assert [k["id"] for k in out["candidates"][0]["checks"]] == [K2, K1]
    assert out["causes"] == []
    assert fm.first_check(out) == K2
    # 점검 · 조치 키가 없는 원인이 앞에 있어도 오류 없이 옮긴다
    g2 = _card(make_handoff, "E029", [("E029-C2", [K2])])
    g2["causes"].insert(0, {"name": "x"})
    out2 = fm.apply_order(g2, K2)
    assert _layout(out2) == [[K2], []]
    assert out2["causes"][1] == {"name": "x"}


def test_first_check_and_order_pinned(make_handoff, make_case):
    g = _card(make_handoff, "E029", _CACHED_E029)
    assert fm.first_check(g) == K1
    assert fm.first_check(_card(make_handoff, "E029", [("E029-C2", []), ("E029-C1", [K1])])) is None
    g["causes"] = []
    assert fm.first_check(g) == K1  # 원인 목록이 비면 candidates[0].checks
    assert fm.first_check({}) is None
    assert fm.first_check({"causes": [], "candidates": []}) is None
    case = make_case("C1", make_handoff("e1"))
    assert fm.order_pinned(case) is False
    case["records"]["judgments"].append({"situation_id": "E029-1", "verdict": "yes", "at": "t"})
    assert fm.order_pinned(case) is False  # field_order가 없다
    case["field_order"] = {"lead": K2}
    assert fm.order_pinned(case) is True
    case["records"]["judgments"] = []
    case["records"]["checks"] = {K2: {"result": "abnormal"}}
    assert fm.order_pinned(case) is False  # 점검 기록만으로는 고정하지 않는다(점검 id로 저장된다)
    for key in ("actions", "results"):
        case["records"][key] = [{"situation_id": "E029-1", "at": "t"}]
        assert fm.order_pinned(case) is True
        case["records"][key] = []


# ---- R2: annotate (스펙 5.2) ----

_R2_HISTORY = "지난 점검 기록(E029): E029-K2 이상 3 · 정상 0, E029-K1 이상 0 · 정상 3 (참고)."


def _r2_cases(make_case, make_handoff):
    """K1 정상 · K2 이상으로 해결 종료한 3건 (test_7 1건, test_0 2건)."""
    marks = {K1: "normal", K2: "abnormal"}
    return [
        _resolved(make_case, make_handoff, "C1", "test_7", marks),
        _resolved(make_case, make_handoff, "C2", "test_0", marks),
        _resolved(make_case, make_handoff, "C3", "test_0", marks),
    ]


def _r2_target(make_case, make_handoff, cases, gun="test_0", groups=_CACHED_E029):
    """ingest가 만드는 것처럼: 순서 결정을 field_order에, 적용한 카드를 guidance에."""
    target = make_case("C9", make_handoff("e9", gun_id=gun))
    order = fm.order_for("E029", cases)
    target["field_order"] = order
    target["guidance"] = fm.apply_order(_card(make_handoff, "E029", groups), order["lead"])
    return target


def test_r2_applied_when_lead_first(make_handoff, make_case):
    cases = _r2_cases(make_case, make_handoff)
    target = _r2_target(make_case, make_handoff, cases)
    m = fm.annotate(target, [*cases, target])["field_memory"]
    assert m["applied"] == ["R2"]
    assert m["note_ko"] == " ".join([_history("test_0", "E029", 0, 2), _r2(K2, {K1: 0, K2: 3}, 2), _R2_HISTORY])
    assert list(m["checks"]) == [K2, K1]  # 지난 점검 기록도 카드 순서(lead 먼저)


def test_r2_absent_when_card_not_moved(make_handoff, make_case):
    cases = _r2_cases(make_case, make_handoff)
    target = _r2_target(make_case, make_handoff, cases, groups=[("E029-C1", [K1]), ("E029-C2", [K2])])
    # 매핑 대체 카드: 조치 보호로 그대로다. field_order는 있다
    assert target["field_order"]["lead"] == K2 and fm.first_check(target["guidance"]) == K1
    m = fm.annotate(target, [*cases, target])["field_memory"]
    assert m["applied"] == []
    assert "순서를 바꿈" not in m["note_ko"]
    assert m["note_ko"].endswith(
        "지난 점검 기록(E029): E029-K1 이상 0 · 정상 3, E029-K2 이상 3 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)."
    )


def test_r1_and_r2_together(make_handoff, make_case):
    false_alarms = [make_case(f"F{i}", make_handoff(f"f{i}", gun_id="test_9"), closure="false_alarm") for i in range(2)]
    cases = _r2_cases(make_case, make_handoff) + false_alarms
    target = _r2_target(make_case, make_handoff, cases, gun="test_9")
    m = fm.annotate(target, [*cases, target])["field_memory"]
    assert m["applied"] == ["R1", "R2"]
    assert m["note_ko"] == " ".join([_r1("test_9", "E029", 2, 0), _r2(K2, {K1: 0, K2: 3}, 2), _R2_HISTORY])
