"""현장 기억(field_memory): 라벨 규칙 · 집계 · 참고 문장(note_ko) · 주석 · 요약 · 챗봇 문장.

안내 카드(guidance)는 RAG가 만든 그대로다. 현장 기억은 문장만 붙이고 순서 · 내용은 바꾸지 않는다.
"""

from __future__ import annotations

import field_memory as fm
import guidance


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
