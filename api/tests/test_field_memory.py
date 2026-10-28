"""현장 기억(field_memory): 라벨 규칙 · 집계 · 안내 순서 · 주석 · 요약 · 챗봇 문장."""

from __future__ import annotations

import field_memory as fm
import guidance


def test_enabled_reads_env_each_call(monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert fm.enabled() is True
    monkeypatch.delenv("FIELD_MEMORY")
    assert fm.enabled() is False


def test_links_from_file():
    assert fm.links() == {"S01": ["S05"], "S02": ["S07"], "S03": ["S06", "S08"]}
    assert fm.links().get("S04", []) == []


def test_alarm_key_uses_rule_code(make_handoff):
    assert fm.alarm_key(make_handoff("e1", code="E029")) == "test_0|E029"
    assert fm.alarm_key(make_handoff("e2", code="E029", source="model")) == "test_0|"


def test_label_false_alarm_close(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1"), closure="false_alarm")
    assert fm.alarm_label(c) == "false_alarm"


def test_label_s10_resolved_is_false_alarm(make_handoff, make_case):
    c = make_case(
        "C1", make_handoff("e1"), closure="resolved", judgments=[("S10", "yes")], results=[("S10", "resolved")]
    )
    assert fm.alarm_label(c) == "false_alarm"


def test_label_s10_yes_then_false_alarm_close(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1"), closure="false_alarm", judgments=[("S10", "yes")])
    assert fm.alarm_label(c) == "false_alarm"


def test_label_resolved_other_is_real(make_handoff, make_case):
    c = make_case(
        "C1", make_handoff("e1"), closure="resolved", judgments=[("S04", "yes")], results=[("S04", "resolved")]
    )
    assert fm.alarm_label(c) == "real"
    m = make_case(
        "C2", make_handoff("e2"), closure="resolved", results=[("MANUAL", "resolved")], manual={"cause": "호스 꺾임"}
    )
    assert fm.alarm_label(m) == "real"


def test_label_unlabelled(make_handoff, make_case):
    for closure in ("handed_over", "unresolved", None):
        c = make_case("C1", make_handoff("e1"), closure=closure, judgments=[("S04", "yes")])
        assert fm.alarm_label(c) is None


def test_decisive_prefers_last_resolved_result(make_handoff, make_case):
    c = make_case(
        "C1",
        make_handoff("e1", code="E012"),
        closure="resolved",
        judgments=[("S01", "yes")],
        results=[("S01", "next"), ("S05", "resolved")],
    )
    assert fm.decisive(c) == "S05"


def test_build_counts_alarms_per_key(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029"), closure="false_alarm"),
        make_case("C3", make_handoff("e3", code="E012"), closure="resolved", results=[("S01", "resolved")]),
    ]
    m = fm.build(cases)
    assert m["alarms"]["test_0|E029"] == {
        "gun_id": "test_0",
        "code": "E029",
        "false_alarm": 2,
        "real": 0,
        "case_ids": ["C1", "C2"],
    }
    assert m["alarms"]["test_0|E012"]["real"] == 1
    assert m["alarms"]["test_0|E012"]["false_alarm"] == 0
    assert m["labels"] == {"C1": "false_alarm", "C2": "false_alarm", "C3": "real"}
    assert m["labeled"] == 3


def test_build_causes_only_real_decisive(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E028"), closure="resolved", results=[("S08", "resolved")]),
        make_case("C2", make_handoff("e2", code="E029"), closure="false_alarm", judgments=[("S04", "yes")]),
        make_case("C3", make_handoff("e3", code="E028"), closure="resolved", results=[("S10", "resolved")]),
        make_case(
            "C4",
            make_handoff("e4", code="E012"),
            closure="resolved",
            results=[("MANUAL", "resolved")],
            manual={"cause": "호스 꺾임"},
        ),
    ]
    m = fm.build(cases)
    assert m["causes"] == {"S03": {"S08": 1}}
    assert m["manual_causes"] == [{"case_id": "C4", "code": "E012", "cause": "호스 꺾임"}]
    # S10으로 해결(C3)은 오알람으로 센다
    assert m["labels"]["C3"] == "false_alarm"
    assert m["alarms"]["test_0|E028"]["false_alarm"] == 1
    assert m["alarms"]["test_0|E028"]["real"] == 1


def test_build_checks_and_missed(make_handoff, make_case):
    cases = [
        make_case(
            "C1",
            make_handoff("e1"),
            closure="false_alarm",
            checks={"S04-K1": {"result": "abnormal"}, "S04-K2": {"result": "skipped"}},
            log={"missed_checks": "에어 호스 확인"},
        ),
        make_case(
            "C2",
            make_handoff("e2"),
            closure="handed_over",
            checks={"S04-K1": {"result": "normal"}, "S04-K2": {"result": "normal"}},
            log={"missed_checks": "  "},
        ),
    ]
    m = fm.build(cases)
    assert m["checks"] == {"S04-K1": {"abnormal": 1, "normal": 1}, "S04-K2": {"abnormal": 0, "normal": 1}}
    assert m["missed_checks"] == [{"case_id": "C1", "code": "E029", "text": "에어 호스 확인"}]


def test_build_ignores_malformed_log_and_checks(make_handoff, make_case):
    # PUT /log · PUT /checks 본문이 이상해도 build()가 멈추지 않는다
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm", log={"missed_checks": ["a", "b"]}),
        make_case("C2", make_handoff("e2"), closure="false_alarm", log={"missed_checks": None}),
        make_case("C3", make_handoff("e3"), closure="false_alarm", checks={"S04-K1": "abnormal"}),
    ]
    m = fm.build(cases)
    assert m["missed_checks"] == []
    assert m["checks"] == {}
    assert m["labeled"] == 3


def test_build_exclude_and_open_cases(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm", checks={"S04-K1": {"result": "abnormal"}}),
        make_case(
            "C2",
            make_handoff("e2"),
            judgments=[("S04", "yes")],
            checks={"S04-K1": {"result": "abnormal"}},
            log={"missed_checks": "에어 호스"},
        ),
    ]
    m = fm.build(cases, exclude="C1")
    assert m == {
        "alarms": {},
        "causes": {},
        "checks": {},
        "manual_causes": [],
        "missed_checks": [],
        "labels": {},
        "labeled": 0,
    }


def _order(g):
    return [c["situation_id"] for c in g["candidates"]]


def test_guidance_cold_start_orders(make_handoff):
    expected = {
        "E029": ["S04", "S10"],
        "E012": ["S01", "S05", "S10"],
        "E016": ["S02", "S07", "S10"],
        "E028": ["S03", "S06", "S08", "S10"],
    }
    for code, order in expected.items():
        g = fm.guidance_for(make_handoff("e1", code=code), [])
        assert g["status"] == "retrieval_only"
        assert _order(g) == order
        assert g["memory"]["baseline"] == order
        assert g["memory"]["applied"] == []
        assert g["memory"]["note_ko"] is None


def test_guidance_candidate_fields(make_handoff):
    g = fm.guidance_for(make_handoff("e1", code="E028"), [])
    by_sid = {c["situation_id"]: c for c in g["candidates"]}
    assert by_sid["S03"]["basis"] == "code"
    assert by_sid["S03"]["field"] == {"confirmed": 0}
    assert by_sid["S06"]["basis"] == "manual_link"
    assert by_sid["S06"]["evidence_level"] == "manual"
    assert by_sid["S10"]["basis"] == "alarm_truth"
    assert "field" not in by_sid["S10"]
    for c in g["candidates"]:
        assert c["checks"] == guidance.candidate(c["situation_id"], c["rank"])["checks"]
    assert [c["rank"] for c in g["candidates"]] == [1, 2, 3, 4]


def test_r1_moves_s10_first(make_handoff, make_case):
    cases = [make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm")]
    g = fm.guidance_for(make_handoff("e2", code="E029"), cases)
    assert _order(g) == ["S10", "S04"]
    assert [c["rank"] for c in g["candidates"]] == [1, 2]
    assert g["memory"]["applied"] == ["R1"]
    assert g["memory"]["alarm_key"] == "test_0|E029"
    assert g["memory"]["alarm"] == {"false_alarm": 1, "real": 0}
    assert g["memory"]["note_ko"].startswith("같은 설비·코드(test_0 E029)에서 사람이 오알람으로 닫은 기록 1건")


def test_r1_counts_s10_resolved(make_handoff, make_case):
    # S10 "맞아요"로 해결 종료한 케이스도 오알람이라 다음 같은 알림에서 R1
    cases = [
        make_case(
            "C1",
            make_handoff("e1", code="E029"),
            closure="resolved",
            judgments=[("S10", "yes")],
            results=[("S10", "resolved")],
        )
    ]
    g = fm.guidance_for(make_handoff("e2", code="E029"), cases)
    assert _order(g) == ["S10", "S04"]
    assert g["memory"]["applied"] == ["R1"]
    assert g["memory"]["alarm"] == {"false_alarm": 1, "real": 0}


def test_r1_tie_does_not_fire(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029"), closure="resolved", results=[("S04", "resolved")]),
    ]
    g = fm.guidance_for(make_handoff("e3", code="E029"), cases)
    assert _order(g) == ["S04", "S10"]
    assert g["memory"]["applied"] == []
    assert "지난 기록: 오알람 1건 · 고장 확인 1건" in g["memory"]["note_ko"]


def test_r1_other_gun_unaffected(make_handoff, make_case):
    cases = [make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm")]
    assert _order(fm.guidance_for(make_handoff("e2", gun_id="test_6", code="E029"), cases)) == ["S04", "S10"]
    assert _order(fm.guidance_for(make_handoff("e3", code="E012"), cases)) == ["S01", "S05", "S10"]


def test_model_only_skips_r1_and_alarm_sentence(make_handoff, make_case):
    # 모델 신호만(규칙 코드 없음): R1도 오알람 문장도 없음 (키 "gun|"이 코드끼리 섞이므로)
    cases = [make_case("C1", make_handoff("e1", code="E012", source="model"), closure="false_alarm")]
    g = fm.guidance_for(make_handoff("e2", source="model", code="E029"), cases)
    assert _order(g) == ["S04", "S10"]
    assert g["memory"]["applied"] == []
    assert g["memory"]["alarm_key"] == "test_0|"
    assert g["memory"]["alarm"] == {"false_alarm": 1, "real": 0}
    assert g["memory"]["note_ko"] is None  # 문장에 "None" 글자가 찍히지 않음


def test_model_only_keeps_past_causes_sentence(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E028", source="model"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E028", source="model"), closure="false_alarm"),
        make_case(
            "C3", make_handoff("e3", code="E028", source="model"), closure="resolved", results=[("S08", "resolved")]
        ),
    ]
    g = fm.guidance_for(make_handoff("e4", source="model", code="E028"), cases)
    assert _order(g) == ["S03", "S06", "S08", "S10"]
    assert g["memory"]["applied"] == []
    assert g["memory"]["note_ko"] == (
        "지난 같은 고장(E028) 확정 원인: 축 보정 틀어짐·간섭에 의한 위치 이탈(S08) 1건 (참고, 순서는 매뉴얼 기준)."
    )


def test_past_causes_reference_only(make_handoff, make_case):
    cases = [make_case("C1", make_handoff("e1", code="E028"), closure="resolved", results=[("S08", "resolved")])]
    g = fm.guidance_for(make_handoff("e2", code="E028"), cases)
    assert _order(g) == ["S03", "S06", "S08", "S10"]
    s08 = next(c for c in g["candidates"] if c["situation_id"] == "S08")
    assert s08["evidence_level"] == "field"
    assert s08["field"] == {"confirmed": 1}
    assert g["memory"]["past_causes"][0] == {
        "situation_id": "S08",
        "name": "축 보정 틀어짐·간섭에 의한 위치 이탈",
        "confirmed": 1,
    }
    assert g["memory"]["note_ko"].endswith("(참고, 순서는 매뉴얼 기준).")


def test_past_causes_sorted_by_count_then_baseline(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", code="E028"), closure="resolved", results=[("S08", "resolved")]),
        make_case("C2", make_handoff("e2", code="E028"), closure="resolved", results=[("S06", "resolved")]),
        make_case("C3", make_handoff("e3", code="E028"), closure="resolved", results=[("S03", "resolved")]),
        make_case("C4", make_handoff("e4", code="E028"), closure="resolved", results=[("S08", "resolved")]),
    ]
    g = fm.guidance_for(make_handoff("e5", code="E028"), cases)
    assert [(p["situation_id"], p["confirmed"]) for p in g["memory"]["past_causes"]] == [
        ("S08", 2),
        ("S03", 1),
        ("S06", 1),
    ]
    assert g["memory"]["note_ko"].endswith(
        "지난 같은 고장(E028) 확정 원인: 축 보정 틀어짐·간섭에 의한 위치 이탈(S08) 2건, 원치 않는 이동(S03) 1건, "
        "기계 마찰·걸림·윤활 부족(S06) 1건 (참고, 순서는 매뉴얼 기준)."
    )


def test_no_code_is_no_evidence(make_handoff):
    g = fm.guidance_for(make_handoff("e1", code=None, source="model"), [])
    assert g == {"status": "no_evidence", "candidates": []}


def test_annotate_excludes_self_and_keeps_order(make_handoff, make_case):
    h = make_handoff("e1", code="E029")
    c1 = make_case("C1", h, closure="false_alarm", checks={"S04-K1": {"result": "abnormal"}})
    c1["guidance"] = fm.guidance_for(h, [])
    c2 = make_case(
        "C2",
        make_handoff("e2", code="E029"),
        closure="false_alarm",
        checks={"S04-K2": {"result": "normal"}, "S01-K1": {"result": "abnormal"}},
    )
    out = fm.annotate(c1, [c1, c2])
    assert "field_memory" not in c1
    assert _order(out["guidance"]) == _order(c1["guidance"]) == ["S04", "S10"]
    assert out["field_memory"] == {
        "alarm_key": "test_0|E029",
        "alarm": {"false_alarm": 1, "real": 0},
        "past_causes": [],
        "checks": {"S04-K2": {"abnormal": 0, "normal": 1}},
        "labeled": 1,
    }


def test_snapshot_shapes(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1", gun_id="test_6", code="E029"), closure="false_alarm"),
        make_case("C2", make_handoff("e2", code="E029"), closure="false_alarm"),
        make_case(
            "C3",
            make_handoff("e3", code="E028"),
            closure="resolved",
            results=[("S03", "resolved")],
            checks={"S03-K1": {"result": "abnormal"}},
        ),
    ]
    s = fm.snapshot(cases)
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
    assert s["past_causes"] == [
        {
            "situation_id": "S03",
            "code": "E028",
            "causes": [{"situation_id": "S03", "name": "원치 않는 이동", "confirmed": 1}],
        }
    ]
    assert s["checks"] == [{"check_id": "S03-K1", "abnormal": 1, "normal": 0}]
    assert s["manual_causes"] == []
    assert s["missed_checks"] == []


def test_chat_note(make_handoff, make_case):
    snap = fm.snapshot([make_case("C1", make_handoff("e1", code="E029"), closure="false_alarm")])
    assert (
        fm.chat_note("S04", snap)
        == "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 1 · 고장 확인 0"
    )
    assert fm.chat_note("S07", snap) is None
    assert fm.chat_note("S01", snap) is None
    assert fm.chat_note("S04", fm.snapshot([])) is None


def test_chat_note_with_past_causes(make_handoff, make_case):
    snap = fm.snapshot(
        [make_case("C1", make_handoff("e1", code="E028"), closure="resolved", results=[("S08", "resolved")])]
    )
    assert fm.chat_note("S03", snap) == (
        "현장 기록(사람 판정, 매뉴얼 근거 아님): E028 알림 — test_0 오알람으로 닫음 0 · 고장 확인 1"
        " / 지난 확정 원인: 축 보정 틀어짐·간섭에 의한 위치 이탈(S08) 1건 (참고)"
    )
