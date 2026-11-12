"""이전 기록(field_memory): 플래그 · 묶는 단위(설비 | 규칙 코드) · 닫힘 라벨 · 같은 단위의 닫힌 케이스 찾기."""

from __future__ import annotations

import field_memory as fm


def test_enabled_reads_env_each_call(monkeypatch):
    monkeypatch.setenv("FIELD_MEMORY", "1")
    assert fm.enabled() is True
    monkeypatch.delenv("FIELD_MEMORY")
    assert fm.enabled() is False


def test_alarm_key_uses_rule_code(make_handoff):
    assert fm.alarm_key(make_handoff("e1", code="E029")) == "test_0|E029"
    assert fm.alarm_key(make_handoff("e2", code="E029", source="model")) == "test_0|"


def test_label_false_alarm_close(make_handoff, make_case):
    c = make_case("C1", make_handoff("e1"), closure="false_alarm", judgments=[("E029", "yes")])
    assert fm.alarm_label(c) == "false_alarm"


def test_label_resolved_is_real(make_handoff, make_case):
    c = make_case(
        "C1", make_handoff("e1"), closure="resolved", judgments=[("E029", "yes")], results=[("E029", "resolved")]
    )
    assert fm.alarm_label(c) == "real"
    # 결정 원인이 없어도 해결 종료면 고장 확인
    assert fm.alarm_label(make_case("C2", make_handoff("e2"), closure="resolved")) == "real"


def test_label_unlabelled(make_handoff, make_case):
    for closure in ("handed_over", "unresolved", None):
        c = make_case("C1", make_handoff("e1"), closure=closure, judgments=[("E029", "yes")])
        assert fm.alarm_label(c) is None


def test_past_same_gun_and_code_closed_only(make_handoff, make_case):
    cases = [
        make_case("C1", make_handoff("e1"), closure="false_alarm"),
        make_case("C2", make_handoff("e2"), closure="resolved"),
        make_case("C3", make_handoff("e3"), closure="unresolved"),  # 라벨 없음
        make_case("C4", make_handoff("e4")),  # 열림
        make_case("C5", make_handoff("e5", gun_id="test_1"), closure="resolved"),  # 다른 설비
        make_case("C6", make_handoff("e6", code="E012"), closure="resolved"),  # 다른 코드
    ]
    target = make_case("C9", make_handoff("e9"))
    assert [c["case_id"] for c in fm.past(target, [*cases, target])] == ["C1", "C2"]


def test_past_excludes_self(make_handoff, make_case):
    c1 = make_case("C1", make_handoff("e1"), closure="false_alarm")
    c2 = make_case("C2", make_handoff("e2"), closure="resolved")
    assert [c["case_id"] for c in fm.past(c1, [c1, c2])] == ["C2"]
