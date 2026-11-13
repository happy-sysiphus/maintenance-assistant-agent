"""GET /api/ml/feedback: 닫힌 조기 경보 케이스의 사실만 ML에 내준다 (정답인지는 ML이 정한다).

설계: ml/docs/superpowers/specs/2026-11-12-feedback-labels-design.md 3절
"""

from __future__ import annotations

from typing import Any

import ml_feedback
import pytest
import store


@pytest.fixture
def warning(make_handoff):
    """조기 경보 핸드오프 (v1.4에서 api가 읽는 필드만)."""

    def _make(event_id: str, gun_id: str = "E01_3", t: str = "2021-09-01T10:00:59Z") -> dict[str, Any]:
        h = make_handoff(event_id, gun_id=gun_id, code=None, t=t, source="model")
        h.update(
            event_type="early_warning",
            window_start=t[:17] + "00Z",
            detector={"name": "baseline_ensemble", "created": "2026-11-01T18:46:47"},
            early_warning={"since": t, "onset_at": t},
        )
        return h

    return _make


def test_item_resolved_legacy_checks(make_case, warning):
    """예전 /close 경로: 고른 경보가 없고, 점검은 records.checks에 {"result": ...} 꼴로 있다."""
    case = make_case(
        "CASE-0001",
        warning("w1"),
        closure="resolved",
        checks={"K1": {"result": "abnormal"}, "K2": {"result": "normal"}},
    )
    case["warning_resolution_status"] = "not_linked"
    assert ml_feedback.item(case, None) == {
        "case_id": "CASE-0001",
        "outcome": "resolved",
        "closed_at": "2021-09-06T22:00:00Z",
        "resolution_status": "not_linked",
        "selected_event_id": None,
        "checks": {"abnormal": 1, "normal": 1, "unchecked": 0},
        "events": [
            {
                "event_id": "w1",
                "representative": True,
                "gun_id": "E01_3",
                "window_start": "2021-09-01T10:00:00Z",
                "onset_at": "2021-09-01T10:00:59Z",
                "detected_at": "2021-09-01T10:00:59Z",
                "detector_created": "2026-11-01T18:46:47",
            }
        ],
    }


def test_item_onset_at(make_case, warning):
    """경보가 시작된 창은 early_warning.onset_at이다. 한 요청에 창이 여럿이면 window_start(마지막 창)보다 앞선다."""
    h = warning("w1", t="2021-09-01T10:19:59Z")
    h["early_warning"]["onset_at"] = "2021-09-01T10:00:59Z"
    old = warning("w2")
    del old["early_warning"]
    odd = warning("w3")
    odd["early_warning"] = "soon"
    for handoff, want in ((h, "2021-09-01T10:00:59Z"), (old, None), (odd, None)):
        event = ml_feedback.item(make_case("CASE-0001", handoff, closure="resolved"), None)["events"][0]
        assert event["onset_at"] == want
    assert ml_feedback.item(make_case("CASE-0001", h, closure="resolved"), None)["events"][0]["window_start"] == (
        "2021-09-01T10:19:00Z"
    )


def test_item_resolved_selected_draft(make_case, warning):
    """/warning/resolve 경로: 고른 반복 경보와 그 경보의 점검 초안을 쓴다."""
    case = make_case("CASE-0001", warning("w1"), closure="resolved")
    case["repeats"] = [{"event_id": "w2", "detected_at": "2021-09-01T14:00:59Z", "handoff": warning("w2")}]
    case["warning_events"] = {
        "w1": {"draft": {"checks": {"c9": "abnormal"}}},
        "w2": {"draft": {"checks": {"c1": "abnormal", "c2": "unchecked"}}},
    }
    case["warning_resolution_status"] = "pending"
    got = ml_feedback.item(case, {"event_id": "w2", "status": "saved"})
    assert got["selected_event_id"] == "w2" and got["resolution_status"] == "saved"
    assert got["checks"] == {"abnormal": 1, "normal": 0, "unchecked": 1}, "고른 경보의 초안만"


def test_item_same_check_counted_once(make_case, warning):
    case = make_case("CASE-0001", warning("w1"), closure="resolved", checks={"c1": {"result": "normal"}, "c2": "normal"})
    case["warning_events"] = {"w1": {"draft": {"checks": {"c1": "abnormal"}}}}
    got = ml_feedback.item(case, None)
    assert got["checks"] == {"abnormal": 1, "normal": 1, "unchecked": 0}, "같은 점검은 초안이 이긴다"
    odd = make_case("CASE-0002", warning("w3"), closure="resolved", checks={"c1": {"result": "skip"}, "c2": None, "c3": 3})
    assert ml_feedback.item(odd, None)["checks"] == {"abnormal": 0, "normal": 0, "unchecked": 0}, "모르는 값은 세지 않는다"


def test_item_false_alarm(make_case, warning):
    got = ml_feedback.item(make_case("CASE-0001", warning("w1"), closure="false_alarm"), None)
    assert got["outcome"] == "false_alarm" and got["selected_event_id"] is None and got["resolution_status"] is None
    assert got["checks"] == {"abnormal": 0, "normal": 0, "unchecked": 0}


def test_item_excluded(make_case, make_handoff, warning):
    assert ml_feedback.item(make_case("CASE-0001", make_handoff("c1"), closure="resolved"), None) is None, "E 코드 케이스"
    assert ml_feedback.item(make_case("CASE-0002", warning("w1")), None) is None, "열린 케이스"
    assert ml_feedback.item(make_case("CASE-0003", warning("w2"), closure="handed_over"), None) is None


def test_item_events(make_case, warning):
    case = make_case("CASE-0001", warning("w1"), closure="false_alarm")
    case["repeats"] = [
        {"event_id": "w2", "detected_at": "2021-09-01T14:00:59Z", "handoff": warning("w2", t="2021-09-01T14:00:59Z")},
        {"event_id": "w3", "detected_at": "2021-09-01T16:00:59Z"},  # 원본이 없는 옛 반복 경보
    ]
    events = ml_feedback.item(case, None)["events"]
    assert [(e["event_id"], e["representative"]) for e in events] == [("w1", True), ("w2", False), ("w3", False)]
    assert events[1]["window_start"] == "2021-09-01T14:00:00Z" and events[1]["detector_created"] == "2026-11-01T18:46:47"
    assert events[2] == {
        "event_id": "w3",
        "representative": False,
        "gun_id": None,
        "window_start": None,
        "onset_at": None,
        "detected_at": "2021-09-01T16:00:59Z",
        "detector_created": None,
    }


def test_items_order(make_case, warning):
    late = make_case("CASE-0001", warning("w1"), closure="resolved")
    late["records"]["closure"]["at"] = "2021-09-08T00:00:00Z"
    early = make_case("CASE-0002", warning("w2"), closure="false_alarm")
    got = ml_feedback.items([late, early, make_case("CASE-0003", warning("w3"))], lambda case_id: None)
    assert [i["case_id"] for i in got] == ["CASE-0002", "CASE-0001"], "닫힌 순서"


def test_route(api, make_case, make_handoff, warning):
    store.put(make_case("CASE-0001", warning("w1"), closure="resolved", checks={"K1": {"result": "abnormal"}}))
    store.put(make_case("CASE-0002", make_handoff("c1"), closure="resolved"))
    r = api.get("/api/ml/feedback")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.0" and [i["case_id"] for i in body["items"]] == ["CASE-0001"]
    assert body["items"][0]["checks"]["abnormal"] == 1 and body["items"][0]["selected_event_id"] is None


# ------------------------------------------------------------------ 최종 검토의 사소한 지적 (2026-11-13)
def test_item_tolerates_odd_fields(make_case, warning):
    """손으로 보낸 핸드오프처럼 모양이 조금 다른 케이스: 읽을 수 있는 만큼 읽는다."""
    h = warning("w1")
    h["detector"] = "x"
    case = make_case("CASE-0001", h, closure="resolved", checks=["K1"])
    case["warning_events"] = {"w1": None}
    got = ml_feedback.item(case, None)
    assert got["events"][0]["detector_created"] is None and got["events"][0]["gun_id"] == "E01_3"
    assert got["checks"] == {"abnormal": 0, "normal": 0, "unchecked": 0}
    case = make_case("CASE-0002", warning("w2"), closure="resolved", checks={"K1": {"result": "abnormal"}})
    case["warning_events"] = {"w2": {"draft": "x"}}
    case["repeats"] = [None, {"event_id": "w3", "handoff": "x"}]
    got = ml_feedback.item(case, None)
    assert got["checks"]["abnormal"] == 1 and [e["event_id"] for e in got["events"]] == ["w2", "w3"]


def test_items_skips_a_case_it_cannot_read(make_case, warning, caplog):
    """읽을 수 없는 케이스 하나가 나머지를 막지 않는다. 건너뛴 것은 로그에 남는다."""
    good = make_case("CASE-0001", warning("w1"), closure="resolved")
    bad = make_case("CASE-0002", warning("w2"), closure="resolved")
    bad["repeats"] = 5
    other = {"case_id": "CASE-0003", "event": "x", "records": None}
    with caplog.at_level("WARNING"):
        got = ml_feedback.items([bad, good, other], lambda case_id: None)
    assert [i["case_id"] for i in got] == ["CASE-0001"]
    assert "CASE-0002" in caplog.text and "CASE-0003" not in caplog.text, "조기 경보가 아닌 것은 그냥 대상이 아니다"


def test_route_reads_the_outbox_once(api, make_case, warning, monkeypatch):
    """닫힌 케이스마다 DB를 따로 열지 않는다: 전송 기록은 한 번에 읽는다."""

    def per_case(case_id: str) -> None:
        raise AssertionError("케이스마다 묻지 않는다")

    for n, eid in ((1, "w1"), (2, "w2")):
        case = make_case(f"CASE-000{n}", warning(eid), closure="resolved")
        delivery = {"handoff": {"event_id": eid}, "pattern_key": "k"} if n == 1 else None
        store.put(case, warning_delivery=delivery)
    monkeypatch.setattr(store, "warning_delivery_for_case", per_case)
    body = api.get("/api/ml/feedback").json()
    got = {i["case_id"]: (i["selected_event_id"], i["resolution_status"]) for i in body["items"]}
    assert got == {"CASE-0001": ("w1", "pending"), "CASE-0002": (None, None)}


def test_store_warning_deliveries(api, make_case, warning):
    """store.warning_deliveries: 모든 케이스의 전송 기록을 한 번에, warning_delivery_for_case와 같은 내용으로."""
    assert store.warning_deliveries() == {}
    for n, eid in ((1, "w1"), (2, "w2"), (3, "w3")):
        case = make_case(f"CASE-000{n}", warning(eid), closure="resolved")
        store.put(case, warning_delivery={"handoff": {"event_id": eid}, "pattern_key": "k"} if n != 2 else None)
    got = store.warning_deliveries()
    assert got == {
        "CASE-0001": {"event_id": "w1", "status": "pending"},
        "CASE-0003": {"event_id": "w3", "status": "pending"},
    }
    assert all(store.warning_delivery_for_case(case_id) == d for case_id, d in got.items())


def test_route_uses_the_store_function(api, make_case, warning, monkeypatch):
    """새 주소는 store의 공개 함수만 쓴다 (store의 내부 이름에 기대지 않는다)."""
    store.put(make_case("CASE-0001", warning("w1"), closure="resolved"))
    fake = {"CASE-0001": {"event_id": "w9", "status": "saved"}}
    monkeypatch.setattr(store, "warning_deliveries", lambda: fake, raising=False)
    found = api.get("/api/ml/feedback").json()["items"][0]
    assert (found["selected_event_id"], found["resolution_status"]) == ("w9", "saved")
    assert not hasattr(ml_feedback, "_deliveries")
