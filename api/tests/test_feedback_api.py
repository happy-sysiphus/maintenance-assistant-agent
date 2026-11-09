"""스펙 11.1(D-9 ML 피드백 루프): 조기 경보 케이스는 주의(warning)로 받고, 모델 단독 critical과 따로 묶는다.

케이스 목록에는 event_type, 이력에는 repeat_event_ids가 붙는다. 조기 경보 핸드오프를 켜지 않으면(ML 기본값)
케이스 묶음과 severity는 그대로다.
"""

from __future__ import annotations

import sqlite3

import store

T = ["2021-09-06T19:30:09Z", "2021-09-06T19:40:09Z", "2021-09-06T19:50:09Z", "2021-09-06T20:00:09Z"]


def _ew(make_handoff, eid: str, t: str) -> dict:
    """v1.3 조기 경보 핸드오프: 종료 코드도 고장 분류도 없다."""
    h = make_handoff(eid, code=None, t=t, source="early_warning")
    h["schema_version"] = "1.3"
    h["event_type"] = "early_warning"
    return h


def _model(make_handoff, eid: str, t: str) -> dict:
    """모델 단독 critical(v1.2, event_type 없음): 종료 코드가 없다."""
    return make_handoff(eid, code=None, t=t, source="model")


def _post(api, handoff: dict) -> str:
    return api.post("/api/ml/handoffs", json=handoff).json()["result"]


def _codes() -> list[tuple]:
    with sqlite3.connect(store.DB_PATH) as c:
        return c.execute("SELECT case_id, code FROM cases ORDER BY case_id").fetchall()


def test_ew_severity_warning(api, make_handoff):
    assert _post(api, _ew(make_handoff, "w1", T[0])) == "new"
    assert _post(api, make_handoff("r1", t=T[1])) == "new"
    assert _post(api, _model(make_handoff, "m1", T[2])) == "new"
    assert [store.get(f"CASE-000{i}")["severity"] for i in (1, 2, 3)] == ["warning", "critical", "critical"]


def test_ew_and_model_critical_grouped_apart(api, make_handoff):
    sent = [_ew(make_handoff, "w1", T[0]), _model(make_handoff, "m1", T[1]),
            _ew(make_handoff, "w2", T[2]), _model(make_handoff, "m2", T[3])]
    assert [_post(api, h) for h in sent] == ["new", "new", "repeat", "repeat"]
    assert [r["event_id"] for r in store.get("CASE-0001")["repeats"]] == ["w2"]
    assert [r["event_id"] for r in store.get("CASE-0002")["repeats"]] == ["m2"]
    assert _codes() == [("CASE-0001", "early_warning"), ("CASE-0002", None)]


def test_list_and_history_fields(api, make_handoff):
    for h in (_ew(make_handoff, "w1", T[0]), _ew(make_handoff, "w2", T[1]), make_handoff("r1", t=T[2])):
        _post(api, h)
    listed = {c["case_id"]: c["event"]["event_type"] for c in api.get("/api/cases").json()["cases"]}
    assert listed == {"CASE-0001": "early_warning", "CASE-0002": "critical"}
    for cid in ("CASE-0001", "CASE-0002"):
        assert api.post(f"/api/cases/{cid}/close", json={"outcome": "false_alarm"}).status_code == 200
    items = {i["case_id"]: i for i in api.get("/api/history").json()["items"]}
    assert {k: v["repeat_event_ids"] for k, v in items.items()} == {"CASE-0001": ["w2"], "CASE-0002": []}
    assert {k: v["event"]["event_type"] for k, v in items.items()} == listed


def test_default_grouping_unchanged(api, make_handoff):
    """조기 경보 핸드오프가 없으면(ML 기본값) 묶음 · severity · code 칸이 지금과 같다 (회귀 방지)."""
    sent = [make_handoff("r1", t=T[0]), make_handoff("r2", t=T[1]),
            _model(make_handoff, "m1", T[2]), _model(make_handoff, "m2", T[3])]
    assert [_post(api, h) for h in sent] == ["new", "repeat", "new", "repeat"]
    assert [store.get(c)["severity"] for c in ("CASE-0001", "CASE-0002")] == ["critical", "critical"]
    assert _codes() == [("CASE-0001", "E029"), ("CASE-0002", None)]
