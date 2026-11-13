from unittest.mock import Mock

import store
import warning_bridge
from rag.warning_catalog import initial_catalog, source_ref
from test_warning_bridge import close_body, path, warning


def test_timeseries_groups_later_code_and_reuses_existing_delivery(api, monkeypatch):
    monkeypatch.setenv("RAG_WARNING_TIMESERIES", "1")
    monkeypatch.setenv("RAG_WARNING_NAMESPACE", "test-timeseries")
    monkeypatch.setenv("RAG_WARNING_EQUIPMENT_SCOPE", "cpx-weld-2g-sa")
    rag = Mock()
    rag.read_decision.return_value = None
    rag.close.return_value = {"status": "saved"}
    monkeypatch.setattr(warning_bridge, "assessment_service", lambda *a, **k: rag)

    first = {**warning("onset"), "episode_id": "e1"}
    assert api.post("/api/ml/handoffs", json=first).json()["result"] == "new"
    code = {**first, "event_id": "code", "event_type": "critical", "early_warning": None,
            "observed_error": {"code": "E016", "observed_at": first["detected_at"]}}
    assert api.post("/api/ml/handoffs", json=code).json()["result"] == "repeat"
    assert api.post("/api/ml/handoffs", json=code).json()["result"] == "duplicate"
    other = {**first, "event_id": "another", "episode_id": "e2"}
    assert api.post("/api/ml/handoffs", json=other).json()["result"] == "new"

    case = store.all_cases()[0]["case_id"]
    assert api.get(f"/api/cases/{case}").json()["preferred_event_id"] == "code"
    assert api.get(path(case)).json()["status"] == "awaiting_assessment"
    rag.assess.assert_not_called()
    card = {"manual_case_id": "manual:E016-C1", "title": "전극 점검", "equipment_scope": "cpx-weld-2g-sa"}
    rag.assess.return_value = {"event_id": "code", "route": "error_code", "status": "ok", "cards": [card]}
    assert api.post(path(case, "/assess"), json={"event_id": "code"}).json()["route"] == "error_code"
    assert api.get(path(case)).json()["cards"] == [card]
    rag.assess.assert_called_once_with("code")

    selection = {"event_id": "code", "manual_case_id": card["manual_case_id"]}
    assert api.post(path(case, "/select"), json=selection).status_code == 200
    assert api.post(path(case, "/resolve"), json=close_body("code")).json()["warning_resolution_status"] == "pending"
    assert warning_bridge.deliver_next()
    rag.close.assert_called_once()
    assert rag.close.call_args.args == ("code",)
    assert rag.close.call_args.kwargs["manual_case_id"] == card["manual_case_id"]
    assert api.get(path(case)).json()["recording_status"] == "saved"
    assert not warning_bridge.deliver_next()


def test_closed_timeseries_case_opens_the_event_the_worker_used(api, monkeypatch):
    monkeypatch.setenv("RAG_WARNING_TIMESERIES", "1")
    rag = Mock()
    monkeypatch.setattr(warning_bridge, "assessment_service", lambda *a, **k: rag)
    first = {**warning("used"), "episode_id": "e1"}
    code = {**first, "event_id": "later-code", "event_type": "critical", "early_warning": None,
            "observed_error": {"code": "E016", "observed_at": first["detected_at"]}}
    for handoff in (first, code):
        assert api.post("/api/ml/handoffs", json=handoff).status_code == 200
    case = store.all_cases()[0]
    assert warning_bridge.preferred_event(case)["event_id"] == "later-code"
    case["status"] = "resolved"
    store.put(case, warning_delivery={"handoff": first, "pattern_key": None})
    assert api.get(f"/api/cases/{case['case_id']}").json()["preferred_event_id"] == "used"


def test_rejected_timeseries_card_leaves_new_pdf_evidence_selectable(api, monkeypatch):
    monkeypatch.setenv("RAG_WARNING_TIMESERIES", "1")
    card = initial_catalog()["manual:E012-C1"]
    source = source_ref("manual82", 44, "The drive can drift due to leakage.", evidence_id="drift")
    candidate = {"evidence_id": "drift", "source": source, "excerpt": source["quote"],
                 "title": "위치 유지 확인", "reason": "위치가 이동했습니다.", "step_kind": "check"}
    rag = Mock()
    rag.read_decision.return_value = None
    rag.assess.return_value = {"event_id": "seen-event", "route": "history", "status": "seen", "card": card}
    rag.manual_candidates.return_value = {"status": "ok", "candidates": [candidate]}
    rag.manual_search.return_value = {"status": "ok", "cards": [card], "hits": [{"source": source}]}
    monkeypatch.setattr(warning_bridge, "assessment_service", lambda *a, **k: rag)

    event = {"event_id": "seen-event"}
    assert api.post("/api/ml/handoffs", json=warning(event["event_id"])).json()["result"] == "new"
    case_id = store.all_cases()[0]["case_id"]
    assert api.post(path(case_id, "/assess"), json=event).json()["status"] == "seen"
    assert api.post(path(case_id, "/reject"), json=event).json()["status"] == "rejected"
    assert api.post(path(case_id, "/candidates"), json=event).json()["candidates"] == [candidate]
    search = api.post(path(case_id, "/manual"), json={**event, "question": "위치가 밀릴 때 확인할 사항"}).json()
    assert search["cards"] == []
    assert search["hits"] == [{"source": source}]
    refreshed = api.get(path(case_id)).json()
    assert refreshed["card"] is None
    assert refreshed["generate_candidates"]["candidates"] == [candidate]
    assert api.post(path(case_id, "/select"), json={**event, "evidence_id": "drift"}).json() == {"manual_source": source}
    assert api.post(path(case_id, "/select"), json={**event, "manual_case_id": card["manual_case_id"]}).status_code == 422
