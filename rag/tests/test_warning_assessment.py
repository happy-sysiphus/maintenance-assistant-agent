"""Time-series routing and opt-in end-to-end verification on shared Aura."""

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from rag import warning_assessment as wa
from rag.config import get_database, graph_driver, load_environment
from rag.documents import mapping
from rag.pipeline.build import write_graph
from rag.warning_memory import WarningMemory, canonical
from rag.warning_series import parse_event, search_query

SCOPE = "cpx-weld-2g-sa"
CASE = "manual:E016-C1"
FIXTURE = Path(__file__).parent / "fixtures" / "early_warning_v14.json"


def handoff(event_id, minute, change=40, *, episode_id=None):
    return {"schema_version": "1.4", "event_id": event_id, "episode_id": episode_id,
            "event_type": "critical", "detected_at": f"2026-11-10T02:{minute:02}:59Z",
            "sensor_features": {"window_start": f"2026-11-10T02:{minute:02}:00Z",
                                "window_end": f"2026-11-10T02:{minute:02}:59Z", "missing_s": [],
                                "channels": {"c10": [[0, 0], [change, 1]], "c5": [[0, 1], [change, 2]]}}}


def prior(event, case=CASE):
    return {"event": event, "event_id": event["event_id"], "case_id": "case-" + event["event_id"],
            "actual_action": "작업자가 기록한 정비 조치", "closed_at": "2026-11-10T02:03:00Z",
            "worker_confirmed": True, "equipment_scope": SCOPE, "sensor_features": event["sensor_features"],
            "manual_case": wa.initial_catalog()[case]}


def seen(event_id):
    return wa._Decision(status="seen", past_event_id=event_id, reason="동일한 시점에 신호가 전환되었습니다.",
                        intervals=[wa._Interval(channel="c10", start_offset_s=-20, end_offset_s=-18)])


def test_actual_v14_restores_all_channels_and_preserves_raw_block():
    actual = json.loads(FIXTURE.read_text(encoding="utf-8"))
    event, series = parse_event(actual, SCOPE)
    assert set(series) == {f"c{i}" for i in range(1, 20)}
    assert event["observed_error"] is None
    assert series["c10"][-20] == 0 and series["c10"][-19] == 1
    assert event["sensor_features"] == actual["sensor_features"]


def test_full_waveform_ranking_without_ml_selection(monkeypatch):
    def full_event(event_id, minute, unknown_value):
        raw = handoff(event_id, minute)
        raw["sensor_features"]["channels"] = {
            **{f"c{i}": [[0, 0]] for i in range(1, 20)},
            **raw["sensor_features"]["channels"],
            "c19": [[0, unknown_value]],
        }
        return parse_event(raw, SCOPE)[0]

    current = full_event("current", 5, 5)
    exact = prior(full_event("z-exact", 2, 5))
    different = prior(full_event("a-different", 1, 10))
    # Older Aura records can still carry the original derived list.
    current["comparison_channels"] = []
    memory = Mock(database="neo4j")
    memory.event.return_value = current
    memory.comparison_history.return_value = [different, exact]
    memory.resolution.return_value = exact
    model = Mock(return_value=seen("z-exact"))
    monkeypatch.setattr(wa, "_claude", model)
    result = wa.WarningAssessment(memory).assess("current")
    assert result["status"] == "seen"
    assert result["candidate_ids"] == ["z-exact", "a-different"]
    assert len(result["candidate_comparisons"][0]["channels"]) == 19
    context = model.call_args.args[1]
    assert len(context["current"]) == len(context["candidates"][0]["samples"]) == 19
    assert context["current"]["c19"] == [5] * 60
    assert "unidentified sensor" not in search_query(result["observations"])


def test_registered_code_bypasses_comparison_and_models(monkeypatch):
    raw = handoff("code", 4)
    raw.pop("sensor_features")
    raw["event_type"] = "early_warning"
    raw["observed_error"] = {"code": "E016", "observed_at": raw["detected_at"]}
    memory = Mock(database="neo4j")
    memory.event.return_value = parse_event(raw, SCOPE)[0]
    retrieve = Mock(return_value={"situations": [mapping()["E016"]], "evidence": [{"id": "code-proof"}]})
    monkeypatch.setattr(wa, "retrieve", retrieve)
    model = Mock(side_effect=AssertionError("code route must not call a model"))
    monkeypatch.setattr(wa, "_claude", model)
    result = wa.WarningAssessment(memory).assess("code")
    assert result["route"] == "error_code"
    assert result["cards"][0]["manual_case_id"] == CASE
    retrieve.assert_called_once_with("E016", "E016", use_vectors=False, database="neo4j", exact_code_only=True)
    memory.comparison_history.assert_not_called()
    model.assert_not_called()
    memory.event.return_value["observed_error"]["code"] = "E999"
    assert wa.WarningAssessment(memory).assess("code")["status"] == "unsupported_code"
    retrieve.assert_called_once()


def test_comparison_ranks_waveforms_and_checks_cited_intervals(monkeypatch):
    current, _ = parse_event(handoff("current", 5, episode_id="episode"), SCOPE)
    current["sensor_features"]["missing_s"] = [42]
    early = prior(parse_event(handoff("early", 1, 23), SCOPE)[0], "manual:E016-C2")
    close = prior(parse_event(handoff("close", 2), SCOPE)[0])
    repeat = prior(parse_event(handoff("repeat", 3, episode_id="episode"), SCOPE)[0])
    memory = Mock(database="neo4j")
    memory.event.return_value = current
    memory.comparison_history.return_value = [early, close, repeat]
    memory.resolution.return_value = close
    model = Mock(return_value=seen("close"))
    monkeypatch.setattr(wa, "_claude", model)
    service = wa.WarningAssessment(memory)
    # The model first names an endpoint that is missing in the current window.
    model.return_value.intervals[0].end_offset_s = -17
    assert service.assess("current")["reason"] == "comparison_failed"
    model.return_value = seen("close")
    result = service.assess("current")
    assert result["status"] == "seen"
    assert result["candidate_ids"] == ["close", "early"]
    assert result["card"]["repair"]["actual_action"] == close["actual_action"]
    assert model.call_args.args[1]["current"]["c10"][42] is None
    model.return_value = seen("not-retrieved")
    assert service.assess("current")["reason"] == "comparison_failed"


def test_insufficient_and_unavailable_are_distinct_from_unseen():
    memory = Mock(database="neo4j")
    memory.event.return_value = parse_event(handoff("current", 5), SCOPE)[0]
    memory.comparison_history.return_value = []
    service = wa.WarningAssessment(memory)
    assert service.assess("current")["status"] == "unseen"
    memory.comparison_history.side_effect = ServiceUnavailable("offline")
    assert service.assess("current")["reason"] == "history_unavailable"
    memory.event.return_value["sensor_features"] = None
    assert service.assess("current")["reason"] == "insufficient_samples"


def test_generated_candidate_requires_an_excerpt_from_retrieved_pdf(monkeypatch):
    memory = Mock(database="neo4j")
    memory.event.return_value = parse_event(handoff("current", 5), SCOPE)[0]
    row = {"id": "pdf-hit", "doc": "manual82", "page": 38, "text": "Check the source reading."}
    monkeypatch.setattr(wa, "retrieve", Mock(return_value={"evidence": [row]}))
    selected = wa._ManualCandidate(evidence_id="pdf-hit", title="문서 근거 후보", reason="관련 관측", step_kind="check",
                                   excerpt=row["text"])
    monkeypatch.setattr(wa, "_claude", Mock(return_value=wa._ManualCandidates(candidates=[selected])))
    service = wa.WarningAssessment(memory)
    result = service.manual_candidates("current")
    assert result["candidates"][0]["review_needed"]
    assert result["candidates"][0]["source"]["quote"] == row["text"]
    assert "manual_case_id" not in result["candidates"][0]
    selected.excerpt = "Unverified instruction"
    assert service.manual_candidates("current")["status"] == "generation_error"


@pytest.fixture
def aura_assessment():
    if os.getenv("RAG_WARNING_TEST_AURA") != "1":
        pytest.skip("set RAG_WARNING_TEST_AURA=1 for the shared Aura check")
    load_environment()
    uri = urlsplit(os.getenv("NEO4J_URI", ""))
    assert uri.scheme == "neo4j+s" and (uri.hostname or "").endswith(".databases.neo4j.io")
    namespace = f"warning-assessment-test-{uuid4()}"
    with graph_driver() as driver:
        memory = WarningMemory(driver, database=get_database(), namespace=namespace)
        memory.initialize()
        try:
            yield wa.WarningAssessment(memory, equipment_scope=SCOPE, use_vectors=False)
        finally:
            with driver.session(database=memory.database) as session:
                session.run("""MATCH (n {namespace:$namespace})
                    WHERE n:WarningEvent OR n:WarningSensorWindow OR n:WarningResolution OR n:WarningManualCase
                    DETACH DELETE n""", namespace=namespace).consume()


def test_aura_code_unseen_first_resolution_seen_and_conflict(aura_assessment, monkeypatch):
    service = aura_assessment
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(service.register(raw)["comparison_channels"]) == 19
    assert service.assess(raw["event_id"])["status"] == "unseen"
    stored = service.memory.event(raw["event_id"])
    assert stored.pop("sensor_features") == raw["sensor_features"]
    with service.memory.driver.session(database=service.memory.database) as session:
        session.run("MATCH (e:WarningEvent {id:$id}) SET e.payload=$payload",
                    id=service.memory._id(raw["event_id"]),
                    payload=canonical({**stored, "comparison_channels": ["c10"]})).consume()
    assert len(service.register(raw)["comparison_channels"]) == 19
    assert len(service.assess(raw["event_id"])["comparison_channels"]) == 19
    with pytest.raises(ValueError, match="different handoff"):
        service.register({**raw, "sensor_features": None})

    known = handoff("known", 1)
    known["observed_error"] = {"code": "E016", "observed_at": known["detected_at"]}
    service.register(known)
    assert service.assess("known")["route"] == "error_code"
    current = handoff("current", 5)
    service.register(current)
    assert service.assess("current")["status"] == "unseen"
    fields = {"case_id": "case-a", "actual_action": "테스트 정비 조치", "worker_confirmed": True,
              "closed_at": "2026-11-10T02:03:00Z", "manual_case_id": CASE}
    assert service.close("known", **fields)["status"] == "saved"
    assert service.close("known", **fields)["status"] == "saved"
    with pytest.raises(ValueError, match="different resolution"):
        service.close("known", **{**fields, "actual_action": "changed"})
    assert len(service.memory.comparison_history(service.memory.event("current"))) == 1
    if os.getenv("RAG_WARNING_TEST_MODEL") != "1":
        monkeypatch.setattr(wa, "_claude", Mock(return_value=seen("known")))
    result = service.assess("current")
    assert result["status"] == "seen", (result["status"], result["reason"], result.get("error"))
    assert result["card"]["repair"]["actual_action"] == fields["actual_action"]
    assert service.read_decision("current")["past_event_id"] == "known"
    changed = deepcopy(current)
    changed["sensor_features"]["channels"]["c5"] = [[0, 50]]
    with pytest.raises(ValueError, match="different handoff"):
        service.register(changed)
    service.register(handoff("conflict", 2))
    service.close("conflict", **{**fields, "case_id": "case-b", "manual_case_id": "manual:E016-C2"})
    monkeypatch.setattr(wa, "_claude", Mock(return_value=seen("known")))
    assert service.assess("current")["reason"] == "conflicting_repairs"


def test_aura_manual_search_and_review_links_source(aura_assessment):
    service = aura_assessment
    service.register(handoff("manual", 1))
    service.register(handoff("current", 5))
    found = service.manual_search("manual", "E012 Timeout compensation",
                                  explain=os.getenv("RAG_WARNING_TEST_MODEL") == "1")
    assert found["status"] == "ok"
    source = next(h["source"] for h in found["hits"] if not h["review_needed"])
    pending = service.close("manual", case_id="repair-manual", actual_action="검증용 실제 조치",
                            worker_confirmed=True, closed_at="2026-11-10T02:03:00Z", manual_source=source)
    assert pending["status"] == "pending_review"
    current = service.memory.event("current")
    assert not service.memory.comparison_history(current)
    reviewed = service.review_resolution("manual", title="검증용 문서 항목", equipment_scope=SCOPE)
    assert reviewed["manual_case"]["source"]["page"] == source["page"]
    assert not service.memory.comparison_history(current)
    # Replaying a historical alert must not see the source before review.
    after_review = {**current, "event_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()}
    assert len(service.memory.comparison_history(after_review)) == 1


def test_disposable_graph_rebuild_preserves_sensor_window_and_resolution():
    uri = os.getenv("RAG_WARNING_TEST_BOLT")
    if not uri:
        pytest.skip("set RAG_WARNING_TEST_BOLT for the disposable Neo4j instance")
    assert uri == "bolt://127.0.0.1:17687"
    with GraphDatabase.driver(uri, auth=None) as driver:
        memory = WarningMemory(driver, database="neo4j", namespace=f"warning-assessment-test-{uuid4()}")
        memory.initialize()
        service = wa.WarningAssessment(memory, equipment_scope=SCOPE)
        event = handoff("prior", 1)
        service.register(event)
        service.close("prior", case_id="repair", actual_action="테스트 조치", worker_confirmed=True,
                      closed_at="2026-11-10T02:03:00Z", manual_case_id=CASE)
        with driver.session(database="neo4j") as session:
            session.run("CREATE (:Rag:Cause {id:'timeseries-rebuild-test'})").consume()
            session.execute_write(write_graph, {}, {})
            assert not session.run("MATCH (n:Rag:Cause {id:'timeseries-rebuild-test'}) RETURN n").single()
        assert memory.event("prior")["sensor_features"] == event["sensor_features"]
        assert memory.resolution("prior")["manual_case"]["manual_case_id"] == CASE
        current = parse_event(handoff("current", 5), SCOPE)[0]
        assert len(memory.comparison_history(current)) == 1


def test_aura_register_backfills_existing_resolution(aura_assessment):
    service = aura_assessment
    old = {**handoff("old", 1), "event_type": "early_warning",
           "early_warning": {"onset_at": "2026-11-10T02:01:59Z"}}
    service.record_resolution(old, case_id="legacy", actual_action="기존 해결 기록",
                              worker_confirmed=True, closed_at="2026-11-10T02:03:00Z", manual_case_id=CASE)
    current, _ = parse_event(handoff("current", 5), SCOPE)
    assert service.memory.comparison_history(current) == []
    service.register(old)
    assert [r["event_id"] for r in service.memory.comparison_history(current)] == ["old"]
