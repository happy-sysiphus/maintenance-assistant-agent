"""v1.4 contract tests and opt-in Neo4j checks for shared Aura or disposable fixtures."""
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from rag import documents, early_warning as ew
from rag.config import get_database, graph_driver, load_environment
from rag.documents import fingerprint, normalized
from rag.pipeline.build import write_graph
from rag.pipeline.ingest import KB_ID
from rag.warning_catalog import initial_catalog, source_ref
from rag.warning_input import analyze_warning
from rag.warning_memory import WarningMemory

LEGACY = "festo-legacy"
US2 = "manual:festo:main-cylinder-us2-load-voltage"
OTHER = "manual:E012-C2"


@pytest.fixture
def public_warning():
    return json.loads((Path(__file__).parent / "fixtures" / "early_warning_v14.json").read_text(encoding="utf-8"))


def test_corpus_version_matches_aura_across_checkout_line_endings(tmp_path, monkeypatch):
    manifest, mapping, processing = documents.DOCUMENTS_PATH, documents.MAPPING_PATH, documents.DATA / "processing.json"
    expected = fingerprint()
    for path in (manifest, mapping, processing):
        (tmp_path / path.name).write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(documents, "DOCUMENTS_PATH", tmp_path / manifest.name)
    monkeypatch.setattr(documents, "MAPPING_PATH", tmp_path / mapping.name)
    monkeypatch.setattr(documents, "DATA", tmp_path)
    assert fingerprint() == expected
    for path in tmp_path.iterdir():
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert fingerprint() == expected


def test_warning_cli_uses_shared_connection_settings(public_warning, tmp_path, monkeypatch):
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps({"handoff": public_warning}), encoding="utf-8")
    monkeypatch.setenv("RAG_WARNING_NAMESPACE", "team-rag")
    monkeypatch.setenv("NEO4J_DATABASE", "team-cloud")
    monkeypatch.setattr(sys, "argv", ["early_warning", "inspect", "--input", str(path)])
    driver = Mock()
    graph = Mock(return_value=nullcontext(driver))
    monkeypatch.setattr(ew, "graph_driver", graph)
    memory = Mock()
    monkeypatch.setattr(ew, "WarningMemory", memory)

    ew.main()

    memory.assert_called_once_with(driver, database="team-cloud", namespace="team-rag")
    memory.return_value.lookup.assert_not_called()


def warning(event_id, gun_id):
    start = "2026-11-10T02:00:00Z"
    end = "2026-11-10T02:00:59Z"
    channels = {"c5": [[0, 4.5], [30, 3.2]], "c3": [[0, 10], [40, 12]], "c10": [[0, 0], [40, 1]]}
    return {
        "schema_version": "1.4", "event_type": "early_warning", "event_id": event_id, "gun_id": gun_id,
        "early_warning": {"onset_at": end},
        "sensor_features": {"window_start": start, "window_end": end, "missing_s": [], "channels": channels},
        "contributing_features": [{"sensor": c, "feature": c + "_mean"} for c in channels],
    }


def test_public_v14_has_three_features_but_only_one_observation(public_warning):
    result = analyze_warning(public_warning)
    assert len(public_warning["contributing_features"]) == 3
    assert result["selected_channels"] == ["c10"]
    assert result["pattern_key"] is None
    assert result["observations"][0]["transitions"] == [
        {"last_before_offset_s": 39, "first_after_offset_s": 40, "from": 0, "to": 1}
    ]
    assert len(result["derived_features"]) == 2


def test_match_needs_three_directional_channels_and_ignores_order():
    event = warning("a", "gun-a")
    key = analyze_warning(event)["pattern_key"]
    assert key == "direction-v1:c10:up|c3:up|c5:down"
    event["contributing_features"].reverse()
    assert analyze_warning(event)["pattern_key"] == key
    event["sensor_features"]["channels"]["c5"] = [[0, 5], [30, 6]]
    assert analyze_warning(event)["pattern_key"] == "direction-v1:c10:up|c3:up|c5:up"
    event["contributing_features"][0] = {"sensor": "c3", "feature": "c3_std"}
    assert analyze_warning(event)["pattern_key"] is None


def test_missing_seconds_and_wrong_window_do_not_form_a_pattern():
    event = warning("a", "gun-a")
    event["sensor_features"]["missing_s"] = list(range(1, 19))
    observations = analyze_warning(event)["observations"]
    assert all(o["first_samples"] == 2 and o["direction"] is None for o in observations)
    assert analyze_warning(event)["pattern_key"] is None
    event["sensor_features"]["window_end"] = "2026-11-10T02:00:58Z"
    with pytest.raises(ValueError, match="onset"):
        analyze_warning(event)


def test_candidate_generation_is_explicit_even_with_one_channel(public_warning, monkeypatch):
    memory = Mock(database="neo4j")
    memory.catalog.return_value = []
    service = ew.EarlyWarningRag(memory)
    case = initial_catalog()[US2]
    primary = case["source"]
    search = Mock(return_value={"evidence": [{**primary, "id": "proof", "text": primary["quote"]}]})
    model = Mock(return_value=ew._Candidates(candidates=[
        ew._Selection(manual_case_id=US2, reason="US2 신호가 전환되어 관련 점검 조건을 확인합니다.")
    ]))
    monkeypatch.setattr(ew, "retrieve", search)
    monkeypatch.setattr(ew, "_claude", model)

    assert service.inspect(public_warning)["status"] == "insufficient_channels"
    model.assert_not_called()
    search.assert_not_called()
    memory.lookup.assert_not_called()
    response = service.generate_candidates(public_warning, equipment_scope=LEGACY)
    assert response["status"] == "ok"
    assert response["candidates"][0]["source"] == primary
    assert response["candidates"][0]["actions"] == case["actions"]
    assert "US2 operating enable" in search.call_args.args[0]
    assert list(model.call_args.kwargs["allowed_ids"]) == [US2]

    model.return_value = ew._Candidates(candidates=[ew._Selection(manual_case_id="invented", reason="추측")])
    rejected = service.generate_candidates(public_warning)
    assert rejected["status"] == "generation_error" and not rejected["candidates"]
    search.return_value = {"evidence": [{**primary, "id": "wrong", "page": 93, "text": primary["quote"]}]}
    model.reset_mock()
    assert service.generate_candidates(public_warning)["status"] == "no_evidence"
    model.assert_not_called()


def test_memory_unavailable_can_still_search_manuals(monkeypatch):
    memory = Mock(database="neo4j")
    memory.lookup.side_effect = ServiceUnavailable("offline")
    memory.catalog.side_effect = ServiceUnavailable("offline")
    service = ew.EarlyWarningRag(memory)
    event = warning("x", "gun-x")
    assert service.inspect(event)["status"] == "memory_unavailable"
    evidence = {"doc": "festo", "page": 94, "text": "Cause: No compressed air supply",
                "review_needed": False, "id": "selected-source"}
    monkeypatch.setattr(ew, "retrieve", lambda *args, **kwargs: {"evidence": [evidence]})
    monkeypatch.setattr(ew, "_claude", lambda *a, **kw: ew._ManualReply(notes=[
        ew._ManualNote(evidence_id="selected-source", text="공급 압력 관련 근거입니다.")
    ]))
    response = service.search_manuals(event, "실린더가 움직이지 않아요", explain=True)
    assert response["status"] == "ok"
    assert response["notes"][0]["source"] == response["hits"][0]["source"]
    assert response["hits"][0]["manual_case_ids"] == []


def test_manual_search_keeps_pdf_hits_when_explanation_fails(monkeypatch):
    memory = Mock(database="neo4j")
    memory.catalog.return_value = []
    service = ew.EarlyWarningRag(memory)
    event = warning("x", "gun-x")
    evidence = {"doc": "festo", "page": 94, "text": "Cause: No compressed air supply",
                "review_needed": False, "id": "selected-source"}
    search = Mock(return_value={"evidence": [evidence]})
    monkeypatch.setattr(ew, "retrieve", search)
    monkeypatch.setattr(ew, "_claude", Mock(side_effect=RuntimeError("warning generation did not complete")))

    response = service.search_manuals(event, "실린더가 움직이지 않아요", explain=True)
    assert response["status"] == "retrieval_only"
    assert response["notes"] == []
    assert [hit["source"]["evidence_id"] for hit in response["hits"]] == ["selected-source"]

    search.side_effect = RuntimeError("manual search unavailable")
    failed = service.search_manuals(event, "실린더가 움직이지 않아요", explain=True)
    assert failed["status"] == "search_error"
    assert failed["hits"] == []


@pytest.fixture
def graph_service():
    uri = os.environ.get("RAG_WARNING_TEST_BOLT")
    if not uri:
        pytest.skip("set RAG_WARNING_TEST_BOLT for the disposable Neo4j instance")
    assert uri == "bolt://127.0.0.1:17687", "tests require the dedicated loopback endpoint"
    with GraphDatabase.driver(uri, auth=None, connection_timeout=5, max_transaction_retry_time=2) as driver:
        memory = WarningMemory(driver, database="neo4j", namespace=f"warning-test-{uuid4()}")
        memory.initialize()
        yield ew.EarlyWarningRag(memory)


@pytest.fixture
def aura_graph_service():
    if os.getenv("RAG_WARNING_TEST_AURA") != "1":
        pytest.skip("set RAG_WARNING_TEST_AURA=1 to verify the configured team Aura")
    load_environment()
    uri = urlsplit(os.getenv("NEO4J_URI", ""))
    assert uri.scheme == "neo4j+s" and (uri.hostname or "").endswith(".databases.neo4j.io"), (
        "Aura verification requires the encrypted team Aura URI"
    )
    namespace = f"warning-test-{uuid4()}"
    with graph_driver() as driver:
        memory = WarningMemory(driver, database=get_database(), namespace=namespace)
        with driver.session(database=memory.database) as session:
            session.run("RETURN 1").consume()
        try:
            yield ew.EarlyWarningRag(memory, use_vectors=False)
        finally:
            with driver.session(database=memory.database) as session:
                session.run(
                    """MATCH (n {namespace:$namespace})
                    WHERE n:WarningPattern OR n:WarningResolution OR n:WarningManualCase
                    DETACH DELETE n""",
                    namespace=namespace,
                ).consume()


def resolved(service, event, selected=US2, *, case_id=None, equipment_scope=LEGACY, **kwargs):
    return service.record_resolution(event, case_id=case_id or "CASE-" + event["event_id"],
                                     actual_action="테스트 작업자가 정비 조치를 수행함",
                                     worker_confirmed=True, closed_at="2026-11-10T03:00:00Z",
                                     equipment_scope=equipment_scope, manual_case_id=selected, **kwargs)


@pytest.mark.parametrize("fixture", ["graph_service", "aura_graph_service"])
def test_live_graph_first_candidate_promotion_and_counterexample(fixture, request, monkeypatch):
    service = request.getfixturevalue(fixture)
    model = Mock(side_effect=AssertionError("automatic lookup must not call a model"))
    monkeypatch.setattr(ew, "_claude", model)
    monkeypatch.setattr(ew, "retrieve", model)
    next_warning = warning("next", None)
    next_warning.pop("gun_id")

    assert service.inspect(next_warning, equipment_scope=LEGACY)["status"] == "new_pattern"
    assert resolved(service, warning("a1", "gun-a"))["status"] == "saved"
    retry = warning("a1", None)
    retry.pop("gun_id")
    assert resolved(service, retry)["status"] == "saved"
    with pytest.raises(ValueError, match="different resolution"):
        resolved(service, warning("a1", "gun-a"), OTHER)
    first = service.inspect(next_warning, equipment_scope=LEGACY)
    assert first["status"] == "candidates"
    assert len(first["history_candidates"][0]["repairs"]) == 1

    resolved(service, warning("a2", "gun-other"), case_id="CASE-a1")
    repeated = service.inspect(next_warning, equipment_scope=LEGACY)
    assert repeated["status"] == "candidates"
    assert repeated["history_candidates"][0]["resolved_cases"] == 1
    assert len(repeated["history_candidates"][0]["repairs"]) == 1

    common_site = ew.EarlyWarningRag(service.memory, equipment_scope=LEGACY)
    resolved(common_site, warning("b", "gun-a"), equipment_scope=None)
    assert service.inspect(next_warning)["status"] == "candidates"
    assert service.inspect(next_warning, equipment_scope=LEGACY, pending=True)["status"] == "pending"
    automatic = common_site.inspect(next_warning)
    assert automatic["status"] == "automatic"
    assert automatic["automatic_card"]["applicable_cases"] == 2

    resolved(service, warning("c", "gun-c"), OTHER, rejected_auto_case_id=US2)
    held = service.inspect(next_warning, equipment_scope=LEGACY)
    assert held["status"] == "counterexample" and held["counterexample_event_ids"] == ["c"]
    service.memory.acknowledge_counterexample("c")
    assert service.inspect(next_warning, equipment_scope=LEGACY)["status"] == "automatic"
    resolved(service, warning("d", "gun-d"), OTHER)
    assert service.inspect(next_warning, equipment_scope=LEGACY)["status"] == "conflict"
    model.assert_not_called()


def test_live_graph_rebuild_preserves_warning_history(graph_service):
    service = graph_service
    resolved(service, warning("first", None))
    resolved(service, warning("second", None))
    with service.memory.driver.session(database=service.memory.database) as session:
        session.run("CREATE (:Rag:Cause {id:'warning-rebuild-fixture'})").consume()
        session.execute_write(write_graph, {}, {})
        assert session.run("MATCH (n:Rag:Cause {id:'warning-rebuild-fixture'}) RETURN count(n) AS n").single()["n"] == 0
    assert service.inspect(warning("next", None), equipment_scope=LEGACY)["status"] == "automatic"


def test_aura_manual_search_and_worker_resolution(aura_graph_service):
    service = aura_graph_service
    event = warning("manual-aura", None)
    response = service.search_manuals(event, "E012 Timeout compensation")
    assert response["status"] == "ok", response
    source = next(hit["source"] for hit in response["hits"] if not hit["review_needed"])
    pending = resolved(service, event, selected=None, manual_source=source)
    assert pending["status"] == "pending_review"
    assert service.inspect(warning("next", None), equipment_scope=LEGACY)["status"] == "new_pattern"
    approved = service.review_resolution(
        event["event_id"], title="Aura 테스트 매뉴얼 항목", equipment_scope=LEGACY,
        checks=[source["quote"][:100].strip()],
    )
    saved = service.inspect(warning("next", None), equipment_scope=LEGACY)
    assert saved["status"] == "candidates"
    assert saved["history_candidates"][0]["manual_case_id"] == approved["manual_case"]["manual_case_id"]


def test_live_graph_manual_review_and_missing_confirmation(graph_service):
    service = graph_service
    event = warning("manual-1", "gun-a")
    source = source_ref("festo", 94, "Cause: No compressed air supply | Remedy: Check compressed air supply",
                        evidence_id="snapshot-manual-94")
    with service.memory.driver.session(database="neo4j") as session:
        session.run(
            """MERGE (c:Rag:Chunk {id:$id})
            SET c.doc=$doc, c.page_number=$page, c.text=$text, c.searchable=true""",
            id=source["evidence_id"], doc=source["doc"], page=source["page"], text=source["quote"],
        ).consume()
    args = {"case_id": "CASE-manual-1", "actual_action": "공급 압력을 점검하고 정상 작동을 확인",
            "closed_at": "2026-11-10T03:00:00Z", "equipment_scope": LEGACY, "manual_source": source}
    with pytest.raises(ValueError, match="confirmation"):
        service.record_resolution(event, worker_confirmed=False, **args)
    with pytest.raises(ValueError, match="search result"):
        service.record_resolution(event, worker_confirmed=True, **{
            **args, "manual_source": {**source, "quote": "not a real result"}
        })
    pending = service.record_resolution(event, worker_confirmed=True, **args)
    assert pending["status"] == "pending_review" and pending["manual_case"] is None
    assert service.inspect(warning("next", "gun-b"), equipment_scope=LEGACY)["status"] == "new_pattern"

    with pytest.raises(ValueError, match="excerpt"):
        service.review_resolution("manual-1", title="공압 공급 점검", equipment_scope=LEGACY,
                                  checks=["made up an instruction"])
    approved = service.review_resolution("manual-1", title="공압 공급 점검", equipment_scope=LEGACY,
                                         checks=["Check compressed air supply"])
    case = approved["manual_case"]
    assert case["manual_case_id"].startswith("manual:reviewed:")
    assert "evidence_id" not in case["source"]
    assert normalized(case["checks"][0]["text"]) in normalized(case["source"]["quote"])
    assert service.record_resolution(event, worker_confirmed=True, **args)["manual_case"] == case
    matched = service.inspect(warning("next", "gun-b"), equipment_scope=LEGACY)
    assert matched["status"] == "candidates" and matched["history_candidates"][0]["manual_case_id"] == case["manual_case_id"]
    second = warning("manual-2", None)
    second.pop("gun_id")
    resolved(service, second, case["manual_case_id"])
    assert service.inspect(warning("next", "gun-c"), equipment_scope=LEGACY)["automatic_card"]["manual_case_id"] == case["manual_case_id"]


def test_live_manual_index_supplies_candidate_evidence(graph_service, public_warning, monkeypatch):
    service = graph_service
    service.use_vectors = False
    driver = service.memory.driver
    source = initial_catalog()[US2]["source"]
    with driver.session(database="neo4j") as session:
        session.run(
            """MERGE (k:RagKnowledgeBase {id:$kb})
            SET k.ready=true, k.version=$version
            MERGE (c:Rag:Chunk {id:'warning-us2-corpus'})
            SET c.doc=$doc, c.page_number=$page, c.text=$text, c.searchable=true""",
            kb=KB_ID, version=fingerprint(),
            doc=source["doc"], page=source["page"], text=source["quote"],
        ).consume()
        session.run("CREATE FULLTEXT INDEX rag_chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]").consume()
        session.run("CALL db.awaitIndex('rag_chunk_text', 20)").consume()

    # The retrieval is real; the model adapter only supplies the selection.
    monkeypatch.setattr("rag.retrieval.graph_driver", lambda: GraphDatabase.driver(
        "bolt://127.0.0.1:17687", auth=None))
    model = Mock(return_value=ew._Candidates(candidates=[
        ew._Selection(manual_case_id=US2, reason="US2 관련 매뉴얼 근거를 확인합니다.")
    ]))
    monkeypatch.setattr(ew, "_claude", model)
    generated = service.generate_candidates(public_warning, equipment_scope=LEGACY)
    assert generated["status"] == "ok", generated
    assert generated["candidates"][0]["source"] == source
    assert model.call_args.kwargs["allowed_ids"].keys() == {US2}
