"""Evaluate manual evidence and stored repair cards on versioned synthetic inputs."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi.testclient import TestClient
import pdfplumber

from rag import warning_assessment as assessment
from rag.config import get_database, graph_driver, load_environment
from rag.documents import fingerprint, manual_path, normalized, verify_manuals
from rag.retrieval import retrieve
from rag.warning_catalog import initial_catalog, validate_manual_selection
from rag.warning_memory import WarningMemory
from rag.warning_series import observations, parse_event, search_query

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).with_name("timeseries_dataset.json")
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
FIELDS = ("event_id", "case_id", "actual_action", "closed_at", "worker_confirmed")
SYNTHETIC_SHIFT = (("c5", 95), ("c3", -22), ("c1", 1), ("c15", 12), ("c3", 23), ("c10", 0))


def load_dataset():
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    path, base = dataset["base"].split("#")
    events = json.loads((ROOT / path).read_text(encoding="utf-8-sig"))["events"]
    templates = {row["handoff"]["event_id"]: row["handoff"] for row in events}
    assert base in templates
    return dataset, templates, base


def match_text(text):
    return " ".join(re.findall(r"[a-z0-9]+", normalized(text).lower().replace("\u00ad", "")))


def check_reference_pdfs(evidence):
    verify_manuals()
    pages = {}
    for key, group in evidence.items():
        for doc, page, anchor in group["refs"]:
            if (doc, page) not in pages:
                with pdfplumber.open(manual_path(doc)) as pdf:
                    pages[doc, page] = match_text(pdf.pages[page - 1].extract_text())
            if match_text(anchor) not in pages[doc, page]:
                raise ValueError(f"{key}: anchor not found on {doc} PDF page {page}")


def handoff(templates, base, item, minute, *, episode=None):
    result = deepcopy(templates[item.get("template", base)])
    start = START + timedelta(minutes=minute)
    end = start + timedelta(seconds=59)
    case_id = item["id"]
    result.update(event_id=case_id, episode_id=episode or f"episode-{case_id}",
                  gun_id="eval-cpx-weld", detected_at=end.isoformat(),
                  early_warning={"onset_at": end.isoformat()})
    block = result["sensor_features"]
    block.update(window_start=start.isoformat(), window_end=end.isoformat())
    channels = block["channels"]
    if item.get("static"):
        channels = {key: changes[:1] for key, changes in channels.items()}
    channels.update(item.get("channels", {}))
    if "only" in item:
        channels = {key: channels[key] for key in item["only"]}
    if item.get("all_missing"):
        channels = {key: [] for key in channels}
        block.update(missing_s=list(range(60)), error=[])
    block["channels"] = channels
    return result


def matching_labels(doc, page, text, labels):
    text = f" {match_text(text)} "
    return {label for label, group in labels.items()
            if any(doc == ref_doc and page == ref_page and f" {match_text(anchor)} " in text
                   for ref_doc, ref_page, anchor in group["refs"])}


def pr(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def score_units(units, gold):
    relevant = sum(bool(labels & gold) for labels in units)
    covered = set().union(*units) & gold
    return {"relevant": relevant, "returned": len(units), "covered": len(covered), "expected": len(gold)}


def aggregate_evidence(rows):
    counts = {field: sum(row[field] for row in rows)
              for field in ("relevant", "returned", "covered", "expected")}
    return {**counts, "precision": counts["relevant"] / counts["returned"] if counts["returned"] else None,
            "recall": counts["covered"] / counts["expected"] if counts["expected"] else None}


def aggregate(rows):
    return pr(*(sum(row[field] for row in rows) for field in ("tp", "fp", "fn")))


def score_search(evidence, gold, labels):
    hits = []
    for row in evidence[:5]:
        found = matching_labels(row["doc"], row["page"], row["text"], labels)
        hits.append({"id": row["id"], "doc": row["doc"], "page": row["page"], "labels": sorted(found)})
    return {"hits_at_5": hits, "metrics": score_units([set(row["labels"]) for row in hits], gold)}


def score_candidates(result, gold, labels):
    rows = []
    for card in result["candidates"]:
        try:
            source = validate_manual_selection(card["source"])
            found = matching_labels(source["doc"], source["page"], card["excerpt"], labels)
            valid = (bool(card["title"].strip() and card["reason"].strip())
                     and card["evidence_id"] == source["evidence_id"]
                     and card["step_kind"] in {"check", "action"}
                     and bool(card["excerpt"].strip()) and normalized(card["excerpt"]) in normalized(source["quote"]))
            rows.append({"title": card["title"], "doc": source["doc"], "page": source["page"],
                         "labels": sorted(found), "ui_contract": valid})
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            rows.append({"labels": [], "ui_contract": False, "error": type(exc).__name__})
    units = [set(row["labels"]) if row["ui_contract"] else set() for row in rows]
    return {"status": result["status"], "error": result.get("error"), "cards": rows,
            "metrics": score_units(units, gold)}


def evaluate_unseen(service, dataset, templates, base, *, hybrid, live_candidates):
    pack = dataset["unseen"]
    labels = pack["evidence"]
    cases = []
    for number, item in enumerate(pack["cases"]):
        doc = handoff(templates, base, item, number)
        event, samples = parse_event(doc, dataset["equipment_scope"])
        assert event["observed_error"] is None
        query = search_query(observations(samples))
        gold = set(item["gold"])
        row = {key: item[key] for key in ("id", "family", "symptom", "gold")}
        row["query"] = query
        methods = {"keyword": False, **({"hybrid": True} if hybrid else {})}
        for method, use_vectors in methods.items():
            result = retrieve(query, use_vectors=use_vectors, database=service.memory.database) if query else {"evidence": []}
            row[method] = score_search(result["evidence"], gold, labels)
        if live_candidates:
            service.register(doc)
            with patch.object(assessment, "retrieve", return_value=result):
                proposed = service.manual_candidates(item["id"])
            row["candidates"] = score_candidates(proposed, gold, labels)
        cases.append(row)
        print(item["id"], flush=True)
    return cases


def repair_fields(record, manual):
    return {**{key: record[key] for key in FIELDS}, "outcome": "resolved",
            "manual_case_id": manual["manual_case_id"], "source": manual["source"],
            "checks": manual["checks"], "actions": manual["actions"]}


def fields_from_card(card):
    if not card:
        return {}
    repair = card.get("repair") or {}
    return {key: val for key, val in (repair | card).items()
            if key in (*FIELDS, "outcome", "manual_case_id", "source", "checks", "actions")}


def score_fields(actual, expected):
    matching = {key for key in expected.keys() & actual.keys() if expected[key] == actual[key]}
    return pr(len(matching), len(actual) - len(matching), len(expected) - len(matching))


def chosen_response(context, *, selected, decision):
    if decision in ("unseen", "invalid_id"):
        selected = "not-in-the-graph" if decision == "invalid_id" else None
    if selected:
        candidate = next((row for row in context["candidates"] if row["event_id"] == selected), None)
        if not candidate and decision != "invalid_id":
            raise ValueError(f"Ground truth {selected} is absent from the actual ranked candidates")
        interval = (candidate["comparison"][0] if candidate else {"channel": "c1"})
        index = [i - 59 for i, (a, b) in enumerate(zip(context["current"][interval["channel"]],
                                                      candidate["samples"][interval["channel"]]))
                 if a is not None and b is not None] if candidate else [-59, 0]
        return assessment._Decision(status="seen", past_event_id=selected, reason="평가에서 고정한 과거 이벤트",
                                    intervals=[assessment._Interval(channel=interval["channel"],
                                                                    start_offset_s=index[0], end_offset_s=index[-1])])
    return assessment._Decision(status="unseen", past_event_id=None, reason="평가에서 고정한 이력 없음", intervals=[])


def evaluate_seen(service, dataset, templates, base, directory):
    # The in-process API has its own SQLite file. History reads and writes use the isolated Aura namespace.
    sys.path.insert(0, str(ROOT / "api"))
    import main as api
    import store

    repairs = {}
    catalog = initial_catalog()
    for number, item in enumerate(dataset["seen"]["repairs"]):
        channel, shift = SYNTHETIC_SHIFT[number]
        changes = [[sec, round(value + shift, 2)] for sec, value in templates[base]["sensor_features"]["channels"][channel]]
        past = handoff(templates, base, {**item, "channels": {channel: changes}}, 60 + number * 5)
        service.register(past)
        closed_at = START + timedelta(minutes=62 + number * 5)
        payload = {"event_id": item["id"], "case_id": f"repair-{item['id']}", "actual_action": item["action"],
                   "worker_confirmed": True, "closed_at": closed_at.isoformat(), "manual_case_id": item["case"]}
        expected = repair_fields(payload, catalog[item["case"]])
        service.close(**payload)
        stored = service.memory.resolution(item["id"])
        repairs[item["id"]] = (past, expected, repair_fields(stored, stored["manual_case"]))

    rows = []
    with patch.object(store, "DB_PATH", directory / "cases.db"), patch.object(api, "score_trace", return_value=[]), \
         patch.object(api.warning_bridge, "assessment_service", return_value=service), \
         patch.dict(os.environ, {"RAG_WARNING_TIMESERIES": "1", "RAG_WARNING_FEEDBACK": "1",
                                 "RAG_WARNING_NAMESPACE": service.memory.namespace,
                                 "RAG_WARNING_EQUIPMENT_SCOPE": dataset["equipment_scope"]}):
        client = TestClient(api.app)
        for number, item in enumerate(dataset["seen"]["cases"]):
            decision = item.get("control")
            selected = item.get("history")
            if selected:
                past, expected, stored = repairs[selected]
                block = past["sensor_features"]
                changed_pressure = [[sec, value + number + 1] for sec, value in block["channels"]["c5"]]
                extra = {"channels": {**block["channels"], "c5": changed_pressure}}
                if number % 4 == 2:
                    extra["only"] = [key for key in block["channels"] if key not in {"c11", "c12", "c19"}]
            else:
                extra = {"all_missing": decision == "missing_samples"}
                expected = stored = {}
            minute = 120 + number if decision != "future" else 58
            episode = "episode-H01" if decision == "same_episode" else None
            if episode:
                # The excluded event would be the closest match if the episode filter were removed.
                extra["channels"] = repairs["H01"][0]["sensor_features"]["channels"]
            doc = handoff(templates, base, {**item, **extra}, minute, episode=episode)
            posted = client.post("/api/ml/handoffs", json=doc)
            posted.raise_for_status()
            if posted.json()["result"] != "new":
                raise RuntimeError(f"{item['id']}: expected a new evaluation case")
            case_id = next(case["case_id"] for case in store.all_cases()
                           if case["event"]["event_id"] == item["id"])
            route = f"/api/cases/{case_id}/warning"
            with patch.object(assessment, "_claude", side_effect=lambda _, context, *a, **k:
                              chosen_response(context, selected=selected, decision=decision)) as llm:
                assessed = client.post(route + "/assess", json={"event_id": item["id"]})
                assessed.raise_for_status()
                reply = client.get(route, params={"event_id": item["id"]})
                reply.raise_for_status()
            result = reply.json()
            actual = fields_from_card(result.get("card"))
            expected_status = {"unseen": "unseen", "future": "unseen"}.get(decision, "uncertain" if decision else "seen")
            if decision == "same_episode":
                # Other resolved episodes are available. Verify that H01 is excluded before selecting.
                expected_status = "unseen"
                assert "H01" not in result.get("candidate_ids", [])
            exact = bool(expected) and expected == stored == actual and result.get("past_event_id") == selected
            emitted = bool(actual)
            accepted = int(exact)
            rows.append({"id": item["id"], "expected_history": selected, "control": decision,
                         "expected_status": expected_status, "status": result["status"], "reason": result.get("reason"),
                         "actual_history": result.get("past_event_id"), "model_calls": llm.call_count,
                         "emitted": emitted, "exact": exact, "fields": score_fields(actual, expected),
                         "mismatches": sorted(key for key in actual.keys() | expected.keys()
                                              if actual.get(key) != expected.get(key)),
                         "cards": pr(accepted, int(emitted) - accepted, int(bool(expected)) - accepted)})
            print(item["id"], result["status"], flush=True)
        client.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hybrid", action="store_true", help="include vector retrieval using synthetic sensor queries")
    parser.add_argument("--live-candidates", action="store_true", help="send retrieved manual excerpts to Claude")
    parser.add_argument("--section", choices=("unseen", "seen", "both"), default="both")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    load_environment()
    uri = urlsplit(os.getenv("NEO4J_URI", ""))
    if uri.scheme != "neo4j+s" or not (uri.hostname or "").endswith(".databases.neo4j.io"):
        parser.error("an encrypted team Aura endpoint is required")
    dataset, templates, base = load_dataset()
    check_reference_pdfs(dataset["unseen"]["evidence"])
    namespace = f"timeseries-eval-{uuid4()}"
    result = {"dataset": dataset["id"], "dataset_sha256": sha256(DATASET.read_bytes()).hexdigest(),
              "synthetic": True, "graph_fingerprint": fingerprint(),
              "hybrid": args.hybrid, "live_candidates": args.live_candidates,
              "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") if args.hybrid else None,
              "candidate_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5") if args.live_candidates else None}
    with graph_driver() as driver, TemporaryDirectory() as directory:
        memory = WarningMemory(driver, database=get_database(), namespace=namespace)
        service = assessment.WarningAssessment(memory, equipment_scope=dataset["equipment_scope"], use_vectors=args.hybrid)
        try:
            if args.section != "seen":
                result["unseen"] = evaluate_unseen(service, dataset, templates, base,
                                                   hybrid=args.hybrid, live_candidates=args.live_candidates)
                for mode in ("keyword", "hybrid", "candidates"):
                    if mode in result["unseen"][0]:
                        result[mode] = aggregate_evidence([row[mode]["metrics"] for row in result["unseen"]])
                if args.live_candidates:
                    result["candidate_statuses"] = dict(Counter(row["candidates"]["status"] for row in result["unseen"]))
            if args.section != "unseen":
                result["seen"] = evaluate_seen(service, dataset, templates, base, Path(directory))
                result["seen_cards"] = aggregate([row["cards"] for row in result["seen"]])
                result["seen_fields"] = aggregate([row["fields"] for row in result["seen"]])
                result["seen_statuses"] = dict(Counter(row["status"] for row in result["seen"]))
        finally:
            with driver.session(database=memory.database) as session:
                session.run("""MATCH (n {namespace:$namespace})
                    WHERE n:WarningEvent OR n:WarningSensorWindow OR n:WarningResolution OR n:WarningManualCase
                    DETACH DELETE n""", namespace=namespace).consume()
                result["remaining_evaluation_nodes"] = session.run(
                    "MATCH (n {namespace:$namespace}) RETURN count(n) AS count",
                    namespace=namespace).single()["count"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
