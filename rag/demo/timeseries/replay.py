"""Replay the synthetic warning dataset in a disposable Aura namespace."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import os

from rag.config import get_database, graph_driver, load_environment
from rag.warning_assessment import WarningAssessment
from rag.warning_memory import WarningMemory

DATASET = Path(__file__).with_name("dataset.json")


def replay(service, events):
    outcomes = []
    for item in events:
        registered = service.register(item["handoff"])
        event_id = registered["event_id"]
        decision = service.assess(event_id)
        expected = item["expected"]
        selected = decision["past_event_id"]
        matched = (decision["status"] in expected["acceptable_statuses"]
                   and selected in (expected["applicable_past_event_ids"] or [None]))
        outcome = {"event_id": event_id, "expected": expected, "decision": decision, "matched": matched}
        if selected is not None:
            stored = service.memory.resolution(selected)
            repair = decision["card"]["repair"]
            outcome["repair_matches_aura"] = (
                all(repair[key] == stored[key] for key in
                    ("event_id", "case_id", "actual_action", "closed_at", "worker_confirmed"))
                and all(decision["card"][key] == value for key, value in stored["manual_case"].items())
            )
        if "simulated_resolution" in item:
            outcome["manual_candidates"] = service.manual_candidates(event_id)
            outcome["manual_search"] = service.manual_search(event_id, item["manual_question"])
            # This ground truth represents a worker's scripted action after searching the manual.
            outcome["resolution"] = service.close(event_id, **item["simulated_resolution"])
        outcomes.append(outcome)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="read settings from an existing RAG .env")
    parser.add_argument("--output", type=Path, help="save the complete JSON report")
    parser.add_argument("--keyword-only", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    load_environment()
    uri = urlsplit(os.getenv("NEO4J_URI", ""))
    if uri.scheme != "neo4j+s" or not (uri.hostname or "").endswith(".databases.neo4j.io"):
        parser.error("configure the encrypted team Aura endpoint")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    namespace = f"warning-reuse-demo-{uuid4()}"
    with graph_driver() as driver:
        memory = WarningMemory(driver, database=get_database(), namespace=namespace)
        memory.initialize()
        try:
            service = WarningAssessment(memory, equipment_scope=dataset["equipment_scope"],
                                        use_vectors=not args.keyword_only)
            results = replay(service, dataset["events"])
        finally:
            with driver.session(database=memory.database) as session:
                session.run("""MATCH (n {namespace:$namespace})
                    WHERE n:WarningEvent OR n:WarningSensorWindow OR n:WarningResolution OR n:WarningManualCase
                    DETACH DELETE n""", namespace=namespace).consume()
                remaining = session.run("MATCH (n {namespace:$namespace}) RETURN count(n) AS count",
                                        namespace=namespace).single()["count"]
    report = {"dataset_id": dataset["dataset_id"], "synthetic": True, "results": results,
              "test_nodes_remaining": remaining}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in results:
        decision = item["decision"]
        print(item["event_id"], "expected", item["expected"]["target"],
              "got", decision["status"], decision["past_event_id"] or "-", "match", item["matched"])
    print("test nodes remaining", remaining)


if __name__ == "__main__":
    main()
