"""WARNING API integration with the shared Aura history and a durable local outbox."""
import asyncio
import copy
import logging
import os
import threading
import time

from fastapi import HTTPException

import store
from rag.config import get_database, graph_driver, load_environment
from rag.early_warning import MEMORY_ERRORS, EarlyWarningRag
from rag.warning_catalog import validate_manual_selection
from rag.warning_input import analyze_warning, required_text
from rag.warning_memory import WarningMemory

log = logging.getLogger(__name__)
_driver = None
_driver_lock = threading.Lock()
_INSPECTION_FIELDS = (
    "event_id", "gun_id", "onset_at", "pattern_version", "pattern_key", "selected_channels",
    "observations", "derived_features", "equipment_scope", "automatic_card",
    "history_candidates", "counterexample_event_ids", "status",
)


def feedback_enabled():
    load_environment()
    return os.getenv("RAG_WARNING_FEEDBACK", "1").lower() in ("1", "true", "on", "yes")


def scope():
    load_environment()
    return os.getenv("RAG_WARNING_EQUIPMENT_SCOPE") or None


def driver():
    global _driver
    with _driver_lock:
        if _driver is None:
            _driver = graph_driver()
        return _driver


def shutdown():
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None


def target():
    database = get_database()
    return {"database": database,
            "namespace": required_text(os.getenv("RAG_WARNING_NAMESPACE"), "RAG_WARNING_NAMESPACE")}


class ApiWarningRag(EarlyWarningRag):
    # Only the outbox worker passes a source that was validated and saved by this API.
    trusted_source = None

    def _verify_manual_hit(self, selection):
        if self.trusted_source is None:
            return super()._verify_manual_hit(selection)
        if selection != self.trusted_source:
            raise ValueError("source was not returned by the event search")


def memory(target_config):
    return WarningMemory(driver(), **target_config)


def service(target_config=None):
    return ApiWarningRag(memory(target_config if target_config is not None else target()),
                         use_vectors=os.getenv("RAG_WARNING_VECTORS", "1") != "0",
                         equipment_scope=scope(), reuse_history=feedback_enabled())


def deliver_next(now=None):
    timestamp = time.time() if now is None else now
    job = store.claim_warning_delivery(timestamp, timestamp + 120)
    if job is None:
        return False
    event_id, attempt, payload = job
    try:
        rag = ApiWarningRag(memory(payload["target"]),
                            equipment_scope=payload["resolution"]["equipment_scope"])
        rag.trusted_source = payload["resolution"].get("manual_source")
        result = rag.record_resolution(payload["handoff"], **payload["resolution"])
    except (ValueError, TypeError, KeyError) as exc:
        log.warning("WARNING delivery rejected for %s (%s)", event_id, type(exc).__name__)
        store.finish_warning_delivery(event_id, attempt, "failed")
    except Exception as exc:
        delay = min(300, 2 ** min(attempt, 9))
        log.warning("WARNING delivery retry for %s in %ss (%s)", event_id, delay, type(exc).__name__)
        store.finish_warning_delivery(event_id, attempt, "pending",
                                      next_attempt_at=(time.time() if now is None else now) + delay)
    else:
        store.finish_warning_delivery(event_id, attempt, result["status"])
    return True


async def deliver_until_stopped(stop):
    while not stop.is_set():
        try:
            if await asyncio.to_thread(deliver_next):
                continue
        except Exception:
            log.exception("Could not process the WARNING outbox")
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except TimeoutError:
            pass


def event_for(case, event_id=None):
    if case["event"].get("event_type") != "early_warning":
        raise HTTPException(409, "early_warning case required")
    events = [case["event"]] + [r["handoff"] for r in case.get("repeats", []) if "handoff" in r]
    eid = event_id or case["event"]["event_id"]
    event = next((e for e in events if e["event_id"] == eid), None)
    if event is None:
        raise HTTPException(404, "event does not belong to this case or its raw payload is unavailable")
    try:
        analyze_warning(event)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return event


def without_rejected(result, state):
    rejected = state.get("rejected_auto_case_id")
    if not rejected:
        return result
    cleaned = {**result}
    if (cleaned.get("selection") or {}).get("manual_case_id") == rejected:
        cleaned.pop("selection")
    if "shown_cards" in cleaned:
        cleaned["shown_cards"] = {key: card for key, card in cleaned["shown_cards"].items() if key != rejected}
    if "generate_candidates" in cleaned:
        cleaned["generate_candidates"] = without_rejected(cleaned["generate_candidates"], state)
    for key in ("history_candidates", "candidates"):
        if key in cleaned:
            cleaned[key] = [card for card in cleaned[key] if card["manual_case_id"] != rejected]
    if (cleaned.get("automatic_card") or {}).get("manual_case_id") == rejected:
        cleaned.update(automatic_card=None, status="candidates")
    if result.get("status") == "ok" and "candidates" in cleaned and not cleaned["candidates"]:
        cleaned["status"] = "no_evidence"
    return cleaned


def inspect(case, event_id=None):
    event = event_for(case, event_id)
    state = case.get("warning_events", {}).get(event["event_id"], {})
    saved = {key: copy.deepcopy(value) for key, value in state.items() if key != "inspection"}
    previous = state.get("inspection")
    delivery = store.warning_delivery_for_case(case["case_id"])
    status = delivery["status"] if delivery else "not_resolved"
    enabled = feedback_enabled()

    if case["status"] in ("resolved", "false_alarm", "unresolved") and previous:
        response = {key: previous[key] for key in _INSPECTION_FIELDS if key in previous}
    else:
        pending = store.warning_pending(analyze_warning(event)["pattern_key"])
        try:
            response = service().inspect(event, pending=pending)
        except (RuntimeError, ValueError):
            response = EarlyWarningRag(None, reuse_history=False).inspect(event)
            if enabled and response["pattern_key"]:
                response["status"] = "pending" if pending else "memory_unavailable"

    if status == "pending_review":
        try:
            record = memory(target()).resolution(delivery["event_id"])
            if record and record["manual_case"]:
                status = "saved"
        except (*MEMORY_ERRORS, RuntimeError, ValueError):
            pass

    return without_rejected(
        {**saved, **response, "feedback_enabled": enabled, "recording_status": status}, state,
    )


def snapshot(case, event_id, result, kind):
    state = case.setdefault("warning_events", {}).setdefault(event_id, {})
    result = without_rejected(result, state)
    stored = {key: result[key] for key in _INSPECTION_FIELDS if key in result} if kind == "inspection" else result
    state[kind] = copy.deepcopy(stored)
    cards = [result.get("automatic_card"), *result.get("history_candidates", []), *result.get("candidates", [])]
    for card in cards:
        if card:
            state.setdefault("shown_cards", {})[card["manual_case_id"]] = copy.deepcopy(card)
    for hit in result.get("hits", []):
        state.setdefault("shown_sources", {})[hit["source"]["evidence_id"]] = copy.deepcopy(hit["source"])
    return result
