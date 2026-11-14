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
from rag.warning_assessment import WarningAssessment
from rag.warning_catalog import validate_manual_selection
from rag.warning_input import analyze_warning, required_text, timestamp
from rag.warning_series import observed_code, parse_event
from rag.warning_memory import WarningMemory

log = logging.getLogger(__name__)
_driver = None
_driver_lock = threading.Lock()
_INSPECTION_FIELDS = (
    "event_id", "gun_id", "onset_at", "pattern_version", "pattern_key", "selected_channels",
    "observations", "derived_features", "equipment_scope", "automatic_card",
    "history_candidates", "counterexample_event_ids", "status",
)
_ASSESSMENT_FIELDS = (
    "event_id", "route", "status", "reason", "observed_error", "observations", "comparison_channels",
    "candidate_ids", "candidate_comparisons", "past_event_id", "model_reason", "intervals",
    "card", "cards", "evidence", "definition", "recording_status",
)
_assessment_locks = tuple(threading.Lock() for _ in range(32))


def timeseries_enabled(handoff):
    load_environment()
    return os.getenv("RAG_WARNING_TIMESERIES") == "1" and handoff.get("schema_version") == "1.4"


def is_timeseries(case):
    return case.get("rag_mode") == "timeseries"


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


def assessment_service(target_config=None, *, equipment_scope=None):
    config = target_config if target_config is not None else target()
    return WarningAssessment(memory(config), use_vectors=os.getenv("RAG_WARNING_VECTORS", "1") != "0",
                             equipment_scope=scope() if equipment_scope is None else equipment_scope,
                             reuse_history=feedback_enabled())


def case_events(case):
    return [case["event"], *(r["handoff"] for r in case.get("repeats", []) if "handoff" in r)]


def preferred_event(case):
    events = case_events(case)
    if not is_timeseries(case):
        return events[0]
    if case["status"] in ("resolved", "false_alarm"):
        delivery = store.warning_delivery_for_case(case["case_id"])
        if delivery:
            return next((event for event in events if event["event_id"] == delivery["event_id"]), events[0])
    codes = [(event, observed_code(event)) for event in events]
    return max(((event, code) for event, code in codes if code),
               key=lambda pair: timestamp(pair[1]["observed_at"], "observed_at"),
               default=(events[0], None))[0]


def deliver_next(now=None):
    timestamp = time.time() if now is None else now
    job = store.claim_warning_delivery(timestamp, timestamp + 120)
    if job is None:
        return False
    event_id, attempt, payload = job
    try:
        fields = dict(payload["resolution"])
        if payload.get("rag_mode") == "timeseries":
            rag = assessment_service(payload["target"], equipment_scope=fields.pop("equipment_scope"))
            rag.register(payload["handoff"])
            result = rag.close(event_id, **fields)
        else:
            rag = ApiWarningRag(memory(payload["target"]), equipment_scope=fields["equipment_scope"])
            rag.trusted_source = fields.get("manual_source")
            result = rag.record_resolution(payload["handoff"], **fields)
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
    if not is_timeseries(case) and case["event"].get("event_type") != "early_warning":
        raise HTTPException(409, "early_warning case required")
    eid = event_id or preferred_event(case)["event_id"]
    event = next((e for e in case_events(case) if e["event_id"] == eid), None)
    if event is None:
        raise HTTPException(404, "event does not belong to this case or its raw payload is unavailable")
    try:
        if is_timeseries(case):
            parse_event(event, scope())
        else:
            analyze_warning(event)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return event


def assess(event, *, retry=False):
    event_id = event["event_id"]
    with _assessment_locks[hash(event_id) % len(_assessment_locks)]:
        rag = assessment_service()
        rag.register(event)
        previous = rag.read_decision(event_id)
        if previous is not None and (not retry or previous["status"] != "uncertain"):
            return previous
        return rag.assess(event_id)


def without_rejected(result, state):
    rejected = state.get("rejected_auto_case_id")
    if not rejected:
        return result
    cleaned = {**result}
    if (cleaned.get("selection") or {}).get("manual_case_id") == rejected:
        cleaned.pop("selection")
    if "shown_cards" in cleaned:
        cleaned["shown_cards"] = {key: card for key, card in cleaned["shown_cards"].items() if key != rejected}
    for key in ("generate_candidates", "search_manuals"):
        if key in cleaned:
            cleaned[key] = without_rejected(cleaned[key], state)
    for key in ("history_candidates", "candidates", "cards"):
        if key in cleaned:
            cleaned[key] = [card for card in cleaned[key] if card.get("manual_case_id") != rejected]
    if (cleaned.get("automatic_card") or {}).get("manual_case_id") == rejected:
        cleaned.update(automatic_card=None, status="candidates")
    if (cleaned.get("card") or {}).get("manual_case_id") == rejected:
        cleaned.update(card=None, route="manual", status="rejected", reason="worker_rejected")
    if result.get("status") == "ok" and "candidates" in cleaned and not cleaned["candidates"]:
        cleaned["status"] = "no_evidence"
    return cleaned


def inspect(case, event_id=None):
    event = event_for(case, event_id)
    state = case.get("warning_events", {}).get(event["event_id"], {})
    saved = {key: copy.deepcopy(value) for key, value in state.items() if key != "inspection"}
    previous = state.get("inspection")
    delivery = store.warning_delivery_for_case(case["case_id"])
    status = "not_linked" if case["status"] == "resolved" else "not_resolved"
    if delivery:
        status = delivery["status"]
    enabled = feedback_enabled()

    if is_timeseries(case):
        response = previous or {"event_id": event["event_id"], "status": "awaiting_assessment"}
    elif case["status"] in ("resolved", "false_alarm", "unresolved") and previous:
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
    fields = _ASSESSMENT_FIELDS if is_timeseries(case) else _INSPECTION_FIELDS
    stored = {key: result[key] for key in fields if key in result} if kind == "inspection" else result
    state[kind] = copy.deepcopy(stored)
    cards = [result.get("automatic_card"), result.get("card"), *result.get("cards", []),
             *result.get("history_candidates", []), *result.get("candidates", [])]
    for card in cards:
        if card and card.get("manual_case_id") and card["manual_case_id"] != state.get("rejected_auto_case_id"):
            state.setdefault("shown_cards", {})[card["manual_case_id"]] = copy.deepcopy(card)
        elif card and card.get("evidence_id"):
            state.setdefault("shown_sources", {})[card["evidence_id"]] = copy.deepcopy(card["source"])
    for hit in result.get("hits", []):
        state.setdefault("shown_sources", {})[hit["source"]["evidence_id"]] = copy.deepcopy(hit["source"])
    return result
