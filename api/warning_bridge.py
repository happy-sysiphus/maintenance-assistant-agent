"""Early-warning API adapter. SQLite owns field history; Neo4j only supplies manuals.

RAG_WARNING_FEEDBACK=0: record resolutions, but never reuse them as recommendations.
"""
import copy
import os
import threading
from fastapi import HTTPException
import store
from rag.warning_input import analyze_warning

_driver = None
_driver_lock = threading.Lock()


def feedback_enabled():
    return os.getenv('RAG_WARNING_FEEDBACK', '0').lower() in ('1', 'true', 'on', 'yes')


def scope():
    return os.getenv('RAG_WARNING_EQUIPMENT_SCOPE') or None


def driver():
    global _driver
    with _driver_lock:
        if _driver is None:
            from rag.config import graph_driver
            _driver = graph_driver()
        return _driver


def shutdown():
    global _driver
    with _driver_lock:
        if _driver is not None:
            _driver.close()
            _driver = None


class SqliteWarningMemory:
    """Implements the existing RAG memory contract without duplicating field data.

    record() stages one immutable resolution; the route commits it together with
    the closed case in store.put. Nothing is written before case validation.
    """
    def __init__(self):
        from rag.config import get_database
        self.database = get_database()
        self.staged = None

    def lookup(self, pattern_key):
        if not feedback_enabled():
            return []
        return [r for r in store.warning_resolutions()
                if r.get('pattern_key') == pattern_key]

    def catalog(self):
        # Catalog definitions remain usable when feedback reuse is disabled.
        return list({r['manual_case']['manual_case_id']: r['manual_case']
                     for r in store.warning_resolutions() if r.get('manual_case')}.values())

    def resolution(self, event_id):
        return next((r for r in store.warning_resolutions() if r['event_id'] == event_id), None)

    def record(self, resolution, case=None):
        previous = self.resolution(resolution['event_id'])
        value = {**resolution, 'manual_case': case, 'counterexample_reviewed': False}
        if previous and previous != value:
            raise ValueError('event_id already has a different resolution')
        self.staged = value
        return value

    def approve(self, event_id, case):
        value = self.resolution(event_id)
        if not value or not value.get('manual_source') or value.get('manual_case_id'):
            raise ValueError('manual review submission required')
        if value.get('manual_case') and value['manual_case'] != case:
            raise ValueError('already reviewed with a different case')
        value['manual_case'] = case
        store.review_warning_resolution(value)
        return value

    def acknowledge_counterexample(self, event_id):
        value = self.resolution(event_id)
        if (not value or not value.get('manual_case') or not value.get('pattern_key')
                or not value.get('rejected_auto_case_id')
                or value['rejected_auto_case_id'] == value['manual_case']['manual_case_id']):
            raise ValueError('reviewed counterexample required')
        value['counterexample_reviewed'] = True
        store.review_warning_resolution(value)

    def manual_chunk(self, chunk_id):
        with driver().session(database=self.database) as session:
            row = session.run('''MATCH (c:Rag:Chunk {id:$id}) WHERE c.searchable=true
                RETURN c.id AS id, c.doc AS doc, c.page_number AS page, c.text AS text,
                c.quote AS quote, c.review_needed AS review_needed, c.source_json AS source_json''',
                id=chunk_id).single()
            return row.data() if row else None


def service():
    from rag.early_warning import EarlyWarningRag
    from rag.warning_catalog import validate_manual_selection

    class ApiWarningRag(EarlyWarningRag):
        # Search sources are persisted by this API, never accepted from request bodies.
        # Checking current manifest revision/hash keeps close independent of graph availability.
        trusted_sources = None

        def _verify_manual_hit(self, selection):
            if self.trusted_sources is not None:
                verified = validate_manual_selection(selection)
                if self.trusted_sources.get(verified['evidence_id']) != verified:
                    raise ValueError('source was not returned by the event search')
                return
            return super()._verify_manual_hit(selection)

    return ApiWarningRag(SqliteWarningMemory(),
        use_vectors=os.getenv('RAG_WARNING_VECTORS', '1') != '0', equipment_scope=scope())


def event_for(case, event_id=None):
    if case['event'].get('event_type') != 'early_warning':
        raise HTTPException(409, 'early_warning case required')
    events = [case['event']] + [r['handoff'] for r in case.get('repeats', []) if 'handoff' in r]
    eid = event_id or case['event']['event_id']
    event = next((e for e in events if e['event_id'] == eid), None)
    if event is None:
        raise HTTPException(404, 'event does not belong to this case or its raw payload is unavailable')
    try:
        analyze_warning(event)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return event


def inspect(case, event_id=None):
    event = event_for(case, event_id)
    saved = copy.deepcopy(case.get('warning_events', {}).get(event['event_id'], {}))
    if case['status'] in ('resolved', 'false_alarm') and saved.get('inspection'):
        response = copy.deepcopy(saved['inspection'])
    else:
        response = service().inspect(event)
    if not feedback_enabled() and response['status'] not in ('insufficient_channels',):
        response['status'] = 'feedback_disabled'
    # A rejection is per-event; do not show a rejected card as automatic again.
    if saved.get('rejected_auto_case_id'):
        response['automatic_card'] = None
        if response['status'] == 'automatic':
            response['status'] = 'candidates'
    resolution = store.warning_resolution_for_case(case['case_id'])
    status = ('saved' if resolution.get('manual_case') else 'pending_review') if resolution else 'not_resolved'
    return {**saved, **response, 'feedback_enabled': feedback_enabled(), 'recording_status': status}


def snapshot(case, event_id, result, kind):
    state = case.setdefault('warning_events', {}).setdefault(event_id, {})
    state[kind] = copy.deepcopy(result)
    cards = ([result.get('automatic_card')] + result.get('history_candidates', []) + result.get('candidates', []))
    for card in cards:
        if card:
            state.setdefault('shown_cards', {})[card['manual_case_id']] = copy.deepcopy(card)
    for hit in result.get('hits', []):
        state.setdefault('shown_sources', {})[hit['source']['evidence_id']] = copy.deepcopy(hit['source'])
    return state
