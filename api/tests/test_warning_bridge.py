"""WARNING API tests with real case/outbox transactions and optional shared Aura."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from neo4j.exceptions import ServiceUnavailable

import store
import warning_bridge as bridge
from warning_test_memory import TestWarningMemory
from rag import early_warning as ew
from rag.config import graph_driver, load_environment
from rag.warning_catalog import initial_catalog, source_ref
from rag.warning_memory import WarningMemory

CASE = 'manual:E012-C2'


def warning(eid='w1', gun='gun-1'):
    fixture = Path(__file__).parents[2] / 'rag/tests/fixtures/early_warning_v14.json'
    event = json.loads(fixture.read_text(encoding='utf-8'))
    event.update(event_id=eid, gun_id=gun)
    event['sensor_features']['channels'] = {'c3': [[0, 10], [40, 12]], 'c5': [[0, 4.5], [30, 3.2]], 'c10': [[0, 0], [40, 1]]}
    event['contributing_features'] = [{'sensor': c, 'feature': c + '_mean'} for c in ('c3', 'c5', 'c10')]
    return event


@pytest.fixture
def memory(monkeypatch):
    graph = TestWarningMemory()
    monkeypatch.setattr(bridge, 'memory', lambda _: graph)
    return graph


@pytest.fixture
def app(api, monkeypatch, memory):
    monkeypatch.delenv('RAG_WARNING_FEEDBACK', raising=False)
    monkeypatch.delenv('RAG_WARNING_EQUIPMENT_SCOPE', raising=False)
    monkeypatch.delenv('RAG_WARNING_REVIEW_TOKEN', raising=False)
    monkeypatch.setenv('RAG_WARNING_NAMESPACE', 'test-history')
    source = initial_catalog()[CASE]['source']
    monkeypatch.setattr(ew, 'retrieve', lambda *a, **k: {'evidence': [{**source, 'id': 'proof', 'text': source['quote']}]})
    monkeypatch.setattr(ew, '_claude', lambda *a, **k: ew._Candidates(candidates=[ew._Selection(manual_case_id=CASE, reason='관측을 바탕으로 점검 조건을 확인합니다.')]))
    return api


def ingest(app, event=None):
    event = event or warning()
    assert app.post('/api/ml/handoffs', json=event).status_code == 200
    return store.all_cases()[-1]['case_id']


def path(cid, tail=''):
    return f'/api/cases/{cid}/warning{tail}'


def choose(app, cid, eid='w1'):
    result = app.post(path(cid, '/candidates'), json={'event_id': eid})
    assert result.status_code == 200, result.text
    assert result.json()['candidates']
    assert app.post(path(cid, '/select'), json={'event_id': eid, 'manual_case_id': CASE}).status_code == 200


def close_body(eid='w1'):
    return {'event_id': eid, 'worker': '작업자', 'actual_action': '공급 압력을 확인하고 정비함', 'worker_confirmed': True, 'log_approved': True}


def test_default_on_sensor_inspection_no_generation(app, monkeypatch):
    cid = ingest(app)
    monkeypatch.setattr(ew, '_claude', lambda *a, **k: pytest.fail('detail GET called model'))
    result = app.get(f'/api/cases/{cid}').json()['warning_rag']
    assert result['feedback_enabled'] is True
    assert result['status'] == 'new_pattern'
    assert len(result['observations']) == 3
    assert result['automatic_card'] is None


def test_only_one_channel_does_not_invent_pattern(app):
    event = warning()
    event['contributing_features'] = [{'sensor': 'c10', 'feature': 'c10_mean'}]
    cid = ingest(app, event)
    assert app.get(path(cid)).json()['status'] == 'insufficient_channels'


def test_candidates_selection_persists(app):
    cid = ingest(app)
    choose(app, cid)
    result = app.get(path(cid)).json()
    assert result['selection'] == {'manual_case_id': CASE}
    assert result['shown_cards'][CASE]['source']['page'] > 0
    assert 'inspection' not in app.get(path(cid)).json()
    inspection = store.get(cid)['warning_events']['w1']['inspection']
    assert 'inspection' not in inspection and 'shown_cards' not in inspection


def test_forged_or_other_event_selection_rejected(app):
    cid = ingest(app)
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'manual_case_id': CASE}).status_code == 422
    assert app.post(path(cid, '/candidates'), json={'event_id': 'foreign'}).status_code == 404
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'evidence_id': 'fake'}).status_code == 422


def test_repeat_keeps_whole_event_and_uses_selected_repeat(app, memory):
    cid = ingest(app)
    repeat = warning('w2')
    repeat['sensor_features']['channels']['c5'] = [[0, 1], [30, 2]]
    app.post('/api/ml/handoffs', json=repeat)
    assert store.get(cid)['repeats'][0]['handoff'] == repeat
    choose(app, cid, 'w2')
    assert app.post(path(cid, '/resolve'), json=close_body('w2')).status_code == 200
    assert app.get('/api/history').json()['items'][0]['repeat_event_ids'] == ['w2']
    assert store.warning_delivery_for_case(cid) == {'event_id': 'w2', 'status': 'pending'}
    assert bridge.deliver_next()
    row = memory.resolution('w2')
    assert row['case_id'] == 'warning:w1'
    assert 'c5:up' in row['pattern_key']


def test_code_less_critical_is_ignored_and_legacy_warning_unaffected(app, make_handoff):
    """C-9: a code-less critical (old ML 10-min model path) makes no case; a legacy warning row keeps its repeats."""
    warning_id = ingest(app, warning())
    with store._conn() as db:
        db.execute("UPDATE cases SET code = NULL WHERE case_id = ?", (warning_id,))
    critical = make_handoff('critical', gun_id='gun-1', code=None, source='model')
    r = app.post('/api/ml/handoffs', json=critical)
    assert r.status_code == 200 and r.json() == {'result': 'ignored_codeless_critical'}
    assert app.post('/api/ml/handoffs', json=critical).json() == {'result': 'duplicate'}  # a resend stays ignored
    assert [c['case_id'] for c in store.all_cases()] == [warning_id]

    repeated = warning('w2')
    assert app.post('/api/ml/handoffs', json=repeated).status_code == 200
    assert store.get(warning_id)['repeats'][0]['handoff'] == repeated
    summaries = {row['case_id']: row for row in app.get('/api/cases').json()['cases']}
    assert list(summaries) == [warning_id]
    assert summaries[warning_id]['event']['event_type'] == 'early_warning'
    assert summaries[warning_id]['severity'] == 'warning'


def test_coded_critical_still_makes_a_case(app, make_handoff):
    """Rule-triggered critical and a model trigger that carries a terminal code both keep the E-code path."""
    for eid, src in (('rule', 'rule'), ('model-coded', 'model')):
        r = app.post('/api/ml/handoffs', json=make_handoff(eid, gun_id=f'g-{eid}', code='E029', source=src))
        assert r.json() == {'result': 'new'}
    cases = store.all_cases()
    assert len(cases) == 2 and all(c['severity'] == 'critical' for c in cases)
    assert all(c['guidance']['candidates'] for c in cases), 'no empty case'


def test_resolution_saved_when_feedback_off_and_retry_idempotent(app, monkeypatch, memory):
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '0')
    cid = ingest(app)
    choose(app, cid)
    body = close_body()
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    assert bridge.deliver_next()
    assert not bridge.deliver_next()
    assert len(memory.rows) == 1
    body['actual_action'] = '다른 내용'
    assert app.post(path(cid, '/resolve'), json=body).status_code == 409
    assert app.get('/api/history').json()['items'][0]['cause']


@pytest.mark.parametrize('field,value', [('worker_confirmed', False), ('log_approved', False), ('actual_action', ' '), ('worker', ' ')])
def test_confirmation_and_real_action_required(app, field, value):
    cid = ingest(app)
    choose(app, cid)
    body = close_body(); body[field] = value
    assert app.post(path(cid, '/resolve'), json=body).status_code == 422
    assert store.warning_delivery_for_case(cid) is None
    assert store.get(cid)['status'] == 'new'


def test_false_alarm_never_becomes_resolved_training_case(app):
    cid = ingest(app)
    assert app.post(f'/api/cases/{cid}/close', json={'outcome': 'false_alarm'}).status_code == 200
    assert store.warning_delivery_for_case(cid) is None
    assert app.post(path(cid, '/candidates'), json={'event_id': 'w1'}).status_code == 409


def test_legacy_warning_close_keeps_history_without_enqueuing_an_aura_resolution(app, memory, monkeypatch):
    event = warning()
    cid = ingest(app, event)
    case_url = f'/api/cases/{cid}'
    outcome = {'outcome': 'resolved'}
    assert app.post(f'{case_url}/close', json=outcome).status_code == 409
    assert app.put(f'{case_url}/log', json={'action': '이력 조치', 'approved': True}).status_code == 200
    closed = app.post(f'{case_url}/close', json=outcome)
    assert closed.status_code == 200
    assert closed.json()['warning_resolution_status'] == 'not_linked'
    assert store.warning_delivery_for_case(cid) is None

    received = []
    inspect = ew.EarlyWarningRag.inspect

    def capture(self, handoff, **kwargs):
        received.append(handoff)
        return inspect(self, handoff, **kwargs)

    monkeypatch.setattr(ew.EarlyWarningRag, 'inspect', capture)
    monkeypatch.setattr(ew, '_claude', lambda *a, **k: pytest.fail('history GET called model'))
    history = app.get('/api/history').json()['items']
    assert history[0]['case_id'] == cid
    assert history[0]['closure']['outcome'] == 'resolved'
    assert not received

    detail = app.get(case_url).json()
    assert received == [event]
    assert detail['warning_rag']['recording_status'] == 'not_linked'
    assert detail['warning_resolution_status'] == 'not_linked'
    assert detail['records'] == closed.json()['records'] == store.get(cid)['records']
    assert not bridge.deliver_next()
    assert memory.resolution('w1') is None


def test_draft_restores_after_reload_and_validates_checks(app):
    cid = ingest(app)
    choose(app, cid)
    assert app.put(path(cid, '/draft'), json={'event_id': 'w1', 'actual_action': '점검 중', 'worker': '배종혁'}).status_code == 200
    assert app.get(path(cid)).json()['draft']['actual_action'] == '점검 중'
    assert app.put(path(cid, '/draft'), json={'event_id': 'w1', 'checks': {'fake': 'normal'}}).status_code == 422


def test_generation_failure_preserves_previous_selection(app, monkeypatch):
    cid = ingest(app); choose(app, cid)
    def fail(*a, **kw):
        raise RuntimeError('provider failure')
    monkeypatch.setattr(ew, '_claude', fail)
    result = app.post(path(cid, '/candidates'), json={'event_id': 'w1'})
    assert result.json()['status'] == 'generation_error'
    assert app.get(path(cid)).json()['selection']['manual_case_id'] == CASE


def test_feedback_off_blocks_history_then_on_uses_two_cases(app, monkeypatch):
    scope = initial_catalog()[CASE]['equipment_scope']
    assert scope
    monkeypatch.setenv('RAG_WARNING_EQUIPMENT_SCOPE', scope)
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '0')
    for i in range(2):
        eid = f'e{i}'
        cid = ingest(app, warning(eid, f'g{i}'))
        choose(app, cid, eid)
        assert app.post(path(cid, '/resolve'), json=close_body(eid)).status_code == 200
        assert bridge.deliver_next()
    cid = ingest(app, warning('e3', 'g3'))
    assert app.get(path(cid)).json()['history_candidates'] == []
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '1')
    response = app.get(path(cid)).json()
    assert response['status'] == 'automatic'
    assert response['automatic_card']['applicable_cases'] == 2
    assert app.post(path(cid, '/reject'), json={'event_id': 'e3'}).status_code == 200
    after = app.get(path(cid)).json()
    assert after['automatic_card'] is None
    assert after['history_candidates'] == []
    next_candidates = app.post(path(cid, '/candidates'), json={'event_id': 'e3'}).json()
    assert next_candidates['history_candidates'] == []
    assert next_candidates['candidates'] == []
    assert app.post(path(cid, '/select'), json={'event_id': 'e3', 'manual_case_id': CASE}).status_code == 422


def test_no_scope_prevents_automatic_reuse(app, monkeypatch):
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '1')
    cid = ingest(app)
    choose(app, cid)
    assert app.post(path(cid, '/resolve'), json=close_body()).status_code == 200
    assert bridge.deliver_next()
    cid = ingest(app, warning('next', 'g2'))
    assert app.get(path(cid)).json()['automatic_card'] is None


def test_manual_search_selection_close_and_review(app, monkeypatch, memory):
    cid = ingest(app)
    case = initial_catalog()[CASE]
    source = source_ref(case['source']['doc'], case['source']['page'], case['source']['quote'], evidence_id='proof')
    monkeypatch.setattr(ew.EarlyWarningRag, 'search_manuals', lambda *a, **k: {'event_id': 'w1', 'status': 'ok', 'hits': [{'source': source}], 'notes': []})
    assert app.post(path(cid, '/manual'), json={'event_id': 'w1', 'question': '압력 확인'}).status_code == 200
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'evidence_id': 'proof'}).status_code == 200
    assert app.post(path(cid, '/resolve'), json=close_body()).status_code == 200
    assert app.get(path(cid)).json()['recording_status'] == 'pending'
    assert bridge.deliver_next()
    row = memory.resolution('w1')
    assert row['manual_case'] is None
    assert app.get(path(cid)).json()['recording_status'] == 'pending_review'
    url = '/api/warning/resolutions/w1/review'
    assert app.post(url, json={'existing_case_id': CASE}).status_code == 404
    monkeypatch.setenv('RAG_WARNING_REVIEW_TOKEN', 'test-secret')
    assert app.post(url, json={'existing_case_id': CASE}).status_code == 403
    result = app.post(url, json={'existing_case_id': CASE}, headers={'x-warning-review-token': 'test-secret'})
    assert result.status_code == 200, result.text
    assert memory.resolution('w1')['manual_case']['manual_case_id'] == CASE
    assert app.get(path(cid)).json()['recording_status'] == 'saved'


def test_atomic_case_close_rolls_back_on_delivery_conflict(app):
    cid = ingest(app); choose(app, cid)
    body = close_body()
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    d = store.get(cid); d['status'] = 'new'
    delivery = {'handoff': warning(), 'pattern_key': None, 'resolution': {'actual_action': '충돌'}}
    with pytest.raises(ValueError, match='already queued'):
        store.put(d, warning_delivery=delivery)
    assert store.get(cid)['status'] == 'resolved'


def test_delivery_retry_reuses_the_same_resolution_and_suppresses_automatic_card(app, memory, monkeypatch):
    monkeypatch.setenv('RAG_WARNING_EQUIPMENT_SCOPE', initial_catalog()[CASE]['equipment_scope'])
    cid = ingest(app)
    choose(app, cid)
    assert app.post(path(cid, '/resolve'), json=close_body()).json()['warning_resolution_status'] == 'pending'
    original_record = memory.record
    attempts = []

    def loses_first_response(resolution, case):
        attempts.append(resolution)
        result = original_record(resolution, case)
        if len(attempts) == 1:
            raise ServiceUnavailable('synthetic lost response after graph commit')
        return result

    monkeypatch.setattr(memory, 'record', loses_first_response)
    assert bridge.deliver_next(now=100)
    later = ingest(app, warning('next'))
    pending = app.get(path(later)).json()
    assert pending['status'] == 'pending'
    assert pending['automatic_card'] is None
    assert not bridge.deliver_next(now=101)
    assert bridge.deliver_next(now=102)
    assert attempts[0] == attempts[1]
    assert len(memory.rows) == 1
    assert app.get(path(cid)).json()['recording_status'] == 'saved'
    assert app.get(path(later)).json()['status'] == 'candidates'


def test_shared_aura_distinguishes_two_api_databases(app, monkeypatch, tmp_path):
    if os.getenv('RAG_WARNING_TEST_AURA') != '1':
        pytest.skip('set RAG_WARNING_TEST_AURA=1 to verify shared Aura')
    load_environment()
    uri = urlsplit(os.getenv('NEO4J_URI', ''))
    assert uri.scheme == 'neo4j+s' and (uri.hostname or '').endswith('.databases.neo4j.io')
    namespace = f'warning-test-{uuid4()}'
    monkeypatch.setenv('RAG_WARNING_NAMESPACE', namespace)
    monkeypatch.setenv('RAG_WARNING_EQUIPMENT_SCOPE', initial_catalog()[CASE]['equipment_scope'])
    target = bridge.target()

    with graph_driver() as driver:
        graph = WarningMemory(driver, **target)
        monkeypatch.setattr(bridge, 'memory', lambda _: graph)
        try:
            for index in range(2):
                monkeypatch.setattr(store, 'DB_PATH', tmp_path / f'api-{index}.db')
                eid = f'shared-{index}'
                cid = ingest(app, warning(eid))
                assert cid == 'CASE-0001'
                if index == 0:
                    choose(app, cid, eid)
                else:
                    history = app.get(path(cid)).json()['history_candidates']
                    assert history[0]['resolved_cases'] == 1
                    assert app.post(path(cid, '/select'),
                                    json={'event_id': eid, 'manual_case_id': CASE}).status_code == 200
                assert app.post(path(cid, '/resolve'), json=close_body(eid)).status_code == 200
                assert bridge.deliver_next()
                assert app.get(path(cid)).json()['recording_status'] == 'saved'
            monkeypatch.setattr(store, 'DB_PATH', tmp_path / 'api-reader.db')
            cid = ingest(app, warning('shared-next'))
            card = app.get(path(cid)).json()['automatic_card']
            assert card['applicable_cases'] == 2
            assert {r['case_id'] for r in card['repairs']} == {'warning:shared-0', 'warning:shared-1'}
        finally:
            with driver.session(database=graph.database) as session:
                session.run(
                    """MATCH (n {namespace:$namespace})
                    WHERE n:WarningPattern OR n:WarningResolution OR n:WarningManualCase
                    DETACH DELETE n""",
                    namespace=namespace,
                ).consume()


def test_critical_existing_path_is_unchanged(app, make_handoff):
    cid = ingest(app, make_handoff('critical'))
    result = app.get(f'/api/cases/{cid}').json()
    assert result['guidance']['status'] == 'ok'
    assert 'warning_rag' not in result
    assert app.get(path(cid)).status_code == 409


def test_invalid_legacy_warning_safe_detail(app):
    event = warning(); event['schema_version'] = '1.3'
    cid = ingest(app, event)
    result = app.get(f'/api/cases/{cid}')
    assert result.status_code == 200
    assert result.json()['warning_rag']['status'] == 'invalid_input'
    assert app.post(path(cid, '/candidates'), json={'event_id': 'w1'}).status_code == 422


def test_close_saves_current_checks_without_prior_draft(app):
    cid = ingest(app); choose(app, cid)
    check = initial_catalog()[CASE]['checks'][0]['id']
    body = {**close_body(), 'checks': {check: 'abnormal'}}
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    state = store.get(cid)['warning_events']['w1']
    assert state['draft']['checks'] == {check: 'abnormal'}


def test_closed_case_keeps_displayed_inspection(app, monkeypatch):
    cid = ingest(app)
    before = app.get(path(cid)).json()
    choose(app, cid)
    assert app.post(path(cid, '/resolve'), json=close_body()).status_code == 200
    monkeypatch.setattr(ew.EarlyWarningRag, 'inspect', lambda *a, **k: pytest.fail('closed card regenerated'))
    after = app.get(path(cid)).json()
    assert after['observations'] == before['observations']
    assert after['selection']['manual_case_id'] == CASE
