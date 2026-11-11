"""End-to-end API contracts with deterministic RAG retrieval/model doubles.

Sensor analysis, SQLite memory, resolution rules and the API routes are real.
Only manual retrieval and external model generation are stubbed.
"""
import copy
import json
import sqlite3
from pathlib import Path
import pytest
import store
import main
import warning_bridge as bridge
from rag import early_warning as ew
from rag.warning_catalog import initial_catalog, source_ref

CASE = 'manual:E012-C2'


def warning(eid='w1', gun='gun-1'):
    event = json.loads((Path(__file__).parents[2] / 'rag/tests/fixtures/early_warning_v14.json').read_text())
    event.update(event_id=eid, gun_id=gun)
    event['sensor_features']['channels'] = {'c3': [[0, 10], [40, 12]], 'c5': [[0, 4.5], [30, 3.2]], 'c10': [[0, 0], [40, 1]]}
    event['contributing_features'] = [{'sensor': c, 'feature': c + '_mean'} for c in ('c3', 'c5', 'c10')]
    return event


@pytest.fixture
def app(api, monkeypatch):
    monkeypatch.delenv('RAG_WARNING_FEEDBACK', raising=False)
    monkeypatch.delenv('RAG_WARNING_EQUIPMENT_SCOPE', raising=False)
    monkeypatch.delenv('RAG_WARNING_REVIEW_TOKEN', raising=False)
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


def test_default_off_sensor_inspection_no_generation(app, monkeypatch):
    cid = ingest(app)
    monkeypatch.setattr(ew, '_claude', lambda *a, **k: pytest.fail('detail GET called model'))
    result = app.get(f'/api/cases/{cid}').json()['warning_rag']
    assert result['status'] == 'feedback_disabled'
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


def test_forged_or_other_event_selection_rejected(app):
    cid = ingest(app)
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'manual_case_id': CASE}).status_code == 422
    assert app.post(path(cid, '/candidates'), json={'event_id': 'foreign'}).status_code == 404
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'evidence_id': 'fake'}).status_code == 422


def test_repeat_keeps_whole_event_and_uses_selected_repeat(app):
    cid = ingest(app)
    repeat = warning('w2')
    repeat['sensor_features']['channels']['c5'] = [[0, 1], [30, 2]]
    app.post('/api/ml/handoffs', json=repeat)
    assert store.get(cid)['repeats'][0]['handoff'] == repeat
    choose(app, cid, 'w2')
    assert app.post(path(cid, '/resolve'), json=close_body('w2')).status_code == 200
    row = store.warning_resolution_for_case(cid)
    assert row['event_id'] == 'w2'
    assert 'c5:up' in row['pattern_key']


def test_resolution_saved_when_feedback_off_and_retry_idempotent(app):
    cid = ingest(app)
    choose(app, cid)
    body = close_body()
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    assert len(store.warning_resolutions()) == 1
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    body['actual_action'] = '다른 내용'
    assert app.post(path(cid, '/resolve'), json=body).status_code == 409
    assert app.get('/api/history').json()['items'][0]['cause']


@pytest.mark.parametrize('field,value', [('worker_confirmed', False), ('log_approved', False), ('actual_action', ' '), ('worker', ' ')])
def test_confirmation_and_real_action_required(app, field, value):
    cid = ingest(app)
    choose(app, cid)
    body = close_body(); body[field] = value
    assert app.post(path(cid, '/resolve'), json=body).status_code == 422
    assert not store.warning_resolutions()
    assert store.get(cid)['status'] == 'new'


def test_false_alarm_never_becomes_resolved_training_case(app):
    cid = ingest(app)
    assert app.post(f'/api/cases/{cid}/close', json={'outcome': 'false_alarm'}).status_code == 200
    assert not store.warning_resolutions()
    assert app.post(path(cid, '/candidates'), json={'event_id': 'w1'}).status_code == 409


def test_old_close_cannot_bypass_evidence(app):
    cid = ingest(app)
    assert app.post(f'/api/cases/{cid}/close', json={'outcome': 'resolved'}).status_code == 409


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
    for i in range(2):
        eid = f'e{i}'
        cid = ingest(app, warning(eid, f'g{i}'))
        choose(app, cid, eid)
        assert app.post(path(cid, '/resolve'), json=close_body(eid)).status_code == 200
    cid = ingest(app, warning('e3', 'g3'))
    assert app.get(path(cid)).json()['history_candidates'] == []
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '1')
    response = app.get(path(cid)).json()
    assert response['status'] == 'automatic'
    assert response['automatic_card']['applicable_cases'] == 2
    assert app.post(path(cid, '/reject'), json={'event_id': 'e3'}).status_code == 200
    assert app.get(path(cid)).json()['automatic_card'] is None


def test_no_scope_prevents_automatic_reuse(app, monkeypatch):
    monkeypatch.setenv('RAG_WARNING_FEEDBACK', '1')
    cid = ingest(app)
    choose(app, cid)
    assert app.post(path(cid, '/resolve'), json=close_body()).status_code == 200
    cid = ingest(app, warning('next', 'g2'))
    assert app.get(path(cid)).json()['automatic_card'] is None


def test_manual_search_selection_close_and_review(app, monkeypatch):
    cid = ingest(app)
    case = initial_catalog()[CASE]
    source = source_ref(case['source']['doc'], case['source']['page'], case['source']['quote'], evidence_id='proof')
    monkeypatch.setattr(ew.EarlyWarningRag, 'search_manuals', lambda *a, **k: {'event_id': 'w1', 'status': 'ok', 'hits': [{'source': source}], 'notes': []})
    assert app.post(path(cid, '/manual'), json={'event_id': 'w1', 'question': '압력 확인'}).status_code == 200
    assert app.post(path(cid, '/select'), json={'event_id': 'w1', 'evidence_id': 'proof'}).status_code == 200
    assert app.post(path(cid, '/resolve'), json=close_body()).status_code == 200
    row = store.warning_resolutions()[0]
    assert row['manual_case'] is None
    assert store.get(cid)['warning_resolution_status'] == 'pending_review'
    url = '/api/warning/resolutions/w1/review'
    assert app.post(url, json={'existing_case_id': CASE}).status_code == 404
    monkeypatch.setenv('RAG_WARNING_REVIEW_TOKEN', 'test-secret')
    assert app.post(url, json={'existing_case_id': CASE}).status_code == 403
    result = app.post(url, json={'existing_case_id': CASE}, headers={'x-warning-review-token': 'test-secret'})
    assert result.status_code == 200, result.text
    assert store.warning_resolutions()[0]['manual_case']['manual_case_id'] == CASE


def test_atomic_case_close_rolls_back_on_resolution_conflict(app):
    cid = ingest(app); choose(app, cid)
    body = close_body()
    assert app.post(path(cid, '/resolve'), json=body).status_code == 200
    d = store.get(cid); d['status'] = 'new'
    conflicting = copy.deepcopy(store.warning_resolutions()[0]); conflicting['actual_action'] = '충돌'
    with pytest.raises(ValueError):
        store.put(d, warning_resolution=conflicting)
    assert store.get(cid)['status'] == 'resolved'


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
