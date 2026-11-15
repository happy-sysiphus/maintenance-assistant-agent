"""Isolated synthetic UI acceptance server; never connects to providers."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'api'), str(root)]
import tempfile
_temp = tempfile.TemporaryDirectory(prefix='warning-ui-test-')
os.environ['API_DB']=str(Path(_temp.name)/'api.db')
os.environ['RAG_WARNING_FEEDBACK']='0'
os.environ['RAG_WARNING_NAMESPACE']='test-ui'
# 빈 값으로 둔다: 지우면 rag/.env의 개인 설정(예: cpx-weld-2g-sa)이 다시 읽혀 테스트 결과가 PC마다 달라진다 (FE 11/12)
os.environ['RAG_WARNING_EQUIPMENT_SCOPE']=''
import main, store, warning_bridge
from warning_test_memory import TestWarningMemory
from rag import early_warning as ew
from rag.warning_catalog import initial_catalog
from fastapi import HTTPException
import json
case_id='manual:E012-C2'
source=initial_catalog()[case_id]['source']
def retrieve(*args, **kw):
    return {'evidence':[{**source,'id':'demo-proof','text':source['quote'],'review_needed':False}]}
def claude(prompt, context, model, **kw):
    if model is ew._Candidates:
        return ew._Candidates(candidates=[ew._Selection(manual_case_id=case_id,reason='전극 위치 상승과 압력 하락이 관측되어 공급 계통의 점검 조건을 확인하는 후보입니다.')])
    return ew._ManualReply(notes=[ew._ManualNote(evidence_id='demo-proof',text='검색된 매뉴얼의 공급 계통 점검 항목을 확인하세요.')])
ew.retrieve=retrieve
ew._claude=claude
memory=TestWarningMemory()
warning_bridge.memory=lambda _: memory
main.ml_get=lambda path: {'points':[]} if '/trace' in path else []
event=json.loads((root/'rag/tests/fixtures/early_warning_v14.json').read_text(encoding='utf-8'))
event.update(event_id='EW-DEMO-0001',gun_id='DEMO-GUN-01')
event['sensor_features']['channels']={'c3':[[0,10],[40,12]],'c5':[[0,4.5],[30,3.2]],'c10':[[0,0],[40,1]]}
event['contributing_features']=[{'sensor':c,'feature':c+'_mean'} for c in ('c3','c5','c10')]
if not store.all_cases(): main.ingest(event)

# Only this isolated acceptance server exposes synthetic setup; api/main.py does not.
from copy import deepcopy
class SyntheticAssessment:
    def __init__(self):
        self.events = {}
        self.decisions = {}
    def register(self, event):
        self.events[event['event_id']] = event
    def read_decision(self, event_id):
        return self.decisions.get(event_id)
    def assess(self, event_id):
        current = self.events[event_id]
        card = {**initial_catalog()[case_id], 'origin': 'error_code' if current.get('observed_error') else 'history',
                'repair': {'event_id': 'prior-repair', 'actual_action': '공급 계통을 정비함', 'closed_at': '2021-01-01T00:00:00Z'}}
        kind = current.get('test_kind')
        result = {'event_id': event_id, 'route': 'manual', 'status': 'unseen', 'card': None, 'cards': [],
                  'observations': [{'channel': 'c3', 'name': 'position', 'observed_seconds': 60, 'first': 10, 'last': 12}]}
        if kind == 'code' and current.get('observed_error'):
            result.update(route='error_code', status='ok', cards=[card], observed_error=current['observed_error'])
        elif kind == 'seen':
            result.update(route='history', status='seen', card=card,
                          candidate_comparisons=[{'event_id': 'prior-repair', 'score': .95, 'channels': [{'channel': 'c3', 'overlap_seconds': 60}]}])
        self.decisions[event_id] = result
        return result
    def manual_candidates(self, event_id):
        return {'event_id': event_id, 'status': 'ok', 'candidates': [{
            'title': '공급 계통 원문 확인', 'reason': '센서 변화에 관련된 PDF 근거예요.', 'step_kind': 'check',
            'excerpt': source['quote'], 'source': {**source, 'evidence_id': 'series-proof'}, 'evidence_id': 'series-proof'}]}
    def manual_search(self, event_id, question, **kwargs):
        return {'event_id': event_id, 'status': 'retrieval_only', 'cards': [{**initial_catalog()[case_id], 'origin': 'manual'}], 'hits': [{'source': {**source, 'evidence_id': 'series-proof'}}], 'notes': []}
    def close(self, event_id, **kwargs):
        return {'status': 'saved'}
assessment = SyntheticAssessment()
warning_bridge.assessment_service = lambda *args, **kwargs: assessment
@main.app.post('/api/__test/timeseries')
def seed_timeseries(kind: str = 'code'):
    sample = deepcopy(event)
    number = store.next_id()
    sample.update(event_id=f'{number}-warning', gun_id=f'{number}-gun', episode_id=f'{number}-episode', test_kind=kind)
    os.environ['RAG_WARNING_TIMESERIES'] = '1'
    try:
        main.ingest(sample)
        if kind == 'code':
            critical = {**sample, 'event_id': f'{number}-critical', 'event_type': 'critical', 'early_warning': None,
                        'observed_error': {'code': 'E012', 'observed_at': sample['detected_at']}}
            main.ingest(critical)
    finally:
        os.environ['RAG_WARNING_TIMESERIES'] = '0'
    return {'case_id': number}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(main.app,host='127.0.0.1',port=int(os.getenv('WARNING_TEST_PORT','19017')),log_level='warning')
