"""Isolated synthetic UI acceptance server; never connects to providers."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'api'), str(root)]
import tempfile
_temp = tempfile.TemporaryDirectory(prefix='warning-ui-test-')
os.environ['API_DB']=str(Path(_temp.name)/'api.db')
os.environ['RAG_WARNING_FEEDBACK']='0'
import main, store
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
main.ml_get=lambda path: {'points':[]} if '/trace' in path else []
event=json.loads((root/'rag/tests/fixtures/early_warning_v14.json').read_text())
event.update(event_id='EW-DEMO-0001',gun_id='DEMO-GUN-01')
event['sensor_features']['channels']={'c3':[[0,10],[40,12]],'c5':[[0,4.5],[30,3.2]],'c10':[[0,0],[40,1]]}
event['contributing_features']=[{'sensor':c,'feature':c+'_mean'} for c in ('c3','c5','c10')]
if not store.all_cases(): main.ingest(event)
if __name__=='__main__':
    import uvicorn
    uvicorn.run(main.app,host='127.0.0.1',port=int(os.getenv('WARNING_TEST_PORT','19017')),log_level='warning')
