"""Regressions for source boundaries, provenance, and field-context isolation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
import pytest
from rag.pipeline.parse import link_continuations
from rag.pipeline.chunk import structured_units, chunk_documents
from rag.maintenance import normalize_context, validate_advice
from rag.documents import mapping
from rag.service import answer_question, cache_path
from rag.response import mapped_causes, render_answer
from rag.retrieval import evidence_parts
from rag.pipeline.parse import reconcile_native_grid

def block(page,rows,headers=None,section="7.1 Error numbers",table_id=None):
    tid = table_id or f"diagnostic4:table:{page}"
    return {"kind":"table","manual_id":"diagnostic4","page_number":page,
        "page_height":842,"bbox":[60,700 if page==1 else 50,540,810 if page==1 else 600],
        "table_id":tid,"logical_table_id":tid,"section":section,
        "headers":headers or ["No.","Error","Description"],"header_detected":page==1,
        "rows":[{"row_id":f"{tid}:row:{i}","cells":cells,"text":" | ".join(cells),
                 "source_parts":[{"doc":"diagnostic4","page":page,"text":" | ".join(cells),"bbox":[60,50,540,810]}]}
                for i,cells in enumerate(rows)]}

def record(case="case-1",code="E012",date="2026-11-01T00:00:00Z"):
    return {"case_id":case,"error_code":code,"closed_at":date,"outcome":"unresolved",
            "checks":{"E012-K2":"normal"},"actions":["케이블 연결 점검"],"note":"같은 오류 재발"}

def test_cross_page_row_keeps_both_page_sources():
    a,b = block(1,[["17","Friction Low","The friction is"]]),block(2,[["","","smaller than the limit."],["18","Friction High","greater than the limit"]])
    joined = link_continuations({1:[a],2:[b]})
    assert joined[0]["pages"]==[1,2]
    assert "smaller than the limit" in a["rows"][0]["search_text"]
    assert b["rows"][0]["continuation_of"]==a["rows"][0]["row_id"]
    assert [p["page"] for p in a["rows"][0]["source_parts"]]==[1,2]

def test_different_section_is_not_a_continued_table():
    a,b = block(1,[["17","Friction Low","The friction is"]]),block(2,[["","","another table"]],section="8 Another chapter")
    assert link_continuations({1:[a],2:[b]})==[]
    assert "continuation_of" not in b["rows"][0]

def doc_with_two_tables():
    spans=[]
    for i in [1,2]:
        b=block(1,[[str(i),"Error","Description"]],table_id=f"t{i}")
        b.update(order=i-1,source_block_id=f"b{i}",review_needed=False,review_type=None,review_reason=None,
                 content=f"[Table t{i}]\n{i} | Error | Description",extractable_text=f"[Table t{i}]\n{i} | Error | Description",
                 start=(i-1)*50,end=i*50,caption="",error_number=None)
        spans.append(b)
    return Document(page_content="",metadata={"manual_id":"diagnostic4","page_number":1,
        "parent_doc_id":"p1","parser_version":"layout","_block_spans":spans})

def test_same_caption_different_physical_tables_never_merge():
    chunks,_,_=chunk_documents([doc_with_two_tables()],1200)
    assert len(chunks)==2
    assert all(len({u["source_block_id"] for u in c.metadata["unit_sources"]})==1 for c in chunks)

def test_numbered_prose_is_not_treated_as_heading():
    b={"kind":"text","content":"1 bar or greater. Teaching is not affected.","extractable_text":"1 bar or greater. Teaching is not affected.",
       "start":0,"end":55,"section":"Compensation","manual_id":"manual82","page_number":38,
       "source_block_id":"b","order":0,"bbox":[0,0,10,10],"review_needed":False,"review_type":None,"review_reason":None}
    doc=Document(page_content=b["content"],metadata={"_block_spans":[b]})
    units,_=structured_units(doc)
    assert units[0].section_title=="Compensation"
    assert units[0].text.startswith("1 bar")

def test_cross_page_evidence_has_truthful_page_specific_text():
    units=[{"row_id":"r17","source_parts":[{"page":1,"text":"Error 17: friction is"},{"page":2,"text":"smaller than the configured limit"}]}]
    rows=evidence_parts({"id":"row17","doc":"diagnostic4","page":1,"text":"complete",
                         "source_json":json.dumps(units),"review_needed":False})
    assert [r["page"] for r in rows]==[1,2]
    assert rows[0]["text"]=="Error 17: friction is"
    assert rows[1]["text"]=="smaller than the configured limit"
    assert all(r["logical_row_ids"]==["r17"] for r in rows)

def test_context_filters_code_future_dates_and_limits_to_five():
    rows=[record(str(i),date=f"2026-10-{20+i:02}T00:00:00Z") for i in range(8)]
    rows += [record("other","E029"),record("future",date="2099-01-01T00:00:00Z")]
    got=normalize_context(rows,"E012")
    assert len(got)==5
    assert [r["case_id"] for r in got]==["7","6","5","4","3"]
    assert normalize_context(rows,None)==[]

def test_context_rejects_malformed_check_values_without_crashing():
    r=record()
    r["checks"]={"E012-K1":{"result":"normal"},"E012-K2":"normal","E029-K1":"abnormal"}
    assert normalize_context([r],"E012")[0]["checks"]=={"E012-K2":"normal"}

def test_advice_cannot_invent_case_or_check_ids():
    rows=normalize_context([record()],"E012")
    advice={"text":"다른 점검을 검토하세요.","case_ids":["not-provided"],"check_ids":["E012-K1"]}
    with pytest.raises(ValueError):validate_advice(advice,rows,mapping()["E012"])
    advice["case_ids"]=["case-1"]
    advice["check_ids"]=["E029-K1"]
    with pytest.raises(ValueError):validate_advice(advice,rows,mapping()["E012"])
    with pytest.raises(ValueError):validate_advice(advice,[],mapping()["E012"])

def test_cache_changes_when_maintenance_result_changes():
    a,b=record(),record()
    b["outcome"]="resolved"
    assert cache_path("질문","E012",[],"db","model",[a])!=cache_path("질문","E012",[],"db","model",[b])

def test_public_api_forwards_field_context():
    from rag import answer_question as public_answer
    context=[record()]
    with patch("rag.service.answer_question",return_value={"status":"ok"}) as actual:
        public_answer("질문",error_code="E012",maintenance_context=context)
    assert actual.call_args.kwargs["maintenance_context"]==context

def test_claude_receives_history_and_field_records_without_reordering_cards():
    item=mapping()["E012"]
    evidence=[]
    from rag.pipeline.build import references
    for ref in references(item):
        evidence.append({"id":"source","doc":ref["doc"],"page":ref["page"],"text":ref["quote"],"review_needed":False})
    advice={"text":"케이블 점검 후 재발했으므로 설정 압력과 공급 압력 대조를 먼저 검토하세요.",
            "case_ids":["case-1"],"check_ids":["E012-K1"]}
    payload={"sections":{"interpretation":"오류 설명","summary":"요약"},"causes":[],"maintenance_advice":advice}
    response=SimpleNamespace(stop_reason="end_turn",content=[SimpleNamespace(type="text",text=json.dumps(payload))],
                             usage=SimpleNamespace(input_tokens=100,output_tokens=50))
    history=[{"role":"user","text":"E012 오류가 재발해"}]
    with patch("rag.service.retrieve",return_value={"evidence":evidence,"situations":[item],"embedding_tokens":10}),patch("rag.service.anthropic_client") as client:
        client.return_value.__enter__.return_value.messages.create.return_value=response
        result=answer_question("그럼 다음엔?",history=history,maintenance_context=[record()])
        sent=json.loads(client.return_value.__enter__.return_value.messages.create.call_args.kwargs["messages"][0]["content"])
    assert result["status"]=="ok"
    assert sent["history"]==history
    assert sent["maintenance_context"][0]["case_id"]=="case-1"
    assert result["causes"]==mapped_causes("E012",item)
    assert result["maintenance_advice"]==advice
    assert "case-1" in result["answer"]

def test_real_pdf_parameter_table_drops_ghost_columns_not_units():
    import pdfplumber
    from rag.documents import manual_path
    b={"headers":["Command","Button","Function"],"header_detected":True,"bbox":[74,100,527,220],
       "table_id":"parameters","manual_id":"manual82","page_number":38}
    with pdfplumber.open(manual_path("manual82")) as pdf:
        reconcile_native_grid(b,pdf.pages[37])
    assert b["headers"]==["Command","Button","Function"]
    assert len(b["rows"])==2
    assert b["rows"][0]["cells"][1]=="mm"
    assert "cap loss" in b["rows"][0]["cells"][2]

def test_real_pdf_side_by_side_input_groups_keep_distinct_signals():
    import pdfplumber
    from rag.documents import manual_path
    b={"headers":["Input 0/Input 1","Input 0/Input 1","Input 2/Input 3","Input 2/Input 3"],
       "header_detected":True,"bbox":[74,145,527,268],"table_id":"inputs","manual_id":"manual82","page_number":23}
    with pdfplumber.open(manual_path("manual82")) as pdf:
        reconcile_native_grid(b,pdf.pages[22])
    assert b["headers"]==["Input 0/Input 1 / Pin","Input 0/Input 1 / Signal","Input 2/Input 3 / Pin","Input 2/Input 3 / Signal"]
    row=next(r for r in b["rows"] if r["cells"][0]=="2")
    assert "x+1" in row["cells"][1] and "x+3" in row["cells"][3]
