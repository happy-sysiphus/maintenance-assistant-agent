"""Adapt one E-code RAG response to the existing case/record API."""

from pathlib import Path
import sys
from threading import Lock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from rag import answer_question
from rag.documents import mapping, manuals
from rag.response import mapped_causes, clean_cite
from rag.service import DEFAULT_QUESTION, PROMPT_VERSION

_CODE_LOCKS = {code: Lock() for code in mapping()}


def event_code(event):
    return (event.get("trigger") or {}).get("rule_code") or (event.get("fault_class") or {}).get("terminal_code")


def situation(code):
    item = mapping().get(code)
    if not item:
        return None
    return {"situation_id":code,"name":item["name"],"definition":item["definition"],
            "check_sequence":[c["text"] for cause in item["causes"] for c in cause["checks"]],
            "coverage_note":item["coverage_note"],
            "evidence":[{"doc":ref["doc"],"page":ref["page"],"rationale":ref["quote"]}
                        for ref in item["definitions"]],"diagnostics":[]}


def all_situations():
    return [situation(code) for code in mapping()]


def candidate(code, rank=1, response=None):
    item = mapping().get(code)
    if not item:
        return None
    # Canonical checks stay present even if a question-specific answer uses fewer.
    checks = {}
    for cause in item["causes"]:
        for c in cause["checks"]:
            crit = c["criterion"]
            checks[c["id"]] = {
                "id":c["id"],"title":c["text"],"cite":clean_cite(c["cite"]),
                **({"criterion":{"kind":crit["kind"],"text":crit["text"],
                                 "quote":crit["cite"]["quote"],**clean_cite(crit["cite"])}}
                   if crit else {}),
            }
    evidence = (response or {}).get("evidence") or [
        {"doc":r["doc"],"page":r["page"],"quote":r["quote"],"rationale":"오류 정의"}
        for r in item["definitions"]]
    return {"situation_id":code,"rank":rank,"name":item["name"],"definition":item["definition"],
            "coverage_note":item["coverage_note"],"checks":list(checks.values()),
            "diagnostics":[],"evidence":evidence,
            "causes":(response or {}).get("causes") or mapped_causes(code,item)}


def adapt(response):
    code = response["error_code"]
    c = candidate(code,response=response) if code else None
    return {**response,"response_version":response.get("response_version",PROMPT_VERSION),
            "candidates":[c] if c else []}


def guidance_for(event):
    """Fast initial response: no DB/OpenAI calls in ML handoff acknowledgement."""
    code = event_code(event)
    item = mapping().get(code)
    if not item:
        return {"status":"no_evidence","error_code":code,"candidates":[]}
    return adapt({"status":"retrieval_only","error_code":code,"answer":None,
                  "sections":{"interpretation":item["definition"],"summary":""},
                  "causes":mapped_causes(code,item),"evidence":[],"review_notes":[item["coverage_note"]]})


def enrich(case):
    """Called only by detail GET, never by record writes or ML push."""
    current = case["guidance"]
    code = event_code(case["event"])
    if (current.get("status") == "ok" and current.get("response_version") == PROMPT_VERSION
            or code not in mapping()):
        return case
    # Legacy S-based cases retain their recorded keys; new cases use E codes.
    if any(c["situation_id"].startswith("S") for c in current.get("candidates",[])):
        return case
    with _CODE_LOCKS[code]:
        response = answer_question(DEFAULT_QUESTION,error_code=code,use_cache=True)
    if response["status"] != "error":
        case["guidance"] = adapt(response)
    else:
        case["guidance"] = {**current,"status":"error","error":response.get("error")}
    return case


def attach_criteria(case):
    """Retain exact saved check keys. Do not overwrite historical worker records."""
    return case


def search_manuals(question):
    response = answer_question(question,use_llm=False)
    return [{"doc":e["doc"],"page":e["page"],"title":manuals()[e["doc"]]["title"],
             "quote":e["quote"],"situation_id":response["error_code"],
             "situation_name":(mapping().get(response["error_code"]) or {}).get("name")}
            for e in response["evidence"]]
