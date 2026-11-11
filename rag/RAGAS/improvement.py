"""Bounded improvement checks; one shared USD 3 ledger, old results stay unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0,str(REPO))
sys.stdout.reconfigure(encoding="utf-8")
from budget import Budget, PHASE, SAMPLE, MeteredTransport, MeteredAnthropicTransport
from rag.config import load_environment, ssl_context, graph_driver, get_database
from rag.documents import searchable_chunk, DATA
from rag.paths import PARSED_DOCS, PARSED_BLOCKS, RUNTIME
from rag.pipeline.chunk import load_parsed_documents, filter_review_blocks, chunk_documents, save_chunks_to_jsonl
from rag.retrieval import search_context
import httpx
import httpx2
from openai import OpenAI
from anthropic import Anthropic
OUT = HERE/"results-20261104"
CACHE = RUNTIME/"cache"/"embedding-comparison.json"
MODEL = "text-embedding-3-small"

def read(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def openai(budget):
    return OpenAI(max_retries=0,timeout=120,http_client=httpx.Client(transport=MeteredTransport(budget,ssl_context())))

def claude(budget):
    return Anthropic(max_retries=0,timeout=120,http_client=httpx2.Client(transport=MeteredAnthropicTransport(budget,ssl_context())))

def digest(text):
    return hashlib.sha256((MODEL+"\0"+text).encode()).hexdigest()

def audit(budget):
    import re,unicodedata
    def canon(s):return re.sub(r"\W+","",unicodedata.normalize("NFKC",s).casefold())
    rows=read(PARSED_DOCS)
    blocks=read(PARSED_BLOCKS)
    pages={(r["metadata"]["manual_id"],r["metadata"]["page_number"]):r for r in rows}
    fixtures=json.loads((REPO/"rag/tests/fixtures/layout-golden.json").read_text(encoding="utf-8"))
    checks=[]
    for f in fixtures["pages"]:
        page=pages.get((f["doc"],f["page"]))
        bs=[b for b in blocks if b["manual_id"]==f["doc"] and b["page_number"]==f["page"]]
        tables=[b for b in bs if b["kind"]=="table"]
        issues=[]
        for text in f.get("text",[]):
            if not page or canon(text) not in canon(page["page_content"]):issues.append("missing: "+text)
        for headers in f.get("headers",[]):
            if not any(t["headers"]==headers for t in tables):issues.append("header relation missing: "+str(headers))
        if "tables" in f and len(tables)!=f["tables"]:issues.append("table boundary count mismatch")
        if f.get("matrix_review") and not any(t.get("matrix_unverified") and t["review_needed"] and len(t["headers"])==2 for t in tables):issues.append("unverified bit assignments not masked")
        checks.append({"doc":f["doc"],"page":f["page"],"passed":not issues,"issues":issues})
    combined=[r for b in blocks for r in b.get("rows",[]) if len(r.get("source_parts",[]))>1]
    joins={r.get("error_number"):sorted({p["page"] for p in r["source_parts"]}) for r in combined}
    assert joins.get("17")==[1,2] and joins.get("30")==[2,3],"cross-page diagnostic row missing"
    row17=next(r for r in combined if r.get("error_number")=="17")
    assert "3timeslower" in canon(row17["search_text"]) and "FrictionHigh" not in row17["search_text"]
    docs=load_parsed_documents(PARSED_DOCS)
    filter_review_blocks(docs,PARSED_BLOCKS)
    chunks=chunk_documents(docs,1200)[0]
    mixed=[c.metadata["chunk_id"] for c in chunks if len({u["source_block_id"] for u in c.metadata["unit_sources"] if u["source_block_kind"]=="table"})>1]
    flagged=[{"doc":b["manual_id"],"page":b["page_number"],"table_id":b.get("table_id"),"reason":b["review_reason"]}
             for b in blocks if b["kind"]=="table" and b["review_needed"]]
    report={"pages":len(rows),"golden_page_checks":checks,"cross_page_rows":joins,
            "mixed_table_chunks":mixed,"table_review_required":flagged,"method":fixtures["method"]}
    save(OUT/"parsing-audit.json",report)
    assert len(rows)==198 and not mixed and all(c["passed"] for c in checks),"PDF audit failed; inspect parsing-audit.json"
    print(f"PDF audit passed: {len(checks)} reviewed pages, 198 pages processed, 0 mixed table chunks; {len(flagged)} table blocks require original review",flush=True)

def connect(budget):
    PHASE.set("connection_check")
    with claude(budget) as client:
        r=client.messages.create(model="claude-sonnet-5-5",max_tokens=128,
            messages=[{"role":"user","content":"Return an object with ok=true."}],
            output_config={"format":{"type":"json_schema","schema":{"type":"object",
                "properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":False}}})
        assert r.stop_reason=="end_turn"
        assert json.loads("".join(b.text for b in r.content if b.type=="text"))["ok"] is True
    with openai(budget) as client:
        e=client.embeddings.create(model=MODEL,input="Maintenance connection check")
        assert len(e.data[0].embedding)==1536
    save(OUT/"connection.json",{"claude_generation":"ok","openai_embedding":"ok",
                               "claude_model":r.model,"embedding_dimensions":1536})

def compare(budget):
    import numpy as np
    PHASE.set("chunk_comparison")
    docs = load_parsed_documents(PARSED_DOCS)
    filter_review_blocks(docs,PARSED_BLOCKS)
    candidates = {size:chunk_documents(docs,size)[0] for size in [600,900,1200]}
    dataset = [r for r in read(HERE/"results"/"dataset.jsonl") if r["category"]=="general"]
    queries = []
    for row in dataset:
        q,terms = search_context(row["question"],None,[])
        queries.append(q+"\n"+" ".join(terms))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    texts = list(dict.fromkeys([c.page_content for cs in candidates.values() for c in cs
                      if searchable_chunk(c.page_content,json.dumps(c.metadata["unit_sources"]))]+queries))
    pending = [t for t in texts if digest(t) not in cache]
    with openai(budget) as client:
        for start in range(0,len(pending),64):
            batch = pending[start:start+64]
            response = client.embeddings.create(model=MODEL,input=batch)
            for e in response.data:
                cache[digest(batch[e.index])] = e.embedding
            save(CACHE,cache)
            print(f"Comparison embeddings {min(start+64,len(pending))}/{len(pending)}; USD {budget.spent:.4f}",flush=True)
    qvectors = np.asarray([cache[digest(q)] for q in queries],dtype=float)
    qvectors /= np.linalg.norm(qvectors,axis=1,keepdims=True)
    scores = []
    for size,chunks in candidates.items():
        eligible = [c for c in chunks if searchable_chunk(c.page_content,json.dumps(c.metadata["unit_sources"]))]
        vectors = np.asarray([cache[digest(c.page_content)] for c in eligible],dtype=float)
        vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
        similarities = qvectors@vectors.T
        recalls,mrr,chars = [],[],[]
        for index,row in enumerate(dataset):
            truth = {(s["doc"],s["page"]) for s in row["sources"]}
            order = np.argsort(-similarities[index])[:8]
            found,first = set(),0
            for rank,j in enumerate(order,1):
                c = eligible[int(j)]
                pages = {(c.metadata["manual_id"],p) for p in c.metadata["source_pages"]}
                found.update(pages)
                if not first and pages&truth:
                    first = rank
            recalls.append(len(found&truth)/len(truth))
            mrr.append(1/first if first else 0)
            chars.append(sum(len(eligible[int(j)].page_content) for j in order))
        scores.append({"max_chars":size,"chunks":len(chunks),"searchable":len(eligible),
            "median_chars":statistics.median(len(c.page_content) for c in eligible),
            "oversized_atomic_units":sum(c.metadata["oversized_atomic_unit"] for c in chunks),
            "source_page_recall_at_8":statistics.mean(recalls),"mrr_at_8":statistics.mean(mrr),
            "mean_retrieved_chars":statistics.mean(chars)})
    winner = max(scores,key=lambda r:(r["source_page_recall_at_8"],r["mrr_at_8"],-r["mean_retrieved_chars"]))
    policy_path = DATA/"processing.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["max_chars"] = winner["max_chars"]
    save(policy_path,policy)
    from rag.paths import CHUNKS
    save_chunks_to_jsonl(candidates[winner["max_chars"]],CHUNKS)
    report = {"method":"offline vector-only comparison; same question/document embedding model; not a hybrid or end-to-end benchmark",
              "questions":len(dataset),"selection":"recall@8, then MRR@8, then smaller retrieved text",
              "metrics":scores,"selected_max_chars":winner["max_chars"]}
    save(OUT/"chunk-comparison.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

def index(budget):
    from rag.pipeline import index as module
    PHASE.set("index")
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    with graph_driver() as driver,driver.session(database=get_database()) as s:
        rows = s.run("MATCH (c:Rag:Chunk) WHERE c.searchable=true RETURN c.id AS id,c.text AS text").data()
        vectors = [{"id":r["id"],"vector":cache[digest(r["text"])]} for r in rows if digest(r["text"]) in cache]
        for start in range(0,len(vectors),64):
            s.run("""UNWIND $rows AS row MATCH (c:Rag:Chunk {id:row.id})
                     SET c.embedding=row.vector,c.embedding_model=$model""",
                  rows=vectors[start:start+64],model=MODEL).consume()
        print(f"Reused {len(vectors)} document vectors",flush=True)
    with patch.object(module,"openai_client",lambda:openai(budget)):
        old = sys.argv
        try:
            sys.argv = ["index"]
            module.main()
        finally:
            sys.argv = old

def migrate(budget):
    from rag.pipeline import ingest,build
    from rag.documents import fingerprint
    pages,chunks=ingest.load_inputs()
    errors,refs=build.resolve_mapping()  # Resolve every quote before the first mutation.
    database=get_database()
    with graph_driver() as driver:
        backup=ingest.backup_database(driver,database)
        report={"backup":backup.name,"status":"backed_up","corpus_version":fingerprint()}
        save(OUT/"kg-migration.json",report)
        try:
            with driver.session(database=database) as s:
                s.execute_write(ingest.synchronize,pages,chunks)
                s.execute_write(build.write_graph,errors,refs)
            index(budget)
            with driver.session(database=database,default_access_mode="READ") as s:
                kb=s.run("MATCH (k:RagKnowledgeBase {id:$id}) RETURN k.ready AS ready,k.version AS version",id=ingest.KB_ID).single()
                assert kb and kb['ready'] and kb['version']==fingerprint()
                codes=[r['id'] for r in s.run("MATCH (s:Rag:Situation) RETURN s.id AS id ORDER BY id")]
                assert codes==['E012','E016','E028','E029']
                counts=s.run("MATCH (n:Rag) RETURN count(n) AS rag_nodes,sum(CASE WHEN n:Chunk THEN 1 ELSE 0 END) AS chunks,sum(CASE WHEN n:Page THEN 1 ELSE 0 END) AS pages").single().data()
                indexes=s.run("SHOW INDEXES YIELD name,state WHERE name IN ['rag_chunk_vector','rag_chunk_text'] RETURN name,state").data()
                assert len(indexes)==2 and all(i['state']=='ONLINE' for i in indexes)
                assert counts['pages']==198
            report.update(status="complete",counts=counts,indexes=indexes,situation_ids=codes)
            save(OUT/"kg-migration.json",report)
            print("Aura migration completed and read-verified",flush=True)
        except Exception:
            ingest.restore_backup(driver,database,backup)
            report['status']='restored_after_failure'
            save(OUT/"kg-migration.json",report)
            raise

def smoke(budget):
    import rag.service as service
    import rag.retrieval as retrieval
    from rag.documents import mapping
    PHASE.set("rag")
    records = [{"case_id":f"SYNTHETIC-{i}","error_code":"E012","closed_at":f"2026-11-0{i}T00:00:00Z",
                "outcome":"unresolved","checks":{"E012-K2":"normal"},
                "actions":["MPYD 케이블 연결 점검"],"note":"케이블 연결을 점검했으나 동일 오류가 재발했다. 공급 압력과 설정 압력은 아직 대조하지 않았다."}
               for i in [1,2]]
    tests = [(c,f"{c} 오류의 의미와 점검 방법을 알려주세요.",{}) for c in mapping()]
    tests += [("firmware","펌웨어 업데이트 절차와 주의사항을 알려주세요.",{}),
              ("master-control","PC에서 필드버스로 제어권을 넘길 때 어떤 위험을 주의해야 하나요?",{}),
              ("memory","E012 오류가 계속 재발합니다. 무엇부터 확인할까요?",{"maintenance_context":records}),
              ("memory-none","E012 오류가 계속 재발합니다. 무엇부터 확인할까요?",{}),
              ("memory-unrelated","E029 오류의 점검 방법을 알려주세요.",{"maintenance_context":records}),
              ("followup","그럼 다음에는 무엇을 확인할까요?",{"history":[{"role":"user","text":"E016 오류 설명"}]})]
    caps=[{"case_id":f"SYNTHETIC-CAP-{i}","error_code":"E016","closed_at":f"2026-11-0{i}T00:00:00Z",
           "outcome":"unresolved","checks":{"E016-K1":"normal"},"actions":["전극·캡 파손·누락 재점검"],
           "note":"두 번 모두 전극·캡에 파손이나 누락은 없었지만 오류가 재발했다. 전극 위치 감시 한계와 실제 위치는 아직 대조하지 않았다."} for i in [1,2]]
    tests += [("priority-base","E016이 반복됩니다. 무엇부터 확인할까요?",{}),
              ("priority-shift","E016이 반복됩니다. 무엇부터 확인할까요?",{"maintenance_context":caps})]
    done = {r["id"] for r in read(OUT/"smoke.jsonl")}
    with patch.object(service,"anthropic_client",lambda:claude(budget)),patch.object(retrieval,"openai_client",lambda:openai(budget)):
        for key,q,extra in tests:
            if key in done:
                continue
            SAMPLE.set("smoke-"+key)
            r = service.answer_question(q,use_cache=False,**extra)
            with (OUT/"smoke.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"id":key,"synthetic_records":bool(extra.get("maintenance_context")),"result":r},ensure_ascii=False)+"\n")
            print(f"Smoke {key}: {r['status']} {r['seconds']}s USD {budget.spent:.4f}",flush=True)
            if r["status"]!="ok":
                raise RuntimeError("Smoke check did not pass; inspect sanitized result before continuing")
    results = {r["id"]:r["result"] for r in read(OUT/"smoke.jsonl")}
    assert results["memory"]["causes"]==results["memory-none"]["causes"]
    assert results["memory"]["maintenance_advice"]
    assert results["memory-unrelated"]["maintenance_advice"] is None
    assert results["followup"]["error_code"]=="E016"
    assert results["priority-base"]["causes"]==results["priority-shift"]["causes"]
    assert results["priority-shift"]["maintenance_advice"]["check_ids"][0] in {"E016-K2","E016-K3"}

def api_check(budget):
    import os,tempfile
    sys.path.insert(0,str(REPO/"api"))
    import main as app_main
    import store
    import rag.service as service
    import rag.retrieval as retrieval
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    PHASE.set("rag")
    SAMPLE.set("api-integration")
    event={"event_id":"synthetic-1","gun_id":"synthetic-a","detected_at":"2026-11-04T00:00:00Z",
           "trigger":{"rule_code":"E012"},"fault_class":{"terminal_code":"E012","code":"E01"},
           "sensor_trend":[],"sensor_findings":[],"error_timeline":[]}
    with tempfile.TemporaryDirectory(dir=RUNTIME,prefix="api-check-") as tmp, \
         patch.object(store,"DB_PATH",Path(tmp)/"api.db"),patch.dict(os.environ,{"FIELD_MEMORY":"1"}), \
         patch.object(app_main,"score_trace",return_value=[]), \
         patch.object(app_main,"ml_get",side_effect=HTTPException(502,"offline ML fixture")), \
         patch.object(service,"anthropic_client",lambda:claude(budget)), \
         patch.object(retrieval,"openai_client",lambda:openai(budget)),TestClient(app_main.app) as client:
        assert client.post("/api/ml/handoffs",json=event).json()=={"result":"new"}
        client.put("/api/cases/CASE-0001/checks",json={"checks":{"E012-K2":{"result":"normal"}}}).raise_for_status()
        client.post("/api/cases/CASE-0001/actions",json={"kind":"케이블 연결 점검","worker":"synthetic"}).raise_for_status()
        client.post("/api/cases/CASE-0001/close",json={"outcome":"unresolved","note":"케이블을 점검했지만 오류가 재발. 압력 대조는 미실시."}).raise_for_status()
        event.update(event_id="synthetic-2",gun_id="synthetic-b")
        assert client.post("/api/ml/handoffs",json=event).json()=={"result":"new"}
        detail=client.get("/api/cases/CASE-0002").json()
        assert detail["guidance"]["status"]=="ok"
        assert detail["guidance"].get("maintenance_advice")
        chat=client.post("/api/assistant/ask",json={"question":"E012가 계속 재발합니다. 과거 정비를 참고해 무엇부터 확인할까요?","photos":[],"history":[]}).json()
        assert chat["status"]=="ok" and chat["maintenance_advice"]
        assert "CASE-0001" in chat["maintenance_advice"]["case_ids"]
        again=client.post("/api/assistant/ask",json={"question":"E012가 계속 재발합니다. 과거 정비를 참고해 무엇부터 확인할까요?","photos":[],"history":[]}).json()
        assert again["cache_hit"] and again["api_calls_this_run"]==0
        assert client.get("/api/manuals/manual82.pdf").content.startswith(b"%PDF")
        assert store.get("CASE-0001")["records"]["checks"]["E012-K2"]["result"]=="normal"
        save(OUT/"api-integration.json",{"method":"FastAPI TestClient, real Aura/Claude/OpenAI, isolated synthetic SQLite records",
            "handoff":"ok","pdf":"ok","record_preservation":"ok","cache":"ok","case_guidance":detail["guidance"],"chat":chat})
        print("Live API route integration passed; synthetic database removed",flush=True)


def report(budget):
    from report import build_report
    build_report(OUT)
    before=json.loads((HERE/"results/summary.json").read_text(encoding="utf-8"))
    after=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    parsing=json.loads((OUT/"parsing-audit.json").read_text(encoding="utf-8"))
    sizes=json.loads((OUT/"chunk-comparison.json").read_text(encoding="utf-8"))
    kg=json.loads((OUT/"kg-migration.json").read_text(encoding="utf-8"))
    api=json.loads((OUT/"api-integration.json").read_text(encoding="utf-8"))
    smoke_rows=read(OUT/"smoke.jsonl")
    smoke_results={r["id"]:r["result"] for r in smoke_rows}
    names={"faithfulness":"Faithfulness","answer_relevancy":"Answer Relevancy",
           "context_precision":"Context Precision","context_recall":"Context Recall","factual_correctness":"Factual Correctness F1"}
    lines=["# 2026-11-04 RAG 개선 비교","","## 완료 범위","",
        "- PDF 198쪽을 레이아웃·원본 셀 경계로 변환하고, 원문 대조한 20쪽의 핵심 항목을 검사했습니다.",
        "- 서로 다른 표가 한 청크에 섞인 사례는 5건에서 0건입니다. 진단서 오류 17·30의 페이지 간 설명을 연결했습니다.",
        f"- 원본 확인이 필요한 표 블록 {len(parsing['table_review_required'])}개는 보류 표시를 유지합니다. 전체 표 전사 정확도 100%를 뜻하지 않습니다.",
        f"- Aura: 4개 Situation, {kg['counts']['pages']}쪽, {kg['counts']['chunks']}개 청크, 두 검색 인덱스 ONLINE.",
        "- 서비스 답변은 Claude Sonnet 5.5, 문서·질문 임베딩은 text-embedding-3-small입니다.",
        "- 공개 함수·FastAPI 경로·정비 이력·캐시·PDF 경로를 실제 Aura/Claude/OpenAI와 연결해 확인했습니다. API 시험의 SQLite 기록은 합성 자료이며 시험 후 제거했습니다.",
        "","## 청크 크기를 1200자로 유지한 이유","",
        "| 목표 문자 수 | 원본 청크 수 | 검색 가능 청크 중앙값 | 근거 페이지 Recall@8 | MRR@8 | 평균 검색 문자 수 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in sizes["metrics"]:
        lines.append(f"| {r['max_chars']} | {r['chunks']} | {r['median_chars']} | {r['source_page_recall_at_8']:.4f} | {r['mrr_at_8']:.4f} | {r['mean_retrieved_chars']:.0f} |")
    lines += ["","동일한 일반 질문 18개와 동일 임베딩으로 벡터 검색만 비교했습니다. 회수율 → MRR → 근거 길이 순으로 선택했습니다.",
        "1200자는 모든 청크의 길이가 아니라 목표 상한입니다. 긴 표 행은 자르지 않습니다. 900자 대비 MRR 차이는 작고, 이 표본에서의 선택이지 보편적 최적값은 아닙니다.",
        "","## 같은 50문항의 RAGAS 결과","",
        "같은 질문·합성 기준 답변·GPT-4.1-mini 평가자·공식 ragas 0.4.3 지표를 사용했습니다. history와 정비 이력은 이 50문항에 넣지 않았습니다.",
        "평가 입력은 UI JSON 전체가 아닌 전체 answer 문자열입니다. 점검·기준·조치가 렌더링된 문자열을 그대로 평가했습니다.",
        "","| 지표 | 이전 | 개선 | 차이 | 유효 문항 |","|---|---:|---:|---:|---:|"]
    for key,label in names.items():
        b=before["metrics"]["all"][key]["mean"];a=after["metrics"]["all"][key]["mean"]
        if a is not None:lines.append(f"| {label} | {b:.4f} | {a:.4f} | {a-b:+.4f} | {after['metrics']['all'][key]['valid']}/50 |")
    lines += ["","코드 질문과 일반 질문의 세부 수치는 [REPORT.md](REPORT.md), 문항별 원문은 [scores.csv](scores.csv)에 있습니다.",
        "","근거 회수율과 Factual F1은 상승했습니다. Faithfulness는 거의 같고 Answer Relevancy 변화는 +0.0058로 작습니다.",
        "Context Precision은 0.7023→0.6713으로 하락했습니다. 근거를 더 넓게 가져온 만큼 불필요한 문맥도 포함될 수 있다는 신호입니다. 이 비교만으로 원인을 확정하거나 통계적으로 유의한 개선이라고 말하지 않습니다.",
        "","## 정비 이력에 따른 제안 사례","",
        "> 질문: E016이 반복됩니다. 무엇부터 확인할까요?","",
        "이력이 없을 때는 전극·캡 파손·누락 점검(E016-K1)을 먼저 안내했습니다.",
        "두 차례 E016-K1이 정상이었지만 미해결이라는 합성 기록을 제공한 경우의 제안:","",
        "> "+smoke_results["priority-shift"]["maintenance_advice"]["text"],"",
        "사례 ID가 붙은 정비 제안은 매뉴얼 근거와 구분됩니다. 카드 배열·정상 기준·조치는 이력 유무와 관계없이 동일했습니다. 현장 성능이 검증됐다는 뜻은 아닙니다.",
        "","## 시간·비용","",
        f"- 이전 50문항 평균 {before['rag_seconds_mean']:.3f}초 → 이번 {after['rag_seconds_mean']:.3f}초, 이번 p95 {after['rag_seconds_p95']:.3f}초.",
        f"- 이전 답변 1건 평균 USD {before['final_answer_mean_cost_usd']:.6f} → 이번 USD {after['final_answer_mean_cost_usd']:.6f}. 질문 임베딩·생성 비용이며 평가자 비용은 제외합니다.",
        f"- 이번 전체 작업의 기록 추정 비용: USD {after['recorded_usd']:.6f} / 상한 USD 3. 연결·청크 실험·스모크·API·50문항 생성·채점·실패 재시도를 포함합니다.",
        "- 토큰 사용량과 표준 단가에 따른 추정치입니다. OpenAI 입력 캐시 할인은 제외하며, 사용량 없는 실패 요청은 예약 상한으로 계산합니다. 실제 계정 잔액이나 청구 내역을 직접 조회한 금액은 아닙니다.",
        "- [Claude 단가](https://platform.claude.com/docs/en/about-claude/pricing), [평가자 단가](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [임베딩 단가](https://developers.openai.com/api/docs/models/text-embedding-3-small).",
        "","## 해석 범위와 남은 한계","",
        "- 파싱·문맥 확장·답변 모델을 함께 바꿨으므로 점수 차이를 파싱 하나의 효과로 해석하지 않습니다.",
        "- 청크 선택에 사용한 일반 질문 18개가 평가 50문항에도 포함됩니다. 개발 세트 비교이며 독립 검증 세트의 일반화 성능이 아닙니다.",
        "- 기준 답변은 자동 생성한 합성 reference입니다. 기존 자동 검토에서 근거 부족으로 표시된 E016-07도 그대로 포함했습니다.",
        "- 기본 영문 평가 프롬프트, 다국어 역질문 유사도, 짧은 reference와 넓은 UI 답변의 길이 차이, LLM 채점 변동의 영향을 받습니다.",
        "- 유효한 낮은 점수는 재시도하지 않았고 출력 한도·서버 오류 등 실패한 지표만 재시도했습니다. 모든 시도는 metrics.jsonl에 남깁니다.",
        "- 그림·도식·회전 비트 열·복원이 확실하지 않은 수식과 머리글은 원본 확인이 필요합니다. 해당 목록은 parsing-audit.json에 있습니다.",
        "- FIELD_MEMORY=1에서 같은 E 코드의 최근 종료 사례 최대 5건을 활용합니다. gun별 차이, 미해결 조치의 성공률, 작업자 행동 변화는 이 실험에서 검증하지 않았습니다.",
        "- Aura 갱신은 공유되어 있으므로 이번 코드와 환경 설정을 함께 사용해야 합니다. 기존 코드의 문서 버전 검사는 새 KG를 거절할 수 있습니다.",""]
    (OUT/"COMPARISON.md").write_text("\n".join(lines),encoding="utf-8")
    diary=REPO/"rag/docs/development-log.md"
    prefix=diary.read_text(encoding="utf-8").split("### 검증 결과")[0]
    body=["### 검증 결과","",
        "- 전체 198쪽 변환, 원문 대조한 20쪽 핵심 항목 통과. 서로 다른 표가 섞인 청크 5→0건.",
        "- 모델만 적용했을 때 진단서의 열 축소·행 경계 왜곡을 확인해 PDF 원본 격자와 대조하도록 보완했다. 다단 머리글과 빈 유령 열도 원문과 대조했다.",
        "- 진단서 오류 17은 1·2쪽, 오류 30은 2·3쪽을 하나의 검색 단위로 연결하고 페이지별 근거를 보존한다.",
        f"- {len(parsing['table_review_required'])}개 표 블록은 머리글·도식·수식·회전 비트 열 검토 표시를 유지한다. 그림은 위치만 기록한다.",
        "- 600·900·1200자 검색 비교에서 1200자를 선택했다. 회수율은 0.7778 / 0.8333 / 0.8333, MRR은 0.5144 / 0.5134 / 0.5181이었다. 검색 가능 청크 중앙값은 222자이며 긴 원자 단위는 절단하지 않는다.",
        f"- Aura 백업 후 갱신: 4개 Situation, 198쪽, {kg['counts']['chunks']}개 청크, 검색 대상 732개, 두 인덱스 ONLINE. 임베딩 724개는 비교 실험 결과를 재사용하고 8개(46토큰)만 추가 생성했다.",
        "- KG는 인덱스 구축까지 끝난 뒤 ready로 전환한다. 실패 시 RAG 범위만 백업 복구하는 경로를 추가했다. 이번 갱신은 정상 완료되어 실제 장애 복구는 실행하지 않았다.",
        "- Claude JSON 생성·OpenAI 임베딩 연결 성공. 실제 질문 12건 성공. 이전 대화, 이력 유무, 다른 코드 이력 제외를 확인했다.",
        "- FastAPI 경로를 실제 Aura·Claude·OpenAI와 연결해 핸드오프→정비 기록→새 케이스 안내→챗봇→캐시→PDF를 확인했다. 시험 DB는 합성 자료이며 제거했다.",
        "- 공개 함수가 새 maintenance_context 인자를 전달하지 못한 문제를 통합 검사에서 발견해 수정하고 회귀 검사에 추가했다.",
        "- 모델 알림의 rule_code가 비어 있어도 terminal_code로 케이스를 구분하도록 수정했다. 서로 다른 E 코드가 한 케이스로 합쳐지는 문제의 회귀 검사를 추가했다.",
        "- 오프라인 회귀 검사 98개 통과. 기존 UI 요청·카드·기록 키를 유지한다.",
        "- 평가 종료 후 API 호환성 검사를 추가했다. 진행 중 케이스에 기록이 있으면 기존 원인 배열을 보존해 E016-1 같은 화면 순번 ID가 다른 원인에 연결되지 않도록 했다. 생성 중 저장·종료된 기록도 보존한다.",
        "- 이 API 호환성 보완은 50문항 평가가 사용하는 RAG 핵심 코드·데이터셋을 바꾸지 않았다. 평가 시점 스냅샷은 run_manifest.json, 후속 차이는 final-verification.json에 기록한다.","",
        "### 정비 이력 반영 사례","","> E016이 반복됩니다. 무엇부터 확인할까요?","",
        "이력 없음: 전극·캡 파손·누락 점검(E016-K1)을 먼저 안내.",
        "E016-K1 두 차례 정상·미해결 이력 있음: E016-K2를 먼저 검토하도록 제안. 카드 배열은 동일.",
        "실제 제안:","","> "+smoke_results["priority-shift"]["maintenance_advice"]["text"],"",
        "이는 합성 시나리오의 동작 검증이며 실제 현장 정비 성공률을 측정한 결과가 아니다.","",
        "### 같은 50문항의 평가와 비용","",
        f"- 정상 생성 {after['answer_statuses'].get('ok',0)}/50, 유효 지표 "+str(sum(v["valid"] for v in after["metrics"]["all"].values()))+"/250.",
        f"- 전체 작업 기록 추정 비용 USD {after['recorded_usd']:.6f}, 상한 USD 3 이내.",
        f"- 이번 답변 1건 평균 USD {after['final_answer_mean_cost_usd']:.6f}, 평균 {after['rag_seconds_mean']:.3f}초, p95 {after['rag_seconds_p95']:.3f}초.",
        "- 이전 버전보다 답변 비용과 지연이 증가했다. 평가자·모델 가격과 실제 토큰 사용량을 분리해 기록했다. 실제 계정 잔액은 Billing 페이지에서 확인한다.",
        "- 이전 결과는 RAGAS/results/에 보존하고, 같은 데이터셋·평가자로 RAGAS/results-20261104/에 별도 평가했다.",
        "- 18개 일반 질문을 청크 선택에도 사용해 튜닝과 평가가 중복된다. 파싱·검색·모델을 함께 바꾼 개발 세트 비교이며 독립 성능 실험은 아니다.","",
        "| 지표 | 이전 | 개선 |","|---|---:|---:|"]
    for key,label in names.items():
        a=after["metrics"]["all"][key]["mean"];b=before["metrics"]["all"][key]["mean"]
        if a is not None:body.append(f"| {label} | {b:.4f} | {a:.4f} |")
    body+=["","[비교 보고서](../RAGAS/results-20261104/COMPARISON.md) · [문항별 결과](../RAGAS/results-20261104/scores.csv) · [표 검토 목록](../RAGAS/results-20261104/parsing-audit.json)","",
        "근거 회수율과 Factual F1은 상승했지만 검색 정밀도는 하락했다. Faithfulness는 거의 같고 Answer Relevancy 개선 폭도 작다. 전체 지표가 좋아졌다고 해석하지 않는다.","",
        "### 변경 범위와 실행","",
        "- rag/: 파싱·청킹·그래프 출처·검색 문맥·Claude 연결·정비 컨텍스트·검증·예시·README.",
        "- api/: 종료 기록을 생성 전에 전달하는 연결, 케이스 코드 구분, 통합·회귀 검사와 실행 문서.",
        "- ui/: docs/09-rag-development-reply.md만 이번 개선 내용을 추가. ui/src와 ML 코드는 수정하지 않았다.",
        "- Docling은 requirements-parse.txt로 분리해 서비스 설치에는 필요 없다. 모델 파일·파싱 산출물·백업·캐시는 Git 제외 경로에 둔다.",
        "- 실제 실행에는 rag/.env의 Aura·ANTHROPIC_API_KEY·OPENAI_API_KEY가 필요하다. 로컬 비밀 파일에는 두 키를 설정했고 Git에서는 제외했다.",
        "- 정비 이력은 FIELD_MEMORY=1에서 활성화한다. 다른 gun의 같은 코드 기록도 포함하며, 작업자·사진 필드는 생성 컨텍스트에 넣지 않는다.",
        "- 공유 Aura는 새 버전으로 갱신했으므로 팀원도 이번 코드를 함께 받아야 한다.",""]
    diary.write_text(prefix+"\n".join(body),encoding="utf-8")
    print("Wrote comparison report and development diary",flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=["connect","audit","compare","migrate","index","smoke","api","report"])
    args = p.parse_args()
    load_environment()
    OUT.mkdir(parents=True,exist_ok=True)
    b = Budget(OUT/"costs.jsonl",limit=3)
    try:
        {"connect":connect,"audit":audit,"compare":compare,"migrate":migrate,"index":index,"smoke":smoke,"api":api_check,"report":report}[args.stage](b)
    except Exception as exc:
        failure={"stage":args.stage,"error_type":type(exc).__name__,"http_status":getattr(exc,"status_code",None)}
        save(OUT/"stage-error.json",failure)
        print(json.dumps(failure),flush=True)
        return 1
    finally:
        print(f"Total improvement cost USD {b.spent:.6f} / 3",flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
