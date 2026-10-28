"""Build the four Situation nodes using reviewed, document-qualified mappings."""

import argparse
import hashlib
import json
from ..config import graph_driver, get_database
from ..documents import ERROR_CODES, mapping, manuals, normalized, fingerprint
from .ingest import KB_ID, load_inputs


def references(value):
    """Yield only explicit source selectors, never LLM text as provenance."""
    if isinstance(value, dict):
        if {"doc", "page", "quote"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def reference_id(ref):
    raw = f"{ref['doc']}|{ref['page']}|{normalized(ref['quote'])}"
    return "rag:evidence:" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def resolve_mapping():
    pages, chunks = load_inputs()
    errors = mapping()
    if set(errors) != set(ERROR_CODES):
        raise ValueError("매핑에는 E012·E016·E028·E029 네 코드가 필요합니다.")
    by_page = {(p["metadata"]["manual_id"], p["metadata"]["page_number"]): p
               for p in pages}
    resolved = {}
    for code, item in errors.items():
        ids = [check["id"] for cause in item["causes"] for check in cause["checks"]]
        if len(ids) != len(set(ids)) or any(not key.startswith(code+"-K") for key in ids):
            raise ValueError(f"{code}: 고정 점검 ID가 중복되거나 잘못됐습니다.")
        if len(item["definitions"]) != 2:
            raise ValueError(f"{code}: 두 신형 문서의 오류 정의가 필요합니다.")
        for ref in references(item):
            key = (ref["doc"], ref["page"])
            page = by_page.get(key)
            quote = normalized(ref["quote"])
            if not page or not quote:
                raise ValueError(f"{code}: 출처가 없습니다: {ref}")
            candidates = [c for c in chunks
                          if (c["metadata"]["manual_id"], c["metadata"]["page_number"]) == key]
            matches = [c for c in candidates if quote in normalized(c["page_content"])]
            raw = normalized(page["raw_text"])
            if not matches and quote not in raw:
                raise ValueError(f"{code}: 인용문이 PDF 원문에 없습니다: {ref}")
            if matches:
                chosen = min(matches, key=lambda c: len(c["page_content"]))
                text = chosen["page_content"]
                review = chosen["metadata"]["review_needed"]
            else:
                offset = raw.index(quote)
                text = raw[max(0, offset-200):offset+len(quote)+500]
                review = False  # Text only; never use inferred visual content.
            resolved[reference_id(ref)] = {
                "id": reference_id(ref), "doc": ref["doc"], "page": ref["page"],
                "quote": ref["quote"], "text": text, "review_needed": review,
            }
    return errors, resolved


def write_graph(tx, errors, evidence):
    # Only the new version's derived nodes are rebuilt; source chunks persist.
    tx.run("""MATCH (n:Rag) WHERE n:Situation OR n:Cause OR n:CheckAction
        OR n:ManualError OR n:Component DETACH DELETE n""").consume()
    tx.run("""MATCH (c:Rag:Chunk) WHERE c.curated=true AND NOT c.id IN $ids DETACH DELETE c""", ids=list(evidence)).consume()
    for row in evidence.values():
        tx.run("""MATCH (p:Rag:Page {id:$page_id})
            MERGE (c:Rag:Chunk {id:$id})
            SET c.embedding=CASE WHEN c.text=$text THEN c.embedding ELSE null END,
                c.embedding_model=CASE WHEN c.text=$text THEN c.embedding_model ELSE null END
            SET c.doc=$doc,c.page_number=$page,c.quote=$quote,c.text=$text,
                c.review_needed=$review_needed,c.curated=true,c.searchable=true
            MERGE (p)-[:HAS_CHUNK]->(c)""",
            page_id=f"rag:page:{row['doc']}:{row['page']}", **row).consume()
    for code, item in errors.items():
        tx.run("""MERGE (s:Rag:Situation {id:$code})
            SET s.code=$code,s.name=$name,s.definition=$definition,
                s.coverage_note=$coverage,s.mapping_json=$mapping""",
            code=code, name=item["name"], definition=item["definition"],
            coverage=item["coverage_note"], mapping=json.dumps(item, ensure_ascii=False)).consume()
        for ref in references(item):
            tx.run("""MATCH (s:Situation {id:$code}),(c:Chunk {id:$cid})
                MERGE (s)-[r:RELEVANT_EVIDENCE]->(c)
                SET r.quote=$quote,r.doc=$doc,r.page_number=$page""",
                code=code, cid=reference_id(ref), **ref).consume()
        for ref in item["definitions"]:
            tx.run("""MATCH (s:Situation {id:$code}),(c:Chunk {id:$cid})
                MERGE (e:Rag:ManualError {id:$id})
                SET e.number=$number,e.name=$name,e.doc=$doc,e.page_number=$page
                MERGE (s)-[:RELEVANT_DIAGNOSTIC]->(e)
                MERGE (e)-[:SUPPORTED_BY]->(c)""",
                code=code, cid=reference_id(ref),
                id=f"rag:diagnostic:{ref['doc']}:{code}", number=int(code[1:]),
                name=ref["quote"], doc=ref["doc"], page=ref["page"]).consume()
        for cause in item["causes"]:
            tx.run("""MATCH (s:Situation {id:$code}),(c:Chunk {id:$cid})
                MERGE (ca:Rag:Cause {id:$id})
                SET ca.name=$name,ca.why=$why
                MERGE (s)-[:HAS_CAUSE]->(ca)
                MERGE (ca)-[:SUPPORTED_BY]->(c)""",
                code=code, cid=reference_id(cause["cite"]), id=cause["id"],
                name=cause["name"], why=cause["why"]).consume()
            for index, check in enumerate(cause["checks"]):
                tx.run("""MATCH (ca:Cause {id:$cause}),(c:Chunk {id:$cid})
                    MERGE (a:Rag:CheckAction {id:$id})
                    SET a.name=$text,a.kind='check',a.step=$step,a.criterion_json=$criterion
                    MERGE (ca)-[:CHECKED_BY]->(a)
                    MERGE (a)-[:SUPPORTED_BY]->(c)""",
                    cause=cause["id"], cid=reference_id(check["cite"]), id=check["id"],
                    text=check["text"], step=index+1,
                    criterion=json.dumps(check["criterion"], ensure_ascii=False)).consume()
                crit = check["criterion"]
                if crit:
                    tx.run("""MATCH (a:CheckAction {id:$id}),(c:Chunk {id:$cid})
                        MERGE (a)-[:SUPPORTED_BY]->(c)""",
                        id=check["id"], cid=reference_id(crit["cite"])).consume()
            for index, action in enumerate(cause["actions"], 1):
                tx.run("""MATCH (ca:Cause {id:$cause}),(c:Chunk {id:$cid})
                    MERGE (a:Rag:CheckAction {id:$id})
                    SET a.name=$text,a.kind='action',a.when=$when
                    MERGE (ca)-[:HAS_CHECK_ACTION]->(a)
                    MERGE (a)-[:SUPPORTED_BY]->(c)""",
                    cause=cause["id"], cid=reference_id(action["cite"]),
                    id=f"{cause['id']}-A{index}", text=action["text"],
                    when=action["when"]).consume()
    # Remove the retired graph only after all new references have resolved.
    tx.run("""MATCH (d:FestoManual {id:'festo:servopneumatic'})
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p)
        OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c)
        WITH collect(DISTINCT c)+collect(DISTINCT p)+collect(DISTINCT d) AS old
        UNWIND old AS n WITH n WHERE n IS NOT NULL AND NOT n:Rag
        DETACH DELETE n""").consume()
    tx.run("""MATCH (n) WHERE NOT n:Rag AND
        (n.manual_id='festo' OR (n:Situation AND n.id IN $ids))
        DETACH DELETE n""", ids=[f"S{i:02}" for i in range(1,11)]).consume()
    tx.run("""MATCH (k:RagKnowledgeBase {id:$id})
        SET k.ready=true,k.version=$version""", id=KB_ID, version=fingerprint()).consume()


def main():
    parser = argparse.ArgumentParser(description="검토된 E 코드 매핑으로 KG 구축 (LLM 호출 없음)")
    parser.add_argument("--database")
    parser.add_argument("--skip-llm", action="store_true",
                        help="호환 옵션: 필수 그래프는 항상 검토된 매핑으로 구축합니다.")
    parser.add_argument("--validate-only", action="store_true", help="DB 변경 없이 원문 매핑 검사")
    args = parser.parse_args()
    errors, refs = resolve_mapping()
    if not args.validate_only:
        with graph_driver() as driver, driver.session(database=get_database(args.database)) as session:
            ready = session.run("MATCH (k:RagKnowledgeBase {id:$id}) RETURN k.id AS id", id=KB_ID).single()
            if not ready:
                raise ValueError("ingest를 먼저 실행하세요.")
            docs = session.run("""MATCH (d:Rag:Manual)
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Rag:Page)
                RETURN d.doc AS doc,d.sha256 AS sha,count(p) AS pages""").data()
            expected = manuals()
            if ({r["doc"] for r in docs} != set(expected)
                    or any(r["sha"] != expected[r["doc"]]["sha256"]
                           or r["pages"] != expected[r["doc"]]["pages"] for r in docs)):
                raise ValueError("DB 문서·페이지가 로컬 원본과 다릅니다. ingest를 먼저 실행하세요.")
            session.execute_write(write_graph, errors, refs)
    print(f"검증 완료: {len(errors)}개 E 코드 / {len(refs)}개 문서 인용")


if __name__ == "__main__":
    main()
