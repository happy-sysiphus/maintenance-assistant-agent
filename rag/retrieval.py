"""Code evidence first, then fulltext/vector retrieval fused by reciprocal rank."""

import json
import os
import re
from neo4j import READ_ACCESS
from .config import graph_driver, get_database, openai_client
from .documents import fingerprint, mapping, normalized
from .pipeline.ingest import KB_ID
from .pipeline.index import VECTOR_INDEX, TEXT_INDEX

TERMS = {
    "보정": ["compensation", "calibration"], "압력": ["pressure"],
    "캡": ["cap", "caps"], "전극": ["electrode"], "마모": ["wear", "abrasion"],
    "드리프트": ["drift"], "누설": ["leakage"], "위치": ["position"],
    "밸브": ["valve"], "케이블": ["cable"], "마찰": ["friction"],
    "교정": ["calibration"], "호밍": ["homing"], "윤활": ["lubrication"],
    "힘": ["force"], "냉각": ["cooling"], "밀링": ["milling"],
    "드레싱": ["milling"], "영점": ["zero"], "공급": ["supply"],
}


def query_text(question, history):
    followup = re.search(r"그럼|그다음|그 다음|다음엔|그건|그거|이 경우|더 자세히", question)
    prior = next((h["text"] for h in reversed(history) if h["role"] == "user"), "")
    return (prior + "\n" + question) if followup and prior else question


def fuse_rankings(branches, limit=8):
    scores = {}
    for ids in branches.values():
        for rank, key in enumerate(dict.fromkeys(ids), 1):
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))[:limit]


def search_context(question, error_code, history):
    query = query_text(question, history)
    context = mapping().get(error_code)
    if context:
        query += "\n" + error_code + " " + context["name"] + " " + " ".join(
            ref["quote"] for ref in context["definitions"])
    terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", query)
    for ko, aliases in TERMS.items():
        if ko in question:
            terms.extend(aliases)
    stopwords = {"the", "is", "are", "for", "and", "or", "not", "when", "with",
                 "from", "this", "than", "within", "reached"}
    terms = list(dict.fromkeys(t.lower() for t in terms if t.lower() not in stopwords))[:24]
    return query, terms


def public_evidence(row):
    text = row["text"]
    # A source chunk may group several diagnostic rows. A curated quote must
    # carry its own row, not neighbouring errors/remedies from the same page.
    quote = row.get("quote")
    if quote:
        matching = [line for line in text.splitlines()
                    if normalized(quote) in normalized(line)
                    and re.match(r"(?:Error number:|Column 1:)",line)]
        if matching:
            text = min(matching,key=len)
    return {"id": row["id"], "doc": row["doc"], "page": row["page"],
            "quote": row.get("quote") or text[:300], "text": text[:1800],
            "review_needed": bool(row.get("review_needed"))}


def retrieve(question, error_code=None, history=None, *, use_vectors=True, database=None):
    history = history or []
    query, terms = search_context(question,error_code,history)
    search = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
    vector = None
    embedding_tokens = 0
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # External calls are optional: no-LLM mode uses exact graph/fulltext only.
    if use_vectors:
        with openai_client() as client:
            response = client.embeddings.create(model=model, input=query + "\n" + " ".join(terms))
            vector = response.data[0].embedding
            embedding_tokens = response.usage.total_tokens
    by_id, branches, required, situations = {}, {}, [], []
    with graph_driver() as driver, driver.session(
            database=get_database(database), default_access_mode=READ_ACCESS) as session:
        kb = session.run("""MATCH (k:RagKnowledgeBase {id:$id})
            RETURN k.ready AS ready,k.version AS version,k.embedding_model AS model""", id=KB_ID).single()
        if not kb or not kb["ready"] or kb["version"] != fingerprint():
            raise RuntimeError("KG와 로컬 문서·매핑 버전이 다릅니다. parse → chunk → ingest → build를 확인하세요.")
        if error_code:
            situations = session.run("""MATCH (s:Rag:Situation {id:$id})
                RETURN s.mapping_json AS mapping""", id=error_code).data()
            if not situations:
                return {"evidence": [], "situations": [], "embedding_tokens": embedding_tokens}
            rows = session.run("""MATCH (s:Situation {id:$id})-[:RELEVANT_EVIDENCE]->(c:Chunk)
                RETURN c.id AS id,c.doc AS doc,c.page_number AS page,c.text AS text,
                    c.quote AS quote,c.review_needed AS review_needed ORDER BY c.doc,c.page_number,c.id""",
                id=error_code).data()
            for row in rows:
                by_id[row["id"]] = row
                required.append(row["id"])
        if search:
            rows = session.run(f"""CALL db.index.fulltext.queryNodes('{TEXT_INDEX}', $search, {{limit:64}})
                YIELD node,score WHERE node:Rag AND node.searchable=true
                RETURN node.id AS id,node.doc AS doc,node.page_number AS page,
                    node.text AS text,node.quote AS quote,node.review_needed AS review_needed,score""",
                search=search).data()
            branches["keyword"] = [r["id"] for r in rows]
            by_id.update({r["id"]: r for r in rows})
        if vector is not None:
            if kb["model"] != model:
                raise RuntimeError("질문과 문서 임베딩 모델이 다릅니다. index를 확인하세요.")
            rows = session.run(f"""CALL db.index.vector.queryNodes('{VECTOR_INDEX}',64,$vector)
                YIELD node,score WHERE node:Rag AND node.searchable=true AND score>=0.30
                RETURN node.id AS id,node.doc AS doc,node.page_number AS page,
                    node.text AS text,node.quote AS quote,node.review_needed AS review_needed,score""",
                vector=vector).data()
            branches["vector"] = [r["id"] for r in rows]
            by_id.update({r["id"]: r for r in rows})
        selected = list(dict.fromkeys(required + fuse_rankings(branches)))
        # Keep exact code evidence complete; apply the budget only to supplements.
        evidence, chars = [], 0
        seen = set()
        for key in selected:
            row = public_evidence(by_id[key])
            unique = (row["doc"], row["page"], row["text"])
            if unique in seen:
                continue
            if key not in required and chars + len(row["text"]) > 18000:
                continue
            seen.add(unique)
            evidence.append(row)
            chars += len(row["text"])
    return {"evidence": evidence, "situations": [json.loads(r["mapping"]) for r in situations],
            "embedding_tokens": embedding_tokens, "routes": list(branches)}
