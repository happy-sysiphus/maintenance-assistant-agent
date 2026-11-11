"""Synchronize four manuals. Source backup precedes the first graph mutation."""

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from neo4j import READ_ACCESS
from ..config import graph_driver, get_database
from ..documents import manuals, fingerprint, verify_manuals, searchable_chunk, DATA
from ..paths import CHUNKS, PARSED_DOCS, RUNTIME

KB_ID = "rag:manuals"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def backup_database(driver, database):
    """Portable snapshot. Rich values are serialized as strings if needed."""
    target = RUNTIME / "backups" / f"aura-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        nodes = session.run("MATCH (n) RETURN elementId(n) AS key,labels(n) AS labels,properties(n) AS properties").data()
        links = session.run("""MATCH (a)-[r]->(b) RETURN elementId(a) AS start,
            elementId(b) AS end,type(r) AS type,properties(r) AS properties""").data()
    target.write_text(json.dumps({"database": database, "nodes": nodes, "relationships": links},
                                ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    saved = json.loads(target.read_text(encoding="utf-8"))
    if len(saved["nodes"]) != len(nodes) or len(saved["relationships"]) != len(links):
        raise RuntimeError("그래프 백업 검증 실패")
    return target


def restore_backup(driver, database, path):
    """Restore only this application's RAG subgraph after a failed migration."""
    saved=json.loads(Path(path).read_text(encoding="utf-8"))
    if saved["database"] != database:
        raise ValueError("백업과 대상 데이터베이스가 다릅니다.")
    nodes=[n for n in saved["nodes"] if "Rag" in n["labels"] or "RagKnowledgeBase" in n["labels"]]
    keys={n["key"]:n["properties"].get("id") for n in nodes}
    if not nodes or not all(keys.values()) or len(set(keys.values()))!=len(keys):
        raise ValueError("RAG 백업의 노드 ID가 없거나 중복됐습니다.")
    groups={}
    for node in nodes:
        labels=tuple(sorted(node["labels"]))
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",label) for label in labels):
            raise ValueError("잘못된 백업 라벨")
        groups.setdefault(labels,[]).append(node["properties"])
    links={}
    for link in saved["relationships"]:
        if (link["start"] in keys) != (link["end"] in keys):
            raise ValueError("외부 그래프와 연결된 백업은 자동 복구하지 않습니다.")
        if link["start"] not in keys:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",link["type"]):
            raise ValueError("잘못된 백업 관계")
        links.setdefault(link["type"],[]).append({"a":keys[link["start"]],"b":keys[link["end"]],"props":link["properties"]})
    def restore(tx):
        tx.run("MATCH (n:Rag) DETACH DELETE n").consume()
        tx.run("MATCH (k:RagKnowledgeBase {id:$id}) DETACH DELETE k",id=KB_ID).consume()
        for labels,rows in groups.items():
            tx.run("UNWIND $rows AS props CREATE (n:"+":".join(labels)+") SET n=props",rows=rows).consume()
        for kind,rows in links.items():
            tx.run("""UNWIND $rows AS row MATCH (a {id:row.a}),(b {id:row.b})
                WHERE (a:Rag OR a:RagKnowledgeBase) AND (b:Rag OR b:RagKnowledgeBase)
                CREATE (a)-[r:"""+kind+"]->(b) SET r=row.props",rows=rows).consume()
    with driver.session(database=database) as session:
        session.execute_write(restore)


def load_inputs():
    verify_manuals()
    pages, chunks = read_jsonl(PARSED_DOCS), read_jsonl(CHUNKS)
    docs = manuals()
    if not pages or not chunks:
        raise ValueError("parse → chunk를 먼저 실행하세요.")
    keys = {(p["metadata"]["manual_id"], p["metadata"]["page_number"]) for p in pages}
    expected = {(doc, page) for doc, info in docs.items() for page in range(1, info["pages"]+1)}
    if keys != expected or len(keys) != len(pages):
        raise ValueError("구축에는 네 문서의 전체 페이지가 필요합니다: parse --all-pages")
    seen = set()
    policy = json.loads((DATA / "processing.json").read_text(encoding="utf-8"))
    for chunk in chunks:
        meta = chunk["metadata"]
        if ((meta["manual_id"], meta["page_number"]) not in keys
                or not chunk["page_content"].strip() or chunk["chunk_id"] in seen):
            raise ValueError("청크 출처·ID 검증 실패")
        if meta.get("max_chars") != policy["max_chars"] or meta.get("parser_version") != policy["parser"]:
            raise ValueError("청크와 processing.json의 파싱·청킹 버전이 다릅니다.")
        if any((meta["manual_id"],p) not in keys for p in meta.get("source_pages",[])):
            raise ValueError("여러 페이지 청크의 출처 페이지가 없습니다.")
        seen.add(chunk["chunk_id"])
    return pages, chunks


def synchronize(tx, pages, chunks):
    tx.run("""MERGE (k:RagKnowledgeBase {id:$id})
        SET k.ready=false,k.graph_ready=false,k.version=$version""", id=KB_ID, version=fingerprint()).consume()
    for doc, info in manuals().items():
        tx.run("""MERGE (d:Rag:Manual {id:$id})
            SET d.doc=$doc,d.title=$title,d.sha256=$sha,d.page_count=$pages,
                d.version=$version,d.role=$role""",
            id=f"rag:manual:{doc}", doc=doc, title=info["title"], sha=info["sha256"],
            pages=info["pages"], version=info["version"], role=info["role"]).consume()
    page_rows = [{
        "id": f"rag:page:{p['metadata']['manual_id']}:{p['metadata']['page_number']}",
        "doc": p["metadata"]["manual_id"], "page": p["metadata"]["page_number"],
        "text": p["raw_text"], "sha": p["metadata"]["document_sha256"],
    } for p in pages]
    tx.run("""UNWIND $rows AS row
        MATCH (d:Manual {id:'rag:manual:'+row.doc})
        MERGE (p:Rag:Page {id:row.id})
        SET p.doc=row.doc,p.page_number=row.page,p.text=row.text,p.sha256=row.sha
        MERGE (d)-[:HAS_PAGE]->(p)""", rows=page_rows).consume()
    rows = []
    for chunk in chunks:
        meta = chunk["metadata"]
        rows.append({"id": "rag:" + chunk["chunk_id"],
                     "page_id": f"rag:page:{meta['manual_id']}:{meta['page_number']}",
                     "page_ids": [f"rag:page:{meta['manual_id']}:{p}" for p in meta.get("source_pages",[meta["page_number"]])],
                     "props": {"doc": meta["manual_id"], "page_number": meta["page_number"],
                               "text": chunk["page_content"], "chunk_index": chunk["chunk_index"],
                               "section":meta["section_title"],"unit_kind":meta["unit_kind"],
                               "source_pages":meta.get("source_pages",[meta["page_number"]]),
                               "review_needed": meta["review_needed"],
                               "review_reasons": meta["review_reasons"],
                               "searchable": searchable_chunk(chunk["page_content"], json.dumps(meta["unit_sources"])),
                               "source_json": json.dumps(meta["unit_sources"], ensure_ascii=False)}})
    tx.run("""UNWIND $rows AS row MATCH (p:Page {id:row.page_id})
        MERGE (c:Rag:Chunk {id:row.id}) SET c += row.props
        MERGE (p)-[:HAS_CHUNK]->(c)""", rows=rows).consume()
    tx.run("""UNWIND $rows AS row MATCH (c:Rag:Chunk {id:row.id})
        UNWIND row.page_ids AS pid MATCH (p:Rag:Page {id:pid}) MERGE (p)-[:HAS_CHUNK]->(c)""",rows=rows).consume()
    tx.run("MATCH (:Rag:Chunk)-[r:NEXT_CHUNK]->(:Rag:Chunk) DELETE r").consume()
    links = [{"a":a["id"],"b":b["id"]} for a,b in zip(rows,rows[1:]) if a["props"]["doc"]==b["props"]["doc"]]
    tx.run("""UNWIND $links AS row MATCH (a:Rag:Chunk {id:row.a}),(b:Rag:Chunk {id:row.b})
        MERGE (a)-[:NEXT_CHUNK]->(b)""",links=links).consume()
    tx.run("""MATCH (c:Rag:Chunk) WHERE coalesce(c.curated,false)=false AND NOT c.id IN $ids
        DETACH DELETE c""", ids=[row["id"] for row in rows]).consume()


def main():
    parser = argparse.ArgumentParser(description="네 문서의 페이지·청크를 Neo4j에 적재")
    parser.add_argument("--database")
    args = parser.parse_args()
    pages, chunks = load_inputs()
    database = get_database(args.database)
    with graph_driver() as driver:
        backup = backup_database(driver, database)
        with driver.session(database=database) as session:
            for label in ("Manual", "Page", "Chunk", "Situation"):
                session.run(f"CREATE CONSTRAINT rag_{label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()
            session.execute_write(synchronize, pages, chunks)
    print(f"백업: {backup}\n적재: {len(pages)}페이지 / {len(chunks)}청크")


if __name__ == "__main__":
    main()
