"""Load selected Festo manual chunks into Neo4j.

Graph: (:FestoManual)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk).
Only this Festo document is synchronized; unrelated documents are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from ..paths import CHUNKS
from ..config import get_database, load_environment
from neo4j import GraphDatabase, Transaction


CHUNKS_PATH = CHUNKS
DOC_ID = "festo:servopneumatic"
DOC_TITLE = "Festo Servopneumatic Drive System for Robot Welding Guns"


def load_chunks(path: Path) -> list[dict]:
    """Validate JSONL input before any database write."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}. Run python -m rag chunk first.")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            metadata = data.get("metadata") or {}
            page_number = metadata.get("page_number")
            chunk_index = data.get("chunk_index")
            chunk_id = data.get("chunk_id")
            content = data.get("page_content")
            if metadata.get("manual_id") != "festo":
                raise ValueError(f"Line {line_no}: only Festo chunks can be loaded.")
            if type(page_number) is not int or page_number < 1:
                raise ValueError(f"Line {line_no}: invalid page_number.")
            page_id = f"festo:page:{page_number}"
            if metadata.get("parent_doc_id") != page_id:
                raise ValueError(f"Line {line_no}: parent_doc_id must be {page_id}.")
            if type(chunk_index) is not int or chunk_index < 0:
                raise ValueError(f"Line {line_no}: invalid chunk_index.")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Line {line_no}: missing chunk_id.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Line {line_no}: empty page_content.")
            if chunk_id in seen_ids or chunk_index in seen_indices:
                raise ValueError(f"Line {line_no}: duplicate chunk_id or chunk_index.")
            seen_ids.add(chunk_id)
            seen_indices.add(chunk_index)
            source = metadata.get("source")
            page = metadata.get("page", page_number - 1)
            if type(page) is not int or page != page_number - 1:
                raise ValueError(f"Line {line_no}: zero-based page disagrees with page_number.")
            if not isinstance(source, str) or not source:
                raise ValueError(f"Line {line_no}: missing source PDF path.")
            review_needed = metadata.get("review_needed")
            review_types = metadata.get("review_types")
            review_reasons = metadata.get("review_reasons")
            review_sources = metadata.get("review_sources")
            unit_sources = metadata.get("unit_sources")
            if type(review_needed) is not bool:
                raise ValueError(f"Line {line_no}: missing boolean review_needed; rerun python -m rag chunk.")
            for field, value in (("review_types", review_types), ("review_reasons", review_reasons)):
                if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
                    raise ValueError(f"Line {line_no}: invalid {field}.")
            if not isinstance(review_sources, list) or any(not isinstance(item, dict) for item in review_sources):
                raise ValueError(f"Line {line_no}: invalid review_sources.")
            if not isinstance(unit_sources, list) or not unit_sources or any(
                not isinstance(item, dict) for item in unit_sources
            ):
                raise ValueError(f"Line {line_no}: invalid unit_sources.")
            if any(type(item.get("review_needed")) is not bool for item in unit_sources):
                raise ValueError(f"Line {line_no}: unit review status is missing.")
            if review_needed != bool(review_sources):
                raise ValueError(f"Line {line_no}: review_needed disagrees with review_sources.")
            if review_needed != any(item["review_needed"] for item in unit_sources):
                raise ValueError(f"Line {line_no}: review_needed disagrees with unit_sources.")
            if metadata.get("source_page") != page_number:
                raise ValueError(f"Line {line_no}: source_page must match page_number.")
            review_block_ids = []
            for item in review_sources:
                block_id = item.get("source_block_id", item.get("block_id"))
                if not isinstance(block_id, str) or not block_id:
                    raise ValueError(f"Line {line_no}: review source missing block ID.")
                if item.get("page_number") != page_number:
                    raise ValueError(f"Line {line_no}: review source page does not match chunk page.")
                review_block_ids.append(block_id)

            rows.append({
                "chunk_id": chunk_id,
                "page_id": page_id,
                "page_number": page_number,
                "page": page,
                "source": source,
                "properties": {
                    "text": content,
                    "chunk_index": chunk_index,
                    "source": source,
                    "page": page,
                    "page_number": page_number,
                    "start_index": metadata.get("start_index"),
                    "end_index": metadata.get("end_index"),
                    "char_count": metadata.get("char_count"),
                    "parent_doc_id": page_id,
                    "manual_id": "festo",
                    "unit_kind": metadata.get("unit_kind"),
                    "section_title": metadata.get("section_title"),
                    "table_title": metadata.get("table_title"),
                    "fault_heading": metadata.get("fault_heading"),
                    "error_numbers": metadata.get("error_numbers") or [],
                    "visual_review_required": metadata.get("visual_review_required", False),
                    # Neo4j accepts scalar properties and primitive lists; retain
                    # the richer provenance as JSON for later source inspection.
                    # Write empty values too so an updated chunk never keeps a
                    # stale review flag from a previous load.
                    "review_needed": review_needed,
                    "review_types": review_types,
                    "review_reasons": review_reasons,
                    "review_block_ids": review_block_ids,
                    "review_pages": sorted({item.get("page_number", page_number) for item in review_sources}),
                    "source_page": page_number,
                    "review_source_json": json.dumps(review_sources, ensure_ascii=False, sort_keys=True),
                    "source_json": json.dumps(review_sources, ensure_ascii=False, sort_keys=True),
                    "unit_source_json": json.dumps(unit_sources, ensure_ascii=False, sort_keys=True),
                },
            })
    if not rows:
        raise ValueError(f"No chunks in {path}; refusing to clear existing Festo data.")
    if seen_indices != set(range(len(rows))):
        raise ValueError("chunk_index values must be contiguous from zero.")
    return sorted(rows, key=lambda row: row["properties"]["chunk_index"])


def create_constraints(session) -> None:
    session.run("""
        CREATE CONSTRAINT festo_manual_id_unique IF NOT EXISTS
        FOR (m:FestoManual) REQUIRE m.id IS UNIQUE
    """).consume()
    session.run("""
        CREATE CONSTRAINT festo_page_id_unique IF NOT EXISTS
        FOR (p:Page) REQUIRE p.id IS UNIQUE
    """).consume()
    session.run("""
        CREATE CONSTRAINT festo_chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.id IS UNIQUE
    """).consume()


def synchronize_document(tx: Transaction, rows: list[dict]) -> None:
    """Atomically upsert the snapshot and rebuild Festo-only chunk ordering."""
    tx.run("""
        MERGE (d:FestoManual {id: $doc_id})
        SET d.title = $title, d.manual_id = 'festo', d.source = $source
    """, doc_id=DOC_ID, title=DOC_TITLE, source=rows[0]["source"]).consume()
    tx.run("""
        MATCH (d:FestoManual {id: $doc_id})
        UNWIND $rows AS row
        MERGE (p:Page {id: row.page_id})
        SET p.page = row.page, p.page_number = row.page_number,
            p.source = row.source, p.manual_id = 'festo', p.parent_doc_id = $doc_id
        MERGE (c:Chunk {id: row.chunk_id})
        SET c += row.properties
        MERGE (d)-[:HAS_PAGE]->(p)
        MERGE (p)-[:HAS_CHUNK]->(c)
    """, doc_id=DOC_ID, rows=rows).consume()
    # Source span and content determine chunk IDs. Remove obsolete chunks only
    # if owned by this manual, preserving other documents in the same database.
    tx.run("""
        MATCH (d:FestoManual {id: $doc_id})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        WHERE NOT c.id IN $active_ids AND c.parent_doc_id STARTS WITH 'festo:page:'
        WITH DISTINCT c
        DETACH DELETE c
    """, doc_id=DOC_ID, active_ids=[row["chunk_id"] for row in rows]).consume()
    tx.run("""
        MATCH (d:FestoManual {id: $doc_id})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)-[r:NEXT_CHUNK]->()
        WITH DISTINCT r
        DELETE r
    """, doc_id=DOC_ID).consume()
    tx.run("""
        MATCH (d:FestoManual {id: $doc_id})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        WITH DISTINCT c ORDER BY c.chunk_index
        WITH collect(c) AS ordered
        WHERE size(ordered) > 1
        UNWIND range(0, size(ordered) - 2) AS i
        WITH ordered[i] AS current_chunk, ordered[i + 1] AS next_chunk
        MERGE (current_chunk)-[:NEXT_CHUNK]->(next_chunk)
    """, doc_id=DOC_ID).consume()
    tx.run("""
        MATCH (d:FestoManual {id: $doc_id})-[r:HAS_PAGE]->(p:Page)
        WHERE NOT p.id IN $active_page_ids
        DELETE r
        WITH p
        WHERE p.manual_id = 'festo' AND NOT EXISTS { MATCH ()-[:HAS_PAGE]->(p) }
        DETACH DELETE p
    """, doc_id=DOC_ID,
        active_page_ids=list({row["page_id"] for row in rows})).consume()


def get_summary(session) -> dict:
    result = session.run("""
        MATCH (d:FestoManual {id: $doc_id})
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH d, count(DISTINCT p) AS pages
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        WITH d, pages, count(DISTINCT c) AS chunks
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(:Chunk)-[r:NEXT_CHUNK]->(:Chunk)
        RETURN pages, chunks, count(DISTINCT r) AS next_chunk_links
    """, doc_id=DOC_ID).single()
    return dict(result) if result else {"pages": 0, "chunks": 0, "next_chunk_links": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest selected Festo chunks into Neo4j")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH,
                        help=f"Input JSONL (default: {CHUNKS_PATH})")
    parser.add_argument("--database", default=get_database(),
                        help="Neo4j target database (default: NEO4J_DATABASE)")
    args = parser.parse_args()
    rows = load_chunks(args.chunks)
    load_environment()
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    database = get_database(args.database)
    if not all((uri, username, password, database)):
        raise RuntimeError("NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD and NEO4J_DATABASE are required.")
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            create_constraints(session)
            session.execute_write(synchronize_document, rows)
            summary = get_summary(session)
    print(f"Neo4j database: {database}")
    print(f"Festo document: {DOC_ID}")
    print(f"Pages: {summary['pages']}, chunks: {summary['chunks']}, "
          f"NEXT_CHUNK links: {summary['next_chunk_links']}")


if __name__ == "__main__":
    main()
