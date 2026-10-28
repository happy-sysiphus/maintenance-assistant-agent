"""Build reusable vector/fulltext indexes. Embed only new/changed chunks."""

import argparse
import os
from ..config import graph_driver, openai_client, get_database, load_environment
from ..documents import searchable_chunk
from .ingest import KB_ID

VECTOR_INDEX = "rag_chunk_vector"
TEXT_INDEX = "rag_chunk_text"


def main():
    parser = argparse.ArgumentParser(description="문서 청크의 임베딩·벡터·키워드 인덱스 구축")
    parser.add_argument("--database")
    parser.add_argument("--metadata-only", action="store_true", help="검색 가능 여부만 갱신 (OpenAI 호출 없음)")
    args = parser.parse_args()
    load_environment()
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    calls, tokens = 0, 0
    with graph_driver() as driver, driver.session(database=get_database(args.database)) as session:
        kb = session.run("MATCH (k:RagKnowledgeBase {id:$id}) RETURN k.ready AS ready", id=KB_ID).single()
        if not kb or not kb["ready"]:
            raise ValueError("build를 먼저 실행하세요.")
        source_rows = session.run("""MATCH (c:Rag:Chunk)
            RETURN c.id AS id,c.text AS text,c.source_json AS source_json,c.curated AS curated""").data()
        flags = [{"id":r["id"],"searchable":bool(r["curated"]) or searchable_chunk(r["text"],r["source_json"])}
                 for r in source_rows]
        session.run("""UNWIND $rows AS row MATCH (c:Chunk {id:row.id})
            SET c.searchable=row.searchable""", rows=flags).consume()
        if args.metadata_only:
            print(f"검색 가능 청크: {sum(r['searchable'] for r in flags)}/{len(flags)}; OpenAI 호출 0회")
            return
        pending = session.run("""MATCH (c:Rag:Chunk)
            WHERE c.searchable=true AND (c.embedding IS NULL OR coalesce(c.embedding_model,'')<>$model)
            RETURN c.id AS id,c.text AS text ORDER BY c.id""", model=model).data()
        if pending:
            with openai_client() as client:
                for offset in range(0,len(pending),64):
                    batch = pending[offset:offset+64]
                    response = client.embeddings.create(model=model, input=[r["text"] for r in batch])
                    rows = [{"id": batch[item.index]["id"], "vector": item.embedding} for item in response.data]
                    session.run("""UNWIND $rows AS row MATCH (c:Chunk {id:row.id})
                        SET c.embedding=row.vector,c.embedding_model=$model""", rows=rows, model=model).consume()
                    calls += 1
                    tokens += response.usage.total_tokens
                    print(f"임베딩 {min(offset+64,len(pending))}/{len(pending)}", flush=True)
        dims = session.run("MATCH (c:Rag:Chunk) WHERE c.searchable=true RETURN DISTINCT size(c.embedding) AS n").data()
        if len(dims) != 1 or not dims[0]["n"]:
            raise ValueError("임베딩이 없거나 차원이 일치하지 않습니다.")
        dimension = dims[0]["n"]
        old = session.run("SHOW INDEXES YIELD name,options WHERE name=$name RETURN options", name=VECTOR_INDEX).single()
        if old and old["options"]["indexConfig"].get("vector.dimensions") != dimension:
            session.run(f"DROP INDEX {VECTOR_INDEX}").consume()
        session.run(f"""CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON c.embedding OPTIONS {{indexConfig: {{
            \u0060vector.dimensions\u0060: $dims,
            \u0060vector.similarity_function\u0060: 'cosine'}}}}""", dims=dimension).consume()
        session.run(f"CREATE FULLTEXT INDEX {TEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]").consume()
        session.run("CALL db.awaitIndexes(60)").consume()
        session.run("""MATCH (k:RagKnowledgeBase {id:$id})
            SET k.embedding_model=$model,k.embedding_dimensions=$dims""",
            id=KB_ID, model=model, dims=dimension).consume()
    print(f"인덱스 완료: {dimension}차원 / API {calls}회 / 입력 {tokens}토큰")


if __name__ == "__main__":
    main()
