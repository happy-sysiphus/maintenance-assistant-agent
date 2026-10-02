"""Build an evidence-linked Festo KG for ML situations S01–S10.

Run python -m rag ingest first. The ML-facing key is S01–S10 only.
Manual error numbers are candidate diagnostics, never observed ML error codes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from ..paths import CHUNKS, KG_CACHE, MAPPING_PATH, RSW_PDF as RSW_SOURCE
from ..config import get_database, load_environment
from typing import Any

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema, NodeType, Pattern, RelationshipType, SchemaFromTextExtractor,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler


CACHE = KG_CACHE
RSW_PDF = str(RSW_SOURCE)
DOC_ID = "festo:servopneumatic"
CACHE_VERSION = "s01-s10-kg-v1"
ERROR_RE = re.compile(r"^Error number:\s*(\d+)\s*\|")
PAIR_RE = re.compile(r"^Cause:\s*(.*?)\s*\|\s*Remedy:\s*(.+)$")

# Curated from RSW PDF pp. 5–8. Festo error links are held separately as candidates.
SITUATIONS = (
    ("S01", "보정 압력 도달 지연", "설정한 보정 압력에 1800 ms 안에 도달하지 못함.", 5),
    ("S02", "전극 파손", "전극 위치가 기준 이동의 영점보다 작음(캡이 빠지거나 깨져 더 많이 들어감).", 6),
    ("S03", "원치 않는 이동", "실제 위치가 실린더 스트로크의 6.5% 넘게 목표에서 벗어남.", 6),
    ("S04", "드리프트", "잠긴 실린더가 멈추지 않고 분당 5 mm 넘게 움직임.", 6),
    ("S05", "공기 공급 부족·누설로 전극 힘 저하", "공기 공급 부족 또는 누설로 전극 힘이 낮거나 힘 도달이 늦어지는 원인 상황.", 7),
    ("S06", "기계 마찰·걸림·윤활 부족", "기계 마찰, 걸림 또는 윤활 부족이 의심되는 원인 상황.", 7),
    ("S07", "전극 캡 마모·드레싱 불량", "전극 캡 마모 또는 드레싱 불량이 의심되는 원인 상황.", 7),
    ("S08", "축 보정 틀어짐·간섭에 의한 위치 이탈", "축 보정 틀어짐 또는 간섭으로 목표 위치에서 벗어나는 원인 상황.", 8),
    ("S09", "냉각수·변압기·용접 제어기 등 주변 설비 이상", "건 자체는 정상으로 움직이나 주변 설비 이상이 의심되는 원인 상황.", 8),
    ("S10", "설정값 변경·잔고장 반복", "설정값 변경이나 반복되는 잔고장을 구분해야 하는 운영 상황.", 8),
)
def digest(text: str, size: int = 18) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:size]


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def component_id(name: str) -> str:
    if norm(name) == "electrode caps":
        return "festo:component:electrode_caps"  # Stable identity for electrode-cap nodes.
    return "festo:component:rule:" + digest(norm(name))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python -m rag parse and python -m rag chunk first.")
    rows, ids = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        meta = raw.get("metadata") or {}
        page, uid = meta.get("page_number"), raw.get("chunk_id")
        if meta.get("manual_id") != "festo" or type(page) is not int or meta.get("parent_doc_id") != f"festo:page:{page}":
            raise ValueError(f"{path}:{number}: invalid Festo page metadata")
        if not isinstance(uid, str) or not uid or uid in ids or not raw.get("page_content"):
            raise ValueError(f"{path}:{number}: invalid or repeated chunk")
        review_needed = meta.get("review_needed")
        unit_sources = meta.get("unit_sources")
        review_sources = meta.get("review_sources")
        if type(review_needed) is not bool or not isinstance(unit_sources, list) or not unit_sources:
            raise ValueError(f"{path}:{number}: missing review-aware unit provenance; rerun python -m rag chunk")
        if not isinstance(review_sources, list) or review_needed != bool(review_sources):
            raise ValueError(f"{path}:{number}: inconsistent visual review provenance")
        if meta.get("source_page") != page:
            raise ValueError(f"{path}:{number}: source_page must equal PDF page_number")
        for unit in unit_sources:
            if (not isinstance(unit, dict) or not isinstance(unit.get("text"), str)
                or not isinstance(unit.get("kind"), str)
                or type(unit.get("review_needed")) is not bool):
                raise ValueError(f"{path}:{number}: invalid unit provenance")
        if review_needed != any(unit["review_needed"] for unit in unit_sources):
            raise ValueError(f"{path}:{number}: unit and chunk review status differ")
        ids.add(uid)
        rows.append({"id": uid, "index": raw["chunk_index"], "page": page,
                     "text": raw["page_content"], "kind": meta.get("unit_kind"),
                     "heading": meta.get("fault_heading") or "",
                     "table_title": meta.get("table_title") or "",
                     "section_title": meta.get("section_title") or "",
                     "review_needed": review_needed,
                     "review_sources": review_sources,
                     "unit_sources": unit_sources})
    if not rows:
        raise ValueError("No selected Festo chunks.")
    return rows


def load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("situations")
    expected = {row[0] for row in SITUATIONS}
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise ValueError(f"Mapping must contain exactly {sorted(expected)}")
    for code, item in mapping.items():
        if not isinstance(item.get("check_source_page"), int) or not item.get("check_sequence"):
            raise ValueError(f"{code}: RSW check sequence and its PDF page are required")
        if not item.get("diagnostics") or not item.get("evidence") or not item.get("components"):
            raise ValueError(f"{code}: diagnostics, evidence and components are required")
        numbers = [row["number"] for row in item["diagnostics"]]
        if len(numbers) != len(set(numbers)) or any(
            row.get("priority") not in {"primary", "related"} or not row.get("rationale")
            for row in item["diagnostics"]
        ):
            raise ValueError(f"{code}: duplicate or invalid diagnostic candidate")
    return mapping


def resolve_mapping(chunks: list[dict[str, Any]], errors: list[dict[str, Any]],
                    mapping: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Resolve every reviewed page/phrase selector before any paid call or DB write."""
    by_error = {row["number"]: row for row in errors}
    rows: dict[str, list[dict[str, Any]]] = {
        "diagnostics": [], "evidence": [], "components": [], "pages": []
    }
    for code, item in mapping.items():
        page_reasons: dict[int, list[str]] = {}
        for candidate in item["diagnostics"]:
            number = candidate["number"]
            if number not in by_error:
                raise ValueError(f"{code}: Festo error {number} is absent from extracted text rows")
            rows["diagnostics"].append({
                "situation_id": code, "error_id": by_error[number]["id"],
                "priority": candidate["priority"], "rationale": candidate["rationale"],
                "manual_page_number": by_error[number]["page_number"],
            })
        for selector in item["evidence"]:
            matches = [chunk for chunk in chunks if chunk["page"] == selector["page"]
                       and norm(selector["contains"]) in norm(chunk["text"])]
            if len(matches) != 1:
                raise ValueError(f"{code}: evidence selector must match one chunk: {selector}; matches={len(matches)}")
            chunk = matches[0]
            rows["evidence"].append({
                "situation_id": code, "chunk_id": chunk["id"], "page_number": chunk["page"],
                "rationale": selector["rationale"],
                "priority": "review_pending_visual_context" if chunk["review_needed"] else "reviewed_reference",
                "assessment": "review_pending_visual_context" if chunk["review_needed"] else "reviewed_reference",
                "review_needed": chunk["review_needed"],
            })
            page_reasons.setdefault(chunk["page"], []).append(selector["rationale"])
        for page, reasons in page_reasons.items():
            page_review = any(row["review_needed"] for row in rows["evidence"]
                              if row["situation_id"] == code and row["page_number"] == page)
            rows["pages"].append({"situation_id": code, "page_id": f"festo:page:{page}",
                                  "page_number": page, "rationale": "; ".join(reasons),
                                  "review_needed": page_review,
                                  "evidence_status": ("review_pending_visual_context" if page_review
                                                      else "safe_text_extracted")})
        for selector in item["components"]:
            matches = [chunk for chunk in chunks if chunk["page"] == selector["page"]
                       and norm(selector["contains"]) in norm(chunk["text"])
                       and norm(selector["name"]) in norm(chunk["text"])]
            if not matches:
                raise ValueError(f"{code}: component name is not source-grounded: {selector}")
            rows["components"].append({
                "situation_id": code, "id": component_id(selector["name"]),
                "name": selector["name"], "chunk_id": matches[0]["id"],
                "page_number": matches[0]["page"],
                "review_needed": matches[0]["review_needed"],
            })
    return rows


def parse_tables(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors, pairs = {}, []
    for chunk in chunks:
        for unit in chunk["unit_sources"]:
            line = unit["text"].strip()
            kind = unit["kind"]
            provenance = {
                "source_block_id": unit.get("source_block_id") or "",
                "source_start": unit.get("start"), "source_end": unit.get("end"),
                "review_needed": unit["review_needed"],
                "review_type": unit.get("review_type") or "",
                "review_reason": unit.get("review_reason") or "",
            }
            if kind == "error_row" and ERROR_RE.match(line):
                fields = {}
                for part in line.split(" | "):
                    if ": " not in part:
                        raise ValueError(f"Malformed manual error row: {line}")
                    key, value = part.split(": ", 1)
                    fields[key] = value.strip()
                number = int(fields["Error number"])
                if number in errors or not fields.get("Error text") or not fields.get("Error elimination"):
                    raise ValueError(f"Duplicate or incomplete manual error {number}")
                errors[number] = {"id": f"festo:error:{number}", "number": number,
                                  "name": fields["Error text"], "description": fields.get("Description", ""),
                                  "error_elimination": fields["Error elimination"],
                                  "ready_signal": fields.get("Ready signal", ""), "row_text": line,
                                  "page_number": chunk["page"], "chunk_id": chunk["id"],
                                  **provenance}
            elif kind == "cause_remedy_row":
                match = PAIR_RE.match(line)
                if match:
                    if not chunk["heading"]:
                        raise ValueError(f"Missing symptom heading in {chunk['id']}")
                    cause, remedy = match.groups()
                    if not cause.strip() or not remedy.strip():
                        raise ValueError(f"Incomplete Cause/Remedy row: {line}")
                    key = f"{chunk['heading']}\0{cause}\0{remedy}"
                    pairs.append({"id": f"festo:pair:{digest(key)}", "symptom": chunk["heading"],
                                  "cause": cause.strip(), "remedy": remedy.strip(), "row_text": line,
                                  "page_number": chunk["page"], "chunk_id": chunk["id"],
                                  **provenance})
    if not pairs:
        raise ValueError("No Cause/Remedy pairs; inspect PDF page 94.")
    return [errors[key] for key in sorted(errors)], pairs


def parse_manual_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Index text extracted from ordinary and mixed tables without interpreting images."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        for unit in chunk["unit_sources"]:
            if unit["kind"] != "table_row" or not unit["text"].strip():
                continue
            source_block_id = unit.get("source_block_id") or ""
            row_text = unit["text"].strip()
            identity = f"{chunk['page']}\0{source_block_id}\0{unit.get('start')}\0{row_text}"
            row_id = f"festo:table-row:{digest(identity)}"
            if row_id in seen:
                raise ValueError(f"Duplicate table row provenance: {row_id}")
            seen.add(row_id)
            rows.append({
                "id": row_id, "row_text": row_text, "page_number": chunk["page"],
                "chunk_id": chunk["id"], "table_title": chunk["table_title"],
                "section_title": chunk["section_title"],
                "source_block_id": source_block_id,
                "source_start": unit.get("start"), "source_end": unit.get("end"),
                "review_needed": unit["review_needed"],
                "review_type": unit.get("review_type") or "",
                "review_reason": unit.get("review_reason") or "",
                "source_method": "extracted_table_text",
            })
    return rows


def select_extra_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One safe excerpt each for maintenance, calibration, caps, valves and faults."""
    targets = [(19, "lubrication intervals", "prose"),
               (62, "axis calibration is used", "prose"),
               (81, "set caps", "prose"),
               (84, "functional test of the blocking valves", "prose"),
               (90, "error number: 12", "error_row")]
    chosen = []
    for page, phrase, kind in targets:
        options = [c for c in chunks if c["page"] == page and c["kind"] == kind
                   and not c["review_needed"]
                   and phrase in c["text"].casefold()
                   and "Column 1: Warning" not in c["text"]]
        if not options:
            raise ValueError(f"Safe LLM chunk missing: PDF page {page}, {phrase}")
        chosen.append(options[0])
    if len({c["id"] for c in chosen}) != 5:
        raise ValueError("Expected five distinct LLM chunks.")
    return chosen


class MeteredLLM(OpenAILLM):
    """No SDK retries; enforce 1 schema + 5 extraction calls, including failures."""

    def __init__(self, model: str) -> None:
        super().__init__(model_name=model, model_params={"max_completion_tokens": 4000},
                         rate_limit_handler=NoOpRateLimitHandler(), max_retries=0)
        self.phase = "schema"
        self.calls = {"schema": 0, "extraction": 0}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        limit = 1 if self.phase == "schema" else 5
        if self.calls[self.phase] >= limit:
            raise RuntimeError(f"Paid {self.phase} call cap reached.")
        self.calls[self.phase] += 1
        result = await super().ainvoke(input, *args, **kwargs)
        if result.usage:
            self.usage["input_tokens"] += result.usage.request_tokens or 0
            self.usage["output_tokens"] += result.usage.response_tokens or 0
            self.usage["total_tokens"] += result.usage.total_tokens or 0
        return result


def key(*parts: Any) -> str:
    return digest(json.dumps([CACHE_VERSION, *parts], ensure_ascii=False, sort_keys=True), 24)


def controlled_schema(inferred: GraphSchema | None) -> GraphSchema:
    descriptions = {n.label: n.description for n in inferred.node_types} if inferred else {}
    return GraphSchema(
        node_types=(NodeType(label="Component", description=(descriptions.get("Component") or "") +
                             " Exact physical part name copied from the source text."),
                    NodeType(label="CheckAction", description=(descriptions.get("CheckAction") or "") +
                             " Exact manual check or calibration action copied from the source text.")),
        relationship_types=(RelationshipType(label="CHECKS_COMPONENT",
                                             description="This action explicitly checks this component."),),
        patterns=(Pattern(source="CheckAction", relationship="CHECKS_COMPONENT", target="Component"),),
        additional_node_types=False, additional_relationship_types=False, additional_patterns=False,
    )


def validate_extra(graph: Any, chunk: dict[str, Any]) -> dict[str, Any]:
    """Write only supported labels and exact spans; discard invented entities."""
    allowed, source = {}, norm(chunk["text"])
    for node in graph.nodes:
        name = node.properties.get("name")
        if node.label not in ("Component", "CheckAction") or not isinstance(name, str):
            continue
        if not name.strip() or norm(name) not in source:
            continue
        node_id = str(node.id)
        if node_id in allowed:
            continue
        prefix = "festo:component:" if node.label == "Component" else "festo:llm-action:"
        identity = norm(name) if node.label == "Component" else chunk["id"] + "\0" + norm(name)
        allowed[node_id] = {"id": prefix + digest(identity), "label": node.label, "name": name.strip(),
                            "chunk_id": chunk["id"], "page_number": chunk["page"]}
    edges = []
    for rel in graph.relationships:
        start, end = allowed.get(rel.start_node_id), allowed.get(rel.end_node_id)
        if rel.type == "CHECKS_COMPONENT" and start and end:
            if start["label"] == "CheckAction" and end["label"] == "Component":
                edges.append({"start": start["id"], "end": end["id"],
                              "chunk_id": chunk["id"], "page_number": chunk["page"]})
    return {"chunk_id": chunk["id"], "page_number": chunk["page"],
            "nodes": list(allowed.values()), "relationships": edges}


def review_extra(record: dict[str, Any]) -> dict[str, Any]:
    """Keep named equipment and meaningful source-grounded actions."""
    equipment = {
        "caps", "electrode caps", "gun", "robot gun", "c-guns", "x-guns",
        "main cylinder", "main cylinders", "main cylinder 1a1", "mpyd",
        "compensating cylinder", "piston rod", "blocking valves 1v2, 1v3, 1v4",
        "transformer", "controller",
    }
    nodes = {}
    for node in record["nodes"]:
        action_words = node["name"].strip().split()
        if (node["label"] == "CheckAction" and len(action_words) >= 2) or (
            node["label"] == "Component" and norm(node["name"]) in equipment
        ):
            nodes[node["id"]] = node
    links = []
    seen = set()
    for rel in record["relationships"]:
        action, component = nodes.get(rel["start"]), nodes.get(rel["end"])
        if not action or not component or component["label"] != "Component":
            continue
        action_name = norm(action["name"])
        component_name = norm(component["name"])
        cap_operation = "cap" in component_name and any(
            phrase in action_name for phrase in (
                "set caps", "geometry check", "measure before milling", "measure after milling")
        )
        explicit_check = ("check" in action_name or "measure" in action_name) and (
            component_name in action_name
        )
        if not (cap_operation or explicit_check):
            continue
        identity = (rel["start"], rel["end"])
        if identity not in seen:
            links.append(rel)
            seen.add(identity)
    return {**record, "nodes": list(nodes.values()), "relationships": links}


async def optional_llm(chunks: list[dict[str, Any]], model: str,
                       summary: dict[str, Any]) -> list[dict[str, Any]]:
    selected = select_extra_chunks(chunks)
    llm = MeteredLLM(model)
    schema_input = "\n\n".join(row["text"][:900] for row in selected)
    cache_path = CACHE / f"스키마_{key(model, schema_input)}.json"
    inferred = None
    if cache_path.is_file():
        inferred = GraphSchema.model_validate(json.loads(cache_path.read_text(encoding="utf-8")))
        summary["cache_hits"] += 1
        summary["schema_status"] = "cached"
    else:
        try:
            inferred = await SchemaFromTextExtractor(llm=llm, use_structured_output=True).run(
                schema_input, examples="Infer a compact graph schema from only the named manual concepts.")
            save_json(cache_path, inferred.model_dump(mode="json"))
            summary["schema_status"] = "inferred"
        except Exception as exc:
            summary["schema_status"] = "failed"
            cause = exc.__cause__
            summary["warnings"].append(
                f"Automatic schema inference failed: {type(exc).__name__}: {exc}"
                + (f"; caused by {type(cause).__name__}: {cause}" if cause else "")
            )
    controlled = controlled_schema(inferred)
    summary["schema_inferred_node_labels"] = [n.label for n in inferred.node_types] if inferred else []
    result = []
    if inferred:
        llm.phase = "extraction"
        extractor = LLMEntityRelationExtractor(
            llm=llm, create_lexical_graph=False, max_concurrency=1, use_structured_output=True)
        for row in selected:
            path = CACHE / f"청크_{key(model, controlled.model_dump(mode='json'), row['id'], row['text'])}.json"
            if path.is_file():
                result.append(review_extra(json.loads(path.read_text(encoding="utf-8"))))
                summary["cache_hits"] += 1
                continue
            try:
                graph = await extractor.run(
                    TextChunks(chunks=[TextChunk(text=row["text"], index=row["index"], uid=row["id"])]),
                    schema=controlled,
                    examples="Copy exact component and action phrases from the text; do not infer fault occurrence.")
                valid = validate_extra(graph, row)
                save_json(path, valid)
                result.append(review_extra(valid))
            except Exception as exc:
                summary["warnings"].append(
                    f"LLM extraction stopped on PDF page {row['page']}: {type(exc).__name__}: {exc}")
                break  # No repeated attempts for a broken setup.
    summary["schema_calls"] = llm.calls["schema"]
    summary["extraction_calls"] = llm.calls["extraction"]
    summary["usage"] = llm.usage
    return result


def preflight(session: Any, chunks: list[dict[str, Any]]) -> None:
    records = list(session.run("""
        MATCH (d:FestoManual {id:$doc})-[:HAS_PAGE]->(:Page)-[:HAS_CHUNK]->(c:Chunk)
        RETURN DISTINCT c.id AS id, c.review_needed AS review_needed
    """, doc=DOC_ID))
    loaded = {record["id"]: record["review_needed"] for record in records}
    expected = {chunk["id"]: chunk["review_needed"] for chunk in chunks}
    if loaded != expected:
        raise RuntimeError("Neo4j chunks or their review flags differ from chunks.jsonl; "
                           "run python -m rag ingest first.")


def constraints(session: Any) -> None:
    for label in ("Situation", "ManualError", "FaultSymptom", "Cause", "CheckAction",
                  "Component", "ManualTableRow"):
        session.run(f"CREATE CONSTRAINT festo_{label.lower()}_id_unique IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()


def write_graph(tx: Any, errors: list[dict[str, Any]], pairs: list[dict[str, Any]],
                table_rows: list[dict[str, Any]],
                extras: list[dict[str, Any]], mapping: dict[str, dict[str, Any]],
                linked: dict[str, list[dict[str, Any]]]) -> None:
    # Rebuild all Festo-derived facts from the current chunk snapshot. Chunk IDs
    # may change when review warnings are added, so old source links cannot be
    # carried over safely even in deterministic-only mode.
    tx.run("""
        MATCH (s:Situation)-[r:RELEVANT_DIAGNOSTIC|RELEVANT_SECTION|RELEVANT_EVIDENCE|CHECKS_COMPONENT]->()
        WHERE s.id IN $ids DELETE r
    """, ids=[row[0] for row in SITUATIONS]).consume()
    tx.run("""
        MATCH (n) WHERE n.manual_id='festo'
          AND (n:ManualError OR n:FaultSymptom OR n:Cause OR n:CheckAction
               OR n:Component OR n:ManualTableRow)
        DETACH DELETE n
    """).consume()
    situations = [{"id": code, "code": code, "name": name, "definition": definition,
                   "source_page": page, "source_document": RSW_PDF,
                   "check_sequence": mapping[code]["check_sequence"],
                   "check_sequence_source_page": mapping[code]["check_source_page"],
                   "source_pages": sorted({page, mapping[code]["check_source_page"]}),
                   "coverage_note": mapping[code].get("coverage_note", "")}
                  for code, name, definition, page in SITUATIONS]
    tx.run("""
        UNWIND $rows AS row MERGE (s:Situation {id:row.id})
        SET s.code=row.code, s.name=row.name, s.definition=row.definition,
            s.source_page=row.source_page, s.source_document=row.source_document,
            s.check_sequence=row.check_sequence,
            s.check_sequence_source_page=row.check_sequence_source_page,
            s.source_pages=row.source_pages, s.coverage_note=row.coverage_note,
            s.source_method='rsw_definition'
    """, rows=situations).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (c:Chunk {id:row.chunk_id})
        MERGE (e:ManualError {id:row.id})
        SET e.number=row.number, e.name=row.name, e.description=row.description,
            e.error_elimination=row.error_elimination, e.ready_signal=row.ready_signal,
            e.row_text=row.row_text, e.page_number=row.page_number,
            e.chunk_id=row.chunk_id, e.manual_id='festo', e.source_method='manual_error_table',
            e.source_block_id=row.source_block_id,
            e.source_start=row.source_start, e.source_end=row.source_end,
            e.review_needed=row.review_needed, e.review_type=row.review_type,
            e.review_reason=row.review_reason
        MERGE (e)-[es:SUPPORTED_BY]->(c)
        SET es.review_needed=row.review_needed, es.source_block_id=row.source_block_id
        MERGE (a:CheckAction {id:'festo:error-action:'+toString(row.number)})
        SET a.name=row.error_elimination, a.page_number=row.page_number,
            a.chunk_id=row.chunk_id, a.manual_id='festo', a.source_method='manual_error_table',
            a.source_block_id=row.source_block_id,
            a.review_needed=row.review_needed, a.review_type=row.review_type,
            a.review_reason=row.review_reason
        MERGE (e)-[ea:HAS_CHECK_ACTION]->(a)
        SET ea.review_needed=row.review_needed
        MERGE (a)-[action_support:SUPPORTED_BY]->(c)
        SET action_support.review_needed=row.review_needed,
            action_support.source_block_id=row.source_block_id
    """, rows=errors).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (c:Chunk {id:row.chunk_id})
        MERGE (f:FaultSymptom {id:'festo:symptom:'+toLower(replace(row.symptom,' ','_'))})
        SET f.name=row.symptom, f.manual_id='festo', f.page_number=row.page_number,
            f.review_needed=coalesce(f.review_needed,false) OR row.review_needed
        MERGE (ca:Cause {id:row.id+':cause'})
        SET ca.name=row.cause, ca.row_text=row.row_text, ca.page_number=row.page_number,
            ca.chunk_id=row.chunk_id, ca.manual_id='festo',
            ca.source_block_id=row.source_block_id,
            ca.review_needed=row.review_needed, ca.review_type=row.review_type,
            ca.review_reason=row.review_reason
        MERGE (a:CheckAction {id:row.id+':action'})
        SET a.name=row.remedy, a.row_text=row.row_text, a.page_number=row.page_number,
            a.chunk_id=row.chunk_id, a.manual_id='festo', a.source_method='cause_remedy_table',
            a.source_block_id=row.source_block_id,
            a.review_needed=row.review_needed, a.review_type=row.review_type,
            a.review_reason=row.review_reason
        MERGE (f)-[fc:HAS_CAUSE]->(ca)
        SET fc.review_needed=row.review_needed
        MERGE (ca)-[car:CHECKED_BY]->(a)
        SET car.review_needed=row.review_needed
        MERGE (f)-[fs:SUPPORTED_BY]->(c)
        SET fs.review_needed=coalesce(fs.review_needed,false) OR row.review_needed
        MERGE (ca)-[cs:SUPPORTED_BY]->(c)
        SET cs.review_needed=row.review_needed, cs.source_block_id=row.source_block_id
        MERGE (a)-[action_support:SUPPORTED_BY]->(c)
        SET action_support.review_needed=row.review_needed,
            action_support.source_block_id=row.source_block_id
    """, rows=pairs).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (c:Chunk {id:row.chunk_id})
        MERGE (t:ManualTableRow {id:row.id})
        SET t.row_text=row.row_text, t.page_number=row.page_number,
            t.chunk_id=row.chunk_id, t.table_title=row.table_title,
            t.section_title=row.section_title, t.manual_id='festo',
            t.source_method=row.source_method,
            t.source_block_id=row.source_block_id,
            t.source_start=row.source_start, t.source_end=row.source_end,
            t.review_needed=row.review_needed, t.review_type=row.review_type,
            t.review_reason=row.review_reason
        MERGE (t)-[r:SUPPORTED_BY]->(c)
        SET r.review_needed=row.review_needed, r.source_block_id=row.source_block_id
    """, rows=table_rows).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (s:Situation {id:row.situation_id})
        MATCH (e:ManualError {id:row.error_id})
        MERGE (s)-[r:RELEVANT_DIAGNOSTIC]->(e)
        SET r.priority=row.priority, r.rationale=row.rationale,
            r.assessment='candidate', r.situation_source_page=s.source_page,
            r.manual_page_number=e.page_number,
            r.source_method='reviewed_mapping',
            r.evidence='RSW situation definition + Festo Table 10.1',
            r.review_needed=coalesce(e.review_needed,false)
    """, rows=linked["diagnostics"]).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (s:Situation {id:row.situation_id})
        MATCH (p:Page {id:row.page_id})
        MERGE (s)-[r:RELEVANT_SECTION]->(p)
        SET r.rationale=row.rationale, r.evidence_status=row.evidence_status,
            r.assessment='reference_for_inspection', r.source_method='reviewed_mapping',
            r.situation_source_page=s.source_page, r.review_needed=row.review_needed
    """, rows=linked["pages"]).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (s:Situation {id:row.situation_id})
        MATCH (c:Chunk {id:row.chunk_id})
        MERGE (s)-[r:RELEVANT_EVIDENCE]->(c)
        SET r.rationale=row.rationale, r.priority=row.priority,
            r.page_number=row.page_number, r.assessment=row.assessment,
            r.source_method='reviewed_mapping', r.review_needed=row.review_needed
    """, rows=linked["evidence"]).consume()
    tx.run("""
        UNWIND $rows AS row MATCH (s:Situation {id:row.situation_id})
        MATCH (c:Chunk {id:row.chunk_id})
        MERGE (co:Component {id:row.id})
        SET co.name=row.name, co.manual_id='festo', co.source_method='reviewed_mapping',
            co.review_needed=coalesce(co.review_needed,false) OR row.review_needed
        MERGE (co)-[cs:SUPPORTED_BY]->(c)
        SET cs.review_needed=row.review_needed
        MERGE (s)-[r:CHECKS_COMPONENT]->(co)
        SET r.assessment='inspection_candidate', r.page_number=row.page_number,
            r.source_method='reviewed_mapping', r.review_needed=row.review_needed
    """, rows=linked["components"]).consume()
    nodes = [n for item in extras for n in item["nodes"]]
    links = [r for item in extras for r in item["relationships"]]
    if nodes:
        tx.run("""
            UNWIND $rows AS row WITH row WHERE row.label='Component'
            MATCH (c:Chunk {id:row.chunk_id}) MERGE (n:Component {id:row.id})
            SET n.name=row.name, n.manual_id='festo', n.source_method='llm_reviewed',
                n.review_needed=false
            MERGE (n)-[r:SUPPORTED_BY]->(c)
            SET r.review_needed=false
        """, rows=nodes).consume()
        tx.run("""
            UNWIND $rows AS row WITH row WHERE row.label='CheckAction'
            MATCH (c:Chunk {id:row.chunk_id}) MERGE (n:CheckAction {id:row.id})
            SET n.name=row.name, n.manual_id='festo', n.source_method='llm_reviewed',
                n.page_number=row.page_number, n.chunk_id=row.chunk_id,
                n.review_needed=false
            MERGE (n)-[r:SUPPORTED_BY]->(c)
            SET r.review_needed=false
        """, rows=nodes).consume()
    if links:
        tx.run("""
            UNWIND $rows AS row
            MATCH (a:CheckAction {id:row.start}) MATCH (c:Component {id:row.end})
            MERGE (a)-[r:CHECKS_COMPONENT]->(c)
            SET r.chunk_id=row.chunk_id, r.page_number=row.page_number,
                r.source_method='llm_reviewed', r.review_needed=false
        """, rows=links).consume()


def db_summary(session: Any) -> dict[str, Any]:
    labels = ("Situation", "ManualError", "FaultSymptom", "Cause", "CheckAction",
              "Component", "ManualTableRow")
    counts = {label: session.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"] for label in labels}
    coverage = [dict(row) for row in session.run("""
        MATCH (s:Situation)
        OPTIONAL MATCH (s)-[:RELEVANT_DIAGNOSTIC]->(e:ManualError)
        WITH s, count(DISTINCT e) AS diagnostic_count
        OPTIONAL MATCH (s)-[:RELEVANT_EVIDENCE]->(c:Chunk)
        WITH s, diagnostic_count, count(DISTINCT c) AS evidence_chunk_count
        OPTIONAL MATCH (s)-[:CHECKS_COMPONENT]->(co:Component)
        RETURN s.id AS situation_id, diagnostic_count, evidence_chunk_count,
               count(DISTINCT co) AS component_count
        ORDER BY situation_id
    """)]
    candidates = [dict(row) for row in session.run("""
        MATCH (s:Situation)-[r:RELEVANT_DIAGNOSTIC]->(e:ManualError)
        RETURN s.id AS situation_id, e.number AS number, e.name AS name,
               r.priority AS priority, r.assessment AS assessment,
               e.page_number AS manual_page
        ORDER BY situation_id, CASE r.priority WHEN 'primary' THEN 0 ELSE 1 END, number
    """)]
    return {"node_counts": counts, "situation_coverage": coverage,
            "diagnostic_candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the S01–S10 Festo KG")
    parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--database", default=get_database(),
                        help="Neo4j target database (default: NEO4J_DATABASE)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Deterministic graph only; explicitly mark schema inference skipped.")
    args = parser.parse_args()
    chunks = load_chunks(args.chunks)
    mapping = load_mapping(MAPPING_PATH)
    errors, pairs = parse_tables(chunks)
    table_rows = parse_manual_rows(chunks)
    linked = resolve_mapping(chunks, errors, mapping)
    pages = sorted({c["page"] for c in chunks})
    load_environment()
    uri, user = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME")
    password, database = os.getenv("NEO4J_PASSWORD"), get_database(args.database)
    if not all((uri, user, password, database)):
        raise RuntimeError("Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE.")
    summary = {"status": "running", "database": database, "selected_pages": pages,
               "chunk_count": len(chunks), "situation_count": len(SITUATIONS),
               "manual_error_rows": len(errors), "cause_remedy_rows": len(pairs),
               "manual_table_rows": len(table_rows),
               "review_pending_table_rows": sum(row["review_needed"] for row in table_rows),
               "reviewed_diagnostic_links": len(linked["diagnostics"]),
               "reviewed_evidence_links": len(linked["evidence"]),
               "reviewed_component_links": len(linked["components"]),
               "schema_calls": 0, "extraction_calls": 0, "cache_hits": 0,
               "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
               "warnings": []}
    # Never make a paid call until the DB snapshot matches our input.
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            preflight(session, chunks)
    extras = []
    if args.skip_llm:
        summary["schema_status"] = "skipped_by_request"
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing; use --skip-llm for deterministic-only mode.")
        extras = asyncio.run(optional_llm(chunks, os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), summary))
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session(database=database) as session:
                constraints(session)
                session.execute_write(write_graph, errors, pairs, table_rows, extras, mapping, linked)
                summary["graph"] = db_summary(session)
        summary["llm_nodes_accepted"] = sum(len(r["nodes"]) for r in extras)
        summary["llm_relationships_accepted"] = sum(len(r["relationships"]) for r in extras)
        summary["status"] = "complete" if not summary["warnings"] else "partial"
    except Exception as exc:
        summary["status"] = "failed"
        summary["warnings"].append(f"Neo4j write failed: {type(exc).__name__}: {exc}")
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
