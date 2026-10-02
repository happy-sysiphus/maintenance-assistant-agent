"""Chunk Festo text while preserving table rows, cause/remedy pairs, and review provenance."""

import argparse
from bisect import bisect_right
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from ..paths import CHUNKS, PARSED_BLOCKS, PARSED_DOCS

from langchain_core.documents import Document


INPUT_PATH = PARSED_DOCS
BLOCKS_PATH = PARSED_BLOCKS
OUTPUT_PATH = CHUNKS
DEFAULT_MAX_CHARS = 1200
REVIEW_WARNING = "[검토 필요: 자동으로 해석되지 않은 시각 정보가 있습니다. 원본 PDF 확인]"

SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\s+\S")
ERROR_ROW_RE = re.compile(r"^Error number:\s*(\d+)\s*\|")
CAUSE_REMEDY_RE = re.compile(r"^Cause:\s*.+\s\|\sRemedy:\s*.+")
RESERVE_RE = re.compile(r"^Error number:\s*\d+\s*\|\s*Error text:\s*Reserve(?:\s*\||$)", re.I)
TABLE_MARKER_RE = re.compile(r"^\[Table(?:\s|;)")
FIGURE_MARKER_RE = re.compile(r"^\[(?:Figure|Fig\.|Uncaptioned image)")


@dataclass(frozen=True)
class Unit:
    """원본 페이지에서 잘라서는 안 되는 최소 텍스트 단위."""

    kind: str
    text: str
    start: int
    end: int
    section_title: str
    table_title: str
    fault_heading: str
    error_number: str | None = None
    segment: int = 0
    source_block_id: str = ""
    source_block_kind: str = ""
    source_block_order: int = 0
    source_bbox: list[float] | None = None
    image_path: str | None = None
    review_status: str | None = None
    review_needed: bool = False
    review_type: str | None = None
    review_reason: str | None = None

    @property
    def group_key(self) -> tuple[str, str, str, str, int]:
        # A chunk may contain both reviewed and ordinary units.
        return self.kind, self.section_title, self.table_title, self.fault_heading, self.segment


def load_parsed_documents(input_path: Path) -> list[Document]:
    if not input_path.exists():
        raise FileNotFoundError(f"파싱 결과가 없습니다: {input_path}. python -m rag parse를 실행하세요.")

    docs = []
    with input_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata", {}))
            if metadata.get("manual_id") != "festo":
                raise ValueError(f"{line_no}행은 Festo 매뉴얼 파싱 결과가 아닙니다.")
            metadata["parent_doc_id"] = row["doc_id"]
            docs.append(Document(page_content=row["page_content"], metadata=metadata))

    if not docs:
        raise ValueError(f"파싱 결과가 비어 있습니다: {input_path}")
    return docs


def filter_review_blocks(docs: list[Document], blocks_path: Path) -> None:
    """블록을 원문 위치에 연결하고 추출 가능한 텍스트와 검토 상태를 보존한다.

    파싱 단계의 두 출력 파일이 서로 다른 실행에서 만들어졌다면 중단한다.
    """
    if not blocks_path.exists():
        raise FileNotFoundError(f"블록별 검토 상태가 없습니다: {blocks_path}. python -m rag parse를 다시 실행하세요.")

    by_page: dict[int, list[dict]] = {}
    with blocks_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            block = json.loads(line)
            page_no = block.get("page_number")
            if not isinstance(page_no, int) or not isinstance(block.get("order"), int):
                raise ValueError(f"{blocks_path}:{line_no} 블록의 페이지나 순서가 잘못됐습니다.")
            if (not isinstance(block.get("source_block_id"), str)
                    or not isinstance(block.get("extractable_text"), str)
                    or not isinstance(block.get("review_needed"), bool)):
                raise ValueError(
                    f"{blocks_path}:{line_no}: review provenance is missing. "
                    "Run python -m rag parse again to regenerate both parsed files."
                )
            if not block["content"].startswith(block["extractable_text"]):
                raise ValueError(f"{blocks_path}:{line_no}: extractable_text is not a content prefix")
            if block["review_needed"] and not (block.get("review_type") and block.get("review_reason")):
                raise ValueError(f"{blocks_path}:{line_no}: review type or reason is missing")
            by_page.setdefault(page_no, []).append(block)

    doc_pages = {doc.metadata["page_number"] for doc in docs}
    if set(by_page) != doc_pages or len(doc_pages) != len(docs):
        raise ValueError("pages.jsonl과 blocks.jsonl의 페이지가 다릅니다. python -m rag parse를 다시 실행하세요.")

    for doc in docs:
        page_no = doc.metadata["page_number"]
        blocks = sorted(by_page[page_no], key=lambda block: block["order"])
        if [block["order"] for block in blocks] != list(range(len(blocks))):
            raise ValueError(f"PDF {page_no}쪽 블록 순서가 중복되거나 누락됐습니다.")
        if "\n".join(block["content"] for block in blocks) != doc.page_content:
            raise ValueError(f"PDF {page_no}쪽 블록 내용이 페이지와 다릅니다. python -m rag parse를 다시 실행하세요.")

        if len({block["source_block_id"] for block in blocks}) != len(blocks):
            raise ValueError(f"PDF {page_no}: duplicate source_block_id")

        spans: list[dict] = []
        review_ends: list[int] = []
        offset = 0
        for index, block in enumerate(blocks):
            end = offset + len(block["content"])
            spans.append({**block, "start": offset, "end": end})
            if block["review_needed"]:
                review_ends.append(end)
            offset = end + (index + 1 < len(blocks))

        # start/end는 공백으로 가린 사본이 아닌 원본 페이지의 오프셋이다.
        # Keep the exact page text. Search only the PDF-extracted prefix of
        # each block; visual notes and vision drafts are not indexed.
        doc.metadata["_block_spans"] = spans
        doc.metadata["_review_ends"] = review_ends
        doc.metadata["review_blocks_on_page"] = len(review_ends)


def _line_records(doc: Document) -> list[tuple[str, int, int, dict, bool]]:
    """Read only PDF-extracted text; never index a visual/LLM draft."""
    records: list[tuple[str, int, int, dict, bool]] = []
    for block in doc.metadata["_block_spans"]:
        extracted = block["extractable_text"]
        if block["kind"] in {"image", "figure"}:
            label = extracted.strip() or f"[Uncaptioned image; PDF page {block['page_number']}]"
            records.append((label, block["start"], block["end"], block, True))
            continue

        cursor = block["start"]
        has_searchable_line = False
        for raw in extracted.splitlines(keepends=True):
            value = raw.rstrip("\r\n").strip()
            records.append((value, cursor, cursor + len(raw), block, False))
            if value and not TABLE_MARKER_RE.match(value):
                has_searchable_line = True
            cursor += len(raw)

        # A reviewed table without readable rows still needs a source marker.
        if block["review_needed"] and not has_searchable_line:
            records.append((f"[Visual table; PDF page {block['page_number']}]",
                            block["start"], block["end"], block, True))
    return records


def make_units(doc: Document, include_reserve: bool) -> tuple[list[Unit], int]:
    """오류 한 행, Cause/Remedy 한 행, 기타 표 한 행을 각각 원자 단위로 만든다."""
    lines = _line_records(doc)

    section_ids = doc.metadata.get("section_ids") or []
    section_title = f"Section {section_ids[-1]}" if section_ids else ""
    table_title = ""
    fault_heading = ""
    last_was_row = False
    skipped_reserve = 0
    units = []
    review_ends = doc.metadata["_review_ends"]
    current_segment = 0

    for index, (value, start, end, block, visual_placeholder) in enumerate(lines):
        if not value:
            continue
        segment = bisect_right(review_ends, start)
        if segment != current_segment:
            table_title = fault_heading = ""
            last_was_row = False
            current_segment = segment
        following = lines[index + 1][0] if index + 1 < len(lines) else ""

        if visual_placeholder:
            kind, error_number = "visual_placeholder", None
            table_title = ""
        elif SECTION_RE.match(value):
            section_title, table_title, fault_heading = value, "", ""
            last_was_row = False
            continue
        elif following == "Cause Remedy":
            fault_heading, table_title = value, ""
            last_was_row = False
            continue
        elif value == "Cause Remedy":
            continue
        elif TABLE_MARKER_RE.match(value):
            table_title = value
            last_was_row = False
            continue
        elif RESERVE_RE.match(value) and not include_reserve:
            skipped_reserve += 1
            last_was_row = True
            continue
        else:
            error = ERROR_ROW_RE.match(value)
            if error:
                kind, error_number = "error_row", error.group(1)
            elif CAUSE_REMEDY_RE.match(value):
                kind, error_number = "cause_remedy_row", None
            elif " | " in value and ":" in value.split(" | ", 1)[0]:
                kind, error_number = "table_row", None
            elif FIGURE_MARKER_RE.match(value):
                kind, error_number = "figure", None
            else:
                kind, error_number = "prose", None

        if kind == "prose" and last_was_row:
            table_title = ""  # 표 뒤의 본문을 이전 표에 묶지 않는다.
        if kind in {"figure", "visual_placeholder"}:
            table_title = ""
        units.append(Unit(
            kind=kind, text=value, start=start, end=end,
            section_title=section_title, table_title=table_title,
            fault_heading=fault_heading, error_number=error_number, segment=segment,
            source_block_id=block["source_block_id"], source_block_kind=block["kind"],
            source_block_order=block["order"], source_bbox=block.get("bbox"),
            image_path=block.get("image_path"), review_status=block.get("review_status"),
            review_needed=block["review_needed"], review_type=block.get("review_type"),
            review_reason=block.get("review_reason"),
        ))
        last_was_row = kind in {"error_row", "cause_remedy_row", "table_row"}

    return units, skipped_reserve


def chunk_prefix(unit: Unit) -> str:
    """문맥 없이 추출된 표 행도 검색했을 때 소속을 알 수 있게 한다."""
    return "\n".join(part for part in
                     (unit.section_title, unit.fault_heading, unit.table_title) if part)


def _chunk_content(units: list[Unit]) -> str:
    prefix = chunk_prefix(units[0])
    body = "\n".join(unit.text for unit in units)
    warning = REVIEW_WARNING if any(unit.review_needed for unit in units) else ""
    return "\n".join(part for part in (prefix, warning, body) if part)


def make_chunk(doc: Document, units: list[Unit], chunk_index: int,
               max_chars: int) -> Document:
    first, last = units[0], units[-1]
    content = _chunk_content(units)
    metadata = {key: value for key, value in doc.metadata.items() if not key.startswith("_")}
    review_units = [unit for unit in units if unit.review_needed]
    reviewed_blocks = dict.fromkeys(unit.source_block_id for unit in review_units)
    review_sources = []
    for block_id in reviewed_blocks:
        unit = next(item for item in review_units if item.source_block_id == block_id)
        review_sources.append({
            "source_block_id": unit.source_block_id,
            "page_number": doc.metadata["page_number"],
            "order": unit.source_block_order,
            "kind": unit.source_block_kind,
            "bbox": unit.source_bbox,
            "image_path": unit.image_path,
            "review_status": unit.review_status,
            "review_type": unit.review_type,
            "review_reason": unit.review_reason,
        })
    metadata.update({
        "chunk_index": chunk_index,
        "start_index": first.start,
        "end_index": last.end,
        "char_count": len(content),
        "unit_kind": first.kind,
        "unit_count": len(units),
        "section_title": first.section_title,
        "table_title": first.table_title,
        "fault_heading": first.fault_heading,
        "error_numbers": [unit.error_number for unit in units if unit.error_number],
        "oversized_atomic_unit": len(content) > max_chars,
        "review_needed": bool(review_units),
        "review_types": list(dict.fromkeys(unit.review_type for unit in review_units if unit.review_type)),
        "review_reasons": list(dict.fromkeys(unit.review_reason for unit in review_units if unit.review_reason)),
        "source_page": doc.metadata["page_number"],
        "review_sources": review_sources,
        "unit_sources": [{
            "text": unit.text,
            "kind": unit.kind,
            "start": unit.start,
            "end": unit.end,
            "source_block_id": unit.source_block_id,
            "source_block_kind": unit.source_block_kind,
            "source_block_order": unit.source_block_order,
            "review_needed": unit.review_needed,
            "review_type": unit.review_type,
            "review_reason": unit.review_reason,
        } for unit in units],
    })
    raw = f"{metadata['parent_doc_id']}\0{first.start}\0{last.end}\0{content}"
    metadata["chunk_id"] = "chunk-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return Document(page_content=content, metadata=metadata)


def chunk_documents(docs: list[Document], max_chars: int,
                    include_reserve: bool = False) -> tuple[list[Document], list[int], int]:
    chunks = []
    skipped_pages = []
    skipped_reserve = 0

    for doc in docs:
        units, skipped = make_units(doc, include_reserve)
        if not units:
            skipped_pages.append(doc.metadata["page_number"])
            continue
        skipped_reserve += skipped
        pending: list[Unit] = []

        def flush() -> None:
            if pending:
                chunks.append(make_chunk(doc, pending, len(chunks), max_chars))
                pending.clear()

        for unit in units:
            if pending:
                proposed = len(_chunk_content([*pending, unit]))
                if unit.group_key != pending[0].group_key or proposed > max_chars:
                    flush()
            pending.append(unit)
        flush()

    return chunks, skipped_pages, skipped_reserve


def save_chunks_to_jsonl(chunks: list[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps({
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Festo 매뉴얼을 표 행 단위로 청킹")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help=f"청크 목표 최대 문자 수 (기본 {DEFAULT_MAX_CHARS}; 표 행은 절단하지 않음)")
    parser.add_argument("--include-reserve", action="store_true",
                        help="내용이 없는 Reserve 오류 행도 포함")
    args = parser.parse_args()
    if args.max_chars < 200:
        parser.error("--max-chars는 200 이상으로 지정하세요")

    docs = load_parsed_documents(INPUT_PATH)
    filter_review_blocks(docs, BLOCKS_PATH)
    chunks, skipped_pages, skipped_reserve = chunk_documents(
        docs, args.max_chars, args.include_reserve)
    if not chunks:
        parser.error("청크가 없습니다. 선택한 페이지와 검토 상태를 확인하세요")
    save_chunks_to_jsonl(chunks, OUTPUT_PATH)

    print(f"입력 PDF 페이지: {len(docs)}개")
    print(f"청크가 없는 페이지: {skipped_pages or '없음'}")
    print(f"검토 상태를 보존한 원본 블록: {sum(doc.metadata['review_blocks_on_page'] for doc in docs)}개")
    print(f"검토 필요한 청크: {sum(chunk.metadata['review_needed'] for chunk in chunks)}개")
    print(f"제외한 Reserve 행: {skipped_reserve}개")
    print(f"생성한 청크: {len(chunks)}개 (원자 단위가 큰 청크: "
          f"{sum(chunk.metadata['oversized_atomic_unit'] for chunk in chunks)}개)")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
