"""Festo PDF를 근거 위치가 남는 텍스트로 변환한다.

사용 예: python -m rag parse --pages 8,18,27,90
그림 설명: python -m rag parse --pages 16 --vision --vision-pages 16
전체 처리: python -m rag parse --all-pages
"""

import argparse
import base64
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pdfplumber
from ..config import load_environment
from ..paths import MANUAL_PDF, PARSING, PARSED_BLOCKS, PARSED_DOCS, REVIEW_IMAGES, VISION_CACHE


PDF = MANUAL_PDF
OUT = PARSING
IMAGES = REVIEW_IMAGES
TABLE_CAPTION = re.compile(r"^Table\s+([A-Za-z]?\d+(?:\.\d+)*):?\s*(.*)$", re.I)
FIGURE_CAPTION = re.compile(r"^(?:Figure|Fig\.)\s+\d+(?:\.\d+)*:\s*.+$", re.I)
SECTION = re.compile(r"^((?:[1-9]|1[01])(?:\.\d+){0,3})\s+([A-Za-z“].+)$")
TECHNICAL = re.compile(r"\b(?:[12][AV]\d|CPX-[A-Z0-9-]+|MPY[DE]|NEBU-[A-Z0-9,.-]+|E\d{3})\b")

# 이 매뉴얼에서는 몇몇 표의 헤더가 표 테두리 바깥에 있어 직접 대응시킨다.
KNOWN_COLUMNS = {
    **{n: ["No.", "Component", "Designation", "Part No."] for n in ("2.2", "2.3", "2.4", "2.5")},
    "5.3": ["Connection", "Tubing inside diameter", "Description"],
    "5.4": ["Design", "Pneumatic connection"],
    "5.5": ["Connection", "Tubing", "Description, C-gun", "Description, X-gun"],
    "10.1": ["Error number", "Error text", "Ready signal", "Description", "Error elimination"],
}


def parse_pages(spec: str, count: int) -> list[int]:
    """PDF 페이지 번호는 1부터 시작하며 8,18,27 또는 89-91을 허용한다."""
    result = set()
    for item in spec.split(","):
        item = item.strip()
        if re.fullmatch(r"\d+", item):
            result.add(int(item))
        elif re.fullmatch(r"\d+-\d+", item):
            first, last = map(int, item.split("-"))
            if first > last:
                raise ValueError(f"잘못된 페이지 범위: {item}")
            result.update(range(first, last + 1))
        else:
            raise ValueError(f"잘못된 페이지 형식: {item}")
    if not result or min(result) < 1 or max(result) > count:
        raise ValueError(f"1~{count} 사이의 PDF 페이지를 지정하세요: {spec}")
    return sorted(result)


def previous_section(pdf, page_no: int) -> str | None:
    """선택 쪽이 앞 절의 연속일 때 직전 절 번호를 가져온다."""
    for index in range(page_no - 2, max(-1, page_no - 8), -1):
        page = pdf.pages[index]
        table_boxes = [table.bbox for table in page.find_tables()]
        for line in reversed(page.extract_text_lines()):
            if any(inside(box, *center(line)) for box in table_boxes):
                continue
            match = SECTION.match(line["text"].strip())
            # 상단의 반복 장 제목보다 실제 본문 절 제목을 우선한다.
            if match and line["top"] >= 55:
                return match.group(1)
    return None


def inside(box, x, y) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def center(obj) -> tuple[float, float]:
    return ((obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2)


def box_list(box) -> list[float]:
    return [round(float(value), 1) for value in box]


def caption_below(table, lines) -> dict | None:
    for line in lines:
        if 0 <= line["top"] - table.bbox[3] <= 14 and TABLE_CAPTION.match(line["text"].strip()):
            return line
    return None


def is_data_table(table, caption, lines) -> bool:
    rows = table.extract()
    cells = [str(cell).strip() for row in rows for cell in row if cell]
    if not cells:
        return False
    if caption:
        return True
    headers = [line["text"].strip() for line in lines
               if 0 <= table.bbox[1] - line["bottom"] <= 30]
    if len(rows[0]) == 2 and "Cause Remedy" in headers:
        return True
    # 제품 사진 격자의 7/8/9/... 숫자만 표로 인식되는 경우를 제외한다.
    return len(rows) >= 3 and sum(len(cell) > 8 for cell in cells) >= 2


def table_text(table, caption: str, page_no: int, lines: list[dict]) -> str:
    rows = table.extract()
    width = max(map(len, rows))
    match = TABLE_CAPTION.match(caption)
    number = match.group(1) if match else ("10.1" if 89 <= page_no <= 91 and width == 5 else "")
    headers = KNOWN_COLUMNS.get(number)
    if not headers or len(headers) != width:
        header_lines = [line["text"].strip() for line in lines
                        if 0 <= table.bbox[1] - line["bottom"] <= 30]
        if width == 3 and "No. Action Comment" in header_lines:
            headers = ["No.", "Action", "Comment"]
        elif width == 3 and "No. Advice Comment" in header_lines:
            headers = ["No.", "Advice", "Comment"]
        elif width == 2 and "Cause Remedy" in header_lines:
            headers = ["Cause", "Remedy"]
        else:
            headers = [f"Column {i}" for i in range(1, width + 1)]
    title = caption or (
        "Table 10.1: Error messages" + (" (continued)" if page_no > 89 else "")
        if number == "10.1" else "Table"
    )
    result = [f"[{title}; PDF page {page_no}]"]
    for row in rows:
        fields = []
        for header, cell in zip(headers, row):
            if not cell or not cell.strip():
                continue
            if number == "5.4" and header == "Pneumatic connection":
                # 위치와 연결 관계는 그림으로 확인해야 하므로 숫자만 확정하지 않는다.
                header = "Diagram labels (connection direction unverified)"
            fields.append(f"{header}: {' '.join(cell.split())}")
        if fields:
            result.append(" | ".join(fields))
    return "\n".join(result)


def matrix_extracted_text(table, page_no: int) -> str:
    """확인 가능한 두 열만 보존하고 회전된 비트 열에는 의미를 붙이지 않는다."""
    result = [f"[Table 10.2: Assignment of error bits; PDF page {page_no}]"]
    for row in table.extract():
        if len(row) >= 2 and row[0] and row[1]:
            result.append(f"Error number: {' '.join(row[0].split())} | "
                          f"Error text: {' '.join(row[1].split())}")
    return "\n".join(result)


def matrix_text(table, page_no: int) -> str:
    """표 본문에는 비트 열을 해석하지 않았다는 경고를 덧붙인다."""
    return matrix_extracted_text(table, page_no) + "\n[Bit column labels and assignments: manual review required]"


def review_fields(status: str | None) -> tuple[bool, str | None, str | None]:
    """기존의 세부 상태를 청킹 단계에서 사용할 공통 검토 사유로 정규화한다."""
    if not status:
        return False, None, None
    if status == "matrix_header_mapping_pending":
        return True, "table_structure", "unverified_table_header_mapping"
    if status in {"draft_review_required", "unsupported_terms_review_required"}:
        return True, "visual_content", "unverified_visual_description"
    return True, "visual_content", "unparsed_image_or_visual_information"


def is_mixed_table(page, table, all_tables) -> bool:
    if any(inside(table.bbox, *center(image)) for image in page.images):
        return True
    # 배관도는 비트맵 이미지가 아니라 중첩된 PDF 도형/격자로 구성될 수 있다.
    return any(other is not table and table.bbox[0] <= other.bbox[0] < other.bbox[2] <= table.bbox[2]
               and table.bbox[1] <= other.bbox[1] < other.bbox[3] <= table.bbox[3]
               for other in all_tables)


def figure_box(page, caption, rejected, lines):
    grids = [table.bbox for table in rejected if 0 <= caption["top"] - table.bbox[3] <= 16]
    if grids:
        return min(grids, key=lambda box: caption["top"] - box[3])
    images = [image for image in page.images if 0 <= caption["top"] - image["bottom"] <= 35]
    if images:
        return (min(i["x0"] for i in images), min(i["top"] for i in images),
                max(i["x1"] for i in images), max(i["bottom"] for i in images))
    previous = [line for line in lines if line["bottom"] < caption["top"] - 18 and len(line["text"].split()) >= 5]
    if previous:
        top = max(previous, key=lambda line: line["bottom"])["bottom"] + 2
        if caption["top"] - top >= 35:
            return (40, top, page.width - 40, caption["top"] - 2)
    return None


def save_crop(page, box, page_no: int, label: str) -> Path:
    IMAGES.mkdir(parents=True, exist_ok=True)
    korean_label = (label.replace("figure_", "그림")
                    .replace("table_", "표")
                    .replace("image_", "이미지")
                    .replace("matrix_", "행렬표"))
    path = IMAGES / f"{page_no:03d}쪽_{korean_label}.png"
    x0, top, x1, bottom = box
    crop = (max(0, x0 - 3), max(0, top - 3), min(page.width, x1 + 3), min(page.height, bottom + 3))
    page.crop(crop).to_image(resolution=180).original.save(path)
    return path


def ask_vision(image: Path, context: str, model: str) -> dict:
    """선택 옵션: 그림 설명/화면 글자 전사. 결과는 항상 사람이 검토한다."""
    from openai import OpenAI

    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    instruction = (
        "Return JSON only: description (string), visible_text (list of EXACT strings copied from image), "
        "manual_terms (list of exact manual terms used in description), source_quotes "
        "(list of exact context phrases supporting names), uncertain (boolean). "
        "This is a Festo welding-gun manual. Describe only visible facts. "
        "Use component names only when they occur verbatim in the context. "
        "When a name or connection is uncertain, use neutral visual words and set uncertain=true. "
        "Do not invent values, connections, part names, or OCR text.\n\n"
        f"Context from this and adjacent manual pages:\n{context[:12000]}"
    )
    response = OpenAI(max_retries=0).responses.create(model=model, max_output_tokens=400,
                                                       input=[{"role": "user", "content": [
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"},
    ]}])
    raw = response.output_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    data = json.loads(raw)
    if not all(key in data for key in ("description", "visible_text", "manual_terms", "source_quotes", "uncertain")):
        raise ValueError("비전 응답에 필수 필드가 없습니다")
    if not all(isinstance(data[key], list) for key in ("visible_text", "manual_terms", "source_quotes")):
        raise ValueError("비전 응답의 목록 필드 형식이 다릅니다")
    data["unsupported_terms_or_quotes"] = [item for key in ("manual_terms", "source_quotes")
                                            for item in data[key]
                                            if not isinstance(item, str) or item.casefold() not in context.casefold()]
    return data


class VisionRunner:
    """명시적으로 선택한 시각 자료만 호출하고, 같은 요청은 로컬 캐시에서 재사용한다."""

    def __init__(self, model: str, max_calls: int):
        self.model = model
        self.max_calls = max_calls
        self.new_calls = 0
        self.cache_hits = 0
        self.cache_dir = VISION_CACHE

    def describe(self, image: Path, context: str) -> tuple[dict | None, str]:
        # 프롬프트나 모델을 바꾸면 캐시 키를 갱신한다.
        digest = hashlib.sha256(b"festo-vision-v1\0" + self.model.encode("utf-8") + b"\0"
                                + context.encode("utf-8") + b"\0" + image.read_bytes()).hexdigest()
        cache_path = self.cache_dir / f"그림설명_{digest}.json"
        if cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8")), "cache"
        if self.new_calls >= self.max_calls:
            return None, "call_limit_reached"
        self.new_calls += 1  # 실패한 요청도 재시도하지 않도록 호출 횟수에 포함
        result = ask_vision(image, context, self.model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result, "api"


def attach_visual(block, page, page_no: int, label: str, context: str, vision: VisionRunner | None):
    # 비전 설명/전사 초안과 검토 문구를 검색 가능한 원문 텍스트와 분리한다.
    block.setdefault("extractable_text", block["content"])
    image = save_crop(page, block["bbox"], page_no, label)
    block["image_path"] = str(image.relative_to(OUT)).replace("\\", "/")
    block["review_status"] = "description_pending"
    if vision is None:
        block["content"] += "\n[그림 내용/이미지 속 글자: 검토 필요. 원본 그림과 주변 원문을 확인하세요.]"
        return
    try:
        draft, source = vision.describe(image, context)
    except Exception as exc:
        block["vision_error"] = str(exc)[:300]
        block["content"] += "\n[그림 설명 생성 실패: 검토 필요]"
        return
    if draft is None:
        block["review_status"] = "vision_call_limit_reached"
        block["content"] += "\n[비전 호출 상한 도달: 그림 설명 미생성]"
        return
    block["vision_draft"] = draft
    block["vision_source"] = source
    block["review_status"] = "unsupported_terms_review_required" if draft["unsupported_terms_or_quotes"] else "draft_review_required"
    visible = [str(value).strip() for value in draft["visible_text"] if str(value).strip()]
    if visible:
        block["content"] += "\n[이미지 글자 전사 초안] " + " | ".join(visible)
    if str(draft["description"]).strip():
        block["content"] += "\n[그림 설명 초안, KG 사용 전 검토] " + str(draft["description"]).strip()


def convert_page(page, page_no: int, context: str, vision: VisionRunner | None,
                 initial_section: str | None = None):
    lines = page.extract_text_lines()
    candidates = page.find_tables()
    selected, rejected, captions = [], [], {}
    for table in candidates:
        caption = caption_below(table, lines)
        if is_data_table(table, caption, lines):
            selected.append(table)
            captions[id(table)] = caption
        else:
            rejected.append(table)

    blocks, excluded_lines = [], set()
    for index, table in enumerate(selected, start=1):
        caption = captions[id(table)]
        label = caption["text"].strip() if caption else ""
        if caption:
            excluded_lines.add(id(caption))
        if page_no in (92, 93) and len(table.extract()[0]) > 8:
            review_top = 100 if page_no == 92 else 80
            block = {"kind": "matrix_table", "page_number": page_no,
                     "bbox": box_list(table.bbox), "caption": label or "Table 10.2: Assignment of error bits",
                     "row_count": len(table.extract()), "content": matrix_text(table, page_no),
                     "extractable_text": matrix_extracted_text(table, page_no),
                     "review_status": "matrix_header_mapping_pending"}
            crop = save_crop(page, (table.bbox[0], review_top, table.bbox[2], table.bbox[3]),
                             page_no, f"matrix_{index}")
            block["image_path"] = str(crop.relative_to(OUT)).replace("\\", "/")
            blocks.append(block)
            continue
        mixed = is_mixed_table(page, table, candidates)
        block = {"kind": "mixed_table" if mixed else "table", "page_number": page_no,
                 "bbox": box_list(table.bbox), "caption": label,
                 "row_count": len(table.extract()), "content": table_text(table, label, page_no, lines)}
        if mixed:
            attach_visual(block, page, page_no, f"table_{index}", context, vision)
        blocks.append(block)

    figure_boxes = []
    for index, line in enumerate(lines):
        if id(line) in excluded_lines:
            continue
        value = line["text"].strip()
        if not value or (value == str(page_no) and line["top"] > page.height * 0.9):
            continue
        if line["top"] < 55 and SECTION.match(value):
            continue  # 각 페이지 상단에 반복 인쇄된 장 제목
        # 10장의 세로 회전된 표 머리글은 PDF 추출 시 'r e b m u'처럼 깨진다.
        # 표 행은 별도로 추출했으므로 깨진 머리글만 제거하고 절 제목/본문은 보존한다.
        if 89 <= page_no <= 93 and selected:
            header_start = {89: 160, 90: 65, 91: 65, 92: 100, 93: 80}[page_no]
            if header_start <= line["top"] < min(table.bbox[1] for table in selected):
                continue
        point = center(line)
        if any(inside(table.bbox, *point) for table in selected):
            continue
        if FIGURE_CAPTION.match(value):
            region = figure_box(page, line, rejected, lines)
            bbox = region or (line["x0"], line["top"], line["x1"], line["bottom"])
            block = {"kind": "figure", "page_number": page_no, "bbox": box_list(bbox),
                     "caption": value, "content": f"[{value}; PDF page {page_no}]"}
            if region:
                figure_boxes.append(region)
                attach_visual(block, page, page_no, f"figure_{index}", context, vision)
            else:
                block["extractable_text"] = block["content"]
                block["content"] += "\n[그림 영역 자동 탐지 실패: 원본 PDF 확인 필요]"
                block["review_status"] = "crop_pending"
            blocks.append(block)
            continue
        if re.fullmatch(r"[\d\s]+", value) and any(inside(table.bbox, *point) for table in rejected):
            continue  # 사진 격자의 7/8/9/... 표시
        blocks.append({"kind": "text", "page_number": page_no,
                       "bbox": box_list((line["x0"], line["top"], line["x1"], line["bottom"])),
                       "content": value})

    # 표나 Figure 캡션에 속하지 않는 이미지도 검토 대상으로 남긴다.
    for index, image in enumerate(page.images, start=1):
        point = center(image)
        if any(inside(table.bbox, *point) for table in selected) or any(inside(box, *point) for box in figure_boxes):
            continue
        block = {"kind": "image", "page_number": page_no,
                 "bbox": box_list((image["x0"], image["top"], image["x1"], image["bottom"])),
                 "content": f"[Uncaptioned image; PDF page {page_no}]", "extractable_text": ""}
        attach_visual(block, page, page_no, f"image_{index}", context, vision)
        blocks.append(block)

    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    current_section = initial_section
    for order, block in enumerate(blocks):
        block["order"] = order
        block["source_block_id"] = f"festo:page:{page_no}:block:{order}"
        block.setdefault("extractable_text", block["content"])
        if not block["content"].startswith(block["extractable_text"]):
            raise ValueError(f"Extractable text is not a prefix: {block['source_block_id']}")
        block["review_needed"], block["review_type"], block["review_reason"] = review_fields(
            block.get("review_status")
        )
        if block["kind"] == "text":
            match = SECTION.match(block["content"])
            if match and block["bbox"][1] >= 55:
                current_section = match.group(1)
        block["section"] = current_section

    content = "\n".join(block["content"] for block in blocks)
    source = page.extract_text() or ""
    missing = sorted((Counter(token.casefold() for token in TECHNICAL.findall(source))
                      - Counter(token.casefold() for token in TECHNICAL.findall(content))).elements())
    visuals = [block for block in blocks if block["review_needed"]]
    section_ids = list(dict.fromkeys(block["section"] for block in blocks if block["section"]))
    audit = {"page_number": page_no, "source_chars": len(source), "converted_chars": len(content),
             "table_count": sum(block["kind"] in {"table", "mixed_table", "matrix_table"} for block in blocks),
             "figure_count": sum(block["kind"] in {"figure", "image"} for block in blocks),
             "visuals_to_review": len(visuals), "missing_technical_tokens": missing,
             "section_ids": section_ids}
    # 이 판정은 자동 점검 통과 여부만 뜻한다. 표의 의미와 KG 적합성은 원본과 대조한다.
    audit["automatic_check_passed"] = bool(content.strip()) and not missing and not visuals
    row = {"doc_id": f"festo:page:{page_no}", "page_content": content,
           "metadata": {"source": str(PDF), "page": page_no - 1, "page_number": page_no,
                        "manual_id": "festo", "section_ids": section_ids,
                        "visual_review_required": bool(visuals),
                        "automatic_check_passed": audit["automatic_check_passed"]}}
    return row, blocks, audit


def save_results(rows, blocks, selected, total):
    OUT.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["metadata"]["partial_run"] = len(selected) != total
    for path, items in ((PARSED_DOCS, rows), (PARSED_BLOCKS, blocks)):
        with path.open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")



def main():
    parser = argparse.ArgumentParser(description="Festo PDF에서 선택 페이지만 위치 기반으로 변환")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", help="PDF 페이지 번호 예: 8,18,27,89-91")
    group.add_argument("--all-pages", action="store_true", help="전체 페이지 변환")
    parser.add_argument("--vision", action="store_true", help="그림 설명 초안 생성 (OpenAI API 과금)")
    parser.add_argument("--vision-pages", help="API 호출을 허용할 페이지. --vision과 함께 필수")
    parser.add_argument("--max-vision-calls", type=int, default=1,
                        help="이번 실행에서 허용할 신규 API 호출 수 (기본값: 1, 캐시 재사용 제외)")
    parser.add_argument("--vision-model", default="gpt-4.1-mini", help="이미지 입력 가능 모델")
    args = parser.parse_args()
    if not PDF.exists():
        parser.error(f"PDF 파일이 없습니다: {PDF}")
    if args.vision:
        load_environment()
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("--vision 실행에는 OPENAI_API_KEY가 필요합니다")
        if not args.vision_pages:
            parser.error("--vision에는 --vision-pages로 API를 사용할 페이지를 지정해야 합니다")
        if args.max_vision_calls < 0:
            parser.error("--max-vision-calls는 0 이상이어야 합니다")
    elif args.vision_pages:
        parser.error("--vision-pages는 --vision과 함께 사용하세요")

    with pdfplumber.open(PDF) as pdf:
        selected = list(range(1, len(pdf.pages) + 1)) if args.all_pages else parse_pages(args.pages, len(pdf.pages))
        vision_pages = parse_pages(args.vision_pages, len(pdf.pages)) if args.vision else []
        if not set(vision_pages).issubset(selected):
            parser.error("--vision-pages는 --pages에서 선택한 페이지에 포함돼야 합니다")
        vision = VisionRunner(args.vision_model, args.max_vision_calls) if args.vision else None
        rows, blocks, audits = [], [], []
        for no in selected:
            print(f"PDF {no}/{len(pdf.pages)}쪽 처리")
            context = "\n".join((pdf.pages[i].extract_text() or "")
                                for i in range(max(0, no - 2), min(len(pdf.pages), no + 1))) if no in vision_pages else ""
            row, page_blocks, audit = convert_page(pdf.pages[no - 1], no, context,
                                                   vision if no in vision_pages else None,
                                                   previous_section(pdf, no))
            rows.append(row)
            blocks.extend(page_blocks)
            audits.append(audit)
    save_results(rows, blocks, selected, len(pdf.pages))
    print(f"저장: {PARSED_DOCS}")
    print(f"위치와 추출 상세: {PARSED_BLOCKS}")
    if vision:
        print(f"신규 API 호출: {vision.new_calls}/{vision.max_calls}; 로컬 캐시 재사용: {vision.cache_hits}")
    if any(item["visuals_to_review"] for item in audits):
        print("Review flags are preserved in blocks.jsonl and passed to chunking and answers.")


if __name__ == "__main__":
    main()
