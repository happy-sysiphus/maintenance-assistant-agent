"""Layout-aware PDF preparation: local models, native text, no picture interpretation."""
import argparse
from collections import Counter
import json
import os
import re
import pdfplumber
from ..documents import manuals, manual_path, verify_manuals
from ..paths import PARSING, PARSED_BLOCKS, PARSED_DOCS, RUNTIME

PARSER_VERSION = "docling-table-structure-v1"
TECHNICAL = re.compile(r"\b(?:CPX-[A-Z0-9-]+|MPY[DE]|E\d{3})\b")

def parse_pages(spec, count):
    result = set()
    for value in spec.split(","):
        parts = value.strip().split("-")
        if len(parts) == 1 and parts[0].isdigit():
            result.add(int(parts[0]))
        elif len(parts) == 2 and all(p.isdigit() for p in parts) and int(parts[0]) <= int(parts[1]):
            result.update(range(int(parts[0]), int(parts[1]) + 1))
        else:
            raise ValueError("잘못된 페이지 범위")
    if not result or min(result) < 1 or max(result) > count:
        raise ValueError("PDF 페이지 범위를 벗어났습니다.")
    return sorted(result)

def clean(text):
    return " ".join(re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", str(text or "")).split())

def row_text(headers, cells):
    return " | ".join(f"{h}: {v}" for h,v in zip(headers,cells) if v)

def convert_table(item, document, doc_id, page, bbox, section):
    data = item.data
    width = data.num_cols
    grid = [[""] * width for _ in range(data.num_rows)]
    header_rows, spans = set(), []
    for cell in data.table_cells:
        r,c = cell.start_row_offset_idx,cell.start_col_offset_idx
        text = clean(cell.text)
        if r < len(grid) and c < width:
            grid[r][c] = text
        if cell.column_header:
            header_rows.update(range(r,cell.end_row_offset_idx))
            for cc in range(c,min(width,cell.end_col_offset_idx)):
                for rr in range(r,min(len(grid),cell.end_row_offset_idx)):
                    grid[rr][cc] = text
        spans.append({"row":r,"column":c,"row_span":cell.row_span,"col_span":cell.col_span,
                      "text":text,"column_header":cell.column_header})
    if grid and not header_rows and any(re.search(r"^(?:No\.?|Nr|Pin|Error|Parameter|Command|Connection|Cause)$",v,re.I) for v in grid[0]):
        header_rows.add(0)
    headers = []
    for c in range(width):
        labels = list(dict.fromkeys(grid[r][c] for r in sorted(header_rows) if grid[r][c]))
        headers.append(" / ".join(labels) or f"Unmapped column {c+1}")
    table_id = f"{doc_id}:{item.self_ref.lstrip('#/').replace('/', ':')}"
    caption = " ".join(clean(ref.resolve(document).text) for ref in item.captions)
    error_table = any(re.search("error",h,re.I) for h in headers)
    rows = []
    for index,cells in enumerate(grid):
        if index in header_rows or not any(cells):
            continue
        text = row_text(headers,cells)
        rows.append({"row_id":f"{table_id}:row:{index}","cells":cells,"text":text,
            "error_number":cells[0] if error_table and re.fullmatch(r"\d+",cells[0]) else None,
            "source_parts":[{"doc":doc_id,"page":page,"bbox":bbox,"text":text}]})
    return {"kind":"table","caption":caption,"table_id":table_id,"logical_table_id":table_id,
            "headers":headers,"header_detected":bool(header_rows),"rows":rows,"cell_spans":spans,
            "row_count":len(rows),"section":section}

def link_continuations(pages):
    joined, previous = [], None
    for page in sorted(pages):
        tables = [b for b in pages[page] if b["kind"]=="table"]
        if previous and tables:
            prior_page,prior = previous
            first = tables[0]
            aligned = abs(first["bbox"][0]-prior["bbox"][0])<18 and abs(first["bbox"][2]-prior["bbox"][2])<18
            near_edges = prior["bbox"][3]>prior["page_height"]*.7 and first["bbox"][1]<first["page_height"]*.3
            same_headers = first["headers"]==prior["headers"] or not first["header_detected"]
            same_section = first["section"] == prior["section"]
            if prior_page+1==page and aligned and near_edges and same_headers and same_section and len(first["headers"])==len(prior["headers"]):
                first["logical_table_id"] = prior["logical_table_id"]
                if not first["header_detected"]:
                    first["headers"] = list(prior["headers"])
                    first["header_detected"] = prior["header_detected"]
                    for row in first["rows"]:
                        row["text"] = row_text(first["headers"],row["cells"])
                        row["source_parts"][0]["text"] = row["text"]
                        if any("error" in h.casefold() for h in first["headers"]) and re.fullmatch(r"\d+",row["cells"][0]):
                            row["error_number"]=row["cells"][0]
                if first["rows"] and prior["rows"]:
                    lead,last = first["rows"][0],prior["rows"][-1]
                    if not lead["cells"][0] and any(lead["cells"][1:]):
                        merged = [clean(a+" "+b) for a,b in zip(last["cells"],lead["cells"])]
                        last["search_text"] = row_text(prior["headers"],merged)
                        last["source_parts"].extend(lead["source_parts"])
                        lead["continuation_of"] = last["row_id"]
                        joined.append({"row_id":last["row_id"],"pages":[prior_page,page]})
        previous = (page,tables[-1]) if tables else None
    return joined

def reconcile_native_grid(block,page):
    """Use native PDF cell boundaries when present, with model-derived headers.

    The model can collapse unused columns or shift a line into the following row.
    Geometry is a second independent source, not an inferred repair by an LLM.
    """
    x0,y0,x1,y1=block["bbox"]
    candidates=[]
    for table in page.find_tables():
        a,b,c,d=table.bbox
        overlap=max(0,min(c,x1)-max(a,x0))*max(0,min(d,y1)-max(b,y0))
        area=max((c-a)*(d-b),(x1-x0)*(y1-y0))
        if area and overlap/area>.55:
            candidates.append((overlap/area,table))
    if not candidates:
        block["extraction_method"]="layout_model"
        return
    table=max(candidates,key=lambda x:x[0])[1]
    raw_grid=table.extract()
    grid=[]
    for row in raw_grid:
        cells=[]
        for value in row:
            value=value or ""
            if re.search(r"P\s*[=<>].*\n.*(?:setpoint|actual|compensation)",value):
                block["formula_layout_unverified"]=True
                kept=[line for line in value.splitlines() if not re.search(r"P\s*[=<>]|^(?:compensation|setpoint|actual)(?:\s|$)",line.strip())]
                value="\n".join(kept)+"\n[Formula layout requires original PDF]"
            cells.append(clean(value))
        grid.append(cells)
    if not grid:
        return
    grid=[row for row in grid if any(row)]
    if not grid:
        return
    width=len(grid[0])
    model_width=len(block["headers"])
    leaf_names={"pin","signal","no.","nr","error number","error text","ready signal",
                "description","parameter","unit","significance","command","button","function",
                "connection","cause","remedy","action","comment"}
    native_leaf=any(any(v.casefold() in leaf_names for v in row) for row in grid[:2])
    nonempty=[i for i in range(width) if any(row[i] for row in grid)]
    if len(nonempty)==model_width and (block["header_detected"] or native_leaf):
        grid=[[row[i] for i in nonempty] for row in grid]
        width=model_width
    headers=list(block["headers"]) if width==model_width else [f"Unmapped column {i+1}" for i in range(width)]
    if width!=model_width:
        block["header_detected"]=False
    if (grid and any(v.casefold() in leaf_names for v in grid[0])
            and all(not v or len(v)<65 and not re.fullmatch(r"[\d.,]+",v) for v in grid[0])):
        # A model often recognizes the spanning group but misses the leaf header row.
        leaves=grid.pop(0)
        headers=[leaf if header.startswith("Unmapped column") else
                 header if not leaf or leaf.casefold() in header.casefold() else header+" / "+leaf
                 for header,leaf in zip(headers,leaves)]
        block["header_detected"]=True
    if width==1:
        headers=["Text"]
        block["header_detected"]=True
    bit_matrix = "error bits" in (block.get("section","")+" "+block.get("caption","")).casefold()
    bit_matrix = bit_matrix or (model_width>8 and all("error" in h.casefold() for h in block["headers"][:2]))
    if width>8 and bit_matrix:
        # Rotated fieldbus bit headings are visibly merged by the model.
        # Keep identifiable error number/text, not unverified bit assignments.
        block["matrix_unverified"]=True
        headers=list(block["headers"][:2])
        grid=[cells[:2] for cells in grid]
        width=2
        block["header_detected"]=True
    rows=[]
    for index,cells in enumerate(grid):
        if not any(cells):
            continue
        # Rotated monitoring labels are header cells, not an error description.
        if width>=4 and not any(cells[:3]) and any(cells[3:]):
            continue
        if all(not v or v.casefold() in headers[i].casefold() for i,v in enumerate(cells)):
            continue
        for c,header in enumerate(headers):
            if "pneumatic connection" in header.casefold() and re.fullmatch(r"[\d\s]+",cells[c] or ""):
                cells[c]="[Diagram present: see original PDF]"
                block["diagram_cells_uninterpreted"]=True
        if (len(cells)>2 and not cells[0] and not cells[1] and rows
                and re.fullmatch(r"\d+",rows[-1]["cells"][0] or "")):
            prior=rows[-1]
            prior["cells"]=[clean(a+" "+b) for a,b in zip(prior["cells"],cells)]
            prior["text"]=row_text(headers,prior["cells"])
            prior["source_parts"][0]["text"]=prior["text"]
            continue
        text=row_text(headers,cells)
        rows.append({"row_id":f"{block['table_id']}:row:{index}","cells":cells,"text":text,
            "error_number":cells[0] if any("error" in h.casefold() for h in headers) and re.fullmatch(r"\d+",cells[0]) else None,
            "source_parts":[{"doc":block["manual_id"],"page":block["page_number"],"bbox":list(table.bbox),"text":text}]})
    block.update(headers=headers,rows=rows,row_count=len(rows),extraction_method="native_grid_with_layout_headers")

def convert_document(doc_id, document, selected):
    info = manuals()[doc_id]
    pages = {p:[] for p in selected}
    section = ""
    for item,_ in document.iterate_items():
        if not item.prov:
            continue
        prov = item.prov[0]
        p = prov.page_no
        if p not in pages:
            continue
        height = document.pages[p].size.height
        box = prov.bbox.to_top_left_origin(height)
        bbox = [round(v,2) for v in (box.l,box.t,box.r,box.b)]
        label = item.label.value
        if label in {"page_header","page_footer"}:
            continue
        value = clean(getattr(item,"text",""))
        heading = label in {"section_header","title"} and value.casefold() not in {"note","warning","caution","danger","information"}
        if heading:
            section = value
        if label=="table":
            block = convert_table(item,document,doc_id,p,bbox,section)
        elif label=="picture":
            block = {"kind":"image","section":section,"caption":"",
                     "content":f"[Figure present; PDF page {p}]","extractable_text":""}
        elif value:
            block = {"kind":"heading" if heading else "text",
                     "section":section,"content":value}
        else:
            continue
        block.update(manual_id=doc_id,page_number=p,bbox=bbox,page_height=height)
        pages[p].append(block)
    with pdfplumber.open(manual_path(doc_id)) as native:
        for p,items in pages.items():
            for block in items:
                if block["kind"]=="table":
                    # Side-by-side tables can share headers; attach the heading in their own column.
                    headings=[h for h in items if h["kind"]=="heading"
                        and h["bbox"][3] <= block["bbox"][1]+5
                        and max(0,min(h["bbox"][2],block["bbox"][2])-max(h["bbox"][0],block["bbox"][0]))
                            > .5*min(h["bbox"][2]-h["bbox"][0],block["bbox"][2]-block["bbox"][0])]
                    if headings:
                        block["section"]=max(headings,key=lambda h:h["bbox"][3])["content"]
                    reconcile_native_grid(block,native.pages[p-1])
    joined = link_continuations(pages)
    rows,blocks,audits = [],[],[]
    with pdfplumber.open(manual_path(doc_id)) as pdf:
        for p,items in pages.items():
            for order,b in enumerate(items):
                b.update(order=order,source_block_id=f"{doc_id}:page:{p}:block:{order}")
                if b["kind"]=="table":
                    b["content"] = "\n".join([f"[Table {b['table_id']}: {b['caption'] or b['section']}; PDF page {p}]"]+[r["text"] for r in b["rows"]])
                b.setdefault("extractable_text",b["content"])
                unverified = b["kind"]=="table" and (b.get("matrix_unverified") or not b["header_detected"] and len(b["headers"])>1)
                visual = b["kind"]=="image" or b.get("diagram_cells_uninterpreted",False) or b.get("formula_layout_unverified",False)
                b["review_needed"] = bool(visual or unverified)
                b["review_type"] = "table_structure" if unverified else "visual_content" if visual else None
                b["review_reason"] = "rotated_bit_matrix_unverified" if b.get("matrix_unverified") else "unverified_table_header_mapping" if unverified else "manual_visual_reference" if visual else None
            content = "\n".join(b["content"] for b in items)
            raw = pdf.pages[p-1].extract_text() or ""
            missing = sorted((Counter(TECHNICAL.findall(raw))-Counter(TECHNICAL.findall(content))).elements())
            meta = {"source":str(manual_path(doc_id)),"manual_id":doc_id,"page":p-1,"page_number":p,
                    "document_sha256":info["sha256"],"section_ids":[],"parser_version":PARSER_VERSION,
                    "partial_run":len(selected)!=info["pages"],"visual_review_required":any(b["review_needed"] for b in items)}
            rows.append({"doc_id":f"{doc_id}:page:{p}","metadata":meta,"page_content":content,"raw_text":raw})
            blocks.extend(items)
            audits.append({"doc":doc_id,"page":p,"blocks":len(items),"tables":sum(b["kind"]=="table" for b in items),
                           "figures":sum(b["kind"]=="image" for b in items),"missing_technical_tokens":missing,
                           "unmapped_tables":[b["table_id"] for b in items if b["review_type"]=="table_structure"]})
    return rows,blocks,audits,joined

def main():
    parser = argparse.ArgumentParser(description="레이아웃·표 구조를 보존하는 로컬 PDF 파싱")
    parser.add_argument("--doc",choices=list(manuals()))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-pages",action="store_true")
    group.add_argument("--pages")
    parser.add_argument("--refresh",action="store_true",help="로컬 Docling 변환 캐시 재생성")
    args = parser.parse_args()
    if args.pages and not args.doc:
        parser.error("--pages에는 --doc가 필요합니다.")
    verify_manuals()
    os.environ.setdefault("HF_HOME",str(RUNTIME/"cache"/"huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    from ..config import load_environment
    load_environment()
    if os.getenv("RAG_SYSTEM_TRUST")=="1":
        import truststore
        truststore.inject_into_ssl()
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling_core.types.doc import DoclingDocument
    options = PdfPipelineOptions(do_ocr=False,do_table_structure=True,
                                generate_page_images=False,generate_picture_images=False)
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.accelerator_options.num_threads = 4
    converter = DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=options)})
    all_rows,all_blocks,audits,joined = [],[],[],[]
    for doc_id,info in manuals().items():
        if args.doc and doc_id!=args.doc:
            continue
        selected = parse_pages(args.pages,info["pages"]) if args.pages else list(range(1,info["pages"]+1))
        cache = PARSING/"docling"/f"{doc_id}.json"
        saved = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() and not args.refresh else {}
        if saved.get("sha256")==info["sha256"] and saved.get("version")==PARSER_VERSION:
            doc = DoclingDocument.model_validate(saved["document"])
        else:
            print(f"Docling converting {doc_id} ({info['pages']} pages)",flush=True)
            result = converter.convert(manual_path(doc_id))
            if result.status.value!="success":
                raise RuntimeError(f"{doc_id}: Docling 변환 미완료 ({result.status.value})")
            doc = result.document
            cache.parent.mkdir(parents=True,exist_ok=True)
            cache.write_text(json.dumps({"sha256":info["sha256"],"version":PARSER_VERSION,
                                         "document":doc.export_to_dict()},ensure_ascii=False),encoding="utf-8")
        r,b,a,j = convert_document(doc_id,doc,selected)
        all_rows.extend(r);all_blocks.extend(b);audits.extend(a);joined.extend(j)
        print(f"Parsed {doc_id}: {len(r)} pages / {len(b)} blocks",flush=True)
    PARSING.mkdir(parents=True,exist_ok=True)
    for path,items in [(PARSED_DOCS,all_rows),(PARSED_BLOCKS,all_blocks)]:
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf-8")
    (PARSING/"audit.json").write_text(json.dumps({"parser":PARSER_VERSION,"pages":audits,
        "joined_rows":joined},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved {len(all_rows)} pages; joined rows: {len(joined)}",flush=True)

if __name__=="__main__":
    main()
