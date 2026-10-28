"""Four versioned documents and their 1-based PDF citation identities."""

import hashlib
import json
import re
from pathlib import Path
from .paths import DATA, DOCUMENTS_PATH, MAPPING_PATH

ERROR_CODES = ("E012", "E016", "E028", "E029")


def manuals() -> dict:
    rows = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows}


def manual_path(doc: str) -> Path:
    return DATA / manuals()[doc]["path"]


def mapping() -> dict:
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["errors"]


def normalized(text: str) -> str:
    return " ".join(re.sub(r"(?<=\w)-\s+(?=\w)", "", text).split())


def searchable_chunk(text, source_json=None):
    """Visual placeholders are provenance markers, not searchable evidence."""
    units = json.loads(source_json) if source_json else []
    if units and all(unit.get("kind") in {"visual_placeholder", "figure"} for unit in units):
        return False
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)
    cleaned = re.sub(r"(?m)^Section\s+[\d.]+\s*$", "", cleaned)
    return len(re.findall(r"[A-Za-z가-힣]", cleaned)) >= 20


def fingerprint() -> str:
    return hashlib.sha256(DOCUMENTS_PATH.read_bytes() + MAPPING_PATH.read_bytes()).hexdigest()


def verify_manuals() -> None:
    for key, doc in manuals().items():
        path = manual_path(key)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != doc["sha256"]:
            raise ValueError(f"{key}: PDF가 없거나 문서 버전이 변경됐습니다: {path.name}")
