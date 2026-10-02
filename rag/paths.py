"""Source assets and generated runtime files, independent of the working directory."""

import os
from pathlib import Path

from .config import ROOT, load_environment


load_environment()
DATA = ROOT / "data"
MANUAL_PDF = DATA / "festo_manual.pdf"
RSW_PDF = DATA / "rsw_situations.pdf"
MAPPING_PATH = DATA / "situation_mapping.json"

_runtime = Path(os.getenv("RAG_RUNTIME_DIR", "var")).expanduser()
RUNTIME = _runtime if _runtime.is_absolute() else ROOT / _runtime

PARSING = RUNTIME / "parsed"
PARSED_DOCS = PARSING / "pages.jsonl"
PARSED_BLOCKS = PARSING / "blocks.jsonl"
REVIEW_IMAGES = PARSING / "images"
VISION_CACHE = RUNTIME / "cache" / "vision"
CHUNKS = RUNTIME / "chunks.jsonl"
KG_CACHE = RUNTIME / "cache" / "kg"
ANSWER_CACHE = RUNTIME / "cache" / "answers"
