"""Environment configuration shared by the service and ingestion pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """Read rag/.env; values provided by the application take precedence."""
    load_dotenv(ROOT / ".env", override=False)


def get_database(database: str | None = None) -> str:
    load_environment()
    value = database if database is not None else os.getenv("NEO4J_DATABASE", "festomanualv2")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Neo4j 데이터베이스 이름은 비워둘 수 없습니다.")
    return value.strip()
