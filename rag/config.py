"""Environment configuration shared by the service and ingestion pipeline."""

import os
import ssl
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """Read rag/.env; values provided by the application take precedence."""
    load_dotenv(ROOT / ".env", override=False)


def ssl_context() -> ssl.SSLContext:
    """Use OS trust when requested; certificate verification stays enabled."""
    load_environment()
    if os.getenv("RAG_SYSTEM_TRUST", "0") == "1":
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context()


def graph_driver():
    from neo4j import GraphDatabase
    load_environment()
    uri = os.getenv("NEO4J_URI", "")
    username, password = os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise RuntimeError("rag/.env의 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD가 필요합니다.")
    # Explicit SSLContext is supported with the unsuffixed encrypted scheme.
    options = {"connection_timeout": 15, "max_transaction_retry_time": 5,
               "notifications_disabled_classifications": ["DEPRECATION"]}
    if uri.startswith(("neo4j+s://", "bolt+s://")):
        uri = uri.replace("+s://", "://", 1)
        options["ssl_context"] = ssl_context()
    return GraphDatabase.driver(uri, auth=(username, password), **options)


def openai_client():
    import httpx
    from openai import OpenAI
    load_environment()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("rag/.env의 OPENAI_API_KEY가 필요합니다.")
    return OpenAI(timeout=90, max_retries=0,
                  http_client=httpx.Client(verify=ssl_context()))


def get_database(database: str | None = None) -> str:
    load_environment()
    value = database if database is not None else os.getenv("NEO4J_DATABASE", "neo4j")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Neo4j 데이터베이스 이름은 비워둘 수 없습니다.")
    return value.strip()
