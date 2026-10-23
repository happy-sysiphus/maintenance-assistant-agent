"""케이스 저장소: SQLite 파일 하나. 케이스 한 건을 FE `CaseDetail` 모양 JSON으로 통째로 저장한다.

데모 규모(케이스 수십 개)라 표 하나로 충분하다. 사람 기록(records)은 ML · RAG에 보내지 않는다.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(os.environ.get("API_DB", Path(__file__).resolve().parent / "state" / "api.db"))
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS cases ("
        " case_id TEXT PRIMARY KEY, gun_id TEXT, code TEXT, status TEXT, detected_at TEXT, data TEXT)"
    )
    # ML이 같은 알림을 다시 보내도(재시도) 한 번만 처리하려고 받은 event_id를 남긴다
    c.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, case_id TEXT)")
    return c


def get(case_id: str) -> dict[str, Any] | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT data FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return json.loads(row[0]) if row else None


def all_cases() -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT data FROM cases").fetchall()
    return [json.loads(r[0]) for r in rows]


def put(d: dict[str, Any]) -> None:
    e = d["event"]
    code = e["trigger"].get("rule_code")
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?)",
            (d["case_id"], e["gun_id"], code, d["status"], e["detected_at"], json.dumps(d, ensure_ascii=False)),
        )


def seen(event_id: str) -> bool:
    with _lock, _conn() as c:
        return c.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,)).fetchone() is not None


def mark_seen(event_id: str, case_id: str) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR IGNORE INTO events VALUES (?, ?)", (event_id, case_id))


def open_case(gun_id: str, code: str | None) -> dict[str, Any] | None:
    """같은 설비 · 같은 코드로 아직 끝나지 않은 케이스 (반복 알림을 묶을 곳)."""
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT data FROM cases WHERE gun_id = ? AND code IS ? AND status NOT IN ('resolved', 'unresolved', 'false_alarm')"
            " ORDER BY detected_at DESC LIMIT 1",
            (gun_id, code),
        ).fetchone()
    return json.loads(row[0]) if row else None


def next_id() -> str:
    with _lock, _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    return f"CASE-{n + 1:04d}"
