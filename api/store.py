"""케이스 저장소: SQLite 파일 하나. 케이스 한 건을 FE `CaseDetail` 모양 JSON으로 통째로 저장한다.

사람 기록(records)은 SQLite에 보존한다. WARNING 해결 종료와 Aura 전송 대기 건은
같은 트랜잭션에 기록한다. 재사용할 해결 이력은 공용 Aura에서 조회한다.
FIELD_MEMORY=1이면 api 안에서 이 기록을 다시 읽어 케이스 조회에 이전 기록을 붙인다(field_memory.py).
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(os.environ.get("API_DB", Path(__file__).resolve().parent / "state" / "api.db"))
_lock = threading.Lock()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS cases ("
        " case_id TEXT PRIMARY KEY, gun_id TEXT, code TEXT, status TEXT, detected_at TEXT, data TEXT)"
    )
    # ML이 같은 알림을 다시 보내도(재시도) 한 번만 처리하려고 받은 event_id를 남긴다
    c.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, case_id TEXT)")
    # 챗봇 답 평가 (11/04 FE). 대화는 브라우저에만 있어 질문 · 답을 같이 받는다. 같은 답을 다시 누르면 마지막 것만 남긴다
    c.execute(
        "CREATE TABLE IF NOT EXISTS feedback ("
        " conversation_id TEXT, message_index INTEGER, helpful INTEGER, at TEXT, data TEXT,"
        " PRIMARY KEY (conversation_id, message_index))"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS warning_outbox ("
        " event_id TEXT PRIMARY KEY, case_id TEXT UNIQUE NOT NULL, pattern_key TEXT,"
        " data TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',"
        " attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL DEFAULT 0)"
    )
    try:
        with c:
            yield c
    finally:
        c.close()


def get(case_id: str) -> dict[str, Any] | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT data FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    return json.loads(row[0]) if row else None


def all_cases() -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT data FROM cases").fetchall()
    return [json.loads(r[0]) for r in rows]


def group_key(e: dict[str, Any]) -> str | None:
    """Group warnings separately from code-less critical alarms for the same equipment."""
    return "early_warning" if e.get("event_type") == "early_warning" else e["trigger"].get("rule_code")


def put(d: dict[str, Any], *, warning_delivery=None) -> None:
    e = d["event"]
    code = group_key(e)
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?)",
            (d["case_id"], e["gun_id"], code, d["status"], e["detected_at"], json.dumps(d, ensure_ascii=False)),
        )

        if warning_delivery is not None:
            try:
                c.execute(
                    "INSERT INTO warning_outbox (event_id, case_id, pattern_key, data) VALUES (?, ?, ?, ?)",
                    (warning_delivery["handoff"]["event_id"], d["case_id"], warning_delivery["pattern_key"],
                     json.dumps(warning_delivery, ensure_ascii=False)),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("warning case or event already queued") from exc


def warning_delivery_for_case(case_id):
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT event_id, status FROM warning_outbox WHERE case_id = ?", (case_id,),
        ).fetchone()
    return {"event_id": row[0], "status": row[1]} if row else None


def warning_pending(pattern_key):
    if pattern_key is None:
        return False
    with _lock, _conn() as c:
        return c.execute(
            "SELECT 1 FROM warning_outbox WHERE pattern_key = ? AND status IN ('pending', 'failed') LIMIT 1",
            (pattern_key,),
        ).fetchone() is not None


def claim_warning_delivery(now, lease_until):
    with _lock, _conn() as c:
        row = c.execute(
            """UPDATE warning_outbox SET attempts = attempts + 1, next_attempt_at = ?
            WHERE event_id = (
                SELECT event_id FROM warning_outbox WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY next_attempt_at, rowid LIMIT 1
            )
            RETURNING event_id, attempts, data""",
            (lease_until, now),
        ).fetchone()
    return (row[0], row[1], json.loads(row[2])) if row else None


def finish_warning_delivery(event_id, attempt, status, *, next_attempt_at=0):
    with _lock, _conn() as c:
        c.execute(
            """UPDATE warning_outbox SET status = ?, next_attempt_at = ?
            WHERE event_id = ? AND attempts = ? AND status = 'pending'""",
            (status, next_attempt_at, event_id, attempt),
        )


def seen(event_id: str) -> bool:
    with _lock, _conn() as c:
        return c.execute("SELECT 1 FROM events WHERE event_id = ?", (event_id,)).fetchone() is not None


def mark_seen(event_id: str, case_id: str) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR IGNORE INTO events VALUES (?, ?)", (event_id, case_id))


def put_feedback(d: dict[str, Any]) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?)",
            (d["conversation_id"], d["message_index"], int(d["helpful"]), d["at"], json.dumps(d, ensure_ascii=False)),
        )


def all_feedback() -> list[dict[str, Any]]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT data FROM feedback ORDER BY at DESC").fetchall()
    return [json.loads(r[0]) for r in rows]


def open_case(gun_id: str, code: str | None) -> dict[str, Any] | None:
    """Find an open case with the same equipment and alarm group."""
    with _lock, _conn() as c:
        # Warnings saved before type-aware grouping may still have a null code.
        row = c.execute(
            """SELECT data FROM cases
               WHERE gun_id = ? AND status NOT IN ('resolved', 'unresolved', 'false_alarm')
                 AND (CASE WHEN code IS NULL AND json_extract(data, '$.event.event_type') = 'early_warning'
                           THEN 'early_warning' ELSE code END) IS ?
               ORDER BY detected_at DESC LIMIT 1""",
            (gun_id, code),
        ).fetchone()
    return json.loads(row[0]) if row else None


def next_id() -> str:
    with _lock, _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    return f"CASE-{n + 1:04d}"
