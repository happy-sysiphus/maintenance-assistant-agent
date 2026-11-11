"""케이스 저장소: SQLite 파일 하나. 케이스 한 건을 FE `CaseDetail` 모양 JSON으로 통째로 저장한다.

데모 규모(케이스 수십 개)라 표 하나로 충분하다. 사람 기록(records)은 SQLite에 보존한다. WARNING 해결 이력은 warning_bridge가 선택적으로 재사용한다.
FIELD_MEMORY=1은 기존 코드별 현장 기억을 제어한다. 별도로 WARNING 해결 이력은 warning_resolutions에 케이스 종료와 함께 저장하며, RAG_WARNING_FEEDBACK=1일 때 warning_bridge를 통해 재사용한다.
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
    c.execute("CREATE TABLE IF NOT EXISTS warning_resolutions (event_id TEXT PRIMARY KEY, case_id TEXT UNIQUE, data TEXT)")
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
    """케이스를 묶는 키 (cases.code 칸): 종료 코드. 코드 없는 조기 경보는 "early_warning", 코드 없는 모델 단독 critical은 None.

    조기 경보와 모델 단독 critical이 같은 설비의 열린 케이스 하나에 섞이지 않게 한다 (ml D-9 스펙 11.1).
    """
    code = e["trigger"].get("rule_code")
    if code:
        return code
    return "early_warning" if e.get("event_type") == "early_warning" else None


def put(d: dict[str, Any], warning_resolution=None) -> None:
    e = d["event"]
    code = group_key(e)
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO cases VALUES (?, ?, ?, ?, ?, ?)",
            (d["case_id"], e["gun_id"], code, d["status"], e["detected_at"], json.dumps(d, ensure_ascii=False)),
        )

        if warning_resolution is not None:
            existing = c.execute("SELECT data FROM warning_resolutions WHERE event_id = ? OR case_id = ?",
                                 (warning_resolution['event_id'], d['case_id'])).fetchone()
            payload = json.dumps(warning_resolution, ensure_ascii=False, sort_keys=True)
            if existing and existing[0] != payload:
                raise ValueError('immutable warning resolution conflict')
            c.execute("INSERT OR IGNORE INTO warning_resolutions VALUES (?, ?, ?)",
                      (warning_resolution['event_id'], d['case_id'], payload))


def warning_resolutions():
    with _lock, _conn() as c:
        return [json.loads(r[0]) for r in c.execute("SELECT data FROM warning_resolutions")]


def warning_resolution_for_case(case_id):
    return next((r for r in warning_resolutions() if r['case_id'] == case_id), None)


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
    """같은 설비 · 같은 묶음 키(group_key)로 아직 끝나지 않은 케이스 (반복 알림을 묶을 곳)."""
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


def review_warning_resolution(value):
    """Only review metadata may change after closure; actions and event stay immutable."""
    with _lock, _conn() as c:
        row = c.execute('SELECT data FROM warning_resolutions WHERE event_id=?', (value['event_id'],)).fetchone()
        if not row:
            raise ValueError('resolution not found')
        original = json.loads(row[0])
        immutable = lambda r: {k: v for k, v in r.items() if k not in ('manual_case', 'counterexample_reviewed')}
        if immutable(original) != immutable(value):
            raise ValueError('cannot alter recorded resolution')
        c.execute('UPDATE warning_resolutions SET data=? WHERE event_id=?',
                  (json.dumps(value, ensure_ascii=False, sort_keys=True), value['event_id']))
