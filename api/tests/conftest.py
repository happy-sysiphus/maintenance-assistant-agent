"""api 테스트 공용: api 폴더를 import 경로에 넣고, 핸드오프 · 케이스 모양과 격리된 TestClient를 만든다.

RAG 호출(answer_question)은 가짜로 바꾼다. rag/.env가 있어도 OpenAI · Neo4j에 가지 않는다.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assistant  # noqa: E402
import guidance  # noqa: E402
import main  # noqa: E402
import store  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from rag.documents import mapping  # noqa: E402
from rag.response import mapped_causes  # noqa: E402

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

# 종료 코드 → 고장 분류 (이름은 아무거나, 테스트는 이름에 기대지 않는다)
_CLASSES = {"E012": "E01", "E016": "E02", "E028": "E03", "E029": "E04"}


def fake_answer_question(
    question: str, *, error_code: str | None = None, history: Any = None, use_cache: bool = False, **kw: Any
) -> dict[str, Any]:
    """rag.answer_question 대역: 매핑의 네 코드는 ok, 그 밖은 no_evidence. 같은 입력이면 같은 답."""
    code = error_code
    if code is None:
        found = re.search(r"E\d{3}", question.upper())
        code = found.group(0) if found else None
    item = mapping().get(code) if code else None
    if item is None:
        return {
            "status": "no_evidence",
            "error_code": code,
            "answer": "질문에 맞는 매뉴얼 근거를 찾지 못했습니다.",
            "sections": {"interpretation": "", "summary": ""},
            "causes": [],
            "evidence": [],
            "review_notes": [],
        }
    evidence = [
        {"doc": r["doc"], "page": r["page"], "quote": r["quote"], "text": r["quote"], "review_needed": False}
        for r in item["definitions"]
    ]
    first = evidence[0]
    return {
        "status": "ok",
        "error_code": code,
        "answer": f"{item['definition']} [{first['doc']} p.{first['page']}]\n\n요약: 테스트 답",
        "sections": {"interpretation": item["definition"], "summary": "테스트 답"},
        "causes": mapped_causes(code, item),
        "evidence": evidence,
        "review_notes": [item["coverage_note"]],
    }


@pytest.fixture
def make_handoff():
    """api가 읽는 필드만 담은 v1.2 핸드오프. source="model"이면 규칙 코드 없음, 분류는 code가 있을 때만."""

    def _make(
        event_id: str,
        gun_id: str = "test_0",
        code: str | None = "E029",
        t: str = "2021-09-06T19:30:09Z",
        source: str = "rule",
    ) -> dict[str, Any]:
        rule_code = code if source == "rule" else None
        fault_class = {"code": _CLASSES.get(code or "", "E00"), "name_ko": "테스트 고장", "terminal_code": code}
        return {
            "schema_version": "1.2",
            "event_id": event_id,
            "gun_id": gun_id,
            "detected_at": t,
            "window_start": t,
            "trigger": {"source": source, "rule_code": rule_code, "rule_trigger_time": t if rule_code else None},
            "fault_class": fault_class if code else None,
            "summary_ko": "테스트 알림",
            "sensor_findings": [],
            "error_timeline": [],
            "sensor_trend": [],
            "caveats": [],
            "context": {},
        }

    return _make


@pytest.fixture
def make_case():
    """main.ingest가 만드는 케이스 모양 (안내는 guidance.guidance_for, E 코드 카드 하나).

    judgments · results는 (E 코드 | "MANUAL", verdict | outcome) 쌍, 시각은 순서대로 붙인다.
    """

    def _make(
        case_id: str,
        handoff: dict[str, Any],
        closure: str | None = None,
        judgments: Iterable[tuple[str, str]] = (),
        results: Iterable[tuple[str, str]] = (),
        checks: dict[str, Any] | None = None,
        log: dict[str, Any] | None = None,
        manual: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        records: dict[str, Any] = {
            "checks": {},
            "judgments": [],
            "actions": [],
            "results": [],
            "field": [],
            "handovers": [],
        }
        for i, (sid, verdict) in enumerate(judgments):
            records["judgments"].append({"situation_id": sid, "verdict": verdict, "at": f"2021-09-06T20:{i:02d}:00Z"})
        for i, (sid, outcome) in enumerate(results):
            records["results"].append({"situation_id": sid, "outcome": outcome, "at": f"2021-09-06T21:{i:02d}:00Z"})
        if checks is not None:
            records["checks"] = checks
        if closure is not None:
            records["closure"] = {"outcome": closure, "at": "2021-09-06T22:00:00Z"}
        if log is not None:
            records["log"] = log
        if manual is not None:
            records["manual"] = manual
        return {
            "case_id": case_id,
            "status": closure or "new",
            "severity": "critical",
            "assignee": None,
            "event": handoff,
            "score_trace": [],
            "guidance": guidance.guidance_for(handoff),
            "records": records,
            "repeats": [],
        }

    return _make


@pytest.fixture
def api(monkeypatch, tmp_path) -> TestClient:
    """임시 DB를 쓰고 ML 서버는 꺼진 것처럼(502), RAG는 가짜 답(fake_answer_question)으로 만든 TestClient."""
    # starlette testclient는 import할 때 DeprecationWarning을 내서, 쓰는 테스트에서만 부른다
    from fastapi.testclient import TestClient

    def _ml_down(path: str) -> Any:
        raise HTTPException(502, "ML 서버 없음 (테스트)")

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(main, "ml_get", _ml_down)
    monkeypatch.setattr(main, "_last_sync", 0.0)
    monkeypatch.setattr(guidance, "answer_question", fake_answer_question)
    monkeypatch.setattr(assistant, "answer_question", fake_answer_question)
    return TestClient(main.app)
