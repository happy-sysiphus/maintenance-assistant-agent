"""케이스 안내(guidance): 고장 코드 → 상황 ID → RAG 매핑의 점검 순서 · 근거 쪽.

RAG 서버 없이 레포의 `rag/data/situation_mapping.json`을 직접 읽는다 (RAG의 "근거만 조회"와 같은 수준).
ML 핸드오프 v1.2부터 상황 ID가 오지 않아서, 종료 코드 → 상황 ID는 아래 임시 표로 바꾼다.
RAG 담당과 입력 방식이 정해지면(ui/docs/07 요청 1) 이 파일만 고친다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MAPPING = REPO / "rag" / "data" / "situation_mapping.json"

# 종료 코드 → 상황 ID (ml/docs/sensor_features_for_rag.md: S01~S04는 E01~E04와 같음). 임시 표.
CODE_TO_SITUATION = {"E012": "S01", "E016": "S02", "E028": "S03", "E029": "S04"}

# 상황 이름 · 정의 (rag/pipeline/build.py의 표를 그대로 옮김)
SITUATIONS = {
    "S01": ("보정 압력 도달 지연", "설정한 보정 압력에 1800 ms 안에 도달하지 못함.", 5),
    "S02": ("전극 파손", "전극 위치가 기준 이동의 영점보다 작음(캡이 빠지거나 깨져 더 많이 들어감).", 6),
    "S03": ("원치 않는 이동", "실제 위치가 실린더 스트로크의 6.5% 넘게 목표에서 벗어남.", 6),
    "S04": ("드리프트", "잠긴 실린더가 멈추지 않고 분당 5 mm 넘게 움직임.", 6),
    "S05": ("공기 공급 부족·누설로 전극 힘 저하", "공기 공급 부족 또는 누설로 전극 힘이 낮거나 힘 도달이 늦어지는 원인 상황.", 7),
    "S06": ("기계 마찰·걸림·윤활 부족", "기계 마찰, 걸림 또는 윤활 부족이 의심되는 원인 상황.", 7),
    "S07": ("전극 캡 마모·드레싱 불량", "전극 캡 마모 또는 드레싱 불량이 의심되는 원인 상황.", 7),
    "S08": ("축 보정 틀어짐·간섭에 의한 위치 이탈", "축 보정 틀어짐 또는 간섭으로 목표 위치에서 벗어나는 원인 상황.", 8),
    "S09": ("냉각수·변압기·용접 제어기 등 주변 설비 이상", "건 자체는 정상으로 움직이나 주변 설비 이상이 의심되는 원인 상황.", 8),
    "S10": ("설정값 변경·잔고장 반복", "설정값 변경이나 반복되는 잔고장을 구분해야 하는 운영 상황.", 8),
}

# Festo 매뉴얼 원문을 확인한 오류 행 (ui/docs/03 "매뉴얼 인용"). 확인한 것만 넣는다.
FESTO_ROWS = {
    12: {
        "name": "Timeout equalizer",
        "page": 90,
        "text": "Equalizer pressure was not reached within 2 seconds. Check cable to the MPYD. Check supply pressure at least 5 bar.",
    },
}


def _mapping() -> dict[str, Any]:
    return json.loads(MAPPING.read_text(encoding="utf-8"))["situations"]


def situation(sid: str) -> dict[str, Any] | None:
    """상황 하나: 이름 · 정의 + 매핑(점검 순서 · 근거 · 오류 번호). FE /api/situations 모양."""
    m = _mapping().get(sid)
    if m is None or sid not in SITUATIONS:
        return None
    name, definition, def_page = SITUATIONS[sid]
    return {
        "situation_id": sid,
        "name": name,
        "definition": definition,
        "definition_page": def_page,
        "check_sequence": m["check_sequence"],
        "check_source_page": m["check_source_page"],
        "coverage_note": m.get("coverage_note"),
        "evidence": [{"page": e["page"], "rationale": e.get("rationale")} for e in m.get("evidence", [])],
        "diagnostics": [
            {**d, **({"manual": FESTO_ROWS[d["number"]]} if d["number"] in FESTO_ROWS else {})}
            for d in m.get("diagnostics", [])
        ],
    }


def all_situations() -> list[dict[str, Any]]:
    return [s for sid in SITUATIONS if (s := situation(sid))]


def candidate(sid: str, rank: int) -> dict[str, Any] | None:
    """원인 후보 카드 하나 (FE `Candidate`). 점검 항목 id는 RAG에 없어 `S01-K1`처럼 붙인다."""
    s = situation(sid)
    if s is None:
        return None
    m = _mapping()[sid]
    return {
        "situation_id": sid,
        "rank": rank,
        "name": s["name"],
        "definition": s["definition"],
        "coverage_note": s["coverage_note"],
        "checks": [{"id": f"{sid}-K{i + 1}", "title": t} for i, t in enumerate(s["check_sequence"])],
        "diagnostics": s["diagnostics"],
        "evidence": [
            {"page": e["page"], "quote": e.get("contains"), "rationale": e.get("rationale")} for e in m.get("evidence", [])
        ],
    }


def guidance_for(event: dict[str, Any]) -> dict[str, Any]:
    """고장 코드가 있으면 그 상황 하나를 후보로. 코드가 없으면(모델 신호만) 근거 없음 → 화면은 수동 모드."""
    code = (event.get("trigger") or {}).get("rule_code") or (event.get("fault_class") or {}).get("terminal_code")
    sid = CODE_TO_SITUATION.get(code or "")
    c = candidate(sid, 1) if sid else None
    if c is None:
        return {"status": "no_evidence", "candidates": []}
    return {"status": "retrieval_only", "candidates": [c]}
