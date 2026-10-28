"""AI 챗봇 답 (임시): RAG 질문 검색이 생기기 전까지, 매핑에서 근거만 골라 답을 조립한다.

ui/src/mocks/handlers.ts의 챗봇 mock을 그대로 옮겼다. 진짜 RAG(LLM)로 바꿀 자리 (ui/docs/07 요청 4).
"""

from __future__ import annotations

import re
from typing import Any

import field_memory
from guidance import CODE_TO_SITUATION, all_situations

OTHER_MANUALS = re.compile(r"(milco|abb|fanuc|yaskawa|화낙|야스카와)", re.I)
FOLLOW_UP = re.compile(r"(그럼|그거|그건|이거|이건|거기|다음|더 자세히|왜 그래|그 다음)")

NOT_FOUND = {
    "status": "no_evidence",
    "situation_ids": [],
    "blocks": [
        {
            "type": "p",
            "parts": [
                {"text": "질문과 맞는 매뉴얼 근거를 찾지 못했습니다. 근거 없이 답하지 않아요. "},
                {"text": "고장 코드(E012 등)나 증상, 부품 이름", "bold": True},
                {"text": "을 넣어 다시 물어봐 주세요."},
            ],
        }
    ],
    "caution": None,
    "evidence": [],
}


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"[^가-힣a-zA-Z0-9]", "", text).lower()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _match(text: str, situations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not text.strip():
        return None
    by_id = {s["situation_id"]: s for s in situations}
    code = re.search(r"E0?(12|16|28|29)\b", text.upper())
    if code:
        return by_id.get(CODE_TO_SITUATION[f"E0{code.group(1)}"])
    err = re.search(r"오류\s*(\d+)", text)
    if err:
        for s in situations:
            if any(d["number"] == int(err.group(1)) for d in s["diagnostics"]):
                return s
    q = _bigrams(text)
    best, best_score = None, 0
    for s in situations:
        name = _bigrams(s["name"])
        body = _bigrams(" ".join([s["definition"], *s["check_sequence"]]))
        score = sum(3 if g in name else 1 if g in body else 0 for g in q)
        if score > best_score:
            best, best_score = s, score
    return best if best_score >= 4 else None


def _pick(question: str, history: list[dict[str, Any]], situations: list[dict[str, Any]]) -> dict[str, Any] | None:
    now = _match(question, situations)
    if now or not FOLLOW_UP.search(question):
        return now
    for h in reversed(history):
        hit = _match(h.get("text", ""), situations) if h.get("role") == "user" else None
        if hit:
            return hit
    return None


def answer(
    question: str, photos: list[Any], history: list[dict[str, Any]], memory: dict[str, Any] | None = None
) -> dict[str, Any]:
    """memory는 field_memory.snapshot (현장 기억을 켰을 때만). 있으면 답 끝에 현장 기록 한 줄을 붙인다."""
    other = OTHER_MANUALS.search(question)
    if other:
        return {
            **NOT_FOUND,
            "blocks": [
                {
                    "type": "p",
                    "parts": [
                        {"text": "지금 연결된 매뉴얼은 "},
                        {"text": "Festo 하나", "bold": True},
                        {"text": f"뿐이라 {other.group(1).upper()} 내용은 찾지 못했습니다. 근거 없이 답하지 않습니다."},
                    ],
                }
            ],
        }
    situations = all_situations()
    s = _pick(question, history, situations)
    if s is None:
        return NOT_FOUND
    blocks: list[dict[str, Any]] = []
    if photos:
        blocks.append(
            {
                "type": "p",
                "parts": [
                    {"text": "사진 속 부품 상태는 "},
                    {"text": "판단하지 않습니다", "bold": True},
                    {"text": ". 사진은 이 대화에 저장해 두었어요."},
                ],
            }
        )
    blocks.append(
        {
            "type": "p",
            "parts": [
                {"text": "상황 정의서의 "},
                {"text": s["name"], "bold": True},
                {"text": " 점검 순서는 다음과 같습니다 "},
                {"cite": {"doc": "rsw", "page": s["check_source_page"]}},
            ],
        }
    )
    blocks.append({"type": "ol", "items": s["check_sequence"]})
    manual = [d for d in s["diagnostics"] if d.get("manual")]
    for d in manual:
        m = d["manual"]
        blocks.append(
            {
                "type": "p",
                "parts": [
                    {"text": f"Festo 매뉴얼 오류 {d['number']}({m['name']}) 원문: “{m['text']}” "},
                    {"cite": {"doc": "festo", "page": m["page"]}},
                ],
            }
        )
    if not manual and s["diagnostics"]:
        nums = ", ".join(str(d["number"]) for d in s["diagnostics"])
        blocks.append(
            {"type": "p", "parts": [{"text": f"관련 Festo 오류 번호 후보: {nums}. 점검 후보일 뿐, 그 오류가 실제로 떴다는 뜻은 아닙니다."}]}
        )
    # 현장 기록은 매뉴얼 근거가 아니라서 cite · evidence에 넣지 않는다
    note = field_memory.chat_note(s["situation_id"], memory) if memory is not None else None
    if note:
        blocks.append({"type": "p", "parts": [{"text": note}]})
    return {
        "status": "retrieval_only",
        "situation_ids": [s["situation_id"]],
        "blocks": blocks,
        "caution": s["coverage_note"],
        "evidence": [
            {"doc": "rsw", "page": s["check_source_page"], "title": f"{s['name']} 정의 · 점검 순서"},
            *({"doc": "festo", "page": e["page"], "title": e["rationale"] or "Festo 매뉴얼"} for e in s["evidence"]),
        ],
    }
