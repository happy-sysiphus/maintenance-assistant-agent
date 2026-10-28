"""Real RAG chatbot; preserve existing UI blocks and include the structured reply.

FIELD_MEMORY=1이면 main이 현장 기억 요약(memory)을 넘긴다. 그때만 답 끝에 현장 기록 문단 하나를 붙인다.
"""

import field_memory
from guidance import answer_question
from rag.response import INLINE_CITE

NOT_FOUND = {"status":"no_evidence","situation_ids":[],"blocks":[],"caution":None,"evidence":[]}


def blocks_from_text(text):
    blocks = []
    for paragraph in (text or "").split("\n\n"):
        if not paragraph.strip():
            continue
        parts, offset = [], 0
        for match in INLINE_CITE.finditer(paragraph):
            if match.start() > offset:
                parts.append({"text":paragraph[offset:match.start()]})
            parts.append({"cite":{"doc":match.group(1),"page":int(match.group(2))}})
            offset = match.end()
        if offset < len(paragraph):
            parts.append({"text":paragraph[offset:]})
        blocks.append({"type":"p","parts":parts})
    return blocks


def answer(question, photos, history, memory=None):
    response = answer_question(question,history=history,use_cache=True)
    caution = list(response["review_notes"])
    if photos:
        caution.append("사진의 부품 상태는 분석하지 않습니다. 답변은 질문과 문서 근거를 사용합니다.")
    blocks = blocks_from_text(response["answer"] or "답변을 생성하지 못했습니다. 다시 시도해 주세요.")
    # 현장 기록은 매뉴얼 근거가 아니라서 evidence · caution · cite에 넣지 않고 문단 하나로만 붙인다
    note = field_memory.chat_note(response.get("error_code"), memory) if memory is not None else None
    if note:
        blocks.append({"type":"p","parts":[{"text":note}]})
    return {
        **response,
        "situation_ids":[response["error_code"]] if response["error_code"] else [],
        "blocks":blocks,
        "caution":" ".join(caution) or None,
        "evidence":[{**e,"title":f"{e['doc']} PDF p.{e['page']}"} for e in response["evidence"]],
    }
