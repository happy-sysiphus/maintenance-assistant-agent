"""GraphRAG service for RSW welding-gun situations S01–S10."""

from typing import Any


def answer_question(
    situation_ids: str | list[str],
    question: str,
    *,
    database: str | None = None,
    use_llm: bool = True,
    use_cache: bool = False,
) -> dict[str, Any]:
    """Retrieve manual evidence and return an answer without writing question history."""
    from .service import answer_question as _answer_question

    return _answer_question(
        situation_ids, question, database=database,
        use_llm=use_llm, use_cache=use_cache,
    )


__all__ = ["answer_question"]
