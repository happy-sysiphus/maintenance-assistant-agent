"""E-code/manual GraphRAG public API. No sensor inputs or stored conversations."""


def answer_question(question, *, error_code=None, history=None, database=None,
                    use_llm=True, use_cache=False):
    from .service import answer_question as answer
    return answer(question,error_code=error_code,history=history,database=database,
                  use_llm=use_llm,use_cache=use_cache)


__all__ = ["answer_question"]