"""UI response schema, citation validation, and deterministic text rendering."""

import re
from copy import deepcopy
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .documents import ERROR_CODES

Code = Literal["E012", "E016", "E028", "E029"]
Doc = Literal["festo", "manual82", "diagnostic4", "whitepaper11"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cite(Record):
    doc: Doc
    page: int = Field(ge=1)


class Criterion(Record):
    text: str
    cite: Cite


class Check(Record):
    id: str | None
    text: str
    cite: Cite
    criterion: Criterion | None


class Action(Record):
    when: str | None
    text: str
    cite: Cite


class Cause(Record):
    situation_id: Code | None
    name: str
    why: str
    checks: list[Check]
    actions: list[Action]


class Sections(Record):
    interpretation: str
    summary: str


class GeneratedAnswer(Record):
    sections: Sections
    causes: list[Cause]


class AnswerValidationError(ValueError):
    def __init__(self, code, message, path):
        super().__init__(message)
        self.code, self.path = code, path


def generation_schema(error_code, item=None):
    """The model selects registered IDs; the application owns stored checks."""
    schema = GeneratedAnswer.model_json_schema()
    cause = schema["$defs"]["Cause"]
    cause["properties"]["situation_id"] = (
        {"type":"string","enum":[error_code]} if error_code else {"type":"null"})
    if error_code:
        ids = [c["id"] for ca in (item or {}).get("causes",[]) for c in ca["checks"]]
        schema["$defs"]["Check"] = {
            "type":"object", "properties":{"id":{"type":"string","enum":ids}},
            "required":["id"], "additionalProperties":False,
        }
    else:
        schema["$defs"]["Check"]["properties"]["id"] = {"type":"null"}
    return schema


def assemble_checks(payload, error_code, item=None):
    if not error_code:
        return payload
    parsed = deepcopy(payload)
    canonical = {c["id"]:c for ca in mapped_causes(error_code,item) for c in ca["checks"]}
    seen = set()
    for i, cause in enumerate(parsed.get("causes",[])):
        checks = []
        for j, selected in enumerate(cause.get("checks",[])):
            key = selected.get("id")
            if key not in canonical:
                raise AnswerValidationError("unknown_check_id","매핑에 없는 점검 ID입니다.",
                                            f"causes[{i}].checks[{j}].id")
            if key not in seen:
                checks.append(deepcopy(canonical[key]))
                seen.add(key)
        cause["checks"] = checks
    return parsed


def clean_cite(ref):
    return {"doc": ref["doc"], "page": ref["page"]}


def mapped_causes(code, item):
    """Stable fallback also supplies deterministic check IDs to the UI."""
    causes = []
    for cause in item["causes"]:
        causes.append({
            "situation_id": code, "name": cause["name"], "why": cause["why"],
            "checks": [{"id": c["id"], "text": c["text"], "cite": clean_cite(c["cite"]),
                        "criterion": {"text": c["criterion"]["text"], "cite": clean_cite(c["criterion"]["cite"])}
                        if c["criterion"] else None} for c in cause["checks"]],
            "actions": [{**a, "cite": clean_cite(a["cite"])} for a in cause["actions"]],
        })
    return causes


def iter_cites(value):
    if isinstance(value, dict):
        if set(value) == {"doc", "page"}:
            yield value
        else:
            for child in value.values():
                yield from iter_cites(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_cites(child)


INLINE_CITE = re.compile(r"\[(festo|manual82|diagnostic4|whitepaper11) p\.(\d+)\]")


def validate_answer(payload, evidence, error_code, item=None):
    parsed = GeneratedAnswer.model_validate(payload).model_dump()
    allowed = {(e["doc"], e["page"]) for e in evidence}
    for cite in iter_cites(parsed):
        if (cite["doc"], cite["page"]) not in allowed:
            raise AnswerValidationError("citation_not_retrieved","조회하지 않은 문서·페이지의 인용입니다.","cite")
    for doc, page in INLINE_CITE.findall(str(parsed)):
        if (doc, int(page)) not in allowed:
            raise AnswerValidationError("citation_not_retrieved","본문에 조회하지 않은 인용이 있습니다.","sections/why")
    valid_ids = {c["id"] for cause in (item or {}).get("causes", []) for c in cause["checks"]}
    seen = set()
    for i, cause in enumerate(parsed["causes"]):
        if cause["situation_id"] != error_code:
            raise AnswerValidationError("wrong_error_code","응답 상황 코드는 입력 E 코드 또는 null과 일치해야 합니다.",f"causes[{i}].situation_id")
        for j, check in enumerate(cause["checks"]):
            if error_code and check["id"] not in valid_ids:
                raise AnswerValidationError("unknown_check_id","매핑에 없는 점검 ID입니다.",f"causes[{i}].checks[{j}].id")
            if check["id"] and check["id"] in seen:
                raise AnswerValidationError("duplicate_check_id","중복 점검 ID입니다.",f"causes[{i}].checks[{j}].id")
            if check["id"]:
                seen.add(check["id"])
    return parsed


def cite_text(cite):
    return f"[{cite['doc']} p.{cite['page']}]"


def render_answer(payload):
    """The explanatory string is a view of the same JSON, never a second answer."""
    sections = payload["sections"]
    parts = [sections["interpretation"]]
    for cause in payload["causes"]:
        parts.append(f"원인·확인 후보: {cause['name']}\n{cause['why']}")
        for check in cause["checks"]:
            line = f"- {check['text']} {cite_text(check['cite'])}"
            if check["criterion"]:
                criterion = check["criterion"]
                line += f"\n  기준·확인 방법: {criterion['text']} {cite_text(criterion['cite'])}"
            parts.append(line)
        for action in cause["actions"]:
            when = f"{action['when']}: " if action["when"] else ""
            parts.append(f"- 조치: {when}{action['text']} {cite_text(action['cite'])}")
    if sections["summary"]:
        parts.append("요약: " + sections["summary"])
    return "\n\n".join(part for part in parts if part)
