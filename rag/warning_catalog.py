"""Reviewed manual cases and stable page-level references for early warnings."""
from copy import deepcopy

from .documents import manuals, mapping, normalized
from .warning_input import required_text
from .warning_memory import stable_key


def source_ref(doc, page, quote, *, evidence_id=None):
    info = manuals().get(doc) if isinstance(doc, str) else None
    if not info or type(page) is not int or not 1 <= page <= info["pages"]:
        raise ValueError("invalid manual document or PDF page")
    result = {"doc": doc, "page": page, "revision": info["version"], "sha256": info["sha256"],
              "quote": required_text(quote, "manual quote", 2400)}
    if evidence_id is not None:
        result["evidence_id"] = required_text(evidence_id, "evidence_id")
    return result


def validate_manual_selection(selection):
    """Selection is a server-owned source returned by the manual search endpoint."""
    if not isinstance(selection, dict):
        raise ValueError("select a manual search result")
    evidence_id = required_text(selection.get("evidence_id"), "evidence_id")
    result = source_ref(selection.get("doc"), selection.get("page"), selection.get("quote"),
                        evidence_id=evidence_id)
    if any(result[key] != selection.get(key) for key in ("revision", "sha256")):
        raise ValueError("manual selection belongs to a different document revision")
    return result


def _with_cite(item):
    result = deepcopy(item)
    result["cite"] = source_ref(**item["cite"])
    if result.get("criterion"):
        result["criterion"]["cite"] = source_ref(**item["criterion"]["cite"])
    return result


def initial_catalog(*, situations=None):
    cases = {}
    for item in mapping().values() if situations is None else situations:
        for cause in item["causes"]:
            primary = source_ref(**cause["cite"])
            checks = [_with_cite(x) for x in cause["checks"]]
            actions = [_with_cite(x) for x in cause["actions"]]
            docs = {x["cite"]["doc"] for x in checks + actions} | {primary["doc"]}
            docs.update(x["criterion"]["cite"]["doc"] for x in checks if x["criterion"])
            # The diagnostic attachment and mixed legacy guidance require equipment review.
            scope = "cpx-weld-2g-sa" if docs == {"manual82"} else "festo-legacy" if docs == {"festo"} else None
            key = "manual:" + cause["id"]
            cases[key] = {"manual_case_id": key, "title": cause["name"], "basis": cause["why"],
                          "source": primary, "checks": checks, "actions": actions,
                          "equipment_scope": scope, "coverage_note": item["coverage_note"]}
    if situations is not None:
        return cases
    # A separately reviewed source row outside the four E codes. No sensor-to-case rule.
    key = "manual:festo:main-cylinder-us2-load-voltage"
    source = source_ref("festo", 94, "Cause: US2 load voltage off | Remedy: Switch on the load voltage")
    cases[key] = {
        "manual_case_id": key, "title": "메인 실린더가 움직이지 않을 때의 US2 부하 전압 확인",
        "basis": "구판 매뉴얼은 메인 실린더가 움직이지 않는 원인 중 하나로 US2 부하 전압 꺼짐을 제시합니다.",
        "source": source, "checks": [],
        "actions": [{"text": "구판 매뉴얼 조치: 부하 전압을 켭니다.",
                     "when": "적용 장비에서 메인 실린더가 움직이지 않고 US2 부하 전압 꺼짐이 확인된 경우",
                     "cite": source}],
        "equipment_scope": "festo-legacy",
        "coverage_note": "구판 장비의 해당 증상에 적용합니다. 센서 전환만으로 부하 전압 상태를 확정할 수 없습니다.",
    }
    return cases


def case_is_current(case):
    manifest = manuals()
    refs = [case["source"]]
    for step in case["checks"] + case["actions"]:
        refs.append(step["cite"])
        if step.get("criterion"):
            refs.append(step["criterion"]["cite"])
    return all(ref["doc"] in manifest
               and ref["sha256"] == manifest[ref["doc"]]["sha256"]
               and ref["revision"] == manifest[ref["doc"]]["version"]
               and 1 <= ref["page"] <= manifest[ref["doc"]]["pages"] for ref in refs)


def source_matches_evidence(source, evidence):
    return (source["doc"], source["page"]) == (evidence["doc"], evidence["page"]) and (
        normalized(source["quote"]) in normalized(evidence["text"])
    )


def reviewed_case(source, *, title, equipment_scope, checks=(), actions=()):
    """Create a stable reviewed case using exact selected PDF excerpts."""
    selected = validate_manual_selection(source)
    # Chunk IDs are useful for the request trace but are not stable identities.
    selected.pop("evidence_id")
    key = "manual:reviewed:" + stable_key(selected["doc"], selected["revision"],
                                         selected["page"], normalized(selected["quote"]))
    title = required_text(title, "title")
    scope = required_text(equipment_scope, "equipment_scope")
    checked = []
    for kind, values in (("check", checks), ("action", actions)):
        if isinstance(values, str) or not isinstance(values, (list, tuple)) or len(values) > 5:
            raise ValueError(f"{kind}: expected up to five source excerpts")
        steps = []
        for excerpt in values:
            text = required_text(excerpt, kind, 1000)
            if normalized(text) not in normalized(selected["quote"]):
                raise ValueError(f"{kind} must be an excerpt of the selected source")
            steps.append({"text": text, "cite": selected})
        checked.append(steps)
    for index, check in enumerate(checked[0], 1):
        check.update(id=f"{key}:check:{index}", criterion=None)
    for action in checked[1]:
        action["when"] = None
    return {"manual_case_id": key, "title": title, "source": selected, "basis": selected["quote"],
            "equipment_scope": scope, "checks": checked[0], "actions": checked[1],
            "coverage_note": "검토자가 지정한 설비 적용 범위의 매뉴얼 항목입니다."}
