# 현장 기록 점검 순서(R2) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 해결 종료된 케이스의 '이상' 점검 기록으로, api가 다음 케이스 카드의 점검 순서를 바꾼다(`FIELD_MEMORY=1`일 때만).

**Architecture:**
- 규칙은 `api/field_memory.py`의 순수 함수다: `order_for`, `apply_order`, `first_check`, `order_pinned`.
- `api/main.py`의 ingest는 새 케이스에서 순서를 한 번 정하고, 최상위 `field_order`와 옮긴 카드를 저장한다.
- get_case는 RAG 카드가 오면 같은 순서를 다시 적용한다. 판단 · 조치 · 결과가 기록된 케이스는 카드를 고정한다.
- annotate는 `R2`와 문장을 붙인다. 드라이버는 카드의 점검 순서를 새 키 `card_checks`로 보인다.

**Tech Stack:** Python 3.11, FastAPI, pytest, LangGraph(드라이버), ruff

**Spec:** `ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md` (사용자 승인 11-07). 이 계획은 스펙을 다시 쓰지 않고 순서와 결정만 적는다.
- 규칙: 3절
- 수명 주기: 4절
- 문장: 5.2
- 드라이버: 5.4
- 시험 목록: 7절
- 시연 대본: 8절
- 다른 파트 요청: 11절

## Global Constraints

작업 범위:

- **main 작업 트리에서 바로 고친다**(사용자 동의 11-07: "main에다 바로 커밋할거니까 바로 올릴수있는 상태로 만들어둬").
- 커밋 · 푸시 · 머지 · PR은 하지 않는다. 사용자가 커밋한다. 다른 브랜치나 worktree를 만들지 않는다.
- 다른 파트 코드는 건드리지 않는다: `rag/`, `ui/`, `api/guidance.py`, `api/assistant.py`. `api/store.py`는 4행의 설명 한 줄만 고친다.

플래그와 비용:

- api의 새 동작은 모두 `FIELD_MEMORY=1`일 때만 생긴다.
- 플래그를 끄고 만든 DB에서는 응답이 HEAD 판과 바이트 단위로 같아야 한다. 예외는 `field_order` 키가 있는 행뿐이다(스펙 4.5).
- 유료 API는 부르지 않는다. `rag/.env`를 읽거나 출력하지 않는다. 시험은 가짜 RAG만 쓴다.

환경:

- 8 GB PC다. 서버를 띄우지 않는다(TestClient만 쓴다). 자기가 띄운 프로세스만 끝내고, 사용자 프로세스는 건드리지 않는다.
- 작업 트리 파일은 CRLF다(`core.autocrlf=true`). 고친 파일은 CRLF를 지키고, 새 파일도 CRLF로 쓴다.
- 줄 길이는 120이다. 기존 ruff 오류는 `ml/tests/test_sensor_features.py:14`의 E402 하나뿐이다.

글자 그대로 쓸 값:

- 상수: `ORDER_LEAD = 3`, `ORDER_GUNS = 2`
- R2 문장(스펙 5.2):
  `현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 {lead} {n}건 · {다른 항목} {m}건(설비 {g}대)이라 {lead}부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로.`
  - 다른 항목은 M 순서로 ` · `를 사이에 두고 이어 쓴다.
  - E016 예: `E016-K3 4건 · E016-K1 0건 · E016-K2 1건(설비 2대)이라`
- 지난 점검 기록 문장의 끝: R2이면 `(참고).`, 아니면 지금 그대로 `(참고, 점검 순서는 매뉴얼 기준).`
- summary.md 한 줄: `작업자 판정은 시연 대본이다. 성능이 아니다.`

인터프리터와 neo4j 막이:

| 이름 | 실행 파일 | 있는 것 · 없는 것 |
|---|---|---|
| API | 저장소 `.venv\Scripts\python.exe` | neo4j 있음, langgraph · pandas 없음. `api/tests/test_driver_contract.py`가 건너뛰어진다 |
| DRV | `py -3.11` | langgraph · pandas 있음, neo4j 없음. api를 불러오는 시험은 neo4j 막이를 `PYTHONPATH`에 둔다 |

- neo4j 막이는 저장소 밖에 둔다: `C:\Users\a0106\AppData\Local\Temp\claude\8ff4f6213c18eb00\ce2a4160-2060-4906-8df3-24cd64ce3986\scratchpad\task3\stubs\neo4j\__init__.py`
- 막이 내용:

```python
"""Import-only stub so py -3.11 (langgraph, pandas, no neo4j) can import rag.service. Any real use fails."""
READ_ACCESS = "READ"


class GraphDatabase:
    @staticmethod
    def driver(*a, **k):
        raise RuntimeError("neo4j stub")
```

## Review Focus

1. **RAG 카드가 늦게 올 때.** 첫 조회에서 RAG가 실패한 뒤 판단을 기록하면, RAG가 돌아와도 카드가 바뀌지 않아야 한다. 이때 RAG도 부르지 않는다. Task 3 `test_r2_pin_after_judgment`.
2. **조치가 있는 원인.** 매핑 대체 E029 카드는 옮기지 않는다. 이때 `field_order`는 저장되지만 R2와 문장은 없다. Task 1 `test_apply_order_layouts`, Task 3 `test_r2_script_mapping_card`.
3. **플래그를 끈 DB.** 응답 바이트, 저장된 guidance, 키 목록이 HEAD 판과 같아야 한다. Task 3 `test_flag_off_no_field_order`, Task 6 비교.
4. **쓰기 응답.** 점검 · 판단 · 조치 · 결과 · 일지 저장 응답에도 옮긴 카드와 `field_order`가 실린다. 화면 캐시가 쓰기 응답으로 바뀌기 때문이다. Task 3 `test_r2_write_responses`.
5. **문장 섞임.** R1, R2, 지난 점검 기록이 한 `note_ko`에 스펙 순서대로 들어간다. Task 2 `test_r1_and_r2_together`.

---

### Task 1: R2 규칙 함수

**Files:**
- Modify: `api/field_memory.py` (상수와 함수 추가. 매핑은 `guidance.mapping()`으로 읽는다)
- Test: `api/tests/test_field_memory.py`

**Interfaces:**
- Produces:
  - `ORDER_LEAD: int = 3`, `ORDER_GUNS: int = 2`
  - `mapping_checks(code: str | None) -> list[str]`: 스펙의 M(매핑 원인 순, 그 안의 점검 순). 매핑에 없는 코드면 `[]`.
  - `order_counts(code: str, cases: list[dict]) -> tuple[dict[str, int], dict[str, set[str]]]`: M의 모든 id에 대해 확인 수(M 순서 dict)와 그 확인이 나온 설비 집합(스펙 3.1).
  - `order_for(code: str | None, cases: list[dict]) -> dict | None`: `{"lead": str, "counts": dict[str, int], "guns": int, "note_ko": str}` 또는 None(스펙 3.2).
  - `apply_order(g: dict, lead: str | None) -> dict`: 스펙 3.3.
    - lead가 None이거나 옮길 수 없으면 g 자체를 돌려준다.
    - 옮기면 deepcopy 위에서 고친 새 dict를 돌려준다.
  - `first_check(g: dict) -> str | None`: 보이는 첫 점검 id. `g["causes"]`의 첫 원인의 첫 점검이고, 원인 목록이 비면 `candidates[0].checks[0]`이다. 없으면 None.
  - `order_pinned(case: dict) -> bool`: `case.get("field_order")`가 있고, `records`의 `judgments` · `actions` · `results` 중 하나라도 비어 있지 않음.

- [ ] **Step 1: 시험을 쓴다.** 파일 안 도우미(K1 · K2 · K3는 `f"{code}-K1"` 꼴):
  - `_resolved(make_case, make_handoff, cid, gun, marks, code="E029", decisive=None, source="rule")`
    - `closure="resolved"`
    - judgment와 result는 `decisive or f"{code}-1"`에 `yes` · `resolved`
    - `checks={id: {"result": r} for id, r in marks.items()}`
  - `_card(make_handoff, code, groups)`
    - `guidance.guidance_for(make_handoff(...))`를 받아 원인을 `groups`로 다시 만든다. `groups`는 (매핑 원인 id, 점검 id 목록) 쌍의 목록이다.
    - 원인의 `situation_id` · `name` · `why` · `actions`는 `rag.response.mapped_causes`에서 그 매핑 원인의 것을 복사한다.
    - 점검은 같은 함수의 점검 dict를 id로 골라 넣는다.
    - 결과를 `g["causes"]`와 `g["candidates"][0]["causes"]`에 넣는다.
  - `_layout(g) -> list[list[str]]`: `g["causes"]`의 원인별 점검 id.

  시험:
  1. `test_order_counts_only_resolved_abnormal`
     - 세는 케이스: g1 · g2 해결 종료, 둘 다 K2 이상.
     - 빼는 케이스(모두 K2 이상): 오알람, 미해결, 도움 요청, 열림, `decisive="MANUAL"` 해결.
     - 빼는 기록: 해결 종료의 K2 정상, K2 건너뜀, dict가 아닌 기록, `E029-K9` 이상, E012 케이스의 `E012-K2` 이상.
     - 기대: `order_counts("E029", cases) == ({K1: 0, K2: 2}, {K1: set(), K2: {"g1", "g2"}})`
  2. `test_order_model_only_counts_as_terminal_code`: `source="model"` 해결 종료 케이스의 K2 이상이 E029로 센다.
  3. `test_order_for_lead_rule` (parametrize, 항목 하나 = 그 점검만 이상인 해결 종료 케이스 하나)
     - `[(g1,K2),(g2,K2),(g1,K2)]` → lead K2, counts `{K1: 0, K2: 3}`, guns 2
     - `[(g1,K2),(g2,K2)]` → None
     - 위 3건 + `(g1,K1)` → None
     - 위 4건 + `(g2,K2)` → lead K2, counts `{K1: 1, K2: 4}`
     - K1 3건 · K2 3건(2대) → None
     - `(g1,K2)` 3건 → None(1대)
     - K1 5건(2대) → None(M의 첫 항목)
  4. `test_order_for_e016_three_checks`
     - K3 4건(g1, g2) + K2 1건 → lead K3
     - K2 3건(g1, g2) + K3 3건(g1, g2) → lead K2
     - K3 5건(g1만) + K2 3건(g1, g2) → None
  5. `test_order_for_note_and_shape`
     - 3번의 첫 줄 결과가 `{"lead": K2, "counts": {K1: 0, K2: 3}, "guns": 2, "note_ko": <R2 문장>}`와 같다.
     - `list(counts) == [K1, K2]`
     - `order_for("E999", …)`와 `order_for(None, …)`은 None이다.
  6. `test_apply_order_layouts` (parametrize: 코드, groups, lead → 기대 `_layout`)
     - E029 `[("E029-C1",[K1,K2]),("E029-C2",[])]`, K2 → `[[K2,K1],[]]`. 원인 1에 재시작 주의 조치가 남는다.
     - E029 `[("E029-C1",[K1]),("E029-C2",[K2])]`, K2 → 바뀌지 않는다(`apply_order(g, K2) is g`, 조치 보호).
     - E016 `[("E016-C1",[K1,K2]),("E016-C2",[K3])]`, K3 → `[[K3],[K1,K2]]`. 같은 카드에서 K2 → `[[K2,K1],[K3]]`.
     - E012 `[("E012-C1",[K1]),("E012-C2",[K2])]`, K2 → `[[K2],[K1]]`
     - E012 `[("E012-C1",[K1,K2])]`, K2 → `[[K2,K1]]`
     - E012 `[("E012-C1",[K1,K2]),("E012-C2",[K2])]`, K2 → `[[K2,K1],[K2]]`(첫 원인만)
     - E029 `[("E029-C1",[K1])]`, K2 → 바뀌지 않는다(`candidates`까지)
     - 옮긴 경우: `candidates[0]["causes"]`의 layout이 같고, `candidates[0]["checks"][0]["id"] == lead`다.
  7. `test_apply_order_keeps_content` (E029 캐시 모양 카드)
     - 호출 뒤에도 입력이 deepcopy와 같다(입력을 바꾸지 않는다).
     - 두 번 적용한 결과가 한 번 적용한 결과와 같다.
     - 원인 dict · 점검 dict를 정렬해 비교하면 같다(위치만 다르다).
     - `status` · `response_version` · `answer`는 그대로다.
     - `apply_order(g, None) is g`
  8. `test_apply_order_empty_causes_and_missing_checks`
     - `g["causes"] = []`이면 `candidates[0]["checks"]`만 lead가 맨 앞이다.
     - `[{"name": "x"}, <E029-C2 + K2>]`(`checks`와 `actions`가 없는 원인이 앞): 오류 없이 `[[], …]` → `[[K2], []]`로 옮긴다.
  9. `test_first_check_and_order_pinned`
     - `first_check`: 원인이 있을 때, 원인 목록이 빌 때, 아무것도 없을 때.
     - `order_pinned`는 `field_order` + judgment면 True, `field_order` + 점검만이면 False, `field_order` 없이 judgment만이면 False.

- [ ] **Step 2: 실패를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_field_memory.py -q -k "order or first_check"` → AttributeError로 실패.
- [ ] **Step 3: 구현한다.**
  - 세기는 `alarm_label(d) == "real"`, `decisive(d) != MANUAL_CAUSE`, `guidance.event_code(d["event"]) == code`로 고른다. 설비는 `d["event"]["gun_id"]`다.
  - 조치 보호: lead가 든 첫 원인의 위치 i에 대해 `any(c.get("actions") for c in causes[:i])`이면 옮기지 않는다.
  - 원인과 점검은 `isinstance(…, dict)`로 거른다. 이상한 모양에서 예외를 내지 않는다.
- [ ] **Step 4: 통과를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_field_memory.py -q` → 모두 통과(기존 시험 포함).
- [ ] **Step 5:** 커밋하지 않는다. `git diff --stat`으로 두 파일만 바뀌었는지 본다.

### Task 2: annotate의 R2

**Files:**
- Modify: `api/field_memory.py` (`annotate`, 맨 위 설명 1-7행)
- Test: `api/tests/test_field_memory.py` (맨 위 설명 3행 포함)

**Interfaces:**
- Consumes: `first_check`, `order_for`, `apply_order` (Task 1)
- Produces:
  - `field_memory.applied`에 `"R2"`가 들어갈 수 있다(R1 뒤).
  - `note_ko`의 순서: 오알람 문장 → `field_order.note_ko` → 지난 점검 기록.

- [ ] **Step 1: 시험을 쓴다.**
  1. `test_r2_applied_when_lead_first`
     - 준비: 해결 종료 3건(test_7, test_0, test_0), 모두 K1 정상 · K2 이상.
     - 대상(test_0): `field_order = order_for("E029", cases)`, guidance는 E029 캐시 모양 카드에 `apply_order(…, K2)`를 적용한 것.
     - `applied == ["R2"]`
     - `note_ko`는 아래 셋을 한 칸씩 띄어 이은 것이다.
       - `같은 설비·코드(test_0 E029) 지난 기록: 오알람 0건 · 고장 확인 2건 (참고).`
       - R2 문장(K2 3건 · K1 0건, 설비 2대)
       - `지난 점검 기록(E029): E029-K2 이상 3 · 정상 0, E029-K1 이상 0 · 정상 3 (참고).`
     - `list(field_memory["checks"]) == [K2, K1]`
  2. `test_r2_absent_when_card_not_moved`: 같은 준비에 매핑 대체 카드를 쓴다(옮겨지지 않는다). `field_order`가 있어도 `applied == []`이고, 문장에 R2가 없고, 끝이 `(참고, 점검 순서는 매뉴얼 기준).`이다.
  3. `test_r1_and_r2_together`
     - 대상 설비 test_9: 오알람 2건(점검 기록 없음). 다른 설비에서 위 해결 종료 3건.
     - `applied == ["R1", "R2"]`
     - `note_ko == R1(test_9, E029, 2, 0) + " " + R2 문장 + " " + 지난 점검 기록(끝 "(참고).")`
- [ ] **Step 2: 실패를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_field_memory.py -q -k r2`
- [ ] **Step 3: 구현한다.**
  - `order = case.get("field_order")`, `r2 = bool(order) and first_check(case.get("guidance") or {}) == order.get("lead")`
  - `applied`는 R1 목록 뒤에 R2를 더한다.
  - 문장: 오알람 문장, R2이면 `order["note_ko"]`, 지난 점검 기록(끝은 Global Constraints) 순이다.
  - 맨 위 설명을 고친다: R1은 문장, R2는 새 케이스의 카드 순서(ingest가 정함, 스펙 경로). annotate는 카드를 바꾸지 않는다.
  - 시험 파일 3행의 "순서 · 내용은 바꾸지 않는다"도 같은 뜻으로 고친다.
- [ ] **Step 4: 통과를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_field_memory.py -q` → 모두 통과.

### Task 3: api 훅 — ingest, get_case, 고정

**Files:**
- Modify: `api/main.py` (ingest 101-130, get_case 239-255와 249행 주석)
- Modify: `api/store.py:4` (설명만)
- Test: `api/tests/test_memory_api.py` (맨 위 설명 3행, 71행 주석 포함)

**Interfaces:**
- Consumes: `order_for`, `apply_order`, `order_pinned` (Task 1), annotate (Task 2)
- Produces: 저장 문서와 모든 응답의 최상위 `field_order`(스펙 4.2 모양)

- [ ] **Step 1: 시험을 쓴다.**

  도우미:
  - `_cached_rag(monkeypatch) -> list`: `guidance.answer_question`을 `fake_answer_question`을 감싼 함수로 바꾼다. 부를 때마다 목록에 코드를 넣는다.
    - E029 원인: 원인 1 = E029-C1(조치 포함) + 점검 [K1, K2], 원인 2 = E029-C2, 점검 없음, 조치 없음.
    - E012 원인: 원인 하나 = E012-C1 + 점검 [K1, K2].
    - 그 밖은 그대로.
  - `_resolve(api, cid, marks, judge)`: `api/tests/test_driver_contract.py:139-174`의 `resolve`와 같은 순서로 해결 종료까지 간다.
    - PUT checks는 `marks`(id → 결과)로 보낸다.
    - 판단 · 조치 · 결과의 `situation_id`는 `judge`다.
    - 일지 승인 뒤 해결 종료한다.
  - `SCRIPT`: 스펙 8절 7줄. 각 줄은 (event_id, gun, code, t, marks, 캐시 카드 judge, 대체 카드 judge)다.
    - 설비는 test_7, test_0×6이고, 시각은 늘어나기만 하면 된다.
    - marks: 1-3 · 5번은 `{K1: normal, K2: abnormal}`, 4번은 `{K2: normal, K1: abnormal}`, 6번은 `{K2: abnormal}`, 7번은 `{E012-K1: abnormal}`.
    - 캐시 카드 judge는 모두 `{code}-1`이다.
    - 대체 카드 judge: K2 이상이 든 줄은 `E029-2`, 4번은 `E029-1`, 7번은 `E012-1`.

  시험:
  1. `test_r2_script_cached_card`
     - 줄마다 POST 핸드오프 → GET → 아래를 본다 → `_resolve`.
     - layout: `[[K1,K2],[]]`×3, `[[K2,K1],[]]`, `[[K1,K2],[]]`, `[[K2,K1],[]]`, 그리고 7번은 `[[E012-K1, E012-K2]]`
     - `applied`: `[]`, `[]`, `[]`, `["R2"]`, `[]`, `["R2"]`, `[]`
     - `field_order`: 4번은 `{"lead": K2, "counts": {K1: 0, K2: 3}, "guns": 2, "note_ko": R2(3, 0, 2)}`, 6번은 counts `{K1: 1, K2: 4}` · guns 2. 나머지는 키가 없다.
     - RAG 호출은 7번이다.
  2. `test_r2_script_mapping_card`: 기본 가짜 RAG(대체 카드 모양)로 같은 대본을 대체 카드 judge로 돌린다.
     - E029 layout은 모두 `[[K1],[K2]]`이고, `applied`는 모두 `[]`다.
     - 4 · 6번 줄의 `field_order`는 1번 시험과 같다.
  3. `test_r2_write_responses`
     - 캐시 카드로 K2 해결 종료 3건(2대) 뒤의 새 케이스.
     - PUT checks, POST judgments · actions · results, PUT log 응답마다 `guidance == store.get(cid)["guidance"]`, `_layout == [[K2,K1],[]]`, `field_order`가 있다.
  4. `test_r2_open_case_keeps_card`
     - 순서가 정해지기 전에 연 케이스 A는 K2 해결 3건 뒤에도 `[[K1,K2],[]]`다.
     - 그 뒤에 연 B는 `[[K2,K1],[]]`이고, K1 해결 2건이 더 닫혀도 그대로다.
  5. `test_r2_repeat_decides_nothing`: B가 열린 채로 같은 설비 · 코드 알림이 오면 `"repeat"`다. B의 guidance와 `field_order`는 그대로다.
  6. `test_r2_rag_error_then_ok`
     - 순서가 정해진 상태에서, 가짜가 처음에는 `{"status": "error", "error": "x", …}`를 내고 그 뒤에는 캐시 모양을 낸다.
     - 첫 GET: 저장 카드는 매핑 카드(조치 보호로 그대로)다.
     - 둘째 GET: `[[K2,K1],[]]`이고 `"R2"`가 붙는다.
     - RAG 호출은 2번이다(플래그를 꺼도 같은 수).
  7. `test_r2_pin_after_judgment`
     - 6번처럼 첫 GET이 오류다. 이어서 POST judgments(`E029-1` yes)를 하고 가짜를 캐시 모양으로 바꾼다.
     - 그다음 GET: 카드가 매핑 카드 그대로이고, 가짜를 더 부르지 않는다.
     - 대조: 점검만 PUT한 케이스는 고정되지 않고 RAG 카드가 온다.
  8. `test_r2_pin_when_judgment_lands_during_generation`: 가짜가 불릴 때 `store.get` · `store.put`으로 그 케이스에 judgment 하나를 넣고 캐시 모양을 돌려준다. GET 뒤 저장 카드는 매핑 카드 그대로다.
  9. `test_r2_history_cause_name_after_move`
     - 기본 가짜(E016 `[[K1,K2],[K3]]`)에서 K3 이상 해결 종료 3건(g1, g2, g1, judge `E016-2`).
     - 새 케이스는 `[[K3],[K1,K2]]`다.
     - K3 이상, judge `E016-1`로 해결 종료하면, `/api/history`의 그 줄 `cause`가 매핑 E016-C2의 이름이다.
  10. `test_flag_off_no_field_order`
      - 플래그를 끄고 캐시 모양으로 대본 1-4번을 돌린다.
      - 저장 문서와 응답 어디에도 `field_order` 키가 없다.
      - ingest 직후 저장 guidance는 `guidance_for(handoff)`와 같다.
      - 4번 GET의 layout은 `[[K1,K2],[]]`다.
  11. `test_flag_on_row_kept_with_flag_off`
      - 플래그를 켜고 4번 줄 케이스까지 만든다. 열린 채로, 판단은 없다.
      - 플래그를 끄고 GET: `[[K2,K1],[]]`이고, `field_order`가 있고, `field_memory`는 없다.
      - `monkeypatch.setattr(guidance, "PROMPT_VERSION", "test-bump")` 뒤 GET: 다시 만들어진 카드도 `[[K2,K1],[]]`다.

  기존 시험은 그대로 통과해야 한다. 해결 종료가 없는 시나리오라 순서가 정해지지 않는다.

- [ ] **Step 2: 실패를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_memory_api.py -q -k "r2 or field_order or kept_with"`
- [ ] **Step 3: 구현한다.** 잠금 순서가 중요하므로 모양을 그대로 따른다.

  ingest(기존 잠금 안, 새 케이스 분기):

  ```python
  g = guidance.guidance_for(handoff)
  order = field_memory.order_for(guidance.event_code(handoff), store.all_cases()) if field_memory.enabled() else None
  d = {..., "guidance": field_memory.apply_order(g, order["lead"]) if order else g, ...}
  if order:
      d["field_order"] = order
  ```

  get_case:

  ```python
  current = load(case_id)
  # R2로 순서를 정한 케이스에 판단 · 조치 · 결과가 있으면 카드를 다시 만들지 않는다 (원인 번호가 위치라서)
  enriched = current if field_memory.order_pinned(current) else guidance.enrich(current)  # RAG 생성은 잠금 밖에서
  with _WRITE_LOCK:
      latest = load(case_id)
      card = enriched["guidance"]
      order = latest.get("field_order")
      if order:
          card = latest["guidance"] if field_memory.order_pinned(latest) else field_memory.apply_order(card, order["lead"])
      if latest["guidance"] != card:
          latest["guidance"] = card
          save(latest)
  ```

  - 그 아래(`out = latest`부터)는 그대로 둔다.
  - 249행 주석 "카드는 그대로"는 "field_memory는 저장하지 않는다. 순서는 ingest가 정한다(R2)" 뜻으로 고친다.
  - `api/store.py:4`의 "(field_memory.py, 안내 카드는 그대로)"는 "(field_memory.py. R2로 순서를 정한 케이스는 field_order와 옮긴 카드를 저장한다)"로 고친다.
  - 시험 파일 3행과 71행의 "카드는 그대로"는 "R1은 카드를 바꾸지 않는다"로 고친다.
- [ ] **Step 4: 통과를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests -q` → 모두 통과, 1 skipped(계약 시험).

### Task 4: 드라이버 `card_checks`

**Files:**
- Modify: `ml/.py/hitl_loop.py`
  - `_row` 187-195, guide 271-281, operator 283-288
  - `_cells` 508-522, `render_view` 566-570, `render_summary` 613-624, 콘솔 718-721
  - 설명 17행 · 272행
- Test: `ml/tests/test_hitl_loop.py` (FakeAPI 176-215와 설명 180행, 376-377행, 설명 394행, 새 시험)
- Modify: `api/tests/test_driver_contract.py` (설명 5행 · 272행, 단언 한 줄 추가)

**Interfaces:**
- Produces:
  - 행 키 `card_checks: list[str] | None` (`_row` 기본값 None)
  - operator payload 키 `card_checks`
  - `_card_text(order: list[str] | None, card_checks: list[str] | None) -> str`
- 규칙:
  - guide: `g["causes"]`의 점검 id를 원인 순으로 모은다. 비면 `candidates[0]["checks"]`의 id를 쓴다. `order`는 그대로 둔다.
  - `_card_text`: `" → ".join(order)`와 `" → ".join(card_checks)` 중 빈 것을 빼고 `" · "`로 잇는다. 둘 다 비면 `"—"`.
  - `_cells`(안내 카드 칸), 상태 줄, 콘솔 ⏸ 줄은 모두 `_card_text`를 쓴다.
  - `render_summary`: 알림 표 바로 뒤에, R2가 든 행이 하나라도 있으면 시연 대본 문장 한 줄과 빈 줄을 넣는다.

- [ ] **Step 1: 시험을 쓴다.**
  1. `test_loop_waits_and_resumes`(376-377행): 기대 payload에 `"card_checks": ["E029-K1", "E029-K2"]`를 더한다. `row["card_checks"]`도 같은 값인지 본다(FakeAPI 카드는 `candidates` 점검만 있다).
  2. `test_guide_reads_card_checks_from_causes`
     - FakeAPI에 선택 인자 `causes: list[list[str]] | None`을 더한다. 주면 guidance 최상위 `causes`를 `[{"checks": [{"id": i} …]} …]`로 만든다.
     - `causes=[["E029-K2"], ["E029-K1"]]`이면 `card_checks == ["E029-K2", "E029-K1"]`이고 `order == ["E029"]`다.
  3. `test_card_text_rendering`
     - `view_state()`의 닫힌 행에 `card_checks = ["E029-K2", "E029-K1"]`, `applied = ["R1", "R2"]`를 준다.
     - 칸 6은 `"E029 · E029-K2 → E029-K1"`, 칸 7은 `"R1, R2"`다.
     - 대기 dict에 `card_checks`를 주면 html에 `"· 안내 카드 E029 · E029-K2 → E029-K1</p>"`가 있다.
     - `card_checks`가 없는 행은 `"E029"` 그대로다. 그래서 684 · 696 · 749 · 881행의 기존 단언은 바뀌지 않는다.
  4. `test_summary_script_line`: `view_state()` 그대로면 시연 대본 문장이 없다. 한 행에 `"R2"`를 넣으면 한 번 나온다.
  5. `test_main_console_card_checks`: `interrupted_state`의 payload에 `card_checks`를 넣으면 콘솔에 `"  ⏸ CASE-0002 사람 확인 대기 · 안내 카드 E029 · E029-K2 → E029-K1: http://ui/cases/CASE-0002"`가 찍힌다.
  6. `api/tests/test_driver_contract.py::test_scenario_test_0`: `[r["card_checks"] for r in rows] == [["E029-K1", "E029-K2"]] * 5 + [["E012-K1", "E012-K2"]]` 한 줄을 더한다. 실제 api + 가짜 RAG로 돌고, 카드는 매핑 모양이다.
- [ ] **Step 2: 실패를 확인한다.** `ml/`에서 `py -3.11 -m pytest tests/test_hitl_loop.py -q`
- [ ] **Step 3: 구현한다.**
  - 위 규칙대로 고친다.
  - 설명 17행은 "카드의 점검 순서(card_checks)와 field_memory 문장을 읽는다"로, 272행은 "guidance.causes의 점검 순서"로 고친다.
  - FakeAPI 설명 180행, 394행 주석, 계약 시험 5 · 272행 설명의 "카드는 그대로"는 "R1은 카드를 바꾸지 않는다" 뜻으로 고친다.
- [ ] **Step 4: 통과를 확인한다.**
  - `ml/`에서 `py -3.11 -m pytest tests/test_hitl_loop.py -q` → 모두 통과.
  - 저장소 루트에서 `PYTHONPATH`에 막이 폴더(`…\scratchpad\task3\stubs`)를 두고 `py -3.11 -m pytest api/tests/test_driver_contract.py -q` → 3 passed.

### Task 5: 문서와 다른 파트 요청

**Files:**
- Create: `ml/docs/card-order-requests.md` (다른 파트 요청)
- Modify: `ml/docs/hitl-loop.md`
  - 5행 요약, ② 29-42, ③ 98-250, ④ 251-270, ⑤ 272-289, ⑥ 291-302
  - 96 · 146 · 176 · 217 · 280 · 293행(11-07 기준)
- Modify: `api/README.md` 64-70행
- Modify: `ml/techspec.md` 35행
- Modify: `ml/history.md` (끝에 48절)
- Modify: `ml/MEMORY.md` (변경 이력 끝에 한 줄)

**Interfaces:**
- Consumes: Task 1-4의 이름(`field_order`, `R2`, `card_checks`)과 Task 6의 시험 수. history와 MEMORY의 수는 Task 6 뒤에 채운다.

- [ ] **Step 1: `ml/docs/card-order-requests.md`를 쓴다.** 팀원이 이 파일 하나만 읽고 움직일 수 있게 쓴다.
  1. 무엇이 바뀌나: 5줄 안팎으로 쓴다.
     - `FIELD_MEMORY=1`일 때만 바뀐다.
     - 해결 종료 이상 기록이 3건 앞서고 설비 2대 이상이면, 다음 케이스 카드에서 그 점검과 원인이 먼저 온다.
     - 조치 보호, 카드 고정, 최상위 `field_order`.
     - 스펙 경로를 적는다.
     - 다른 파트와 합의 없이 바뀌었다는 것과 머지 전 알림을 적는다.
  2. FE 11 · 12 · 13: 스펙 11절 그대로 쓴다.
     - `field_order` 타입은 글자 그대로 쓴다.
     - 보일 조건은 "카드 첫 점검 == `field_order.lead`"다.
     - 고칠 주석 세 곳과 `OverviewPage.tsx:84` 안내를 적는다.
  3. RAG 세 항목, api 담당 한 항목, 멘토 · 매핑 두 항목: 스펙 11절 그대로 쓴다.
  - 항목마다 무엇을, 왜, 어디(file:line)를 쓴다.
- [ ] **Step 2: `ml/docs/hitl-loop.md`를 고친다.**
  - 5행 요약: R1 문장에 더해 R2(카드 순서)를 한 문장으로 쓴다.
  - ② "카드는 그대로" → "카드 순서(R2)"
    - 규칙 한 줄, 조치 보호, 고정, `field_order`를 DB에 저장한다는 것을 적고 스펙 경로를 단다.
    - `field_memory`는 지금처럼 저장하지 않는다.
  - ③ 새 시나리오 "test_7 → test_0 (R2)"를 둔다.
    - 준비: `rag/.env`와 Neo4j, 드라이런, 24시간 경계(스펙 8절).
    - 실행: `py -3.11 .py/hitl_loop.py test/test_7.csv test/test_0.csv`
    - 작업자 카드 한 줄: "앞 항목이 정상이면 다음 항목까지 기록".
    - 스펙 8절 7줄 표. 데이터 시각은 t7 E029 09-08 17:55, t0 E029 09-04 16:45 · 09-06 19:30 · 09-07 19:30 · 09-08 19:30 · 09-10 19:30, t0 E012 09-11 02:46이다.
    - 기존 두 표의 카드 칸은 `E029 · E029-K1 → E029-K2`, `E012 · E012-K1 → E012-K2`, `E028 · E028-K1 → E028-K2`로 바꾼다. 나머지 칸은 그대로다.
  - ④: 끝에 "11-13은 `ml/docs/card-order-requests.md`" 목록을 한 줄씩 둔다. 내용을 겹쳐 쓰지 않는다.
  - ⑤: "카드는 그대로라는 결정 때문이다" → "카드 순서는 api가 R2로 정한다".
  - ⑥: "문장만이다"를 고친다. R1은 문장이고 R2는 카드 순서다. 큰 한계 셋(이유가 화면에 없음, E029는 RAG 캐시 카드에서만, 위 항목 유리)을 적고 스펙 10절을 가리킨다.
  - 96 · 146 · 176 · 217 · 280 · 293행에서 카드가 바뀌지 않는다는 문장을 R2에 맞게 고친다.
- [ ] **Step 3: 나머지 문서를 고친다.**
  - `api/README.md` 64-70행:
    - "내용과 순서는 바꾸지 않습니다" → "내용은 바꾸지 않습니다. 순서는 R2 조건을 만족한 새 케이스만 바뀝니다"
    - "DB에는 저장하지 않습니다"는 `field_memory`에만 해당한다. `field_order`는 저장한다고 쓴다.
    - R2 한 줄과 스펙 경로를 더한다.
  - `ml/techspec.md:35` D-8 줄: "RAG 카드는 그대로" → "카드 순서는 R2로(2026-11-07, `history.md` 48절)"
  - `ml/history.md` 48절 "D-8 확장 R2 — 현장 기록으로 점검 순서 바꾸기 (2026-11-07)"
    - 사용자 결정 넷과 받아들인 셋(스펙 2절)
    - 검토 두 번(워크플로 id)
    - 바뀐 파일, Task 6 시험 수, 남은 것(FE 요청, 실제 화면 시연), 스펙 · 계획 경로
  - `ml/MEMORY.md` 변경 이력: `- 2026-11-07: **D-8 R2 카드 순서**(history.md 48절, 설계 docs/superpowers/specs/2026-11-07-feedback-card-order-design.md): …` 한 줄(같은 꼴).
- [ ] **Step 4: 점검한다.** `git grep -n "카드는 그대로\|순서도 내용도\|순서는 바꾸지" -- api ml`
  - 남은 줄은 모두 R1만 말하거나 옛 기록(history 36절, MEMORY 10-28 항목)이어야 한다.
  - `ui/`의 주석은 FE 요청 12로 넘긴다.

### Task 6: 마무리 확인

- [ ] **Step 1: 시험 전체를 돌린다.**
  - API: `.venv\Scripts\python.exe -m pytest api/tests -q` → 통과, 1 skipped
  - DRV + 막이: `py -3.11 -m pytest api/tests -q` → 계약 시험 포함 모두 통과
  - ml: `ml/`에서 `py -3.11 -m pytest -q` → 기존 124 + 새 시험 모두 통과
- [ ] **Step 2: ruff를 돌린다.**
  - `ml/`에서 `py -3.11 -m ruff check .` → 기존 E402 하나만 나온다.
  - api: `py -3.11 -m ruff check api --line-length 120`의 결과를 HEAD 판(아래 임시 폴더)과 비교한다. 새 오류가 없어야 한다.
- [ ] **Step 3: 플래그 끈 바이트 비교.**
  - `git show HEAD:api/<파일>`로 `main.py` · `field_memory.py` · `store.py` · `guidance.py` · `assistant.py`를 스크래치 폴더에 꺼낸다.
  - 스크립트 하나로 두 판에 같은 대본을 돌린다.
    - 플래그 끔, 캐시 모양 가짜 RAG, `main.now`를 고정 수열로 바꾼다.
    - 스펙 8절 7줄과 `/api/history` · `/api/cases`를 부른다.
  - 모든 응답 JSON 글자와 마지막 DB 덤프가 같아야 한다.
- [ ] **Step 4: 작업 트리를 확인한다.**
  - `git status`와 `git diff --stat`: 계획의 파일만 바뀌었고, `rag/` · `ui/` 변경이 없다.
  - 고친 파일은 CRLF 그대로다(파일 전체가 바뀐 diff가 없다).
  - 스펙 · 계획 · 요청 문서 세 개는 새 파일로 보인다.
- [ ] **Step 5: 보고한다.**
  - 스테이징 목록(파일별 한 줄)
  - 시험 수
  - "reset이나 checkout을 하면 커밋 안 된 변경이 사라진다"는 경고
  - 남은 일: FE 요청 11-13 전달, 실제 화면 시연(Neo4j 필요)
