# FE가 받을 데이터 형식 — ML · RAG 담당에게 요청할 것 (레포 기준)

2026-10-02 · FE 담당

## 한눈에 보기

대시보드 와이어프레임에 필요한 값은 ML 쪽에 거의 다 있고, RAG 쪽은 내용은 있지만 글(마크다운) 대신 필드(JSON)로 받아야 화면을 만들 수 있습니다. 케이스 · 점검 · 판단 · 조치 저장은 ML도 RAG도 아닌 `api/` 서버의 일입니다. (기준: maintenance-assistant-agent 레포, 10월 2일 커밋)

| 화면 · 칸 | 필요한 값 | 어디서 | 지금 |
| --- | --- | --- | --- |
| 작업함 목록 | 설비, 코드, 발생 시각, 추정 고장 유형, 한 줄 요약 | ML 핸드오프 | 있음 |
| 작업함 진행 상태 · 담당 | 케이스 상태 | api 저장 | 없음 |
| 원인 찾기 상단 "무슨 일이 났나" | 코드, 시각, 센서 증상(한국어), 데이터 상태 | ML 핸드오프 | 있음 |
| 〃 코드 지속 시간 (예: 16초) | 코드가 켜진 시작 · 끝 | ML | **요청** |
| 〃 작은 그래프 | 분 단위 점수 · 편차 · 코드 이벤트 | ML `/guns/{id}/trace` | 있음, 단 최근 6시간만 → **요청** |
| 후보 카드 (순서 · 이름) | 상황 ID 순서, 상황 이름 · 정의 | 순서는 ML, 이름은 RAG | 있음 (두 곳에 나뉘어) |
| 점검 항목 | 점검 순서 목록 | RAG `evidence` | 있음, 단 글자만 → **요청** |
| 매뉴얼 인용 · 이상일 때 할 일 | 오류 행의 조치 문구, 쪽수, 원문 | RAG `evidence` | 있음, 짧은 인용 **요청** |
| AI 요약 | 답변 요약 부분 | RAG `answer` | 글 통째 → **요청** |
| 근거 없음 (수동 모드) | 상태값 | RAG `status` | 있음 |
| 판단 · 조치 · 결과 저장 | 사람 입력 | api 저장 | 없음 |
| 결과 칸 "재발 없음" 자동 표시 | 조치 뒤 같은 설비의 새 규칙 발화 여부 | ML `/handoffs`, trace | 계산 가능 → **확인 요청** |
| 정비일지 · 이력 · 인계 | 위 기록 모음 | api 저장 | 없음 |
| 설비 상태 | 기준 수집 중, 판단 보류, 드리프트 | ML `/guns`, `/stats` | 있음 |
| 수동 모드 문서 검색 | 키워드 검색 | RAG | 없음 → **요청 또는 제외** |

아래 JSON의 값은 모두 예시입니다. 필드 이름은 레포에 이미 있는 것을 그대로 썼고, 새로 요청하는 필드만 제가 이름을 붙였습니다.

## ML 담당에게

핸드오프와 조회 API는 지금 모양 그대로 쓰겠습니다. 요청은 6가지이고, 이 중 1 · 2번이 있어야 화면 연결을 시작할 수 있습니다.

### 그대로 쓸 것

| ML API | 쓰는 화면 |
| --- | --- |
| 핸드오프 (`RagHandoff` v1.0) | 작업함 목록, 원인 찾기 상단 |
| `GET /guns/{id}/trace` | 신호 그래프 (평소 대비 편차로 그림) |
| `GET /guns`, `GET /guns/{id}/stats` | 설비 상태 |
| `GET /handoffs?gun_id=` | 조치 뒤 재발 확인 |

핸드오프에서 화면이 쓰는 필드: `event_id`, `gun_id`, `detected_at`, `trigger.source`, `trigger.rule_code`, `summary_ko`, `sensor_findings[].text_ko`, `fault_class.code` · `name_ko`, `situation_ids`, `context.non_welding_share` · `alarm_held` · `gun_norm`, `caveats`.

### 요청 1 (필수) · 서버를 띄울 수 있게

레포에 모델 파일(`models/baseline_ensemble.joblib`)과 `test/*.csv`가 없어서 클론만으로는 서버가 안 뜹니다. 둘 중 하나를 부탁드립니다.

- 모델 파일과 테스트 CSV를 받을 수 있는 곳 (드라이브 링크 등)
- 또는 테스트 8건을 재생한 결과 파일: `replay.py --out-dir`로 뽑은 응답과 건별 trace JSON. FE는 이걸 mock으로 먼저 개발합니다

### 요청 2 (필수) · 핸드오프를 api 서버로 보내기

`RSW_RAG_URL`을 RAG가 아니라 api 서버 주소(예: `http://api:9000/diagnose`)로 설정하는 안입니다. api가 받아서 케이스를 만들고 RAG를 부릅니다. 응답은 바로 돌려드립니다.

```json
{ "event_id": "0d711c472d8d500337556b05133273c2", "status": "ok" }
```

질문: README 5.10.6의 7칸 리포트를 이 응답에 꼭 담아야 하나요? RAG 답변은 몇 초가 걸려서, 접수만 먼저 응답하는 쪽이 안전해 보입니다.

### 요청 3 · 핸드오프에 코드 이력과 센서 추세 넣기 (techspec D-3)

케이스를 나중에 열면 trace가 이미 지나갔거나(최근 6시간), 서버 재시작으로 비어 있습니다. 케이스에 그래프와 코드 지속 시간을 남기려면 핸드오프에 아래 두 필드가 필요합니다.

```json
{
  "error_timeline": [
    { "code": "E012", "start": "2021-09-11T02:46:08Z", "end": "2021-09-11T02:46:24Z",
      "duration_s": 16, "class_hint": "E01" }
  ],
  "sensor_trend": [
    { "feature": "c5_mean", "sensor": "c5", "sensor_name_ko": "보정(밸런스) 압력",
      "points": [ { "t": "2021-09-11T02:16:00Z", "z": -0.2 },
                  { "t": "2021-09-11T02:17:00Z", "z": -0.4 } ] }
  ]
}
```

일정이 안 되면 대안: api가 핸드오프를 받은 직후 `/guns/{id}/trace`를 불러 케이스에 복사해 둡니다. 이 경우 코드 지속 시간은 화면에서 뺍니다.

### 요청 4 · 시각 형식

응답 시각에 `Z`가 없습니다(`2021-09-03T09:30:59`). 끝에 `Z`를 붙여 주시거나, "전부 UTC"라고만 확인해 주세요. 확인되면 FE가 한국 시간으로 바꿉니다.

### 요청 5 · 한국어 센서 이름

`sensor_name_ko`가 핸드오프에만 있습니다. trace의 `features[]`와 `contributing_features[]`에도 같은 필드를 넣어 주시면 FE가 이름표를 따로 들고 있지 않아도 됩니다.

```json
{ "feature": "c5_mean", "sensor": "c5", "sensor_name": "Balance pressure",
  "sensor_name_ko": "보정(밸런스) 압력", "statistic": "mean", "share": 0.41 }
```

### 요청 6 · 조치 뒤 "재발 없음" 확인 방법

화면의 결과 칸에 "조치 후 종료 코드 재발 없음"을 자동으로 표시하려고 합니다. 계획은 이렇습니다: 조치 저장 시각 이후에 `GET /handoffs?gun_id=`에 새 이벤트가 없고, trace `events`에 `rule` · `rule_repeat`가 없으면 "재발 없음".

- 이 방법이 맞는지, 몇 분을 봐야 하는지 (규칙 쿨다운 30분 기준?)
- 조치를 저장할 때마다 `DELETE /guns/{id}`(기준 재수집)를 불러야 하는지, 부품 교체 같은 큰 조치에만 부르는지
- 공개 데이터에는 고장 뒤 10분까지만 있어서 데모에서는 이 표시를 "시뮬레이션"으로 달 예정입니다

참고로 핸드오프 `caveats`의 수치(AUROC 0.64, 0/8건)가 현재 `models/baseline_ensemble_test_metrics.json`(0.713, 2/8건)과 다릅니다. 화면에 그대로 보여줄 문구라 확인 부탁드립니다.

## RAG 담당에게

근거 조회 결과(`evidence`)는 화면 재료로 충분합니다. 핵심 요청은 하나입니다: 원인 후보 · 점검 항목 · 요약을 마크다운 글이 아니라 필드로 나눠 주세요. 화면이 후보마다 카드를, 점검 항목마다 정상 / 이상 버튼을 만들기 때문입니다.

### 그대로 쓸 것

`answer_question(situation_ids, question)`은 api 서버가 함수로 부릅니다. RAG 쪽에 웹 서버를 만들 필요는 없습니다.

| RAG 반환값 | 화면에서 |
| --- | --- |
| `evidence[].name` · `definition` | 후보 카드 제목과 설명 |
| `evidence[].check_sequence` | 점검 항목 |
| `evidence[].diagnostics[]` (`number`, `name`, `error_elimination`, `page_number`, `priority`) | "매뉴얼이 말하는 조치" |
| `evidence[].direct_evidence[]` (`text`, `page_number`, `rationale`) | 오른쪽 근거 목록 |
| `evidence[].coverage_note` | 주의 문구 (예: RSW 1800 ms와 Festo 2초는 다른 기준) |
| `review_needed`, `review_notes` | "원본 확인 필요" 배지 |

| `status` | 화면 동작 |
| --- | --- |
| `ok` | 가이드 모드 (후보 · 점검 · 요약 표시) |
| `retrieval_only` | 근거와 점검 항목만 표시, AI 요약 숨김 |
| `needs_citation_review` | 답변을 보여주되 "인용 검토 필요" 배지 |
| `unknown_situation_id`, `no_evidence` | 수동 모드로 전환 |
| `error` | 오류 표시 + 다시 시도 |

### 요청 1 (필수) · 예시 출력을 레포에

S01\~S10 각각의 `answer_question` 결과 JSON을 `rag/docs/`에 올려 주세요. ML이 `ml/docs/*.example.json`을 둔 것처럼요. FE와 api는 Neo4j나 OpenAI 키 없이 이 파일로 먼저 개발합니다.

```bash
python -m rag ask --situation-ids S01 --question "어떤 순서로 점검해야 하나요?" --json --output rag/docs/S01.example.json
```

### 요청 2 (필수) · 답변을 필드로 나누기

지금 `answer`는 네 제목(오류 상황 해석 / 원인 후보 / 점검 체크리스트 / 답변 요약)이 들어간 글 한 덩어리입니다. 최소안은 제목별로 잘라 주는 것입니다.

```json
{
  "answer_sections": {
    "S01": {
      "interpretation": "…",
      "cause_candidates": "…",
      "checklist": "…",
      "summary": "…"
    }
  }
}
```

권장안은 원인 후보와 체크리스트를 목록으로 주는 것입니다. 화면의 후보 카드와 점검 버튼에 바로 연결됩니다.

```json
{
  "situation_id": "S01",
  "interpretation": "…",
  "cause_candidates": [
    { "id": "S01-C1", "name": "공급 압력 부족", "why": "…", "how_to_tell": "…",
      "priority": "primary", "citations": [{ "doc": "Festo", "page": 90 }] }
  ],
  "checklist": [
    { "id": "S01-K1", "step": 1, "title": "본관 공급 압력 측정", "detail": "…",
      "if_abnormal": null, "citations": [{ "doc": "Festo", "page": 90 }] }
  ],
  "summary": "…"
}
```

- 원인 후보 이름은 상황 정의서 PDF의 "원인 후보" 목록(S01: 공급 압력 부족, MPYD · 케이블 불량, 배관 누설, 보정 실린더 불량)이 재료인데, 지금 매핑 JSON과 그래프에는 들어 있지 않습니다
- `if_abnormal`은 점검이 "이상"일 때 할 일. 매뉴얼에 없으면 `null` (지어내지 않기)
- 인용은 글 속 `[Festo PDF p.90]` 대신 `citations` 배열로

### 요청 3 · 점검 항목에 id와 근거 연결

`check_sequence`가 글자 목록이라, 사람이 입력한 점검 결과를 저장할 키가 없고 항목별 근거 쪽수도 알 수 없습니다. `situation_mapping.json`을 이렇게 바꾸면 요청 2 권장안 없이도 화면이 됩니다.

```json
{
  "check_sequence": [
    { "id": "S01-K1", "title": "본관 공급 압력 측정", "evidence_pages": [90], "diagnostic_numbers": [12] },
    { "id": "S01-K2", "title": "필터·레귤레이터 확인", "evidence_pages": [], "diagnostic_numbers": [] }
  ]
}
```

### 요청 4 · 질문 없이 부르기

대시보드는 챗봇이 아니라서, 케이스가 생기면 사람이 질문하기 전에 자동으로 호출합니다. `question`이 필수라 api가 고정 질문을 넣을 예정인데, 어떤 문구가 좋을지 알려 주세요. 그리고 ML 핸드오프의 `summary_ko` · `sensor_findings`를 같이 넘길 선택 인자가 있으면 답변이 그 사건에 맞게 나올 것 같습니다.

```python
answer_question(
    situation_ids=["S01", "S05"],
    question="이 상황의 원인 후보와 점검 순서를 알려주세요.",
    context={"summary_ko": "보정(밸런스) 압력(c5) 평소보다 낮음; …", "event_id": "…"},  # 새 선택 인자
)
```

### 요청 5 · 짧은 인용문

`direct_evidence[].text`는 최대 1,200자 덩어리라 인용 박스에 넣기 깁니다. 매핑 JSON의 `contains` 문구(예: "Check cable to the MPYD")를 `quote` 필드로 같이 돌려주시면 그걸 원문 인용으로 보여주고, 전체 덩어리는 "더 보기"로 둡니다.

### 요청 6 · 확인할 것

- 답변 생성에 보통 몇 초 걸리는지, `use_cache=True`를 써도 되는지. 느리면 api가 `use_llm=False`로 근거를 먼저 보여주고 요약은 뒤에 붙입니다
- `components[]`의 `priority` · `rationale`이 비어 나올 것으로 보입니다 (`build.py`가 그 값을 저장하지 않음). 실행해 보지는 못했습니다
- 수동 모드의 "문서 검색"(키워드로 매뉴얼 찾기)이 가능한지. 어렵다면 데모 화면에서 뺍니다
- 후보를 "아님"으로 판단했을 때는 RAG 변경이 필요 없습니다. api가 다음 상황 ID로 넘어갑니다

## ML과 RAG가 같이 정해야 하는 것

같은 값이 양쪽에 있거나 서로 기대하는 형식이 엇갈리는 곳이 6가지 있습니다. 오른쪽 열은 FE 제안이고, 세 파트가 한 번 모여서 확정하면 됩니다.

| 항목 | ML 쪽 | RAG 쪽 | FE 제안 |
| --- | --- | --- | --- |
| 원인 후보의 단위 | 상황 ID 순서 (`situation_ids`: S01, S05 …) | Festo 오류 번호 후보 + 글 속 원인 | 후보 카드 1개 = 상황 ID 1개. 순서는 ML, 이름 · 점검 · 근거는 RAG |
| RAG 응답 모양 | README 5.10.6의 7칸 리포트를 기대 | `answer` 글 + `evidence` | 하나로 통일. FE는 RAG의 `evidence` 구조 + RAG 요청 2 형태를 선호 |
| 핸드오프를 받는 곳 | RAG의 `POST /diagnose`로 보낼 준비가 됨 | 받는 서버가 없음 | api 서버가 받고, api가 RAG 함수를 호출 |
| 상황 ID의 이름 · 정의 | ID만 있음 | 이름 · 정의가 있음 (`build.py`) | 이름은 RAG 것만 사용. S01\~S10을 바꿀 때는 두 파트가 같이 수정 |
| 해석 한계 문구 | `caveats` (모델 성능, 규칙 한계) | `coverage_note`, `review_notes` (매뉴얼 적용 한계) | 둘 다 받아 한 칸에 출처를 나눠 표시. 적용 등급 배지는 `coverage_note`로 대체 |
| 사건을 잇는 키 | `event_id` | 없음 | api가 `event_id`로 케이스를 보관. RAG 호출에도 실어 보냄 (RAG 요청 4) |

센서 한국어 이름은 ML의 이름표(`SENSOR_KO`)를 기준으로 쓰겠습니다. RAG 답변 속 센서 이름이 이와 다르면 알려 주세요.

## FE가 최종으로 받고 싶은 모양 (api 서버)

FE는 ML과 RAG를 직접 부르지 않고 api 서버 하나만 부릅니다. 케이스 한 건을 열 때 `GET /cases/{id}`가 아래처럼 오면 원인 찾기 · 조치 화면이 완성됩니다.

```json
{
  "case_id": "CASE-0001",
  "status": "in_progress",
  "event": {
    "event_id": "…",
    "gun_id": "test_0",
    "detected_at": "2021-09-11T02:46:24Z",
    "trigger": { "source": "rule", "rule_code": "E012" },
    "summary_ko": "…",
    "sensor_findings": [{ "sensor": "c5", "text_ko": "…", "share": 0.41 }],
    "fault_class": { "code": "E01", "name_ko": "보정 압력 도달 지연" },
    "caveats": ["…"]
  },
  "guidance": {
    "status": "ok",
    "summary": "…",
    "candidates": [
      {
        "situation_id": "S01",
        "rank": 1,
        "state": "reviewing",
        "name": "보정 압력 도달 지연",
        "definition": "…",
        "coverage_note": "…",
        "checks": [{ "id": "S01-K1", "title": "본관 공급 압력 측정", "result": null }],
        "diagnostics": [{ "number": 12, "priority": "primary", "error_elimination": "…", "page_number": 90 }],
        "evidence": [{ "id": "…", "quote": "…", "page_number": 90, "review_needed": false }]
      }
    ]
  },
  "records": { "judgments": [], "actions": [], "result": null }
}
```

| 묶음 | 내용 | 채우는 곳 |
| --- | --- | --- |
| `event` | ML 핸드오프를 그대로 | ML |
| `guidance` | RAG 결과를 상황 ID별로 | RAG (`state`만 api가 사람 판단으로 갱신) |
| `records` | 사람이 입력한 판단 · 조치 · 결과 | api 저장 |

FE가 보내는 것은 네 가지입니다: 점검 결과(정상 / 이상 / 미실시), 후보 판단(맞음 / 아님 / 모르겠음), 실제 수행 조치, 결과(해결됨 / 다른 조치 / 다음 후보). 모두 api가 저장하고 ML · RAG에는 보내지 않습니다. 값은 전부 예시입니다.

## 저장 API — FE가 정한 안 (2026-10-08, api 담당이 정해지기 전까지)

api 서버가 아직 없어서 FE가 주소와 모양을 정하고 MSW(mock)로 먼저 만들었습니다. 코드는 `ui/src/mocks/handlers.ts`, 타입은 `ui/src/types/case.ts`에 있습니다. api를 만드는 사람이 이대로 쓰면 FE는 고칠 것이 없고, 바꾸면 위 두 파일만 맞추면 됩니다.

| 주소 | 하는 일 | 보내는 것 |
| --- | --- | --- |
| `GET /api/cases` | 작업함 목록 | — (응답: `{ "cases": [...] }`, 발생 시각 최근순) |
| `GET /api/cases/{id}` | 케이스 한 건 (`event`, `score_trace`, `guidance`, `records`) | — |
| `PUT /api/cases/{id}/checks` | 점검 결과 저장. 보낸 항목만 덮어씀 | `{ "checks": { "S01-K1": { "result": "abnormal", "memo": "4.2 bar" } } }` |
| `POST /api/cases/{id}/judgments` | 원인 판단 추가 | `{ "situation_id": "S01", "verdict": "yes" }` |
| `POST /api/cases/{id}/actions` | 조치 기록 추가 | 아래 예시 |
| `POST /api/cases/{id}/results` | 결과 추가 | `{ "situation_id": "S01", "outcome": "resolved" }` |
| `PUT /api/cases/{id}/log` | 정비일지 저장. `approved: true`면 승인, `false`면 승인 취소 | 아래 예시 |
| `POST /api/cases/{id}/close` | 해결 종료(`resolved`, 일지 승인 뒤에만) 또는 미해결로 저장(`unresolved`) | `{ "outcome": "resolved" }` |
| `GET /api/history` | 정비 이력: 해결 종료했거나 미해결로 저장한 케이스 | — (응답: `{ "items": [...] }`, 종료 최근순) |
| `PUT /api/cases/{id}/manual` | 수동 모드 기록 (통째로 덮어씀) | `{ "checks": [{ "title": "…", "result": "abnormal", "memo": "…" }], "cause": "…" }` |
| `POST /api/cases/{id}/field` | 현장 확인 입력. 결과는 점검 결과에 합치고 확인한 사람을 남김 | `{ "by": "…", "checks": { "S01-K1": { "result": "normal" } } }` |
| `POST /api/cases/{id}/handover` | 도움 요청. 케이스 기록은 이미 api에 있으므로 요청 내용만 보냄 | `{ "to": "공정 담당", "urgency": "급함", "reasons": ["원인을 못 찾음"], "note": "…" }` |
| `GET /api/guns` | 설비 목록. ML `GET /guns` 그대로 + `open_cases`(해결 종료 전 케이스) | — (응답: `{ "guns": [...] }`) |
| `GET /api/manuals/festo/search?q=` | 수동 모드 "매뉴얼에서 직접 찾기". 지금은 RAG 매핑의 근거 쪽(설명 · 인용 · 원인 이름)에서 찾음 | — (응답: `{ "items": [{ "page", "title", "quote", "situation_id", "situation_name" }] }`) |
| `GET /api/manuals/festo.pdf` | Festo 매뉴얼 원본 PDF (`Content-Type: application/pdf`). FE는 `#page=90`처럼 쪽을 붙여 연다 | — |

```json
{
  "situation_id": "S01",
  "kind": "재체결",
  "did": "…",
  "parts": [{ "name": "…", "qty": 1 }],
  "worker": "…",
  "started_at": "2026-10-08T05:10:00.000Z",
  "ended_at": "2026-10-08T05:25:00.000Z"
}
```

정비일지 (`PUT /log`). 칸은 FE가 앞 단계 기록으로 미리 채우고 사람이 고친 값이다 (AI 생성 아님).

```json
{
  "date": "2026-10-08",
  "work_time": "14:10 ~ 14:25",
  "worker_gun": "… · test_0",
  "problem": "E012 발생 (데이터 시각 2021-09-11 11:46). ML 요약: …",
  "cause": "보정 압력 도달 지연 — …",
  "action": "재체결 — …",
  "missed_checks": "…",
  "recurrence": "처음",
  "approved": true
}
```

- 값의 종류: 점검 `result` = `normal` / `abnormal` / `skipped`, 판단 `verdict` = `yes` / `no` / `unsure`, 결과 `outcome` = `resolved` / `retry` / `next`, 조치 `kind` = 조정 / 교체 / 청소 / 재체결 / 기타.
- 판단 · 조치 · 결과는 덮어쓰지 않고 쌓습니다(다시 판단하면 마지막 것이 유효). 서버가 각 기록에 `at`(저장 시각)을 붙입니다.
- 저장 요청의 응답은 바뀐 케이스 전체(`GET /api/cases/{id}`와 같은 모양)입니다. FE는 이 응답으로 화면을 바로 갱신합니다.
- 처음 무언가를 저장하면 케이스 `status`를 `new` → `in_progress`로 바꾸고 `records.started_at`을 남깁니다.
- 케이스 `status`: `new` 새 고장 → `in_progress` 점검 중 → (결과 "해결됐어요") `logging` 일지 작성 → (일지 승인) `log_approved` 일지 승인됨 → (해결 종료) `resolved`. 미해결로 저장하면 `unresolved`. `resolved`는 작업함에서 빠지고 이력에만 보이며, 일지는 더 고칠 수 없습니다(409). 일지 반복 여부 `recurrence` = 처음 / 반복 / 모름.
- 승인한 일지는 칸이 잠깁니다. 고치려면 승인을 취소하고 다시 승인합니다.
- 시각: 조치의 `started_at` · `ended_at`과 `at`은 작업 시각(사람이 일한 시각, UTC ISO)입니다. `event`의 시각(데이터 시각)과 섞지 않습니다.
- 점검 항목 id(`S01-K1` 등)는 RAG에 아직 없어 FE가 임시로 붙였습니다 (위 RAG 요청 3).
- 매뉴얼 쪽수는 RAG의 `page_number`(PDF 파일의 쪽 순서, 1부터)를 그대로 씁니다. 원본은 `rag/data/festo_manual.pdf`이고, api가 생기기 전에는 개발 서버가 mock 모드에서만 이 파일을 내려줍니다(`ui/vite.config.ts`). 빌드 결과물에는 들어가지 않습니다.
- 수동 모드에서 직접 찾은 원인으로 조치 · 결과를 저장할 때 `situation_id`는 `"MANUAL"`입니다.
- 도움 요청을 보내면 `status`가 `handed_over`가 되고 `records.closure`에 `handed_over`가 남습니다. 작업함에 남고 이력의 "도움 요청"에도 보입니다.
- RAG에 요청할 것: 매뉴얼 전문 키워드 검색 (위 RAG 요청 표의 "수동 모드 문서 검색"). 생기면 `/manuals/festo/search`가 그 결과를 돌려주면 됩니다. 다음 화면을 만들 때 이 표에 추가합니다.
