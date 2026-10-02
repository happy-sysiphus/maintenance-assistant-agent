# 06. RAG 담당에게 — 어떤 형식으로 주면 되나요

2026-10-02 · FE 담당

캔버스의 "API ③ RAG · LLM" 보드(`/cases/{id}/guidance` 예시)는 **팀 레포가 올라오기 전에 그린 옛 초안**입니다. 그 모양 그대로 맞춰 주실 필요는 없습니다.

결론: **지금 `answer_question()`이 주는 `evidence` 구조는 그대로 두고, 아래 네 가지만 더해 주시면 됩니다.** 화면용 모양으로 바꾸는 일(후보 순서, 상태, 버전 등)은 api 서버가 합니다.

## 캔버스 초안 ↔ 지금 RAG가 주는 것

| 캔버스 초안의 필드 | 지금 RAG에 있는 것 | 누가 채우나 |
|---|---|---|
| `candidates[].id` · `rank` | 상황 ID 순서 (ML `situation_ids`) | ML · api |
| `candidates[].name` | `evidence[].name` · `definition` | **RAG (이미 있음)** |
| `candidates[].status` (reviewing / excluded) | 없음 | api (사람 판단을 저장) |
| `candidates[].signals` | ML 핸드오프 `sensor_findings` | ML |
| `checklist[].title` | `evidence[].check_sequence` (글자 목록) | **RAG (있음 · 요청 3)** |
| `checklist[].id` · `evidence` | 없음 | **RAG (요청 3)** |
| `checklist[].input` · `unit` | 없음 | 안 주셔도 됨 (화면은 정상 / 이상 / 미실시 선택으로 통일) |
| `evidence[].doc` · `page` | `direct_evidence[].page_number`, `diagnostics[].page_number` | **RAG (이미 있음)** |
| `evidence[].quote` | `direct_evidence[].text` (최대 1,200자) | **RAG (요청 4)** |
| `evidence[].grade` · `reason` | 없음. 대신 `coverage_note` | 안 주셔도 됨 (`coverage_note`로 대체) |
| `summary.text` · `cites` | `answer` 안의 "답변 요약" 부분 | **RAG (요청 2)** |
| `mode` (guide / manual) | `status` | api (`no_evidence` · `unknown_situation_id` → 수동 모드) |
| `version` | 없음 | api |

## 요청 1 (필수) · 예시 출력을 레포에

S01~S10 각각의 `answer_question` 결과 JSON을 `rag/docs/`에 올려 주세요. FE와 api는 Neo4j·OpenAI 키 없이 이 파일로 먼저 개발합니다.

```bash
python -m rag ask --situation-ids S01 --question "이 상황의 원인 후보와 점검 순서를 알려주세요." --json --output rag/docs/S01.example.json
```

## 요청 2 (필수) · 답변을 제목별로 나눠서

지금 `answer`는 네 제목이 들어간 마크다운 글 한 덩어리입니다. 화면은 "답변 요약"만 따로 AI 요약 카드에 쓰기 때문에, 제목별로 잘라 주시면 됩니다. 기존 `answer`는 그대로 두고 필드만 추가하면 됩니다.

```json
{
  "status": "ok",
  "answer": "## 오류 상황 해석\n…(지금 그대로)…",
  "answer_sections": {
    "S01": {
      "interpretation": "…",
      "cause_candidates": "…",
      "checklist": "…",
      "summary": "…"
    }
  },
  "evidence": [ "…(지금 그대로)…" ]
}
```

- 키는 상황 ID입니다. 여러 ID를 한 번에 물으면 ID마다 하나씩.
- 각 값은 그 제목 아래의 글(마크다운)입니다. 인용 `[Festo PDF p.90]`은 그대로 두셔도 됩니다.
- "판독 구분", "원본 확인 필요" 문단은 `answer_sections`에 넣지 말고 지금처럼 `review_notes`로만 주세요.

## 요청 3 · 점검 항목에 id와 근거 쪽수

화면은 점검 항목마다 사람이 정상 / 이상 / 미실시를 고르고 저장합니다. 지금 `check_sequence`는 글자 목록이라 저장할 키가 없고, 항목별 근거 쪽수도 알 수 없습니다. `situation_mapping.json`과 `evidence[].check_sequence`를 아래처럼 바꿔 주세요.

지금:

```json
"check_sequence": ["본관 공급 압력 측정", "필터·레귤레이터 확인"]
```

요청:

```json
"check_sequence": [
  { "id": "S01-K1", "title": "본관 공급 압력 측정", "evidence_pages": [90], "diagnostic_numbers": [12] },
  { "id": "S01-K2", "title": "필터·레귤레이터 확인", "evidence_pages": [], "diagnostic_numbers": [] }
]
```

- `id`는 한 번 정하면 바꾸지 않는 값 (`상황ID-K순번`).
- 연결되는 근거가 없으면 빈 배열로 두세요. 지어내지 않습니다.
- 쪽수와 번호는 예시입니다.

## 요청 4 · 짧은 인용문

`direct_evidence[].text`는 덩어리가 길어서 인용 박스에 넣기 어렵습니다. 매핑 JSON에 이미 있는 `contains` 문구를 `quote`로 같이 돌려주세요.

```json
{
  "id": "…",
  "page_number": 90,
  "quote": "Check cable to the MPYD",
  "text": "…(지금 그대로, 최대 1,200자)…",
  "rationale": "…",
  "review_needed": false
}
```

## 안 바꾸셔도 되는 것

- 웹 서버: `answer_question()`은 api 서버가 파이썬 함수로 부릅니다.
- `status` 6종: 그대로 씁니다 (`ok` / `retrieval_only` / `unknown_situation_id` / `no_evidence` / `needs_citation_review` / `error`).
- 후보를 "아님"으로 판단했을 때의 처리: api가 다음 상황 ID로 넘어갑니다. RAG는 상태를 기억할 필요가 없습니다.
- 적용 등급(`grade`), 입력 형식(`input` · `unit`), `version`, `mode`: 주지 않으셔도 됩니다.

## 확인 부탁드리는 것

- 답변 생성에 보통 몇 초 걸리는지, `use_cache=True`를 써도 되는지
- 대시보드는 질문 없이 자동으로 호출합니다. 고정 질문으로 어떤 문구가 좋을지
- ML 핸드오프의 `summary_ko` · `event_id`를 같이 넘길 선택 인자를 둘 수 있는지 (없어도 진행 가능)
- `components[]`의 `priority` · `rationale`이 비어 나오는 것으로 보입니다 (실행해 보지는 못했습니다)

자세한 배경과 ML 쪽 요청은 `05-fe-api-requests.md`에 있습니다.
