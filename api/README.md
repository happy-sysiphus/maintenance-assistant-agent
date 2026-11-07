# 정비 지원 API

UI와 ML·RAG를 연결하는 FastAPI 서버입니다.
RAG 요청별 개발 답변은 [ui/docs/09-rag-development-reply.md](../ui/docs/09-rag-development-reply.md)를 참고하세요.

## 실행

저장소 루트에서 다음 순서로 실행합니다.

```powershell
python -m pip install -r api/requirements.txt
# rag/.env에 개인적으로 전달받은 Aura·OpenAI 설정을 입력
cd api
python -m uvicorn main:app --port 9000
```

화면은 ui/.env.local에 VITE_USE_MOCK=false, VITE_API_PROXY_TARGET=http://localhost:9000을 설정합니다.
ML_URL의 기본값은 http://localhost:8000, API_DB의 기본값은 api/state/api.db입니다.
ML이 꺼져 있어도 챗봇·PDF와 저장된 케이스는 사용할 수 있습니다.

## 연결과 저장

- ML 알림: GET /handoffs를 당겨 오거나 POST /api/ml/handoffs로 수신합니다.
- RAG 입력: 알림의 trigger.rule_code 또는 fault_class.terminal_code만 사용합니다. 센서는 보내지 않습니다.
- 신규 케이스: E 코드당 후보 하나와 고정 점검 ID를 즉시 생성합니다. ML 수신 응답은 LLM 생성을 기다리지 않습니다.
- GET /api/cases/{id}: 최초 상세 조회에서 한국어 구조화 안내를 생성·저장합니다. 동일 질문은 캐시를 재사용합니다.
- guidance에는 status·candidates와 함께 sections·causes·evidence·answer 등 실제 RAG 필드가 있습니다.
- 기존 S 기반 저장 케이스의 기록 키는 자동 변환하지 않습니다. 새 케이스부터 E 코드 키를 사용합니다.
- 점검·판단·조치·결과·일지·도움 요청은 기존 SQLite에 저장합니다. 생성 중 추가된 기록도 보존합니다.
- 같은 event_id는 중복 처리하지 않고, 같은 설비·코드의 열린 케이스는 반복으로 묶습니다.

## 챗봇·문서

POST /api/assistant/ask:

```json
{"question":"그럼 다음에는?","history":[{"role":"user","text":"E029 오류를 설명해 주세요."}],"photos":[]}
```

기존 blocks·caution·situation_ids와 함께 구조화 RAG 필드도 반환합니다.
일반 질문은 error_code=null, situation_ids=[]입니다. 사진은 분석하지 않습니다.

| 경로 | 내용 |
|---|---|
| GET /api/situations | E 코드 네 개의 목록 |
| GET /api/manuals/{doc}.pdf | festo / manual82 / diagnostic4 / whitepaper11 원본 |
| GET /api/manuals/festo/search?q= | 네 문서의 키워드 검색. 경로 이름은 기존 UI 호환용 |
| GET /api/health | ML 상태와 RAG 연결 방식 |

문서 검색 결과에는 doc·page가 함께 있습니다. PDF의 page는 1부터 시작합니다.
기존 rsw PDF 경로는 제거했습니다.

## UI 연동에서 바뀐 부분

화면 수정은 UI 담당 범위입니다. S ID 기반 원인 후보 전환 대신 E 코드당 안내 하나를 사용합니다.
ManualDoc와 문서 이름표를 네 종류로 확장하고, guidance.causes의 한국어 조치·근거를 표시해야 합니다.
기존 저장 주소와 candidates/checks의 기본 모양은 유지합니다. 새 답변의 필드 상세는 RAG README와 실제 예시를 참고하세요.

## 검증

저장소 루트에서 python -B -m unittest discover -s rag/tests -v.
API 의존성 파일은 RAG 의존성도 포함합니다. 테스트 DB는 임시 디렉터리를 사용합니다.

## 현장 기억 (D-8, FIELD_MEMORY=1)

`FIELD_MEMORY=1`로 켜면 닫힌 케이스의 사람 판정을 다시 읽어 참고 문장을 붙이고, 새 케이스 카드의 점검 순서를 정합니다(R2).
끄면 위 동작과 같습니다. 단, 켜고 만든 케이스(field_order가 있는 행)는 끈 뒤에도 순서와 고정을 지킵니다(스펙 4.5). 시연은 새 API_DB로 합니다.
RAG 안내 카드의 내용은 바꾸지 않습니다. 순서는 R2 조건을 만족한 새 케이스만 바뀝니다.
문장은 참고용이고 매뉴얼 근거가 아닙니다.
라벨은 "오알람으로 닫기" = 오알람, "해결 종료" = 고장 확인입니다. 미해결·도움 요청은 세지 않습니다.

- GET /api/cases/{id}: 응답에 field_memory(applied·note_ko·checks 등)를 붙입니다. field_memory는 DB에 저장하지 않습니다.
  - 같은 설비·코드에서 오알람으로 닫은 기록이 고장 확인보다 많으면 "오알람(운영 원인)인지부터 확인" 문장을 씁니다(R1).
  - 카드의 첫 점검이 field_order.lead이면 applied에 R2를 더하고, 순서를 바꾼 이유 문장을 씁니다.
  - 이 카드 점검 항목(E0xx-Kn)의 지난 이상·정상 횟수를 함께 적습니다. 지금 보는 케이스는 빼고 셉니다.
- R2(케이스를 만들 때 한 번):
  - 조건: 같은 코드의 해결 종료 케이스(카드 밖 원인으로 해결한 것은 뺌)에서 '이상'으로 가장 많이 기록된 점검이 매핑에서 앞의 모든 점검보다 3건 이상 많고, 설비 2대 이상에서 나와야 합니다.
  - 동작: 그 점검과 그 원인을 맨 앞에 둡니다.
  - 저장: 결정은 케이스 최상위 field_order에 저장되고 케이스 상세 응답과 케이스 쓰기 응답에 실립니다(목록에는 없습니다).
  - 예외: 앞세울 원인 앞의 원인에 앞세울 원인이 갖지 않은 조치가 있으면 바꾸지 않습니다(같은 조치를 가지면 옮김). 순서를 정한 케이스에 판단·조치·결과가 기록되면 카드를 다시 만들지 않습니다.
  - 문서: 설계는 ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md, 다른 파트 요청은 ml/docs/card-order-requests.md에 있습니다.
- GET /api/memory: 사람 라벨, 설비·코드별 오알람·고장 확인 수, 점검 항목별 수. 켰을 때만 있고 /docs에는 없습니다.
- 챗봇: 답 끝에 현장 기록 문단 하나를 붙입니다. evidence·caution·인용에는 넣지 않습니다.
- 닫기: 오알람 이유(reason)·메모(note)를 closure에 남깁니다.
- 알림 수신: 한 번에 하나씩 처리합니다.

코드는 field_memory.py, 테스트는 tests/입니다. 루프 전체 설명은 ml/docs/hitl-loop.md에 있습니다.

```powershell
cd api
..\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider   # 저장소 .venv
py -3.11 -m pytest tests -q -p no:cacheprovider                     # langgraph·pandas가 있는 Python
```

저장소 .venv에는 langgraph·pandas가 없어 test_driver_contract.py는 건너뜁니다.
두 번째 줄은 그 Python에 requirements.txt(RAG 의존성 포함)를 설치한 뒤 실행합니다. 드라이버 계약 테스트까지 돕니다.

## 11/04 FE 추가 (UI 연동)

현장 기억을 화면에 붙이면서 api에 더한 것입니다. 기존 응답 필드는 바꾸지 않고 추가만 했습니다.

- GET /api/cases/{id}: `field_memory.recent`(켰을 때만). 같은 설비·코드로 오알람·해결 종료된 케이스 최근 3건
  `{case_id, outcome, closed_at, cause, reason}`. 해결이면 판단한 원인 이름, 오알람이면 닫을 때 고른 사유.
  화면은 note_ko 문장 대신 이 목록을 "이전 기록"으로 보입니다(상황 화면 오른쪽).
- 닫힌 케이스 읽기 전용: 해결 종료·오알람 케이스에 기록·닫기·도움 요청을 보내면 409 "closed".
  다시 닫으면 마지막 closure가 현장 기억 라벨을 덮어쓰기 때문입니다. 미해결·도움 요청은 이어서 할 수 있습니다.
- 저장 잠금: 케이스를 고치는 주소와 알림 수신(ingest)은 `_WRITE_LOCK` 하나로 한 번에 하나씩 처리합니다(플래그 무관).
  상세 조회는 RAG 생성을 잠금 밖에서 하고, 안내가 바뀌었을 때만 저장합니다(그냥 열면 저장 안 함).
  hitl-loop.md ④ 요청 9(ingest 경쟁)·10(상세 조회가 닫기를 덮음)에 대한 처리입니다.
- 챗봇 피드백: POST /api/assistant/feedback가 질문·답·상태·코드와 함께 SQLite `feedback` 표에 저장합니다
  (같은 답은 마지막 평가만). GET /api/assistant/feedback로 최근 순 목록을 봅니다. 현장 기억·RAG에는 아직 쓰지 않습니다.

## 브라우저에서 직접 질문하기

API 실행 후 http://127.0.0.1:9000/docs 에서 POST /api/assistant/ask를 펼칩니다.
Try it out을 누른 뒤 Request body의 E029 예시 질문을 수정하고 Execute를 누릅니다.
Request body가 없으면 API 서버를 재시작하고 페이지를 새로고침합니다.
