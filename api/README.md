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

## 브라우저에서 직접 질문하기

API 실행 후 http://127.0.0.1:9000/docs 에서 POST /api/assistant/ask를 펼칩니다.
Try it out을 누른 뒤 Request body의 E029 예시 질문을 수정하고 Execute를 누릅니다.
Request body가 없으면 API 서버를 재시작하고 페이지를 새로고침합니다.
