# api

UI와 ml · rag를 연결하는 API 서버. UI는 이 서버만 호출한다.

> **지금 있는 코드는 FE(한혜원)가 만든 데모용 초안이다 (2026-10-23).** api 담당이 정해지지 않아서, 화면을 실제 ML 데이터로 돌려 보려고 먼저 만들었다. 구조 · 주소 · 임시 결정에 다른 의견이 있으면 말해 주세요. 담당이 정해지면 넘기거나 같이 고친다.
> 팀원별로 알아야 할 것: [`ui/docs/08-api-draft.md`](../ui/docs/08-api-draft.md)

## 켜는 법

ML 서버 → api 서버 → 화면 순서로 켠다.

```bash
# 1) ML 서버 (ml/README 0절: 모델 .joblib을 ml/models/에, test CSV를 ml/test/에)
cd ml && uvicorn main:app --port 8000

# 2) api 서버
cd api
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # 처음 한 번
.venv/bin/uvicorn main:app --port 9000

# 3) 화면: ui/.env.local 에 VITE_USE_MOCK=false, VITE_API_PROXY_TARGET=http://localhost:9000
cd ui && npm run dev            # http://localhost:5173

# 4) 고장 상황 흘려보내기 (ML이 감지 → api가 케이스로 만듦)
cd ml && python .py/replay.py test/test_0.csv --chunk-s 600     # 약 2분, 알림 12건
```

| 환경 변수 | 기본값 | 뜻 |
|---|---|---|
| `ML_URL` | `http://localhost:8000` | ML 서버 주소 |
| `API_DB` | `api/state/api.db` | SQLite 파일 (깃에 안 올라감). 지우면 처음부터 |

## 파일

| 파일 | 하는 일 |
|---|---|
| `main.py` | 주소 19개. 모양은 FE mock(`ui/src/mocks/handlers.ts`)과 같다 (`ui/docs/05` "저장 API", "AI 챗봇 API") |
| `store.py` | SQLite 저장. 케이스 한 건 = JSON 한 줄. 같은 `event_id`는 한 번만 처리 |
| `guidance.py` | 고장 코드 → 상황 ID → 점검 순서 · 근거 쪽. **RAG를 붙일 자리** |
| `check_criteria.json` | 점검 항목별 정상 기준 · 확인 방법 (FE 초안: 매뉴얼 · 정의서 원문에 적힌 것만, RAG 확인 전). `guidance.py`가 점검 항목에 붙인다 |
| `assistant.py` | AI 챗봇 답 (임시, 매핑에서 근거만 골라 조립) |

## 흐름

```
ML 서버 ──(고장 알림: GET /handoffs 를 api가 5초마다 당겨 옴, 또는 ML이 POST /api/ml/handoffs 로 push)──▶ api
api ── 케이스 생성: event = 알림 그대로, score_trace = ML trace 30분, guidance = guidance.py
화면 ◀── GET /api/cases, /api/cases/{id}, /api/guns(ML /guns + 열린 케이스) …
화면 ──▶ 점검 · 판단 · 조치 · 일지 · 도움 요청 · 사진 저장 (SQLite. ML · RAG에는 보내지 않음)
```

ML push로 받으려면 ML 서버를 `RSW_RAG_URL=http://localhost:9000/api/ml/handoffs`로 켠다 (이름은 RAG지만 알림을 보낼 주소 설정이다).

## 임시 결정 (바뀔 수 있음)

- **고장 코드 → 상황 ID 표** (`guidance.CODE_TO_SITUATION`): E012→S01, E016→S02, E028→S03, E029→S04. ML 핸드오프 v1.2부터 상황 ID가 오지 않아서 api가 바꾼다. RAG 입력 방식이 정해지면 고친다 (`ui/docs/07` 요청 1).
- **원인 후보는 현상 하나**: 코드의 상황(S01~S04) 하나와 그 점검 순서만 준다. 원인 상황(S05~S09)을 고르는 근거가 지금은 없다. 코드 없는 알림(모델 신호만)은 `no_evidence` → 화면은 수동 모드.
- **반복 알림 묶기**: 같은 설비 · 같은 코드(코드 없음끼리도)로 아직 끝나지 않은 케이스가 있으면 새 케이스 대신 `repeats`에 쌓는다. test_0은 알림 12건 → 케이스 3건.
- **긴급도**: ML은 critical일 때만 알림을 보내므로 케이스 `severity`는 모두 `critical`.
- **점검 항목 정상 기준**: RAG에 항목별 기준이 없어 `check_criteria.json` 초안을 쓴다. 항목과 쪽을 짝지은 건 FE 판단이다 (`ui/docs/07` 요청 3).
- **오알람**: `POST /api/cases/{id}/close`에 `false_alarm`. 작업함 · 열린 케이스 묶기에서 빠지고 이력에 남는다.
- **RAG 없이 근거만**: `rag/data/situation_mapping.json`을 직접 읽는다 (RAG의 "근거만 조회"와 같은 수준). LLM 답은 없다. Festo 원문은 확인한 오류 12만 넣었다.

## 아직 없는 것

- RAG `answer_question` 호출 (Neo4j · OpenAI 키 필요), 원인 후보 여러 개, 한국어 조치 문장
- 자동 테스트 (10/23 수동으로 저장 주소 전부 확인)
- 사진은 DB에 data URL 글자로 저장 (데모 규모용, 나중에 파일 저장으로)
- ML trace는 최근 6시간만 있어서, 알림을 늦게 당겨 오면 그래프(`score_trace`)가 비어 있다 → push로 받으면 해결
- 화면에 반복 횟수(`repeat_count`) 표시
