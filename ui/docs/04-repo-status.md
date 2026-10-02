# 04. 팀 레포 진행 상태 (2026-10-02 커밋 기준)

레포: `happy-sysiphus/maintenance-assistant-agent`. 커밋 5개(마지막 `b1a9db1`, 10-02 19:48 KST).
이 문서는 그 시점의 스냅샷이다. 레포가 바뀌었으면 다시 확인할 것.

읽은 범위: 코드와 실험 결과는 보조 에이전트가 전부 읽고 보고한 것을 정리했다. Claude가 직접 읽은 것은 `ml/techspec.md`, `ml/test.md` 앞 절반, `ml/README.md`의 3절·5.9절·5.10절, 각 폴더 README, 예시 JSON이다. `ml/history.md`, `ml/MEMORY.md`는 자세히 읽지 못했다. ML 테스트는 실행하지 못했다.

## 한눈에

| 폴더 | 올라온 것 | 진행 정도 |
|---|---|---|
| `ml/` | FastAPI 이상 감지 서버, 학습·전처리 코드, 실험 결과, 문서 6종, 테스트 26개 | 서버·문서·테스트 완성. 예측 성능은 스스로 "불충분" 판정 |
| `rag/` | 매뉴얼 → Neo4j 적재 파이프라인, 답변 함수, S01~S10 매핑 | 함수와 CLI까지 완성. 웹 서버·테스트 없음 |
| `api/` | README 한 줄 ("UI와 ml·rag를 연결하는 API 서버, UI는 이 서버만 호출") | 시작 전 |
| `ui/` | README 한 줄 | 시작 전 |

| 흐름 단계 | 상태 |
|---|---|
| 센서 → 이상 감지 → critical 판정 | 됨 (ML) |
| critical → 핸드오프 문서 생성 | 됨 (ML) |
| 핸드오프를 받는 쪽 | 없음 (ML은 `POST /diagnose`로 보낼 준비가 됨) |
| 상황 ID → 매뉴얼 근거·답변 | 됨 (RAG, 파이썬 함수로만) |
| 화면이 부를 서버 | 없음 |
| 케이스·점검·판단·조치·일지 저장 | 없음 |

## ML (`ml/`)

### 서버 (`ml/main.py`, FastAPI)

실행: `ml/`에서 `uvicorn main:app --reload` (기본 8000번). 인증 없음. **CORS 설정 없음** → 브라우저에서 직접 호출 불가.
**모델 파일(`models/baseline_ensemble.joblib`)과 데이터(`train/`, `test/`)가 레포에 없어서 클론만으로는 서버가 뜨지 않는다.**

| 메서드 · 주소 | 하는 일 | FE 화면 |
|---|---|---|
| `POST /predict` | 센서 묶음(최대 1200행)을 받아 판정. 기록이 모자라면 202 | FE는 직접 안 부름 (재생 스크립트가 부름) |
| `GET /handoffs?gun_id=&delivery=&limit=` | critical 이벤트 문서 목록 | 작업함, 재발 확인 |
| `GET /handoffs/{event_id}` | 이벤트 한 건 | 원인 찾기 상단 |
| `POST /handoffs/preview` | AnomalyResult → 핸드오프 변환 (개발용) | — |
| `GET /guns` | 설비별 상태 (워밍업, 마지막 점수, 드리프트 경고) | 설비 상태 |
| `GET /guns/{id}/stats`, `GET /stats` | 일별 알람 통계 | 설비 상태 |
| `GET /guns/{id}/trace?minutes=` | 최근 5~360분의 분 단위 점수·임계값·편차·코드 이벤트 | 신호 그래프 |
| `GET /guns/{id}/trace/view` | 한국어 추적 차트 HTML (`static/trace.html`) | 참고용 |
| `DELETE /guns/{id}` | 설비 기록을 지우고 워밍업 다시 시작 (정비 후) | 조치 저장 후 |
| `GET /model`, `GET /health` | 모델 설정·지표, 서버 상태 | 공통 |

### 판정 규칙

- 60초 창마다 점수를 낸다. 최신 응답은 최대 59초 늦을 수 있다.
- `critical` = 알람이 180초 이상 지속, 또는 종료 코드 규칙 발화. `warning` = 알람 1회, 또는 반복 규칙.
- 종료 코드 규칙: E012·E016·E028·E029. 30분 쿨다운. 같은 코드가 24시간 안에 다시 뜨면 `rule_repeat`로 낮춰 warning, 핸드오프 없음.
- 비용접 비율이 0.5를 넘는 창과 워밍업 중인 설비의 모델 알람은 보류(`alarm_held`).
- 핸드오프는 critical 에피소드의 첫 요청과 (반복이 아닌) 규칙 발화마다 1회 만든다.

### 핸드오프 (`RagHandoff` v1.0) — FE가 주로 쓰는 문서

예시: `ml/docs/rag_handoff.example.json`. 주요 필드:
`event_id`, `gun_id`, `detected_at`, `trigger{source, rule_code, anomaly_score, threshold, alarm_duration_s, sustained}`, `summary_ko`, `sensor_findings[]{sensor, sensor_name_ko, direction, deviation_z, share, text_ko}`, `fault_class{code, name_ko, terminal_code, situation_id, basis}`, `symptoms[]`, `situation_ids[]`, `context{non_welding_share, alarm_held, gun_norm, …}`, `detector`, `caveats[]`.

- 전달 방식: `/predict` 응답 안, `GET /handoffs`(pull), 또는 환경 변수 `RSW_RAG_URL`을 주면 그 주소로 POST(push, 재시도 포함).
- 시각은 `Z`가 없는 UTC 문자열이다 (예: `2021-09-03T09:30:59`).
- `caveats`의 수치(AUROC 0.64, 0/8건)는 옛 값이다. 현재 지표 파일은 0.713, 2/8건.

### 저장

- 디스크(`state/`): 판정 상태, 규칙 이력, 일별 통계, 핸드오프 기록(30일 보관).
- 메모리만: 최근 센서 버퍼, trace, 진행 중인 워밍업 → 서버를 재시작하면 사라진다.

### 성능 (`ml/test.md`, `ml/models/*.json`, 테스트 8개 설비)

- 종료 코드 규칙: 8개 중 8개에서 고장 약 9분 전 발화. critical 오발화 0.18회/일/설비.
- 모델 단독: AUROC 0.713(규칙 이전 구간 0.671), 고장 전 1시간 창의 7.5%만 잡음. 30분 이상 미리 알리는 경우 8개 중 0개.
- 정상 구간 오알람 0.6%.
- 문서의 결론: 신호가 사실상 마지막 10분(종료 코드)에만 있다. 믿을 만한 키는 종료 코드에서 온 첫 번째 상황 ID다.
- 센서 증상 → 상황 ID 규칙(P1~P9)은 매뉴얼 기반이며 데이터로 검증되지 않았다.

### ML이 적어 둔 남은 일 (`ml/techspec.md`)

- 필수 D-2: 온톨로지·RAG와 입력 형식·주소 합의
- D-3: 핸드오프 v1.1 — `error_timeline[]`, `sensor_trend[]`, `recent_events[]` 등 추가
- D-4: RAG 응답 형식 검증
- C-3: 설비별 고장 유형 정보로 오발화 거르기 (현장 정보 없으면 보류)
- B-8: 설정값 전환 구간 오알람 (해법 못 찾음)

### 기타

- `ml/.py/replay.py`: CSV를 실시간처럼 `/predict`로 흘려보내는 스크립트. 예: `python .py/replay.py test/test_0.csv --start-hours 156 --chunk-s 600`
- `ml/.py/mock_rag.py`: 연동 연습용 가짜 RAG 서버 (`POST /diagnose`)
- `ml/pic code/`: 절반은 이 프로젝트와 무관한 다른 연구 그림

## RAG (`rag/`)

### 구성

- 문서: Festo 매뉴얼(`rag/data/festo_manual.pdf`, 101쪽 중 32쪽 파싱)과 팀 문서 "RSW 용접건 MVP 오류 상황 정의서"(`rsw_situations.pdf`, 11쪽).
- 파이프라인: `python -m rag parse → chunk → ingest → build` (PDF → 표 행 단위 청크 → Neo4j → S01~S10 그래프).
- 매핑: `rag/data/situation_mapping.json` — 상황마다 점검 순서(3~5단계), Festo 오류 번호 후보(primary/related), 근거 쪽수, 부품, 주의 문구(`coverage_note`).
- 필요 환경: Neo4j, OpenAI API 키 (`rag/.env`). 기본 모델 `gpt-5.4-mini`.

### 답변 함수

```python
from rag import answer_question
result = answer_question(situation_ids=["S02"], question="…")   # 동기 함수
```

- 반환: `status`, `answer`(마크다운 글), `evidence[]`, `missing_ids`, `citation_check`, `review_notes`, `usage`, `error` 등.
- `answer` 형식: `## 오류 상황 해석` → `## 원인 후보` → `## 점검 체크리스트` → `## 답변 요약`. 인용은 `[Festo PDF p.N]`, `[RSW PDF p.N]`. 앞뒤에 "판독 구분", "원본 확인 필요" 문단이 붙을 수 있다.
- `evidence[i]`: `situation_id`, `name`, `definition`, `check_sequence[]`(글자 목록), `coverage_note`, `diagnostics[]`(`number`, `name`, `description`, `error_elimination`, `page_number`, `priority`, `rationale`), `direct_evidence[]`(`text` 최대 1200자, `page_number`, `rationale`, `review_needed`), `components[]`.

| status | 뜻 |
|---|---|
| `ok` | 답변 생성과 인용 검사 완료 |
| `retrieval_only` | LLM 없이 근거만 조회 (`use_llm=False`) |
| `unknown_situation_id` | 그래프에 없는 ID |
| `no_evidence` | 상황은 있으나 연결된 근거 없음 |
| `needs_citation_review` | 답변은 있으나 인용 검토 필요 |
| `error` | DB·LLM 실행 실패 |

### 아직 없는 것

- 웹 서버 (HTTP 주소) — 파이썬 함수와 CLI만 있음
- ML 핸드오프를 받는 코드 (`event_id`, 센서 증상을 받는 인자 없음)
- 구조화된 원인 후보·점검 항목 (글 속에만 있음). 상황 정의서의 "원인 후보" 목록은 데이터에 들어 있지 않음
- 후속 입력 처리 (점검 결과, "이 후보는 아님") — 함수는 상태가 없음
- 벡터 검색 — 상황 ID로 정해진 조회만 함. 질문 문장은 LLM 설명에만 쓰임
- Festo 외 문서 (MILCO, ABB, FANUC, Yaskawa)
- 테스트, 실행 결과 예시
- (추정) `components[]`의 `priority`·`rationale`이 비어 나옴 — 저장 때 값을 넣지 않음. 실행해 확인하지는 못함

## ML과 RAG가 엇갈리는 곳

- **응답 형식**: ML README 5.10.6은 7칸 리포트 JSON(`cause_candidates`, `recommended_actions` 등)을 기대하는데, RAG는 마크다운 글 + `evidence`를 준다.
- **원인 후보의 뜻**: ML 예시에서는 상황 ID, RAG에서는 Festo 오류 번호와 글 속 설명.
- **넘기는 정보**: ML은 센서 증상·요약·주의 문구까지 넘기는데, RAG는 상황 ID와 질문만 받는다.
- 지금 두 파트를 잇는 것은 `situation_ids` 하나뿐이다.

## 백엔드(`api/`)가 필요한 이유

1. RAG는 파이썬 함수라 브라우저에서 부를 수 없다.
2. RAG가 쓰는 OpenAI 키와 Neo4j 비밀번호를 프론트에 둘 수 없다.
3. 케이스·점검 결과·후보 판단·실제 조치·결과·일지·이력을 저장할 곳이 없다.
4. ML 핸드오프를 받아 RAG를 부르는 주체가 없다.
5. ML 서버에 CORS가 없다.

데모용 최소 범위: FastAPI + SQLite(또는 JSON 파일), 주소 10개 안쪽 — 핸드오프 수신 → 케이스 생성 → RAG 함수 호출 → ML trace·guns 중계 → 점검·판단·조치·결과 저장.
누가 만들지는 미정이다. Claude 추천은 "API 모양은 FE가 정하고, 구현은 ML 담당에게 먼저 부탁, 안 되면 FE가 최소 범위로".
