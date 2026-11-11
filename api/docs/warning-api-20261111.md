# WARNING API·UI 연동 구현 — 2026-11-11

## 구현 범위

- WARNING(v1.4 `early_warning`) 원본을 API → 기존 RAG 함수로 전달한다.
- 상세 조회는 센서 분석과 과거 이력 조회만 한다. 후보 생성·Claude 호출은 버튼 POST에서만 한다.
- UI는 기존 케이스 상세에서 `event_type`으로 WARNING 화면을 선택한다.
- CRITICAL은 기존 E 코드 기반 안내를 유지한다. 코드 없는 CRITICAL의 센서 기반 RAG 분석은 이번 범위 밖이다.
- ML 모델·특징·임계값·학습 로직은 수정하지 않는다.
- 피드백은 구현하되 `RAG_WARNING_FEEDBACK=0`이 기본이다. 비활성 상태에서도 해결 기록은 저장한다.

## 저장소 선택

기존 정비 기록은 API SQLite가 원본이므로, RAG의 `EarlyWarningRag`에 SQLite 메모리 어댑터를 연결했다. Neo4j로 해결 이력을 복제하지 않는다. Neo4j는 기존 문서 KG·검색에만 사용한다. 문서의 Neo4j 전송 대기·백오프 요구는 이 구현에서는 이중 쓰기가 없으므로 필요 없다.

기본 파일: `api/state/api.db` (`API_DB`로 변경 가능).

| 위치 | 저장 내용 |
|---|---|
| `cases.data.event` | 대표 경보 전체 원본 |
| `cases.data.repeats[].handoff` | 반복 경보 전체 원본(이번 버전부터) |
| `cases.data.warning_events[event_id]` | 실제 표시한 카드·검색 근거, 선택, 거절, 점검·조치 초안 |
| `cases.data.records` | 기존 조치·결과·일지·종료 필드와 동일한 구조로 최종 결과 저장 |
| `warning_resolutions` | `event_id` PK, `case_id` UNIQUE, 센서 패턴·선택 근거·실제 조치·확인·검토 상태 |

`warning_resolutions`는 RAG용 조회·검토 레코드이며 케이스 종료와 같은 SQLite 트랜잭션으로 저장한다. 실패 시 둘 다 롤백한다. 동일 종료 재시도는 같은 결과를 반환하며 다른 내용으로 재종료하면 409다. 기존 ML 평가 스크립트는 여전히 `/api/history` 또는 케이스 DB로 종료 결과를 읽을 수 있다.

## API 계약

아래 `:id`는 케이스 ID이며 요청 본문은 JSON이다.

| 메서드 | 주소 | 동작 |
|---|---|---|
| GET | `/api/cases/:id` | WARNING은 `warning_rag` 필드로 센서 소견 추가, CRITICAL은 기존 동작 |
| GET | `/api/cases/:id/warning?event_id=...` | 선택한 대표/반복 경보의 소견·저장 상태 조회 |
| POST | `/api/cases/:id/warning/candidates` | RAG 원인 후보 요청·결과 보존 |
| POST | `/api/cases/:id/warning/manual` | 센서 문맥+질문으로 매뉴얼 검색, 선택 가능한 근거 보존 |
| POST | `/api/cases/:id/warning/select` | 서버가 실제 표시한 사례 또는 근거 선택 |
| POST | `/api/cases/:id/warning/reject` | 표시한 자동 카드의 거절 기록 |
| PUT | `/api/cases/:id/warning/draft` | 작업자·실제 조치·선택 카드의 점검 결과 임시 저장 |
| POST | `/api/cases/:id/warning/resolve` | 작업자 확인·일지 승인 후 원자적 종료·해결 이력 저장 |
| POST | `/api/warning/resolutions/:event_id/review` | 운영자: 수동 근거를 기존/새 검토 사례에 연결 |
| POST | `/api/warning/resolutions/:event_id/acknowledge` | 운영자: 검토한 반례의 자동 표시 보류 해제 |

후보·거절 요청:
```json
{"event_id":"EW-0042"}
```
검색 요청:
```json
{"event_id":"EW-0042","question":"공급 압력을 어떻게 확인하나요?","explain":true}
```
선택 요청(둘 중 하나만):
```json
{"event_id":"EW-0042","manual_case_id":"manual:E012-C2"}
```
```json
{"event_id":"EW-0042","evidence_id":"검색 응답에 포함된 실제 ID"}
```
임시 저장:
```json
{"event_id":"EW-0042","worker":"작업자","actual_action":"실제로 수행한 조치","checks":{}}
```
해결 종료:
```json
{
  "event_id":"EW-0042",
  "actual_action":"실제 점검·조치 내용을 작성",
  "worker":"작업자",
  "worker_confirmed":true,
  "log_approved":true,
  "checks":{}
}
```
`checks`는 선택한 카드에서 받은 실제 점검 ID와 `normal|abnormal|unchecked` 값으로 구성한다. 임의 점검 ID는 거절한다. 카드 순번 대신 `event_id`+`manual_case_id`로 선택을 식별한다. 서버가 보존한 근거만 선택 가능하며 클라이언트가 인용문을 만들어 전송할 수 없다.

수동 근거 해결은 `pending_review`로 저장한다. 운영자가 검토하기 전에는 자동 추천에 사용하지 않는다. 검토 API는 `RAG_WARNING_REVIEW_TOKEN` 미설정이면 404, 설정 시 `X-Warning-Review-Token` 헤더가 필요하다. 토큰을 UI 환경변수에 넣지 않는다. 검토 본문은 기존 사례 연결이면 `{"existing_case_id":"manual:E012-C2"}`, 신규면 `title`, `equipment_scope`, 선택한 PDF 인용 안에 존재하는 `checks`/`actions` 발췌를 전달한다. 운영자 검토 화면은 이번 UI 범위에 포함하지 않는다.

오알람은 기존 `/close`를 사용하며 RAG 해결 사례에 포함하지 않는다. WARNING의 해결 종료는 새 `/warning/resolve`를 사용한다. 기존 주소로 해결 종료하면 409로 새 경로를 안내한다. 기존 UI의 CRITICAL 종료 경로는 유지한다.

## 화면 동작

1. WARNING 상세: 조기 경보·원인 미확정, 센서별 방향/관측 초/상태 전환 구간을 표시한다.
2. 대표·반복 경보를 선택한다. 저장하지 않은 조치가 있으면 경보 전환을 잠근다.
3. 과거 자동 카드 또는 원인 후보 보기 → 매뉴얼 후보·근거·적용 조건·과거 실제 조치를 표시한다.
4. 수동 검색 결과의 PDF를 열거나 해당 근거를 선택한다.
5. 선택한 카드의 점검 정상/이상, 작업자, 실제 조치를 임시 저장할 수 있다. 새로고침 후 복원한다.
6. 해결 확인과 정비일지 미리보기 승인 후 종료한다. 입력 내용을 최종 저장하므로 별도 임시 저장을 잊어도 조치가 보존된다.
7. 종료 후 읽기 전용으로 표시한다. 수동 근거는 검토 대기를 표시한다.

센서 방향은 RAG 원래 모듈의 첫/마지막 구간 중앙값 비교이며 인과·고장 확정이 아니다. 상태 전환 시간은 센서 블록 시작으로부터의 구간으로 표시한다.

## 피드백 OFF / ON

- OFF: 과거 해결 후보 조회·자동 추천을 하지 않는다. 센서 소견·매뉴얼 후보/검색·해결 기록 저장은 가능하다.
- ON: SQLite 해결 이력을 원래 RAG 패턴 규칙으로 조회한다. 서로 다른 방향성 있는 3채널이 필요하다.
- 자동 표시: 현재 매뉴얼 버전·설비 범위가 맞고, 같은 패턴+사례의 서로 다른 해결 케이스가 2건 이상이어야 한다. 같은 케이스 반복은 중복 집계하지 않는다.
- 설비 범위 미설정, 센서 부족, 미검토 수동 근거, 반례, 복수 적격 사례 등은 자동 표시하지 않는다.
- 한 케이스에서 자동 카드를 거절한 뒤 같은 카드를 해결 사례로 다시 제출하지 못하게 한다.
- OFF 동안 기록한 신규 해결도 나중에 ON으로 바꾸면 활용 가능하다. 이 수정 이전의 구형 기록은 실제 선택 근거가 없으므로 자동으로 매핑하지 않는다.

## 실행

기존 API·RAG 의존성 설치:
```bash
python -m pip install -r api/requirements.txt
```
RAG 문서 KG와 기존 키는 `rag/.env`의 설정을 사용한다. API WARNING 옵션은 셸 환경변수로 지정한다(`api/.env.warning.example`는 예시이며 자동으로 읽지 않음).
```bash
export RAG_WARNING_FEEDBACK=0
cd api
uvicorn main:app --port 9000
```
Windows PowerShell:
```powershell
$env:RAG_WARNING_FEEDBACK='0'
cd api
python -m uvicorn main:app --port 9000
```
피드백을 켤 때는 실제 설치 설비의 매뉴얼 적용 범위를 확인한다. 예를 들어 신형 장비 범위가 맞을 때만:
```powershell
$env:RAG_WARNING_FEEDBACK='1'
$env:RAG_WARNING_EQUIPMENT_SCOPE='cpx-weld-2g-sa'
```
ML에서 WARNING 핸드오프를 생성하려면 `RSW_EARLY_WARNING_HANDOFFS=1`이 필요하다(기존 ML 동작). API push는 `RSW_RAG_URL=http://localhost:9000/api/ml/handoffs`, 또는 기존 pull 경로를 사용한다.

UI `ui/.env.local`:
```dotenv
VITE_USE_MOCK=false
VITE_API_BASE_URL=/api
VITE_API_PROXY_TARGET=http://localhost:9000
```
```bash
cd ui
npm ci
npm run dev
```
매뉴얼 후보는 기존 문서 KG 및 Claude 연결이 필요하며 벡터 검색을 켜면 OpenAI 임베딩도 사용한다. `RAG_WARNING_VECTORS=0`으로 키워드 검색만 사용할 수 있지만 후보 생성은 여전히 Claude를 사용한다. 기본 상세 센서 분석·임시 저장·검증된 해결 저장은 외부 모델 호출 없이 가능하다.

## 검증과 한계

```bash
python -m pytest api/tests rag/tests -q -p no:cacheprovider
cd ui
npm run test:warning
npm run build
npm run lint
```
UI 통합 테스트는 실제 FastAPI와 임시 SQLite를 띄우고 React DOM 이벤트로 소견→후보→선택→임시 저장/재마운트→수동 검색/선택→해결 종료/읽기 전용을 검증한다. 문서 검색과 Claude만 합성 응답으로 대체한다. API 테스트는 피드백 ON/OFF, 반복 경보, 근거 조작 거절, 검토, 중복 종료, 트랜잭션 롤백, CRITICAL 회귀를 포함한다.

실제 Neo4j·Claude·OpenAI·운영 ML 연결, 운영 배포, 브라우저 레이아웃/클릭은 이 환경에서 검증하지 않았다. 브라우저가 로컬 서버에 접근하지 못해 화면 이미지를 실측 결과처럼 제공하지 않는다. PDF의 화면 구성도는 구현을 설명하는 도식이다. 기존 기본 mock 데이터에는 신규 WARNING 시나리오가 없으므로 실제 API 모드로 확인해야 한다.
