# WARNING API·UI 연동 구현 — 2026-11-11

## 구현 범위

- WARNING(v1.4 `early_warning`) 원본을 API → 기존 RAG 함수로 전달한다.
- 상세 조회는 센서 분석과 과거 이력 조회만 한다. 후보 생성·Claude 호출은 버튼 POST에서만 한다.
- UI는 기존 케이스 상세에서 `event_type`으로 WARNING 화면을 선택한다.
- CRITICAL은 기존 E 코드 기반 안내를 유지한다. 코드 없는 CRITICAL의 센서 기반 RAG 분석은 이번 범위 밖이다.
- ML 모델·특징·임계값·학습 로직은 수정하지 않는다.
- `RAG_WARNING_FEEDBACK=1`이 기본이다. 비활성 상태에서도 해결 기록은 공용 Aura에 저장한다.
- 수동 모드는 질문별 매뉴얼 검색과 근거 선택을 제공한다.

## 저장소 선택

API는 케이스 원본, 작업자 기록, 종료 상태를 SQLite에 보존한다. `EarlyWarningRag`는 공용 Aura에서 재사용할 해결 이력을 저장·조회·검토한다. 모든 API 인스턴스는 같은 `NEO4J_DATABASE`와 `RAG_WARNING_NAMESPACE`를 사용한다.

기본 파일: `api/state/api.db` (`API_DB`로 변경 가능).

| 위치 | 저장 내용 |
|---|---|
| `cases.data.event` | 대표 경보 전체 원본 |
| `cases.data.repeats[].handoff` | 반복 경보 전체 원본(이번 버전부터) |
| `cases.data.warning_events[event_id]` | 실제 표시한 카드·검색 근거, 선택, 거절, 점검·조치 초안 |
| `cases.data.records` | 기존 조치·결과·일지·종료 필드와 동일한 구조로 최종 결과 저장 |
| SQLite `warning_outbox` | 선택한 경보 원본·종료 요청·Aura 전송 상태 |
| Aura `WarningResolution`, `WarningPattern`, `WarningManualCase` | 공용 해결 이력·센서 패턴·매뉴얼 사례 |

종료와 전송 대기 등록은 하나의 SQLite 트랜잭션에서 처리한다. 백그라운드 작업이 원본과 종료 시각을 고정한 채 Aura로 전송하며, 연결 실패는 2초부터 최대 300초까지 간격을 늘려 재시도한다. 동일 종료 재요청은 기존 상태를 반환하며 다른 내용으로 재종료하면 409다. 전송이 중단되면 API를 다시 띄웠을 때 SQLite에 남은 대기 건부터 이어서 처리한다. 이전 브랜치의 SQLite 전용 이력은 기존 DB에 보존된다. 기존 ML 평가 스크립트는 여전히 `/api/history` 또는 케이스 DB로 종료 결과를 읽을 수 있다.

Aura의 집계 키는 대표 ML 경보 ID를 기준으로 만든 `warning:<대표 event_id>`다. 다른 API에서도 화면 번호는 각각 `CASE-0001`일 수 있다. 대표 이벤트가 다르면 별개 해결 건으로 세고, 반복 경보를 선택한 경우에는 해당 반복의 센서 패턴으로 저장하면서 집계 키는 대표 이벤트를 사용한다.

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
| POST | `/api/cases/:id/warning/resolve` | 작업자 확인·일지 승인 후 케이스 종료와 Aura 전송 대기 등록 |
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

종료 응답의 `warning_resolution_status`와 상세의 `recording_status`로 전송 상태를 확인한다. `pending`은 Aura 반영 대기, `saved`는 반영 완료, `pending_review`는 Aura에 저장된 수동 근거의 검토 대기, `failed`는 데이터 충돌 등 운영자 확인이 필요한 상태다. 같은 API에 해당 패턴의 반영 대기나 실패가 남으면 자동 카드를 보류한다. 각 API의 SQLite 대기 상태는 해당 API에서 확인하며, Aura에 반영된 뒤부터 모든 API가 같은 이력을 조회한다.

수동 근거는 운영자가 검토하기 전까지 자동 추천에 사용하지 않는다. 검토 API는 `RAG_WARNING_REVIEW_TOKEN` 미설정이면 404, 설정 시 `X-Warning-Review-Token` 헤더가 필요하다. 토큰을 UI 환경변수에 넣지 않는다. 검토 본문은 기존 사례 연결이면 `{"existing_case_id":"manual:E012-C2"}`, 신규면 `title`, `equipment_scope`, 선택한 PDF 인용 안에 존재하는 `checks`/`actions` 발췌를 전달한다. 운영자 검토 화면은 이번 UI 범위에 포함하지 않는다.

오알람은 기존 `/close`를 사용하며 RAG 해결 사례에 포함하지 않는다. WARNING의 해결 종료는 새 `/warning/resolve`를 사용한다. 기존 주소로 해결 종료하면 409로 새 경로를 안내한다. 기존 UI의 CRITICAL 종료 경로는 유지한다.

## 화면 동작

1. WARNING 상세: 조기 경보·원인 미확정, 센서별 방향/관측 초/상태 전환 구간을 표시한다.
2. 대표·반복 경보를 선택한다. 저장하지 않은 조치가 있으면 경보 전환을 잠근다.
3. 과거 자동 카드 또는 원인 후보 보기 → 매뉴얼 후보·근거·적용 조건·과거 실제 조치를 표시한다.
4. 수동 검색 결과의 PDF를 열거나 해당 근거를 선택한다.
5. 선택한 카드의 점검 정상/이상, 작업자, 실제 조치를 임시 저장할 수 있다. 새로고침 후 복원한다.
6. 해결 확인과 정비일지 미리보기 승인 후 종료한다. 입력 내용을 최종 저장하므로 별도 임시 저장을 잊어도 조치가 보존된다.
7. 종료 후 읽기 전용으로 표시한다. Aura 반영 상태를 갱신하고, 수동 근거는 저장된 뒤 검토 대기를 표시한다.

센서 방향은 RAG 원래 모듈의 첫/마지막 구간 중앙값 비교이며 인과·고장 확정이 아니다. 상태 전환 시간은 센서 블록 시작으로부터의 구간으로 표시한다.

## 피드백 OFF / ON

- OFF: 과거 해결 후보 조회·자동 추천을 하지 않는다. 센서 소견·매뉴얼 후보/검색·해결 기록 저장은 가능하다.
- ON(기본): 공용 Aura 해결 이력을 원래 RAG 패턴 규칙으로 조회한다. 서로 다른 방향성 있는 3채널이 필요하다.
- 자동 표시: 현재 매뉴얼 버전·설비 범위가 맞고, 같은 패턴+사례의 서로 다른 해결 케이스가 2건 이상이어야 한다. 같은 케이스 반복은 중복 집계하지 않는다.
- 설비 범위 미설정, 센서 부족, 미검토 수동 근거, 반례, 복수 적격 사례 등은 자동 표시하지 않는다.
- 자동 카드를 거절하면 해당 경보에서 후보 목록과 선택에서 제거한다.
- OFF 동안 기록한 신규 해결도 나중에 ON으로 바꾸면 활용 가능하다. 이 수정 이전의 구형 기록은 실제 선택 근거가 없으므로 자동으로 매핑하지 않는다.

## 실행

기존 API·RAG 의존성 설치:
```bash
python -m pip install -r api/requirements.txt
```
공용 Aura 접속 정보와 키는 `rag/.env`에 설정한다. `RAG_WARNING_NAMESPACE=team-rag`처럼 팀이 합의한 이력 이름을 사용한다. API 실행 전에 [공용 Aura 이력 스키마](../../rag/docs/early-warning-integration.md#처음-한-번-준비)가 준비돼 있어야 한다. 환경변수는 셸에서 덮어쓸 수 있다.
```bash
cd api
uvicorn main:app --port 9000
```
Windows PowerShell:
```powershell
cd api
python -m uvicorn main:app --port 9000
```
자동 카드에 사용할 설비의 매뉴얼 적용 범위를 지정한다. 예를 들어 실제 장비가 해당 범위에 속할 때는 다음과 같이 설정한다.
```powershell
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
UI 통합 테스트는 실제 FastAPI와 임시 SQLite를 띄우고 React DOM 이벤트로 소견→후보→선택→임시 저장/재마운트→수동 검색/선택→해결 종료→반영·검토 상태를 검증한다. 문서 검색, Claude, Aura 이력은 테스트 대역을 사용한다. API 회귀 테스트는 반복 경보, 거절 카드 재선택 차단, 전송 실패 재시도와 중복 방지, 근거 검토, CRITICAL을 포함한다.

공용 Aura API 검증은 `RAG_WARNING_TEST_AURA=1`을 설정한 뒤 `python -m pytest api/tests/test_warning_bridge.py -k shared_aura_distinguishes`로 실행한다. 각자 `CASE-0001`인 합성 케이스 두 건이 테스트 전용 Aura 이력에서 서로 다른 해결로 집계되고, 세 번째 API가 자동 카드를 받는지 확인한다. 테스트가 만든 이력만 종료 시 정리한다. 실제 현장 경보의 후보 적중률과 운영 배포는 별도로 검증한다.
