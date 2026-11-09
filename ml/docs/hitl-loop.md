# HITL 피드백 루프 시연 (D-8): 현장 기억 · LangGraph 드라이버

2026-10-28 · ML 담당 (같은 날 RAG E 코드 전환 뒤 다시 씀, 2026-11-07 R2 더함)

요약:
- 사람이 케이스를 닫은 기록이 다음 안내로 돌아오지 않았다. 그래서 api에 "현장 기억"을 붙였다.
- `FIELD_MEMORY=1`일 때만 켜지고, 끄면 api는 현장 기억이 없는 판(RAG 담당 판 + FE 변경)과 같다. 켜고 만든 케이스의 R2 순서는 꺼도 남는다.
- 현장 기억이 하는 일은 둘이다.
  - R1: 오알람 기록으로 **참고 문장**을 만든다.
  - R2(11-07): 해결 종료된 케이스의 '이상' 점검 기록으로 새 케이스 카드의 **점검 순서**를 바꾼다. 카드의 글 · 근거 · 조치는 바꾸지 않는다. 다른 파트 요청은 `ml/docs/card-order-requests.md`에 따로 있다.
- ML은 바뀌지 않는다. 지금처럼 사실(코드 · 고장 유형 · 센서)만 보낸다.
- ML 쪽 고리(현장 라벨로 ML 운영 규칙을 판정하는 보고서)는 D-9다: `docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md`.
- 재생 데이터로 고리가 도는지 보이는 시연이다. 성능 수치가 아니다.

| 절 | 내용 |
|---|---|
| ① | 끊긴 곳 |
| ② | 바뀌는 것 (api, `FIELD_MEMORY=1`일 때만) |
| ③ | 시연법과 작업자 카드 |
| ④ | FE 요청 |
| ⑤ | RAG 연동 제안 |
| ⑥ | 한계와 현장 이야기 |

## ① 끊긴 곳

플래그를 끄면 지금도 이렇다.

- `api/store.py:3`: 사람 기록(records)은 ML · RAG에 보내지 않는다. 저장만 하고, 다시 읽는 곳이 없었다.
- `ui/src/types/case.ts:187-188`: FE 타입 주석도 같다("api가 저장, ML · RAG에는 보내지 않음"). 켰을 때 api 현장 기억이 다시 읽는다는 말도 붙어 있다.
- `api/main.py:116`: 케이스를 만들 때 안내는 코드의 매핑 카드다(`guidance.guidance_for`, `api/guidance.py:69-77`).
- `api/guidance.py:80-96`: 첫 상세 조회(`GET /api/cases/{id}`)에서 RAG 답으로 카드를 채운다. 입력은 E 코드와 고정 질문뿐이다. 지난 케이스는 보지 않는다.
- `api/guidance.py:62-66`: 카드는 E 코드 하나에 하나다(`candidates[0].situation_id` = E 코드). 점검 id는 `E0xx-Kn`이다.
- `api/main.py:531-550`: 챗봇 피드백은 질문 · 답과 함께 SQLite `feedback` 표에 저장한다(플래그 무관). 응답은 `{"ok": true}`다. 안내 · 챗봇에 다시 쓰는 곳은 없다.

그래서 같은 설비의 같은 오알람을 다섯 번 닫아도, 다섯 번째 안내가 첫 번째와 같았다.

## ② 바뀌는 것 (api, `FIELD_MEMORY=1`일 때만)

### 플래그

- `FIELD_MEMORY=1`이면 켜진다. 부를 때마다 환경 변수를 읽는다(`api/field_memory.py:23-24`의 `enabled()`).
- 끄면 아래가 모두 RAG 담당 판과 같다(`api/tests/test_memory_api.py`의 flag-off 테스트).
  - `/api/health` 키, `/api/memory`(어떤 메서드 · 끝 슬래시든 없는 주소와 같은 404, openapi에도 없음), close 본문(다른 키는 무시), 저장된 guidance(= `guidance.guidance_for(handoff)`), 챗봇 응답(인자 셋 그대로).
  - 예외: 켜고 만든 케이스(`field_order`가 있는 행)는 끈 뒤에도 순서와 고정을 지킨다(스펙 4.5). 그래서 시연은 새 `API_DB`로 한다.
- 파일: `api/field_memory.py`(라벨 · 집계 · 참고 문장 · 카드 순서). 훅은 `api/main.py`·`api/assistant.py`에 있다.

### 카드 순서(R2, 11-07)

설계는 `ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md`에 있다.

- 규칙: 같은 코드의 해결 종료 케이스에서 '이상'으로 기록된 점검을 센다(카드 밖 원인으로 해결한 것은 뺀다). 가장 많은 점검이 매핑에서 그 앞의 모든 점검보다 3건 이상 많고 설비 2대 이상에서 나왔으면, 다음 새 케이스는 그 점검과 그 원인부터 보인다(`order_for` · `apply_order`).
- 글 · 근거 · 기준 · 조치는 RAG가 만든 그대로다. 위치만 바뀐다.
- 조치 보호: 앞세울 점검이 든 원인 앞의 원인에 앞세울 원인이 갖지 않은 조치가 있으면 바꾸지 않는다(같은 조치를 가지면 옮긴다). 그래서 E029는 두 점검을 한 원인에 묶은 RAG 캐시 카드에서만 바뀌고, 매핑 대체 카드에서는 바뀌지 않는다.
- 정하는 때: ingest가 새 케이스를 만들 때 한 번. 열린 케이스는 다른 케이스가 닫혀도 바뀌지 않는다(원인 번호가 카드 위치라서).
- 저장: 결정은 케이스 최상위 `field_order`(lead · counts · guns · note_ko)로 DB에 저장하고, 옮긴 카드도 저장한다. 케이스 상세 응답과 케이스 쓰기 응답에 실린다(목록 `/api/cases` · `/api/history`에는 없다).
- 카드 고정: `field_order`가 있는 케이스에 판단 · 조치 · 결과 기록이 생기면 상세 조회가 카드를 다시 만들지 않는다. 아직이면 RAG 카드가 올 때 같은 순서를 다시 적용한다.
- R1(오알람 문장)은 카드를 바꾸지 않는다.

### 참고 문장의 자리

- 참고 문장은 GET 케이스 응답의 최상위 `field_memory`에만 붙는다. `field_memory`는 DB에 저장하지 않는다.
- 확률로 말하지 않는다. 지난 기록 · 점검 기록 문장에는 "참고"를 붙인다. R1 · R2 문장에는 없다(R2는 "현장 기록(사람 판정, 매뉴얼 근거 아님)"으로 시작한다). 매뉴얼 근거(cite · evidence)로 내보내지 않는다.

### 라벨 (닫힌 케이스만, `api/field_memory.py:46-53`)

| closure | 라벨 |
|---|---|
| `false_alarm` | `false_alarm` |
| `resolved` (후보 밖 원인 `MANUAL` 포함) | `real` |
| `unresolved`, `handed_over`, closure 없음 | 라벨 없음 |

- 예전의 "S10으로 해결 = 오알람" 규칙은 없앴다. 화면이 더는 S10을 기록할 수 없다.
- `MANUAL`로 해결하면 `real`로 센다. 적은 원인 글은 `manual_causes`에 따로 남는다.
- 집계 키는 `설비|규칙 코드`다(`alarm_key`, `api/field_memory.py:27-29`). 규칙 코드는 작업자가 화면에서 보는 코드(`trigger.rule_code`)다.

### 참고 문장 (`note_ko`, `api/field_memory.py`의 `annotate`)

센 수는 지금 보는 케이스 자신을 뺀다. 문장은 아래 순서로 한 칸 띄어 잇는다. 없으면 `None`.

1. **R1** (규칙 코드가 있고 오알람 > 고장 확인): `applied`에 `"R1"`.
   - "같은 설비·코드({설비} {코드})에서 사람이 오알람으로 닫은 기록 N건 · 고장 확인 M건 → 오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
2. **기록은 있는데 R1이 아님** (규칙 코드가 있고 N + M > 0).
   - "같은 설비·코드({설비} {코드}) 지난 기록: 오알람 N건 · 고장 확인 M건 (참고)."
3. **R2** (`field_order`가 있고 카드의 첫 점검이 `field_order.lead`): `applied`에 `"R2"`(R1과 함께면 `["R1", "R2"]`). 문장은 `field_order.note_ko`다. 수는 케이스가 생길 때 센 값이다.
   - "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 E029-K2 3건 · E029-K1 0건(설비 2대)이라 E029-K2부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로."
4. **점검 기록** (이 카드 점검 id 중 기록이 있는 것, 설비 무관, 모든 닫힘의 이상 · 정상): 카드 순서대로, 기록 있는 id만. 볼 때마다 다시 센다.
   - "지난 점검 기록({코드}): E028-K1 이상 1 · 정상 0, … (참고, 점검 순서는 매뉴얼 기준)." R2가 붙으면 끝이 "(참고)."다.

- 규칙 코드가 없는 알림(모델 신호만)에는 R1과 1·2번 문장을 쓰지 않는다. 키 `설비|`를 모든 코드가 같이 쓰기 때문이다. 3 · 4번은 붙을 수 있다(R2는 `fault_class.terminal_code`로 센다).
- R1은 문장만 바꾼다. 카드 순서를 바꾸지 않는다. 순서는 R2만 바꾼다(위 "카드 순서").

### 새 필드 · 주소

| 어디 | 필드 | 뜻 |
|---|---|---|
| `GET /api/cases/{id}` | `field_memory` | `alarm_key`, `alarm`(오알람 · 고장 확인 수), `applied`(`[]` · `["R1"]` · `["R2"]` · `["R1", "R2"]`), `note_ko`, `checks`(이 카드 점검 id별 지난 이상 · 정상 수, 카드 순서), `labeled`, `recent`(같은 설비 · 코드로 오알람 · 해결 종료된 최근 3건, FE 추가, `api/main.py:276-300`). 자기 자신을 뺀 집계 |
| 케이스 응답(상세 · 쓰기) | `field_order` | R2 결정(lead · counts · guns · note_ko). R2 조건을 넘은 새 케이스만, DB에 저장 |
| `POST /api/cases/{id}/close` 본문 | `reason`, `note` | 빈 문자열이 아니면 closure에 남긴다(`api/main.py:361-366`) |
| `GET /api/memory` | 전체 요약 | `enabled`, `labeled`, `labels`, `alarms[{key, gun_id, code, false_alarm, real, case_ids}]`, `checks[{check_id, abnormal, normal}]`, `manual_causes`, `missed_checks`(일지의 놓친 점검 글). 끄면 어떤 메서드 · 끝 슬래시든 없는 주소와 같은 404, openapi에 없음(`api/main.py:447-464`) |
| `GET /api/health` | `memory: "on"` | 켰을 때만(`api/main.py:233-234`) |

- `field_memory`는 케이스를 열 때마다 새로 센다. RAG 카드 생성(enrich) 뒤에 붙인다(`api/main.py:246-270`).
- 그래서 이미 열린 케이스도 다시 열면 문장이 바뀔 수 있다. 카드의 점검 순서는 케이스가 생길 때 정한 그대로다(R2).

### 챗봇

- `POST /api/assistant/ask`: 켜면 api가 `field_memory.snapshot`을 `memory=`로 넘긴다(`api/main.py:521-523`).
- 답의 E 코드(`error_code`)에 기록이 있으면 답 끝에 문단 하나를 붙인다(`api/assistant.py:36-39`, `chat_note` `api/field_memory.py:316-331`).
  - 예: "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 N · 고장 확인 M / 지난 점검: … (참고)". 시연에서 설비 이름은 파일 이름이다.
  - E 코드는 RAG가 질문에서 고른다(`rag/service.py:74`의 `select_code`). 오류 응답에도 `error_code`가 남는다(`rag/service.py:112-113`, `:186-188`). 그래서 RAG 키가 없어도 문단은 붙는다.
  - 코드 없는 일반 질문에는 붙지 않는다. evidence · caution · cite에는 넣지 않는다.
- 피드백(`/api/assistant/feedback`)은 api가 SQLite `feedback` 표에 저장한다(④ 요청 5 반영, 플래그 무관). 현장 기억 · 챗봇 문단에는 아직 쓰지 않는다.

### ingest 잠금

- 켜면 ingest 전체(seen → put)를 `_INGEST_LOCK`으로 감싼다(`api/main.py:42-43`, `:105`). 플래그와 상관없이 `_WRITE_LOCK`도 잡는다(`:44-47`).
- push와 5초 당기기가 겹쳐도 같은 알림으로 케이스가 둘 생기지 않는다.
- 끄면 `_INGEST_LOCK`은 없지만 `_WRITE_LOCK`이 ingest를 한 번에 하나씩 돌린다(④ 요청 9 반영).
- 끄면 저장 카드는 `guidance.guidance_for(handoff)` 그대로다. 켜면 R2 조건을 넘은 새 케이스만 `field_order`를 저장하고, 카드는 그 순서를 적용해 저장한다(`api/main.py`의 ingest). 조치 보호에 걸리면 카드는 그대로다.

## ③ 시연법

사람만 닫는다. 시뮬레이션 작업자는 없다. 재생은 600초 청크다.

### 준비

- 드라이버: `py -3.11`(langgraph, pandas). 처음 한 번 `py -3.11 -m pip install "langgraph>=1.0"`(`ml/requirements.txt`의 선택 줄. 따옴표가 없으면 셸이 `>`를 리다이렉트로 읽는다).
- api: RAG 의존성(neo4j 등)이 있는 Python으로 켠다. api가 `rag`를 import한다.
  - 저장소 `.venv`(`.venv/Scripts/python.exe`), 또는 `pip install -r api/requirements.txt`를 한 Python.
- `rag/.env`는 없어도 된다.
  - 없으면 카드는 매핑에서 오고(`api/guidance.py:69-77`), RAG 글은 오류로 보인다(`api/guidance.py:92-95`). 루프는 그대로 돈다.
  - 있으면 케이스의 첫 상세 조회마다 RAG 답을 만든다(수 초).
- 모델 파일과 test CSV는 `ml/README.md` 0절대로 둔다.
- 실행마다 이름을 하나 정한다(예: `1028a`). ML 상태와 api DB를 매번 새로 쓴다.

### 켜는 순서 (bash)

```bash
# 1) ML 서버: RSW_RAG_URL 없이(api가 당겨 감), 새 상태 폴더
cd ml && RSW_STATE_DIR=state/d8-1028a uvicorn main:app --port 8000

# 2) api 서버: 현장 기억 켜기, 새 DB, RAG 의존성이 있는 Python
cd api && FIELD_MEMORY=1 API_DB=state/d8-1028a.db ../.venv/Scripts/python.exe -m uvicorn main:app --port 9000

# 3) 화면: ui/.env.local 에 VITE_USE_MOCK=false, VITE_API_PROXY_TARGET=http://localhost:9000
cd ui && npm run dev            # http://localhost:5173

# 4) 드라이버: 알림마다 멈추고, 화면에서 케이스를 닫으면 이어 간다
cd ml && py -3.11 .py/hitl_loop.py test/test_0.csv

#    두 번째 시나리오: 먼저 ML · api를 새 이름으로 다시 켠다(예: d8-1028b. 1)·2)의 1028a 자리)
#    파일은 시작 시각 순으로 재생, 설비 = 파일 이름
cd ml && py -3.11 .py/hitl_loop.py test/test_5.csv test/test_4.csv
```

- 옵션: `--api`(기본 `http://127.0.0.1:9000`), `--ml`(기본 `http://127.0.0.1:8000`), `--ui`(케이스 링크용), `--out`(산출물 폴더).
- `--dry-run`: api 없이 멈추지 않고 끝까지 재생한다. 파일 · 알림마다 한 줄씩, 끝에 설비별 핸드오프 건수와 합계를 출력한다.
  - 출력 예: `test_0: 6 (rule E029 x5, rule E012 x1)`. 기대 건수는 test_0 6 · test_5 2 · test_4 3이다.
  - ML은 새 `RSW_STATE_DIR`로 켠 뒤에 돌린다. 재생의 `reset`(`DELETE /guns/{id}`, `ml/main.py:2088-2100`)은 설비 상태만 지우고 outbox는 그대로 둔다. 그래서 앞 실행의 핸드오프가 건수에 섞인다.
  - 드라이런 뒤 실제 실행 전에는 ML을 또 다른 새 폴더로 다시 켠다. 드라이런이 outbox를 채웠기 때문이다.
- 시작 점검(`startup_check`)이 아래를 차례로 확인하고, 하나라도 어긋나면 멈춘다. 산출물 폴더는 점검을 통과한 뒤에 만든다.
  - ML `/health`: 추적 중인 설비 0, outbox 비어 있음, `rag_url` 없음.
  - api `/api/health`: `memory`가 `on`, `ml`이 `ok`(아니면 api의 `ML_URL`을 확인한다).
  - api 케이스 · 이력 비어 있음. ML에 문제가 없을 때만 본다. `GET /api/cases`가 ML outbox를 api로 당겨 오기 때문이다(`api/main.py:238-240`).
  - 서버가 꺼져 있으면 그 서버 주소와 "먼저 그 서버를 켜세요"를 보인다.
  - 고치는 법: ML이 깨끗하지 않으면 새 `RSW_STATE_DIR` 폴더(예: `state/d8-<새 이름>`)로 ML을, api가 비어 있지 않으면 새 `API_DB` 파일(예: `state/d8-<새 이름>.db`)로 api를 다시 켠다.
- 산출물: `ml/results/D-8/runs/<YYYYmmdd-HHMMSS>/`(git 제외)
  - `loop_view.html`: 2초마다 새로고침되는 현황.
    - 상태 줄: 재생 중이면 마지막 알림의 데이터 시각. ⏸ 대기 중이면 기다리는 케이스 링크, `안내 카드 E029 · E029-K1 → E029-K2`(후보 · 카드의 점검 순서), 그 아래 안내 문장(`note_ko`).
    - 링크는 대기 중인 케이스 하나만 상태 줄에 있다. 알림 행에는 링크가 없다. 링크는 이름 붙은 탭 하나(`d8-ui`)를 다시 쓰므로 현황 탭은 그대로 열려 있다.
    - 알림 표 칸: 데이터 시각, 설비, 코드, 출처, 상태, 케이스, 안내 카드(후보 · 점검 순서), 기억 반영(R1 · R2), 닫힘, 사람 라벨, 데이터 라벨(닫힌 뒤), 일치, 안내 문장, 기억 변화. "안내 카드" 칸은 드라이버의 첫 상세 조회(guide) 때의 카드라서, 뒤에 RAG가 카드를 바꾸면(예: 첫 조회의 RAG 오류 뒤) 이 칸과 summary.md가 화면과 다를 수 있다.
    - 현장 기억 표 둘: 설비|코드별 오알람 · 고장 확인 · 케이스, 점검 항목별 이상 · 정상.
    - ⚠ 줄: 도움 요청 경고(작업자 카드 참고). 경고가 있는 행은 상태 칸이 주황이다.
    - 끝나면 노드 강조와 링크 없이 "끝남 · 알림 N건 · 요약 summary.md"(회색) 또는 "중단됨: …"(빨강, Ctrl+C 포함)을 남긴다.
  - `loop_log.jsonl`(시작할 때 비운다. 값은 화면 낱말이 아니라 원래 토큰), `summary.md`(알림 표, R2 줄이 있으면 "작업자 판정은 시연 대본이다. 성능이 아니다.", 경고, 현장 기억, 오류), `workflow.mmd`(그래프 그림).

### PowerShell 메모

창마다 한 줄씩 돌린다(저장소 루트에서). Windows PowerShell 5.1에는 `&&`가 없어 `;`로 잇는다.

```powershell
cd ml; $env:RSW_STATE_DIR="state/d8-1028a"; uvicorn main:app --port 8000
cd api; $env:FIELD_MEMORY="1"; $env:API_DB="state/d8-1028a.db"; ..\.venv\Scripts\python.exe -m uvicorn main:app --port 9000
cd ui; npm run dev
cd ml; py -3.11 .py/hitl_loop.py test/test_0.csv
```

- PowerShell에서 환경 변수를 `""`로 두면 값이 지워진다. 그래서 `RSW_STATE_DIR=""`(메모리만)는 기본 `state/`로 돌아간다. 새 폴더 이름을 준다.
- 같은 창에 `RSW_RAG_URL`이 남아 있으면 `Remove-Item Env:RSW_RAG_URL`로 지운다.
- (2026-11-01, C-8) ③ 표는 조기 경보 핸드오프를 끈 상태(`RSW_EARLY_WARNING_HANDOFFS` 없음, 기본) 기준이다.
  - 켜면 api가 조기 경보도 케이스로 받아 건수가 바뀐다.
  - 드라이버(`.py/hitl_loop.py`)는 조기 경보 핸드오프를 건너뛴다(`docs/early-warning-handoff.md`).

### 드라이버가 하는 일

- 그래프(LangGraph): advance → ingest → guide → ⏸ operator → learn → evaluate → advance … → report.
  - advance: 재생 스레드(Feeder)의 큐에서 다음 알림을 받는다. 앞 알림의 재생은 여기서 풀어 준다.
  - ingest: api를 동기화하고, 새 알림마다 케이스를 찾는다. 열린 케이스에 "반복"으로 묶인 알림은 멈추지 않는다. 묶은 곳이 도움 요청 케이스면 ⚠ 경고를 단다.
  - guide: 케이스를 한 번 읽어 안내 카드(`guidance.candidates`의 E 코드 = `order`, `guidance.causes`의 점검 순서 = `card_checks`, 원인 목록이 빌 때만 `candidates[0].checks`)와 `field_memory`의 `applied`·`note_ko`를 남긴다(`ml/.py/hitl_loop.py`의 guide). 이 GET이 첫 상세 조회라 api가 여기서 RAG 답을 만든다.
  - operator: `interrupt()`만 부르고 멈춘다(`ml/.py/hitl_loop.py`의 operator). 이어 갈 때 이 노드는 처음부터 다시 돌므로 부수 효과를 두지 않는다.
    - 조회는 그래프 밖에서 한다: `run_loop`가 `wait_for_closure`로 기다리고, 닫히면 `Command(resume=…)`로 이어 간다.
    - `wait_for_closure`는 `GET /api/history`를 2초마다 본다. 케이스 상세는 RAG를 부를 수 있어 조회에 쓰지 않는다. 닫힌 것을 본 뒤에 상세를 한 번 읽는다(판단 · 결과).
    - 조회 중 api가 잠깐 안 닿거나 5xx면 다음 조회까지 기다린다. 4xx는 실행을 끝낸다.
  - learn: `GET /api/memory`를 다시 읽어 사람 라벨과 바뀐 집계(`alarms`, `checks`)를 남긴다. 도움 요청으로 닫혔으면 ⚠ 경고를 단다.
  - evaluate: 닫힌 뒤에만 데이터 기준 라벨을 붙인다(파일 끝 1시간 안이면 `pre_failure`, 아니면 `far`). 사람에게 답을 미리 보이지 않으려는 것이다.
- `run_loop`가 Feeder를 켜고 끈다.
- 드라이버는 api 모듈을 import하지 않는다. HTTP로만 붙는다.

### 작업자 카드

```
E029 (오알람)
  루프 현황(loop_view) 상태 줄의 케이스 링크 → 개요 → "오알람으로 닫기" → 사유 칩 하나 → "오알람으로 닫기"

진짜 알림 (예: E012, E028)
  점검 시작 → 점검 항목(E0xx-Kn) 하나 이상 기록(앞 항목이 정상이면 다음 항목까지 기록)
  → 판단(이상이 나온 원인 "맞아요") → 조치 → "해결됐어요"
  → 일지(반복 여부) 승인 → "해결 종료"

코드 없는 알림 (모델 신호만)
  R1은 없다. 카드가 없으면 개요에서 "오알람으로 닫기" 또는 "직접 점검해서 기록"
  (test_0 · test_5 · test_4를 600초 청크로 돌리면 나오지 않는다)

하지 않는 것
  - "도움 요청"은 쓰지 않는다. 도움 요청 케이스는 열린 채 남아, 같은 설비 · 코드의 다음 알림을
    반복으로 묶는다(멈춤 · R1 없음). 드라이버는 ⚠ 경고를 띄운다.
    눌렀다면: 다음 알림 전에 작업함의 "도움 요청" 칸에서 그 케이스를 열고 개요의 "오알람으로 닫기"로 닫는다.
    그다음 알림부터 다시 새 케이스가 된다.
  - "잘 모르겠어요"는 쓰지 않는다
  - 진행 중에는 닫힌 케이스를 열지 않는다
```

- 도움 요청 흡수: `api/main.py:393-401`(status `handed_over`), `api/store.py:90-98`(resolved · unresolved · false_alarm이 아니면 묶음).
- 되돌리는 길: 작업함 "도움 요청" 칸(`ui/src/pages/InboxPage.tsx:36`), 개요의 "오알람으로 닫기"(`ui/src/pages/case/OverviewPage.tsx:113`, 팝업 `ui/src/pages/case/parts.tsx:410-488`).
- 카드 없는 안내의 개요 버튼: `ui/src/pages/case/OverviewPage.tsx:99-102`(직접 점검해서 기록), `:113`(오알람으로 닫기).
- 화면은 E 코드 카드 하나의 원인(`guidance.causes`)을 원인 카드로 풀어 보인다(`ui/src/lib/guidance.ts:27-56`). 판단 id는 `E029-1` 꼴이다.

### 볼 것

- 두 번째 E029부터 안내 문장이 나온다: "… 오알람으로 닫은 기록 1건 · 고장 확인 0건 → 오알람(운영 원인)인지부터 확인. …". 세 번째는 2건, 네 번째는 3건이다.
  - 오알람으로만 닫으므로 R2(점검 순서)는 생기지 않는다. 카드는 매번 `E029 · E029-K1 → E029-K2` 그대로다.
  - 케이스 화면은 아직 문장을 보이지 않는다(④ 요청 1). 그래서 드라이버가 보인다: `loop_view.html`의 상태 줄과 "안내 문장" 칸, 콘솔 ⏸ 줄 아래.
- 챗봇에 "E029 알림 진짜야?"라고 물으면 답 끝에 "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 N · 고장 확인 0" 문단이 붙는다. 닫을 때마다 N이 는다.
- `loop_view.html`이 알림마다 ⏸로 멈추고, 닫으면 이어 간다.

아래 기대값은 `api/tests/test_driver_contract.py`가 한 프로세스 안에서 확인한다(가짜 RAG, 실제 서버 아님). 10-29에는 실제 서버(ML · api, RAG 키 없음)에 대본 작업자를 붙여 두 시나리오를 끝까지 돌렸다. 결과는 두 표와 같았다(`ml/history.md` 36절).

**test_0** (파일 끝 2021-09-11T02:56:08Z)

| 알림 (데이터 시각 UTC) | 닫기 | 카드 | `applied` | `note_ko` | 라벨 (사람 / 데이터) |
|---|---|---|---|---|---|
| E029 09-04 16:45 | 오알람 | E029 · E029-K1 → E029-K2 | `[]` | 없음 | false_alarm / far |
| E029 09-06 19:30 | 오알람 | E029 · E029-K1 → E029-K2 | `["R1"]` | R1, 오알람 1건 · 고장 확인 0건 | false_alarm / far |
| E029 09-07 19:30 | 오알람 | E029 · E029-K1 → E029-K2 | `["R1"]` | R1, 2건 · 0건 | false_alarm / far |
| E029 09-08 19:30 | 오알람 | E029 · E029-K1 → E029-K2 | `["R1"]` | R1, 3건 · 0건 | false_alarm / far |
| E029 09-10 19:30 | 오알람 | E029 · E029-K1 → E029-K2 | `["R1"]` | R1, 4건 · 0건 | false_alarm / far |
| E012 09-11 02:46 | 진짜(E012-K1 이상 → 판단 E012 맞아요 → 조치 → 해결 → 일지 승인 → 해결 종료) | E012 · E012-K1 → E012-K2 | `[]` | 없음(같은 키 기록 없음) | real / pre_failure |

- 끝난 뒤 `/api/memory`: alarms `test_0|E029` 오알람 5 · 고장 확인 0, `test_0|E012` 0 · 1. checks `E012-K1` 이상 1 · 정상 0.

**test_5 → test_4**

| 알림 | 닫기 | 카드 | `applied` | `note_ko` | 라벨 (사람 / 데이터) |
|---|---|---|---|---|---|
| t5 E029 09-28 06:05 | 오알람 | E029 · E029-K1 → E029-K2 | `[]` | 없음 | false_alarm / far |
| t5 E028 10-03 02:15 | 진짜(E028-K1 이상) | E028 · E028-K1 → E028-K2 | `[]` | 없음 | real / pre_failure |
| t4 E029 10-12 | 오알람 | E029 · E029-K1 → E029-K2 | `[]` | 없음(다른 설비라 R1 없음) | false_alarm / far |
| t4 E029 10-16 | 오알람 | E029 · E029-K1 → E029-K2 | `["R1"]` | R1, test_4 E029 1건 · 0건 | false_alarm / far |
| t4 E028 10-18 | 진짜 | E028 · E028-K1 → E028-K2 | `[]` | "지난 점검 기록(E028): E028-K1 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)." 오알람 문장 없음 | real / pre_failure |

- t4 E028의 점검 기록은 t5의 기록이다. 점검 기록은 설비를 가리지 않고 센다. 오알람 문장은 설비 × 코드로 센다.

**도움 요청**: 첫 E029를 도움 요청으로 넘기면 라벨이 없고 ⚠ 경고가 뜬다. 다음 E029는 그 케이스에 반복으로 묶이고(멈춤 없음) "도움 요청 상태라 … 오알람으로 닫기" 경고가 붙는다.

### R2 시나리오: test_7 → test_0 (11-07)

준비:

- **RAG 캐시 카드가 필요하다.** `rag/.env`가 있고 Neo4j가 켜져 있어야 한다. 캐시를 읽을 때도 그래프 버전을 Neo4j로 확인하기 때문이다(`rag/service.py:126-127`).
  - 키나 Neo4j가 없으면 E029는 매핑 대체 카드가 되고, 조치 보호로 순서가 바뀌지 않는다(`api/tests/test_memory_api.py::test_r2_script_mapping_card`).
  - E029 · E012 카드는 이미 캐시에 있어 OpenAI 비용이 들지 않는다.
- **먼저 드라이런으로 알림 순서와 수를 본다:** `cd ml && py -3.11 .py/hitl_loop.py test/test_7.csv test/test_0.csv --dry-run`
  - 기대: test_7은 E029 1건, test_0은 E029 5건과 E012 1건이다. 11-07 검토에서 600초 청크로 다시 계산한 값이다.
  - 파일은 첫 행 시각 순으로 하나씩 재생한다(test_7 09-01, test_0 09-04).
- **24시간 경계:** test_0의 셋째 · 넷째 E029는 직전 발화 뒤 86,401초와 86,400초에 뜬다(아래 "현장 이야기"). 반복 창이 1초만 길어도 넷째가 반복으로 묶여 대본이 밀린다.
- **실행:** ML · api를 새 상태 폴더 · 새 DB로 켠 뒤 `cd ml && py -3.11 .py/hitl_loop.py test/test_7.csv test/test_0.csv`

대본: 사람이 아래대로 닫는다. 캐시 카드에서는 모든 줄이 원인 1 "맞아요"이고, 조치 화면에 재시작 주의가 나온다. "앞선 확인"은 그 알림이 케이스가 될 때까지 닫힌 케이스에서 센 E029 확인 수다.

| # | 알림 (데이터 시각 UTC) | 앞선 확인 (K2 · K1, 설비) | 카드(점검 순서) | `applied` | 닫기 |
|---|---|---|---|---|---|
| 1 | t7 E029 09-08 17:55 | 0 · 0 | K1 → K2 | `[]` | 해결 종료: K1 정상, K2 이상 |
| 2 | t0 E029 09-04 16:45 | 1 · 0, 1대 | K1 → K2 | `[]` | 같음 |
| 3 | t0 E029 09-06 19:30 | 2 · 0, 2대 | K1 → K2 | `[]` | 같음 |
| 4 | t0 E029 09-07 19:30 | 3 · 0, 2대 | K2 → K1 | `["R2"]` | 해결 종료: K2 정상, K1 이상(반대 기록) |
| 5 | t0 E029 09-08 19:30 | 3 · 1, 2대(차이 2) | K1 → K2 | `[]` | 해결 종료: K1 정상, K2 이상 |
| 6 | t0 E029 09-10 19:30 | 4 · 1, 2대 | K2 → K1 | `["R2"]` | 해결 종료: K2 이상 |
| 7 | t0 E012 09-11 02:46 | — | K1 → K2 | `[]` | test_0 표와 같다(E012-K1 이상) |

- 4번 줄의 `note_ko`에는 R2 문장이 들어간다: "현장 기록(사람 판정, 매뉴얼 근거 아님): 이 케이스가 생길 때 해결 종료 케이스의 이상 기록이 E029-K2 3건 · E029-K1 0건(설비 2대)이라 E029-K2부터 보도록 순서를 바꿈. 내용은 매뉴얼 그대로." 6번 줄은 4건 · 1건이다.
  - R2가 붙으면 지난 점검 기록 문장의 끝은 "(참고)."다.
- 3 ~ 6번 줄에 "같은 설비·코드(test_0 E029) 지난 기록: 오알람 0건 · 고장 확인 N건 (참고)."이 붙는다(N은 1 ~ 4). 2번 줄은 test_0의 첫 E029라 이 문장이 없다. 해결 종료만 있어 R1은 없다.
- 4 · 6번 줄에서는 첫 쪽 버튼이 매뉴얼 44쪽에서 오류표 2쪽으로 바뀐다. 바뀐 이유는 화면에 아직 없다(④ 11). 드라이버의 상태 줄 · 안내 문장 칸으로 보인다.
- 이 표는 `api/tests/test_memory_api.py::test_r2_script_cached_card`가 한 프로세스 안에서 확인한다(가짜 RAG, 실제 서버 아님). 실제 서버와 화면으로는 아직 돌리지 않았다.
- 수치는 재생 데이터에서 고리가 도는지 보는 확인이다. 어느 점검이 이상인지는 대본이 정한다. 성능이 아니다.

## ④ FE 요청

화면은 `field_memory.recent`("이전 기록")와 닫을 때 고른 오알람 사유 · 메모를 보인다(`ui/src/pages/case/parts.tsx:283-319`, `ui/src/pages/case/CaseLayout.tsx:23-35`). `note_ko` · `alarm` · `checks` · `applied`와 `field_order`는 아직 보이지 않는다. 필드는 모두 추가만이라 지금 화면이 깨지지는 않는다. 아래는 UI가 E 코드 카드 하나로 바뀐 뒤(`ui/docs/09-rag-development-reply.md` "검증 및 UI에 남은 변경")를 기준으로 한다.

1. **카드에 안내 문장**: `field_memory.note_ko`를 카드 위나 아래에 한 줄로 보인다. "현장 기록 · 참고"로 표시해 매뉴얼 근거와 구분한다. 지금 카드 목록은 `ui/src/pages/case/parts.tsx:321-358`(CauseList)다.
   - 원하면 CaseFacts(`parts.tsx:250-278`)에 "지난 알림: 오알람 N · 고장 확인 M"(`field_memory.alarm`)을 더한다.
2. **점검 항목별 지난 기록** (`parts.tsx:62-132`, StepPanel): 점검 id 옆에 "지난 점검에서 이상 N · 정상 M"(`field_memory.checks[id]`). 기록이 없는 id는 키가 없다.
3. **오알람 사유 칩 + 위치**: 사유 칩 · 메모는 FE가 반영했다(팝업, 사유 필수, `ui/src/pages/case/parts.tsx:410-488`). 오알람 버튼은 아직 개요 오른쪽 칸에 있다(`ui/src/pages/case/OverviewPage.tsx:113`). 상단 "도움 요청"(`ui/src/pages/case/CaseLayout.tsx:61-65`) 옆으로 옮기는 것만 남았다.
4. **닫힌 케이스는 읽기 전용**: FE가 반영했다. 해결 종료 · 오알람 케이스는 화면에서 보기만 되고(`ui/src/pages/case/CaseLayout.tsx:11-21`), api는 기록 · 닫기 · 도움 요청을 409 "closed"로 막는다(`api/main.py:201-211`). 미해결 · 도움 요청은 이어서 한다.
5. **챗봇 피드백 본문**: FE가 반영했다. 화면이 질문 · 답 · 상태 · 코드를 같이 보내고(`ui/src/pages/AssistantPage.tsx:107-122`), api가 SQLite `feedback` 표에 저장한다(`api/main.py:531-550`). 현장 기억 · RAG에는 아직 쓰지 않는다.
6. **출처 표시**: 현장 기억은 사람 판정이다. 챗봇의 현장 기록 문단과 카드의 문장을 매뉴얼 근거와 다른 모양으로 보인다. 나중에 시뮬레이션 기록이 섞이면 사람 / 시뮬레이션도 구분한다.
7. **주석 · mock · 타입**: 주석(`ui/src/types/case.ts:187-188`)과 `FieldMemory` 타입(`case.ts:330-379`)은 FE가 맞췄다. mock(`ui/src/mocks/handlers.ts`)에는 아직 `field_memory`가 없다(`ui/docs/05-fe-api-requests.md:278`의 약속).
8. **08 수치** (`ui/docs/08-api-draft.md:10`): "알림 13건 → 케이스 4건"(test_0 · test_3)은 10/23 확인 수치다. 지금 번들은 test_0에서만 알림 6건이다(아무것도 닫지 않으면 케이스 2건). 바뀐 번들로 다시 센다.
9. **ingest 경쟁 상태**: FE가 반영했다. ingest는 플래그와 상관없이 `_WRITE_LOCK` 안에서 돈다(`api/main.py:105`). push와 당기기를 함께 써도 케이스는 하나다.
10. **상세 조회 저장 경쟁** (api 담당): FE가 반영했다. `GET /api/cases/{id}`는 RAG 보강을 잠금 밖에서 하고, `_WRITE_LOCK` 안에서 다시 읽어 안내가 바뀌었을 때만 저장한다(`api/main.py:246-262`). 쓰기 주소도 같은 잠금을 잡아(`:50-58`) 같은 순간의 닫기가 덮이지 않는다.
    - 리허설에서 대본 작업자가 드라이버의 상세 조회와 같은 순간에 닫자 닫기가 사라졌다(`ml/history.md` 36절). 화면도 케이스를 열 때와 창으로 돌아올 때마다 상세를 다시 읽는다(react-query 기본값, `ui/src/lib/useCase.ts:12-15`).
    - 기록 저장 주소도 모두 같은 `_WRITE_LOCK`을 잡는다. 케이스 단위가 아니라 api 전체에 하나다.

11-13(R2, 11-07)은 `ml/docs/card-order-requests.md` 2절에 따로 적었다. 머지 전에 FE 담당이 그 문서를 본다.

11. **순서를 바꾼 이유 보이기**: 카드의 첫 점검이 `field_order.lead`이면 `field_order.note_ko`를 보인다.
12. **타입 · 주석 · 안내**: `field_order` 타입, 틀리게 된 주석 세 곳, 카드 고정 케이스의 "새로고침하면 다시 만들어요" 안내.
13. **(제안) 아래 항목까지 기록하기**: 위 항목만 기록이 쌓이는 쏠림을 줄인다.

예전 요청 중 "S10 출구"와 "후보별 근거 수준(`evidence_level`)"은 뺐다. 카드가 E 코드 하나라 S10 후보도, 후보별 근거 수준도 없다.

## ⑤ RAG 연동 제안

지금 api는 현장 기억을 RAG로 보내지 않는다. 챗봇 문단은 api가 RAG 답 뒤에 붙인다. RAG가 직접 쓰게 한다면 아래를 제안한다. RAG 담당과 합의한 뒤 정한다.

- `answer_question(question, *, error_code=None, history=None, database=None, use_llm=True, use_cache=False)`(`rag/service.py:101-102`)에 `field_history=None`을 더한다.
  - 기본 `None`이면 지금과 같다.
  - 넘기면 생성 입력 `context`(`rag/service.py:152-155`)에 `evidence`·`reviewed_guidance` 밖의 따로 된 키로 둔다. 인용(cite)할 수 없는 근거다.
  - 넘길 모양은 api `GET /api/memory`(= `field_memory.snapshot`)에서 그 코드 부분만 쓰면 된다(`chat_note`와 같은 고르기).
  - 먼저 챗봇 경로(`api/assistant.py:31`)에만 넘긴다. 카드 생성(`api/guidance.py:91`)에는 넘기지 않는다. 카드의 점검 순서는 api가 R2로 정하고(②), 카드의 글은 RAG 그대로 두기 때문이다. R2에 관한 RAG 요청은 `ml/docs/card-order-requests.md` 3절에 있다.
- 프롬프트 규칙(`SYSTEM_PROMPT`, `rag/service.py:20`): "현장 기록은 매뉴얼 근거가 아니다. 인용하지 않는다. 확률로 말하지 않는다."
  - `PROMPT_VERSION`(`rag/service.py:18`)을 올린다. api는 이 값이 다르면 카드를 다시 만든다(`api/guidance.py:84`).
  - 캐시 키(`cache_path`, `rag/service.py:86-91`)에 `field_history`를 넣는다. 기록이 바뀔 때마다 새 생성(비용 · 수 초)이 든다.
- Neo4j(쓴다면)
  - `(:FieldCase {case_id, gun_id, code, label, closed_at})`: 닫힌 케이스 한 건. 점검 결과는 `(:FieldCase)-[:CHECKED {result}]->(:FieldCheck {id: "E028-K1"})`.
  - `:Rag` 라벨을 쓰지 않는다. build가 `:Rag`의 Situation · Cause · CheckAction · ManualError · Component를 지우고 다시 만든다(`rag/pipeline/build.py:72-73`). 그래서 RAG 노드에 건 관계도 함께 지워진다.
  - E 코드는 `:Rag:Situation`에 관계로 걸지 않고 `code` 속성으로 둔다. 조회할 때 맞춘다.
  - `:Situation` 라벨(S01~S10)과 `manual_id='festo'`도 쓰지 않는다(`rag/pipeline/build.py:142-144`가 지운다).
  - 벡터 · 키워드 검색은 `node:Rag`만 본다(`rag/retrieval.py:103`, `:113`). 그래서 현장 노드는 인용 검색에 섞이지 않는다.

## ⑥ 한계

- **R1은 문장이다.** R1은 "오알람인지부터 확인"하라는 문장일 뿐이다. 작업자가 읽지 않으면 효과가 없다. 화면 표시는 ④ 요청 1 뒤에야 생긴다.
- **R2는 카드 순서다.** 큰 한계는 셋이다. 나머지는 스펙 10절(`ml/docs/superpowers/specs/2026-11-07-feedback-card-order-design.md`)에 있다.
  - 화면에 바뀐 이유가 없다. 이유 없이 바뀐 순서는 매뉴얼 순서처럼 읽힌다. ④ 11 전까지는 시연용이다.
  - E029는 RAG 캐시 카드(두 점검을 원인 1에 묶은 카드)에서만 바뀐다. 매핑 대체 카드에서는 재시작 주의가 있는 원인 1을 밀 수 없다(조치 보호).
  - 위 항목이 유리하다. '다음' 버튼은 한 항목만 기록해도 눌려, 위 항목만 기록이 쌓일 수 있다(④ 13).
- **RAG 지연 · 키.** 키가 있으면 케이스의 첫 상세 조회마다 RAG 답을 만든다(코드별 약 4~12초, `ui/docs/09-rag-development-reply.md` 5절). 드라이버의 guide와 화면의 첫 열기가 그만큼 기다린다. 상태가 `ok`가 아니면 열 때마다 다시 시도한다(`api/guidance.py:84-86`). R2 카드 고정 케이스는 다시 시도하지 않는다(`api/main.py:250`). 키가 없으면 카드는 매핑, RAG 글은 오류다.
  - 드라이버의 api 연결은 120초까지 기다린다(`ml/.py/hitl_loop.py`의 `main()`). RAG 한 번(OpenAI 최대 90초 + Neo4j 연결 15초)보다 길게 잡았다.
  - 새 알림이 케이스로 안 보이면, 드라이버는 같은 설비의 열린 케이스 상세를 한 번씩 읽어 '반복'으로 묶였는지 본다(`ml/.py/hitl_loop.py`의 ingest). 알림마다 한 번이고 되풀이 조회는 아니다. 다만 카드가 아직 `ok`가 아니면 이 조회도 RAG를 다시 부른다. 묶임 확인이 느리면 이 때문이다.
- **재생 데이터다.** 현장이 아니다. 성능이 아니다.
  - 데이터 기준 라벨(`pre_failure`/`far`)은 답을 아는 것이다. 사람 라벨과의 일치는 고리가 도는지 보는 용도다.
- **집계 키가 설비 × 코드다.** 다른 설비의 같은 코드는 따로 센다(t4의 첫 E029는 R1이 없다). 코드 없는 알림은 R1을 쓰지 않는다. 점검 기록은 설비를 가리지 않는다.
- **원인 정답이 없다.** 데이터에 원인 라벨이 없다. "고장 확인"은 사람이 해결 종료했다는 뜻이지 검증됐다는 뜻이 아니다.
- **닫힌 케이스는 다시 닫을 수 없다.** 다시 닫으면 마지막 closure가 라벨을 덮어쓰므로, 해결 종료 · 오알람 케이스는 읽기 전용이다(화면, api 409 "closed"). 미해결 · 도움 요청은 다시 닫을 수 있고, 그때 closure를 덮어쓴다.
- **데모 규모의 경쟁 상태.** 집계는 부를 때마다 케이스 전체를 다시 읽는다(수십 건 규모). ingest와 기록 저장 주소는 `_WRITE_LOCK` 하나로 한 번에 하나씩 돈다. 상세 조회의 현장 기억은 잠금 밖에서 센다(`api/main.py:264-269`). 그래서 같은 순간의 닫기가 그 응답의 문장에 빠질 수 있다. 다시 열면 새로 센다. 상세 조회의 저장이 닫기를 덮는 경쟁(④ 10)은 막혔다. 사람이 누르는 속도로는 드물다.

### 현장 이야기: 사람 기억이 24시간 규칙을 0~1초 차로 메운다

- test_0의 E029는 09-06·07·08·10일 19:30(데이터 시각, UTC)에 뜬다. 한국 시각으로는 다음 날 새벽 04:30이다.
- 하루 간격이면 간격이 86,400초 또는 86,401초다.
- ML의 반복 규칙은 "직전 발화 뒤 `rule_repeat_s`(24시간) **미만**"이다(`ml/main.py:895-896`). 86,400초는 미만이 아니다. 그래서 매번 새 critical 알림이 된다(`ml/docs/fe-api-draft-reply.md` 3절).
- ML은 이 규칙을 바꾸지 않는다. 사람이 "오알람"으로 닫은 기억이 다음 날 같은 시각의 알림에 "오알람(운영 원인)인지부터 확인" 문장을 붙인다.
- 시연의 수치는 재생 데이터에서 이 고리가 도는 모습이다. 성능으로 읽지 않는다.
