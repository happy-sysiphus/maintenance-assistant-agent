# HITL 피드백 루프 시연 (D-8): 현장 기억 · LangGraph 드라이버

2026-10-28 · ML 담당 (같은 날 RAG E 코드 전환 뒤 다시 씀)

요약: 사람이 케이스를 닫은 기록이 다음 안내로 돌아오지 않았다. 그래서 api에 "현장 기억"을 붙였다. `FIELD_MEMORY=1`일 때만 켜지고, 끄면 api는 RAG 담당 판(E 코드 RAG)과 똑같다. 현장 기억은 **참고 문장만** 만든다. RAG 안내 카드의 내용과 순서는 바꾸지 않는다. ML은 바뀌지 않는다. 지금처럼 사실(코드 · 고장 유형 · 센서)만 보낸다. 재생 데이터로 고리가 도는지 보이는 시연이다. 성능 수치가 아니다.

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
- `ui/src/types/case.ts:155`: FE 타입 주석도 같다("api가 저장, ML · RAG에는 보내지 않음").
- `api/main.py:107`: 케이스를 만들 때 안내는 코드의 매핑 카드다(`guidance.guidance_for`, `api/guidance.py:69-77`).
- `api/guidance.py:80-96`: 첫 상세 조회(`GET /api/cases/{id}`)에서 RAG 답으로 카드를 채운다. 입력은 E 코드와 고정 질문뿐이다. 지난 케이스는 보지 않는다.
- `api/guidance.py:62-66`: 카드는 E 코드 하나에 하나다(`candidates[0].situation_id` = E 코드). 점검 id는 `E0xx-Kn`이다.
- `api/main.py:431-433`: 챗봇 피드백은 `{"ok": true}`만 돌려준다. 저장하지 않는다.

그래서 같은 설비의 같은 오알람을 다섯 번 닫아도, 다섯 번째 안내가 첫 번째와 같았다.

## ② 바뀌는 것 (api, `FIELD_MEMORY=1`일 때만)

### 플래그

- `FIELD_MEMORY=1`이면 켜진다. 부를 때마다 환경 변수를 읽는다(`api/field_memory.py:21-22`의 `enabled()`).
- 끄면 아래가 모두 RAG 담당 판과 같다(`api/tests/test_memory_api.py`의 flag-off 테스트).
  - `/api/health` 키, `/api/memory`(어떤 메서드 · 끝 슬래시든 없는 주소와 같은 404, openapi에도 없음), close 본문(다른 키는 무시), 저장된 guidance(= `guidance.guidance_for(handoff)`), 챗봇 응답(인자 셋 그대로).
- 파일: `api/field_memory.py`(라벨 · 집계 · 참고 문장). 훅은 `api/main.py`·`api/assistant.py`에 있다.

### 카드는 그대로

- 카드(guidance)의 내용 · 순서는 바꾸지 않는다. 현장 기억은 카드에 끼어들지 않는다.
- 참고 문장은 GET 케이스 응답의 최상위 `field_memory`에만 붙는다. DB에는 저장하지 않는다(`api/main.py:215-216`).
- 확률로 말하지 않는다. "참고"를 붙인다. 매뉴얼 근거(cite · evidence)로 내보내지 않는다.

### 라벨 (닫힌 케이스만, `api/field_memory.py:44-51`)

| closure | 라벨 |
|---|---|
| `false_alarm` | `false_alarm` |
| `resolved` (후보 밖 원인 `MANUAL` 포함) | `real` |
| `unresolved`, `handed_over`, closure 없음 | 라벨 없음 |

- 예전의 "S10으로 해결 = 오알람" 규칙은 없앴다. 화면이 더는 S10을 기록할 수 없다.
- `MANUAL`로 해결하면 `real`로 센다. 적은 원인 글은 `manual_causes`에 따로 남는다.
- 집계 키는 `설비|규칙 코드`다(`alarm_key`, `api/field_memory.py:25-27`). 규칙 코드는 작업자가 화면에서 보는 코드(`trigger.rule_code`)다.

### 참고 문장 (`note_ko`, `api/field_memory.py:119-152`)

센 수는 지금 보는 케이스 자신을 뺀다. 문장은 아래 순서로 한 칸 띄어 잇는다. 없으면 `None`.

1. **R1** (규칙 코드가 있고 오알람 > 고장 확인): `applied == ["R1"]`.
   - "같은 설비·코드({설비} {코드})에서 사람이 오알람으로 닫은 기록 N건 · 고장 확인 M건 → 오알람(운영 원인)인지부터 확인. 아니면 카드의 점검 순서대로."
2. **기록은 있는데 R1이 아님** (규칙 코드가 있고 N + M > 0): `applied == []`.
   - "같은 설비·코드({설비} {코드}) 지난 기록: 오알람 N건 · 고장 확인 M건 (참고)."
3. **점검 기록** (이 카드 점검 id 중 기록이 있는 것, 설비 무관): 카드 순서대로, 기록 있는 id만.
   - "지난 점검 기록({코드}): E028-K1 이상 1 · 정상 0, … (참고, 점검 순서는 매뉴얼 기준)."

- 규칙 코드가 없는 알림(모델 신호만)에는 R1과 1·2번 문장을 쓰지 않는다. 키 `설비|`를 모든 코드가 같이 쓰기 때문이다. 3번은 붙을 수 있다.
- R1은 문장만 바꾼다. 카드 순서를 바꾸지 않는다.

### 새 필드 · 주소

| 어디 | 필드 | 뜻 |
|---|---|---|
| `GET /api/cases/{id}` | `field_memory` | `alarm_key`, `alarm`(오알람 · 고장 확인 수), `applied`(`["R1"]` 또는 `[]`), `note_ko`, `checks`(이 카드 점검 id별 지난 이상 · 정상 수, 카드 순서), `labeled`. 자기 자신을 뺀 집계 |
| `POST /api/cases/{id}/close` 본문 | `reason`, `note` | 빈 문자열이 아니면 closure에 남긴다(`api/main.py:271-276`) |
| `GET /api/memory` | 전체 요약 | `enabled`, `labeled`, `labels`, `alarms[{key, gun_id, code, false_alarm, real, case_ids}]`, `checks[{check_id, abnormal, normal}]`, `manual_causes`, `missed_checks`(일지의 놓친 점검 글). 끄면 어떤 메서드 · 끝 슬래시든 없는 주소와 같은 404, openapi에 없음(`api/main.py:347-364`) |
| `GET /api/health` | `memory: "on"` | 켰을 때만(`api/main.py:196-197`) |

- `field_memory`는 케이스를 열 때마다 새로 센다. RAG 카드 생성(enrich) 뒤에 붙인다(`api/main.py:209-216`).
- 그래서 이미 열린 케이스도 다시 열면 문장이 바뀔 수 있다. 카드는 그대로다.

### 챗봇

- `POST /api/assistant/ask`: 켜면 api가 `field_memory.snapshot`을 `memory=`로 넘긴다(`api/main.py:421-423`).
- 답의 E 코드(`error_code`)에 기록이 있으면 답 끝에 문단 하나를 붙인다(`api/assistant.py:36-39`, `chat_note` `api/field_memory.py:168-183`).
  - 예: "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 N · 고장 확인 M / 지난 점검: … (참고)". 시연에서 설비 이름은 파일 이름이다.
  - E 코드는 RAG가 질문에서 고른다(`rag/service.py:74`의 `select_code`). 오류 응답에도 `error_code`가 남는다(`rag/service.py:112-113`, `:186-188`). 그래서 RAG 키가 없어도 문단은 붙는다.
  - 코드 없는 일반 질문에는 붙지 않는다. evidence · caution · cite에는 넣지 않는다.
- 피드백(`/api/assistant/feedback`)은 여전히 저장하지 않는다(④ 요청 5).

### ingest 잠금

- 켜면 ingest 전체(seen → put)를 lock 하나로 감싼다(`api/main.py:41-42`, `:88`).
- push와 5초 당기기가 겹쳐도 같은 알림으로 케이스가 둘 생기지 않는다.
- 끄면 RAG 담당 판 그대로 잠그지 않는다(④ 요청 9).
- 카드는 켜도 꺼도 `guidance.guidance_for(handoff)`다(`api/main.py:107`).

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
- `--dry-run`: api 없이 멈추지 않고 끝까지 재생한다. 설비별 핸드오프 건수만 출력한다.
  - 출력 예: `test_0: 6 (rule E029 x5, rule E012 x1)`. 기대 건수는 test_0 6 · test_5 2 · test_4 3이다.
  - ML은 새 `RSW_STATE_DIR`로 켠 뒤에 돌린다. 재생의 `reset`(`DELETE /guns/{id}`, `ml/main.py:1916-1928`)은 설비 상태만 지우고 outbox는 그대로 둔다. 그래서 앞 실행의 핸드오프가 건수에 섞인다.
  - 드라이런 뒤 실제 실행 전에는 ML을 또 다른 새 폴더로 다시 켠다. 드라이런이 outbox를 채웠기 때문이다.
- 시작 점검(`startup_check`)이 아래를 차례로 확인하고, 하나라도 어긋나면 멈춘다. 산출물 폴더는 점검을 통과한 뒤에 만든다.
  - ML `/health`: 추적 중인 설비 0, outbox 비어 있음, `rag_url` 없음.
  - api `/api/health`: `memory`가 `on`, `ml`이 `ok`(아니면 api의 `ML_URL`을 확인한다).
  - api 케이스 · 이력 비어 있음. ML에 문제가 없을 때만 본다. `GET /api/cases`가 ML outbox를 api로 당겨 오기 때문이다(`api/main.py:201-203`).
  - 서버가 꺼져 있으면 그 서버 주소와 "먼저 그 서버를 켜세요"를 보인다.
  - 고치는 법: ML이 깨끗하지 않으면 새 `RSW_STATE_DIR` 폴더(예: `state/d8-<새 이름>`)로 ML을, api가 비어 있지 않으면 새 `API_DB` 파일(예: `state/d8-<새 이름>.db`)로 api를 다시 켠다.
- 산출물: `ml/results/D-8/runs/<YYYYmmdd-HHMMSS>/`(git 제외)
  - `loop_view.html`: 2초마다 새로고침되는 현황.
    - 상태 줄: 재생 중이면 마지막 알림의 데이터 시각. ⏸ 대기 중이면 기다리는 케이스 링크, `안내 카드 E029`, 그 아래 안내 문장(`note_ko`).
    - 링크는 대기 중인 케이스 하나만 상태 줄에 있다. 알림 행에는 링크가 없다. 링크는 이름 붙은 탭 하나(`d8-ui`)를 다시 쓰므로 현황 탭은 그대로 열려 있다.
    - 알림 표 칸: 데이터 시각, 설비, 코드, 출처, 상태, 케이스, 안내 카드, 기억 반영(R1), 닫힘, 사람 라벨, 데이터 라벨(닫힌 뒤), 일치, 안내 문장, 기억 변화.
    - 현장 기억 표 둘: 설비|코드별 오알람 · 고장 확인 · 케이스, 점검 항목별 이상 · 정상.
    - ⚠ 줄: 도움 요청 경고(작업자 카드 참고). 경고가 있는 행은 상태 칸이 주황이다.
    - 끝나면 노드 강조와 링크 없이 "끝남 · 알림 N건 · 요약 summary.md"(회색) 또는 "중단됨: …"(빨강, Ctrl+C 포함)을 남긴다.
  - `loop_log.jsonl`(시작할 때 비운다. 값은 화면 낱말이 아니라 원래 토큰), `summary.md`(알림 표, 경고, 현장 기억, 오류), `workflow.mmd`(그래프 그림).

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

### 드라이버가 하는 일

- 그래프(LangGraph): advance → ingest → guide → ⏸ operator → learn → evaluate → advance … → report.
  - advance: 재생 스레드(Feeder)의 큐에서 다음 알림을 받는다. 앞 알림의 재생은 여기서 풀어 준다.
  - ingest: api를 동기화하고, 새 알림마다 케이스를 찾는다. 열린 케이스에 "반복"으로 묶인 알림은 멈추지 않는다. 묶은 곳이 도움 요청 케이스면 ⚠ 경고를 단다.
  - guide: 케이스를 한 번 읽어 안내 카드(`guidance.candidates`의 E 코드)와 `field_memory`의 `applied`·`note_ko`를 남긴다(`ml/.py/hitl_loop.py:269-279`). 이 GET이 첫 상세 조회라 api가 여기서 RAG 답을 만든다.
  - operator: `interrupt()`만 부르고 멈춘다(`ml/.py/hitl_loop.py:281-286`). 이어 갈 때 이 노드는 처음부터 다시 돌므로 부수 효과를 두지 않는다.
    - 조회는 그래프 밖에서 한다: `run_loop`가 `wait_for_closure`로 기다리고, 닫히면 `Command(resume=…)`로 이어 간다(`:395-417`, `:420-446`).
    - `wait_for_closure`는 `GET /api/history`를 2초마다 본다. 케이스 상세는 RAG를 부를 수 있어 조회에 쓰지 않는다. 닫힌 것을 본 뒤에 상세를 한 번 읽는다(판단 · 결과).
    - 조회 중 api가 잠깐 안 닿거나 5xx면 다음 조회까지 기다린다. 4xx는 실행을 끝낸다.
  - learn: `GET /api/memory`를 다시 읽어 사람 라벨과 바뀐 집계(`alarms`, `checks`)를 남긴다. 도움 요청으로 닫혔으면 ⚠ 경고를 단다.
  - evaluate: 닫힌 뒤에만 데이터 기준 라벨을 붙인다(파일 끝 1시간 안이면 `pre_failure`, 아니면 `far`). 사람에게 답을 미리 보이지 않으려는 것이다.
- `run_loop`가 Feeder를 켜고 끈다.
- 드라이버는 api 모듈을 import하지 않는다. HTTP로만 붙는다.

### 작업자 카드

```
E029 (오알람)
  루프 현황(loop_view) 상태 줄의 케이스 링크 → 개요 → "오알람으로 닫기" → "오알람 맞아요"

진짜 알림 (예: E012, E028)
  점검 시작 → 점검 항목(E0xx-Kn) 하나 이상 기록 → 판단(E 코드 카드 "맞아요") → 조치 → "해결됐어요"
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

- 도움 요청 흡수: `api/main.py:301-308`(status `handed_over`), `api/store.py:70-74`(resolved · unresolved · false_alarm이 아니면 묶음).
- 되돌리는 길: 작업함 "도움 요청" 칸(`ui/src/pages/InboxPage.tsx:36`), 개요의 "오알람으로 닫기"(`ui/src/pages/case/OverviewPage.tsx:92`).
- 카드 없는 안내의 개요 버튼: `ui/src/pages/case/OverviewPage.tsx:78-92`.
- 화면은 아직 S 기반 후보 화면이다. E 코드 카드 하나로 바꾸는 것은 UI 담당 몫이다(`ui/docs/09-rag-development-reply.md` "검증 및 UI에 남은 변경").

### 볼 것

- 두 번째 E029부터 안내 문장이 나온다: "… 오알람으로 닫은 기록 1건 · 고장 확인 0건 → 오알람(운영 원인)인지부터 확인. …". 세 번째는 2건, 네 번째는 3건이다.
  - 카드는 매번 `E029` 하나 그대로다.
  - 케이스 화면은 아직 문장을 보이지 않는다(④ 요청 1). 그래서 드라이버가 보인다: `loop_view.html`의 상태 줄과 "안내 문장" 칸, 콘솔 ⏸ 줄 아래.
- 챗봇에 "E029 알림 진짜야?"라고 물으면 답 끝에 "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 N · 고장 확인 0" 문단이 붙는다. 닫을 때마다 N이 는다.
- `loop_view.html`이 알림마다 ⏸로 멈추고, 닫으면 이어 간다.

아래 기대값은 `api/tests/test_driver_contract.py`가 한 프로세스 안에서 확인한다(가짜 RAG, 실제 서버 아님). 10-29에는 실제 서버(ML · api, RAG 키 없음)에 대본 작업자를 붙여 두 시나리오를 끝까지 돌렸다. 결과는 두 표와 같았다(`ml/history.md` 36절).

**test_0** (파일 끝 2021-09-11T02:56:08Z)

| 알림 (데이터 시각 UTC) | 닫기 | 카드 | `applied` | `note_ko` | 라벨 (사람 / 데이터) |
|---|---|---|---|---|---|
| E029 09-04 16:45 | 오알람 | E029 | `[]` | 없음 | false_alarm / far |
| E029 09-06 19:30 | 오알람 | E029 | `["R1"]` | R1, 오알람 1건 · 고장 확인 0건 | false_alarm / far |
| E029 09-07 19:30 | 오알람 | E029 | `["R1"]` | R1, 2건 · 0건 | false_alarm / far |
| E029 09-08 19:30 | 오알람 | E029 | `["R1"]` | R1, 3건 · 0건 | false_alarm / far |
| E029 09-10 19:30 | 오알람 | E029 | `["R1"]` | R1, 4건 · 0건 | false_alarm / far |
| E012 09-11 02:46 | 진짜(E012-K1 이상 → 판단 E012 맞아요 → 조치 → 해결 → 일지 승인 → 해결 종료) | E012 | `[]` | 없음(같은 키 기록 없음) | real / pre_failure |

- 끝난 뒤 `/api/memory`: alarms `test_0|E029` 오알람 5 · 고장 확인 0, `test_0|E012` 0 · 1. checks `E012-K1` 이상 1 · 정상 0.

**test_5 → test_4**

| 알림 | 닫기 | 카드 | `applied` | `note_ko` | 라벨 (사람 / 데이터) |
|---|---|---|---|---|---|
| t5 E029 09-28 06:05 | 오알람 | E029 | `[]` | 없음 | false_alarm / far |
| t5 E028 10-03 02:15 | 진짜(E028-K1 이상) | E028 | `[]` | 없음 | real / pre_failure |
| t4 E029 10-12 | 오알람 | E029 | `[]` | 없음(다른 설비라 R1 없음) | false_alarm / far |
| t4 E029 10-16 | 오알람 | E029 | `["R1"]` | R1, test_4 E029 1건 · 0건 | false_alarm / far |
| t4 E028 10-18 | 진짜 | E028 | `[]` | "지난 점검 기록(E028): E028-K1 이상 1 · 정상 0 (참고, 점검 순서는 매뉴얼 기준)." 오알람 문장 없음 | real / pre_failure |

- t4 E028의 점검 기록은 t5의 기록이다. 점검 기록은 설비를 가리지 않고 센다. 오알람 문장은 설비 × 코드로 센다.

**도움 요청**: 첫 E029를 도움 요청으로 넘기면 라벨이 없고 ⚠ 경고가 뜬다. 다음 E029는 그 케이스에 반복으로 묶이고(멈춤 없음) "도움 요청 상태라 … 오알람으로 닫기" 경고가 붙는다.

## ④ FE 요청

화면은 아직 새 필드를 보이지 않는다. 필드는 모두 추가만이라 지금 화면이 깨지지는 않는다. 아래는 UI가 E 코드 카드 하나로 바뀐 뒤(`ui/docs/09-rag-development-reply.md` "검증 및 UI에 남은 변경")를 기준으로 한다.

1. **카드에 안내 문장**: `field_memory.note_ko`를 카드 위나 아래에 한 줄로 보인다. "현장 기록 · 참고"로 표시해 매뉴얼 근거와 구분한다. 지금 카드 목록은 `ui/src/pages/case/parts.tsx:219-249`(CauseList)다.
   - 원하면 CaseFacts(`parts.tsx:183-211`)에 "지난 알림: 오알람 N · 고장 확인 M"(`field_memory.alarm`)을 더한다.
2. **점검 항목별 지난 기록** (`parts.tsx:63-128`, StepPanel): 점검 id 옆에 "지난 점검에서 이상 N · 정상 M"(`field_memory.checks[id]`). 기록이 없는 id는 키가 없다.
3. **오알람 사유 칩 + 위치** (`parts.tsx:295-323`, `ui/src/pages/case/CaseLayout.tsx:33-35`): close 본문에 `reason`·`note`를 보낸다. 오알람 버튼을 상단 "도움 요청" 옆에 둔다.
4. **닫힌 케이스는 읽기 전용**: 지금은 닫은 뒤에도 기록을 바꾸고 다시 닫을 수 있다. 마지막 closure가 라벨이 된다(⑥).
5. **챗봇 피드백 본문** (`ui/src/pages/AssistantPage.tsx:106`): 지금은 `conversation_id`·`message_index`·`helpful`뿐이다. api가 어떤 질문과 답에 대한 평가인지 알 수 없다. 질문과 답을 함께 보낸다.
6. **출처 표시**: 현장 기억은 사람 판정이다. 챗봇의 현장 기록 문단과 카드의 문장을 매뉴얼 근거와 다른 모양으로 보인다. 나중에 시뮬레이션 기록이 섞이면 사람 / 시뮬레이션도 구분한다.
7. **주석 · mock · 타입**: `ui/src/types/case.ts:155` 주석("ML · RAG에는 보내지 않음")에 "api 현장 기억은 읽는다"를 더한다. `handlers.ts`·`case.ts`에 `field_memory`를 맞춘다(`ui/docs/05-fe-api-requests.md:278`의 약속).
8. **08 수치** (`ui/docs/08-api-draft.md:10`): "알림 13건 → 케이스 4건"(test_0 · test_3)은 10/23 확인 수치다. 지금 번들은 test_0에서만 알림 6건이다(아무것도 닫지 않으면 케이스 2건). 바뀐 번들로 다시 센다.
9. **ingest 경쟁 상태**: 플래그를 끄면 ingest lock이 없다. push와 당기기를 함께 쓰면 케이스가 둘 생길 수 있다(`ml/docs/fe-api-draft-reply.md` 2절). 플래그 없이도 lock을 쓸지 api 담당이 정한다.
10. **상세 조회 저장 경쟁** (api 담당): `GET /api/cases/{id}`는 RAG 보강 → 케이스 다시 읽기 → 저장 순서다(`api/main.py:210-214`, RAG 담당 판). 다시 읽은 뒤 저장하기 전에 들어온 쓰기(닫기 등)는 이 저장에 덮어써진다. 플래그와 상관없다.
    - 리허설에서 대본 작업자가 드라이버의 상세 조회와 같은 순간에 닫자 닫기가 사라졌다(`ml/history.md` 36절). 화면도 케이스를 열 때와 창으로 돌아올 때마다 상세를 다시 읽는다(react-query 기본값, `ui/src/lib/useCase.ts:11-13`).
    - 기록 저장 주소도 모두 읽기 → 고치기 → 저장이고 케이스 단위 lock이 없다.
    - 조회는 guidance만 저장하거나 바뀐 것이 없으면 저장하지 않기, 쓰기는 케이스별 lock 같은 방법을 api 담당이 정한다.

예전 요청 중 "S10 출구"와 "후보별 근거 수준(`evidence_level`)"은 뺐다. 카드가 E 코드 하나라 S10 후보도, 후보별 근거 수준도 없다.

## ⑤ RAG 연동 제안

지금 api는 현장 기억을 RAG로 보내지 않는다. 챗봇 문단은 api가 RAG 답 뒤에 붙인다. RAG가 직접 쓰게 한다면 아래를 제안한다. RAG 담당과 합의한 뒤 정한다.

- `answer_question(question, *, error_code=None, history=None, database=None, use_llm=True, use_cache=False)`(`rag/service.py:101-102`)에 `field_history=None`을 더한다.
  - 기본 `None`이면 지금과 같다.
  - 넘기면 생성 입력 `context`(`rag/service.py:152-155`)에 `evidence`·`reviewed_guidance` 밖의 따로 된 키로 둔다. 인용(cite)할 수 없는 근거다.
  - 넘길 모양은 api `GET /api/memory`(= `field_memory.snapshot`)에서 그 코드 부분만 쓰면 된다(`chat_note`와 같은 고르기).
  - 먼저 챗봇 경로(`api/assistant.py:31`)에만 넘긴다. 카드 생성(`api/guidance.py:91`)에는 넘기지 않는다. 카드는 그대로라는 결정 때문이다.
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

- **문장만이다.** 카드 순서도 점검 순서도 바꾸지 않는다. R1은 "오알람인지부터 확인"하라는 문장일 뿐이다. 작업자가 읽지 않으면 효과가 없다. 화면 표시는 ④ 요청 1 뒤에야 생긴다.
- **RAG 지연 · 키.** 키가 있으면 케이스의 첫 상세 조회마다 RAG 답을 만든다(코드별 약 4~12초, `ui/docs/09-rag-development-reply.md` 5절). 드라이버의 guide와 화면의 첫 열기가 그만큼 기다린다. 상태가 `ok`가 아니면 열 때마다 다시 시도한다(`api/guidance.py:84-86`). 키가 없으면 카드는 매핑, RAG 글은 오류다.
  - 드라이버의 api 연결은 120초까지 기다린다(`ml/.py/hitl_loop.py:685`). RAG 한 번(OpenAI 최대 90초 + Neo4j 연결 15초)보다 길게 잡았다.
  - 새 알림이 케이스로 안 보이면, 드라이버는 같은 설비의 열린 케이스 상세를 한 번씩 읽어 '반복'으로 묶였는지 본다(`ml/.py/hitl_loop.py:253-258`). 알림마다 한 번이고 되풀이 조회는 아니다. 다만 카드가 아직 `ok`가 아니면 이 조회도 RAG를 다시 부른다. 묶임 확인이 느리면 이 때문이다.
- **재생 데이터다.** 현장이 아니다. 성능이 아니다.
  - 데이터 기준 라벨(`pre_failure`/`far`)은 답을 아는 것이다. 사람 라벨과의 일치는 고리가 도는지 보는 용도다.
- **집계 키가 설비 × 코드다.** 다른 설비의 같은 코드는 따로 센다(t4의 첫 E029는 R1이 없다). 코드 없는 알림은 R1을 쓰지 않는다. 점검 기록은 설비를 가리지 않는다.
- **원인 정답이 없다.** 데이터에 원인 라벨이 없다. "고장 확인"은 사람이 해결 종료했다는 뜻이지 검증됐다는 뜻이 아니다.
- **마지막 closure가 이긴다.** 닫힌 케이스를 다시 닫으면 closure를 덮어쓴다. 라벨도 바뀐다.
- **데모 규모의 경쟁 상태.** 집계는 부를 때마다 케이스 전체를 다시 읽는다(수십 건 규모). lock은 ingest만 감싼다. 닫기와 새 알림이 동시에 오면, 방금 닫은 기록이 새 문장에 빠질 수 있다. 다시 열면 새로 센다. 상세 조회의 저장이 같은 순간의 닫기를 덮는 경쟁도 있다(④ 10). 사람이 누르는 속도로는 드물다.

### 현장 이야기: 사람 기억이 24시간 규칙을 0~1초 차로 메운다

- test_0의 E029는 09-06·07·08·10일 19:30(데이터 시각, UTC)에 뜬다. 한국 시각으로는 다음 날 새벽 04:30이다.
- 하루 간격이면 간격이 86,400초 또는 86,401초다.
- ML의 반복 규칙은 "직전 발화 뒤 `rule_repeat_s`(24시간) **미만**"이다(`ml/main.py:806-807`). 86,400초는 미만이 아니다. 그래서 매번 새 critical 알림이 된다(`ml/docs/fe-api-draft-reply.md` 3절).
- ML은 이 규칙을 바꾸지 않는다. 사람이 "오알람"으로 닫은 기억이 다음 날 같은 시각의 알림에 "오알람(운영 원인)인지부터 확인" 문장을 붙인다.
- 시연의 수치는 재생 데이터에서 이 고리가 도는 모습이다. 성능으로 읽지 않는다.
