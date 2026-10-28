# HITL 피드백 루프 시연 (D-8): 현장 기억 · LangGraph 드라이버

2026-10-28 · ML 담당

요약: 사람이 케이스를 닫은 기록이 다음 안내로 돌아오지 않았다. 그래서 api에 "현장 기억"을 붙였다. `FIELD_MEMORY=1`일 때만 켜지고, 끄면 api는 예전과 똑같다. ML은 바뀌지 않는다. 지금처럼 사실(코드 · 고장 유형 · 센서)만 보낸다. 재생 데이터로 고리가 도는지 보이는 시연이다. 성능 수치가 아니다.

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
- `api/main.py:101`: 안내(guidance)는 케이스를 만들 때 한 번 정한다. 지난 케이스는 보지 않는다.
- `api/guidance.py:119`: 후보는 코드의 현상 하나뿐이다. 원인 상황(S05~S08)과 운영 원인(S10)은 없다.
- `api/main.py:407-409`: 챗봇 피드백은 `{"ok": true}`만 돌려준다. 저장하지 않는다.

그래서 같은 설비의 같은 오알람을 다섯 번 닫아도, 다섯 번째 안내가 첫 번째와 같았다.

## ② 바뀌는 것 (api, `FIELD_MEMORY=1`일 때만)

### 플래그

- `FIELD_MEMORY=1`이면 켜진다. 부를 때마다 환경 변수를 읽는다(`api/field_memory.py`의 `enabled()`).
- 끄면 아래가 모두 예전과 같다(`api/tests/test_memory_api.py`의 `test_flag_off_identical`).
  - `/api/health` 키, `/api/memory`(404, openapi에도 없음), close 본문, guidance, 저장된 케이스 JSON.
- 새 파일: `api/field_memory.py`(집계 · 안내), `api/cause_links.json`(원인 → 고장 연결).

### 기본 후보

- 원인 → 고장 연결은 정의서 p.4-5에서 가져왔다: S05→S01, S07→S02, S06→S03, S08→S03.
  - 매뉴얼 기반 추정이다. 데이터로 확인한 관계가 아니다(`cause_links.json`의 `note`).
- 기본 후보 순서: [현상] + 연결 원인(파일 순서) + [S10].

| 코드 | 기본 후보 |
|---|---|
| E012 | S01, S05, S10 |
| E016 | S02, S07, S10 |
| E028 | S03, S06, S08, S10 |
| E029 | S04, S10 |

### 라벨 (닫힌 케이스만)

| closure | 라벨 |
|---|---|
| `false_alarm` | `false_alarm` |
| `resolved`, 결정 원인이 S10 | `false_alarm` |
| 그 밖의 `resolved` (후보 밖 원인 `MANUAL` 포함) | `real` |
| `unresolved`, `handed_over`, closure 없음 | 라벨 없음 |

- 결정 원인: 마지막 `resolved` 결과의 상황 ID. 없으면 마지막 "맞아요"(yes) 판단의 상황 ID.
- S10 규칙: S10(설정값 변경 · 잔고장 반복)으로 해결했으면 고장이 아니라 운영 원인이다. 그래서 오알람으로 센다.
- `MANUAL`은 `real`로 세지만 원인 집계에는 넣지 않는다. 적은 글은 `manual_causes`에 따로 남는다.

### R1: 순서를 바꾸는 규칙은 이것 하나

- 같은 설비 · 같은 규칙 코드에서 오알람이 고장 확인보다 많으면 S10을 맨 앞에 둔다.
  - 규칙 코드는 작업자가 화면에서 보는 코드(`trigger.rule_code`)다.
- 나머지 후보 순서와 후보 안 점검 순서는 그대로다.
- 안내 문장(`note_ko`): "같은 설비·코드(…)에서 사람이 오알람으로 닫은 기록 N건 · 고장 확인 M건 → 운영 원인(S10)부터 확인. 계속 반복되면 다음 카드(현상)로."
  - 기록은 있는데 R1이 아니면: "같은 설비·코드(…) 지난 기록: 오알람 N건 · 고장 확인 M건 (순서는 매뉴얼 기준)."
- 규칙 코드가 없는 알림(모델 신호만)에는 R1을 쓰지 않고, 오알람 문장도 쓰지 않는다. 지난 확정 원인 참고만 붙는다.

### 참고 정보 (순서를 바꾸지 않는다)

- 지난 같은 고장의 확정 원인 건수: 예) "지난 같은 고장(E028) 확정 원인: …(S08) 1건 (참고, 순서는 매뉴얼 기준)."
- 확률로 말하지 않는다. "참고" 또는 "매뉴얼 근거 아님"을 붙인다.
- 현장 기록은 매뉴얼 근거(cite · evidence)로 내보내지 않는다.

### 새 필드 · 주소

| 어디 | 필드 | 뜻 |
|---|---|---|
| 케이스 `guidance` | `memory` | `alarm_key`(설비\|코드), `alarm`(오알람 · 고장 확인 수), `applied`(`["R1"]` 또는 `[]`), `baseline`, `past_causes`, `note_ko` |
| 후보 | `basis` | `code`(현상) · `manual_link`(정의서 연결 원인) · `alarm_truth`(S10) |
| 후보 | `evidence_level` | 연결 원인만. 정의서뿐이면 `manual`, 사람이 한 번 이상 확정했으면 `field` |
| 후보 | `field` | S10 밖 후보만. `{"confirmed": N}` 지난 확정 건수 |
| `GET /api/cases/{id}` | `field_memory` | 자기 자신을 뺀 집계: `alarm_key`, `alarm`, `past_causes`, `checks`(이 케이스 점검 항목의 지난 이상 · 정상 수), `labeled` |
| `POST /api/cases/{id}/close` 본문 | `reason`, `note` | 빈 문자열이 아니면 closure에 남긴다 |
| `GET /api/memory` | 전체 요약 | `labeled`, `labels`, `alarms`, `past_causes`, `checks`, `manual_causes`, `missed_checks`(일지의 놓친 점검 글). 끄면 404 |
| `GET /api/health` | `memory: "on"` | 켰을 때만 |

- 안내는 케이스를 만들 때(ingest) 한 번 정한다. 이미 열린 케이스의 후보 순서는 나중에 바뀌지 않는다.
- `field_memory`는 케이스를 열 때마다 새로 센다(`api/main.py:211`).
- 코드: `api/main.py:101-105`(ingest), `:194`(health), `:266`(close), `:342-347`(`/api/memory`).

### 챗봇

- `POST /api/assistant/ask`: 질문이 고장 상황(S01~S04)에 맞고, 그 코드에 기록이 있을 때만 답 끝에 한 문단을 붙인다(`api/assistant.py:139`).
  - 기록 = 라벨이 붙은 닫힌 케이스(오알람 · 고장 확인)나 지난 확정 원인. 없으면 문단도 없다.
  - 예: "현장 기록(사람 판정, 매뉴얼 근거 아님): E029 알림 — test_0 오알람으로 닫음 N · 고장 확인 M". 시연에서 설비 이름은 파일 이름이다.
  - cite · evidence에는 넣지 않는다.
- 피드백(`/api/assistant/feedback`)은 여전히 저장하지 않는다(④ 요청 7).

### ingest 잠금

- 켜면 ingest 전체(seen → put)를 lock 하나로 감싼다(`api/main.py:36`, `:82`).
- push와 5초 당기기가 겹쳐도 같은 알림으로 케이스가 둘 생기지 않는다.
- 끄면 예전 그대로 잠그지 않는다(④ 요청 11).

## ③ 시연법

사람만 닫는다. 시뮬레이션 작업자는 없다. 재생은 600초 청크다.

### 준비

- 처음 한 번: `py -3.11 -m pip install "langgraph>=1.0"`(`ml/requirements.txt`의 선택 줄. 따옴표가 없으면 셸이 `>`를 리다이렉트로 읽는다).
- 모델 파일과 test CSV는 `ml/README.md` 0절대로 둔다.
- 실행마다 이름을 하나 정한다(예: `1028a`). ML 상태와 api DB를 매번 새로 쓴다.

### 켜는 순서 (bash)

```bash
# 1) ML 서버: RSW_RAG_URL 없이(api가 당겨 감), 새 상태 폴더
cd ml && RSW_STATE_DIR=state/d8-1028a uvicorn main:app --port 8000

# 2) api 서버: 현장 기억 켜기, 새 DB
cd api && FIELD_MEMORY=1 API_DB=state/d8-1028a.db uvicorn main:app --port 9000

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
  - ML은 새 `RSW_STATE_DIR`로 켠 뒤에 돌린다. 재생의 `reset`(`DELETE /guns/{id}`, `ml/main.py:1905-1917`)은 설비 상태만 지우고 outbox는 그대로 둔다. 그래서 앞 실행의 핸드오프가 건수에 섞인다.
  - 드라이런 뒤 실제 실행 전에는 ML을 또 다른 새 폴더로 다시 켠다. 드라이런이 outbox를 채웠기 때문이다.
- 시작 점검(`startup_check`)이 아래를 차례로 확인하고, 하나라도 어긋나면 멈춘다. 산출물 폴더는 점검을 통과한 뒤에 만든다.
  - ML `/health`: 추적 중인 설비 0, outbox 비어 있음, `rag_url` 없음.
  - api `/api/health`: `memory`가 `on`, `ml`이 `ok`(아니면 api의 `ML_URL`을 확인한다).
  - api 케이스 · 이력 비어 있음. ML에 문제가 없을 때만 본다. `GET /api/cases`가 ML outbox를 api로 당겨 오기 때문이다(`api/main.py:199-201`).
  - 서버가 꺼져 있으면 그 서버 주소와 "먼저 그 서버를 켜세요"를 보인다.
  - 고치는 법: ML이 깨끗하지 않으면 새 `RSW_STATE_DIR` 폴더(예: `state/d8-<새 이름>`)로 ML을, api가 비어 있지 않으면 새 `API_DB` 파일(예: `state/d8-<새 이름>.db`)로 api를 다시 켠다.
- 산출물: `ml/results/D-8/runs/<YYYYmmdd-HHMMSS>/`(git 제외)
  - `loop_view.html`: 2초마다 새로고침되는 현황.
    - 상태 줄: 재생 중이면 마지막 알림의 데이터 시각. ⏸ 대기 중이면 기다리는 케이스 링크, 후보 순서, 안내 문장(`note_ko`).
    - 링크는 대기 중인 케이스 하나만 상태 줄에 있다. 알림 행에는 링크가 없다. 링크는 이름 붙은 탭 하나(`d8-ui`)를 다시 쓰므로 현황 탭은 그대로 열려 있다.
    - 알림 표 칸: 데이터 시각, 설비, 코드, 출처, 상태, 케이스, 후보 순서, 기본 순서, 기억 반영(R1), 닫힘, 사람 라벨, 데이터 라벨(닫힌 뒤), 일치, 안내 문장, 기억 변화.
    - ⚠ 줄: 도움 요청 경고(작업자 카드 참고). 경고가 있는 행은 상태 칸이 주황이다.
    - 끝나면 노드 강조와 링크 없이 "끝남 · 알림 N건 · 요약 summary.md"(회색) 또는 "중단됨: …"(빨강, Ctrl+C 포함)을 남긴다.
  - `loop_log.jsonl`(시작할 때 비운다. 값은 화면 낱말이 아니라 원래 토큰), `summary.md`(알림 표, 경고, 현장 기억, 오류), `workflow.mmd`(그래프 그림).

### PowerShell 메모

창마다 한 줄씩 돌린다(저장소 루트에서). Windows PowerShell 5.1에는 `&&`가 없어 `;`로 잇는다.

```powershell
cd ml; $env:RSW_STATE_DIR="state/d8-1028a"; uvicorn main:app --port 8000
cd api; $env:FIELD_MEMORY="1"; $env:API_DB="state/d8-1028a.db"; uvicorn main:app --port 9000
cd ui; npm run dev
cd ml; py -3.11 .py/hitl_loop.py test/test_0.csv
```

- PowerShell에서 환경 변수를 `""`로 두면 값이 지워진다. 그래서 `RSW_STATE_DIR=""`(메모리만)는 기본 `state/`로 돌아간다. 새 폴더 이름을 준다.
- 같은 창에 `RSW_RAG_URL`이 남아 있으면 `Remove-Item Env:RSW_RAG_URL`로 지운다.

### 드라이버가 하는 일

- 그래프(LangGraph): advance → ingest → guide → ⏸ operator → learn → evaluate → advance … → report.
  - advance: 재생 스레드(Feeder)의 큐에서 다음 알림을 받는다. 앞 알림의 재생은 여기서 풀어 준다.
  - ingest: api를 동기화하고, 새 알림마다 케이스를 찾는다. 열린 케이스에 "반복"으로 묶인 알림은 멈추지 않는다. 묶은 곳이 도움 요청 케이스면 ⚠ 경고를 단다.
  - guide: 케이스의 후보 순서와 `guidance.memory`를 남긴다.
  - operator: `interrupt()`만 부르고 멈춘다(`ml/.py/hitl_loop.py:273-278`). 이어 갈 때 이 노드는 처음부터 다시 돌므로 부수 효과를 두지 않는다.
    - 조회는 그래프 밖에서 한다: `run_loop`가 `wait_for_closure`로 케이스가 닫힐 때까지 2초마다 조회하고, 닫히면 `Command(resume=…)`로 이어 간다(`:398-424`, `:375-395`).
    - 조회 중 api가 잠깐 안 닿거나 5xx면 다음 조회까지 기다린다. 4xx는 실행을 끝낸다.
  - learn: `GET /api/memory`를 다시 읽어 사람 라벨과 바뀐 집계를 남긴다. 도움 요청으로 닫혔으면 ⚠ 경고를 단다.
  - evaluate: 닫힌 뒤에만 데이터 기준 라벨을 붙인다(파일 끝 1시간 안이면 `pre_failure`, 아니면 `far`). 사람에게 답을 미리 보이지 않으려는 것이다.
- `run_loop`가 Feeder를 켜고 끈다.
- 드라이버는 api 모듈을 import하지 않는다. HTTP로만 붙는다.

### 작업자 카드

```
E029 (오알람)
  루프 현황(loop_view) 상태 줄의 케이스 링크 → 개요 → "오알람으로 닫기" → "오알람 맞아요"

진짜 알림 (예: E012, E028)
  점검 시작 → 점검 항목 하나 이상 기록 → 판단 → 조치 → "해결됐어요"
  → 일지(반복 여부) 승인 → "해결 종료"

코드 없는 알림 (모델 신호만)
  R1은 없다. 안내에 후보가 없으면 개요에서 "오알람으로 닫기" 또는 "직접 점검해서 기록"
  (test_0 · test_5 · test_4를 600초 청크로 돌리면 나오지 않는다)

하지 않는 것
  - "도움 요청"은 쓰지 않는다. 도움 요청 케이스는 열린 채 남아, 같은 설비 · 코드의 다음 알림을
    반복으로 묶는다(멈춤 · R1 없음). 드라이버는 ⚠ 경고를 띄운다.
    눌렀다면: 다음 알림 전에 작업함의 "도움 요청" 칸에서 그 케이스를 열고 개요의 "오알람으로 닫기"로 닫는다.
    그다음 알림부터 다시 새 케이스가 된다.
  - "잘 모르겠어요"는 쓰지 않는다
  - 진행 중에는 닫힌 케이스를 열지 않는다
```

- 도움 요청 흡수: `api/main.py:296-303`(status `handed_over`), `api/store.py:68`(resolved · unresolved · false_alarm이 아니면 묶음).
- 되돌리는 길: 작업함 "도움 요청" 칸(`ui/src/pages/InboxPage.tsx:36`), 개요의 "오알람으로 닫기"(`ui/src/pages/case/OverviewPage.tsx:92`).
- 후보 없는 안내의 개요 버튼: `ui/src/pages/case/OverviewPage.tsx:77-92`.

### 볼 것

- 두 번째 E029부터 후보가 S10이 먼저다(`[S10, S04]`). 안내 문장에 지난 오알람 건수가 나온다.
  - 케이스 화면은 아직 안내 문장을 보이지 않는다(④ 요청 2). 그래서 드라이버가 보인다: `loop_view.html`의 상태 줄과 "안내 문장" 칸, 콘솔 ⏸ 줄 아래.
- 챗봇에 "E029 알림 진짜야?"라고 물으면 답 끝에 "현장 기록(사람 판정, 매뉴얼 근거 아님)" 문단이 붙는다.
- `loop_view.html`이 알림마다 ⏸로 멈추고, 닫으면 이어 간다.

| 실행 | 알림 (데이터 시각 UTC) | 기대 안내 |
|---|---|---|
| test_0 | E029 09-04 16:45 · 09-06/07/08/10 19:30 (오알람으로 닫음) → E012 09-11 02:46 (진짜) | 첫 E029 [S04, S10] → 2~5번째 [S10, S04] + 오알람 1~4건 문장 → E012 [S01, S05, S10] |
| test_5 → test_4 | t5 E029 09-28 06:05 (오알람) → t5 E028 10-03 02:15 (진짜, 예: S08 확정) → t4 E029 10-12 (다른 설비라 R1 없음) → t4 E029 10-16 (R1) → t4 E028 10-18 (진짜) | t4 E028은 [S03, S06, S08, S10] 그대로. 참고 "S08 확정 1건", S08 `evidence_level` manual → field |

## ④ FE 요청

화면은 아직 새 필드를 보이지 않는다. 필드는 모두 추가만이라 지금 화면이 깨지지는 않는다.

1. **CaseFacts 줄** (`ui/src/pages/case/parts.tsx:183-211`): "지난 알림: 오알람 N · 고장 확인 M"(`field_memory.alarm`).
2. **CauseList** (`parts.tsx:219-249`): 후보별 참고 내역(`field.confirmed`)과 근거 수준 DotTag(`manual` 회색 / `field` 주황). `guidance.memory.note_ko` 한 줄.
3. **StepPanel** (`parts.tsx:63-128`): 점검 항목마다 "지난 점검에서 이상 N · 정상 M"(`field_memory.checks`).
4. **오알람 사유 칩 + 위치** (`parts.tsx:295-323`, `ui/src/pages/case/CaseLayout.tsx:33-35`): close 본문에 `reason`·`note`를 보낸다. 오알람 버튼을 상단 "도움 요청" 옆에 둔다.
5. **S10 출구**: S10 "맞아요" 뒤에 "오알람(운영 원인)으로 닫기"를 둔다. 남은 원인 없음 화면(`ui/src/pages/case/NoCausePage.tsx:50`의 선택지)에도 오알람을 둔다.
6. **닫힌 케이스는 읽기 전용**: 지금은 닫은 뒤에도 기록을 바꾸고 다시 닫을 수 있다. 마지막 closure가 라벨이 된다(⑥).
7. **챗봇 피드백 본문** (`ui/src/pages/AssistantPage.tsx:106`): 지금은 `conversation_id`·`message_index`·`helpful`뿐이다. api가 어떤 질문과 답에 대한 평가인지 알 수 없다. 질문과 답을 함께 보낸다.
8. **출처 표시**: 현장 기억은 사람 판정이다. 나중에 시뮬레이션 기록이 섞이면 사람 / 시뮬레이션을 화면에 구분한다.
9. **주석 · mock · 타입**: `ui/src/types/case.ts:155` 주석("ML · RAG에는 보내지 않음")에 "api 현장 기억은 읽는다"를 더한다. `handlers.ts`·`case.ts`에 새 필드를 맞춘다(`ui/docs/05-fe-api-requests.md:278`의 약속).
10. **08 수치** (`ui/docs/08-api-draft.md:10`): "알림 13건 → 케이스 4건"(test_0 · test_3)은 10/23 확인 수치다. 지금 번들은 test_0에서만 알림 6건이다(아무것도 닫지 않으면 케이스 2건). 바뀐 번들로 다시 센다.
11. **ingest 경쟁 상태**: 플래그를 끄면 ingest lock이 없다. push와 당기기를 함께 쓰면 케이스가 둘 생길 수 있다(`ml/docs/fe-api-draft-reply.md` 2절). 플래그 없이도 lock을 쓸지 api 담당이 정한다.

## ⑤ RAG 연동 제안

지금 api는 현장 기억을 RAG로 보내지 않는다. 붙인다면 아래를 제안한다. RAG 담당과 합의한 뒤 정한다.

- `answer_question(..., field_history=None)` (`rag/service.py:652`)
  - 기본 `None`이면 지금과 같다.
  - 넘기면 `manual_evidence`(`rag/service.py:596`) 밖의 따로 된 칸에 둔다. 인용(cite)할 수 없는 근거다.
  - 넘길 모양은 api `GET /api/memory`(= `field_memory.snapshot`)를 그대로 써도 된다.
- 프롬프트 규칙: "현장 기록은 매뉴얼 근거가 아니다. 확률로 말하지 않는다."
  - `PROMPT_VERSION`(`rag/service.py:26`)을 올린다. 캐시 키(`answer_cache_path`, `rag/service.py:253`)에도 `field_history`를 넣는다.
- Neo4j
  - `(:Situation)-[:LEADS_TO {evidence_level, confirmed}]->(:Situation)`: `cause_links.json`의 연결.
  - `(:FieldCase)`: 닫힌 케이스 한 건.
  - build가 지우는 관계(`rag/pipeline/build.py:475-484`: RELEVANT_DIAGNOSTIC · RELEVANT_SECTION · RELEVANT_EVIDENCE · CHECKS_COMPONENT와 festo 노드)에 들지 않는다. 다시 빌드해도 남는다.

## ⑥ 한계

- **재생 데이터다.** 현장이 아니다. 성능이 아니다.
  - 데이터 기준 라벨(`pre_failure`/`far`)은 답을 아는 것이다. 사람 라벨과의 일치는 고리가 도는지 보는 용도다.
- **집계 키가 설비 × 코드다.** 다른 설비의 같은 코드는 따로 센다(t4의 첫 E029는 R1이 없다). 코드 없는 알림은 R1을 쓰지 않는다.
- **원인 정답이 없다.** 데이터에 원인 라벨이 없다. `evidence_level: field`는 "사람이 확정했다"이지 검증됐다는 뜻이 아니다.
- **S10 점검 3번째와의 긴장.** 정의서 S10의 세 번째 점검은 "반복되면 S01~S08 해당 항목으로 재분류"다(`rag/data/situation_mapping.json:503`). R1은 반복 오알람을 S10으로 먼저 보낸다. 그래서 안내 문장 끝에 "계속 반복되면 다음 카드(현상)로"를 붙였다.
- **마지막 closure가 이긴다.** 닫힌 케이스를 다시 닫으면 closure를 덮어쓴다. 라벨도 바뀐다.
- **데모 규모의 경쟁 상태.** 집계는 부를 때마다 케이스 전체를 다시 읽는다(수십 건 규모). lock은 ingest만 감싼다. 닫기와 새 알림이 동시에 오면, 방금 닫은 기록이 새 안내에 빠질 수 있다.

### 현장 이야기: 사람 기억이 24시간 규칙을 0~1초 차로 메운다

- test_0의 E029는 09-06·07·08·10일 19:30(데이터 시각, UTC)에 뜬다. 한국 시각으로는 다음 날 새벽 04:30이다.
- 하루 간격이면 간격이 86,400초 또는 86,401초다.
- ML의 반복 규칙은 "직전 발화 뒤 `rule_repeat_s`(24시간) **미만**"이다(`ml/main.py:795-796`). 86,400초는 미만이 아니다. 그래서 매번 새 critical 알림이 된다(`ml/docs/fe-api-draft-reply.md` 3절).
- ML은 이 규칙을 바꾸지 않는다. 사람이 "오알람"으로 닫은 기억이 다음 날 같은 시각의 알림에서 S10을 먼저 보이게 한다.
- 시연의 수치는 재생 데이터에서 이 고리가 도는 모습이다. 성능으로 읽지 않는다.
