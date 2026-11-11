# ML 피드백 루프(D-9) 1단계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 현장 사람 기록과 종료 코드로 "ML 조기 경보 규칙과 건 프로필을 바꿀지"를 판정하는 보고서(`report.md`·`report.json`)를 만들고, 오프라인 실행 두 번으로 검증한다.

**Architecture:**
- ML 서버는 `RSW_WINDOW_LOG=1`일 때 새로 채점한 창마다 한 줄을 `<상태 폴더>/windows/`에 쓴다(추가만, 기본 끔).
- `ml/.py/feedback_eval.py`는 파일을 모르는 순수 함수다. 하는 일:
  - B-18 평가를 한 설비 여러 고장으로 넓힌 라벨
  - 설비 복원 추출 CI
  - 7.4 판정, 7.5 합치기, 7.6 프로필
- `ml/.py/feedback_loop.py`는 상태 폴더 · 닫힌 케이스 · 실험실 캐시를 읽어 평가부를 부르고 보고서를 쓴다.
- api는 조기 경보 케이스를 `warning`으로 받고 따로 묶으며, 목록에 `event_type`·`repeat_event_ids`를 더한다(11.1).
- 루프는 제안만 한다. 반영은 사람이 한다.

**Tech Stack:** Python 3.11, pandas, numpy, FastAPI TestClient, sqlite3, httpx, pytest, ruff

**Spec:** `ml/docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md` (사용자 승인 11-08, 7.5 한 줄을 고친 뒤). 이 계획은 스펙을 다시 쓰지 않는다. 순서, 이름, 스펙이 정하지 않은 결정만 적는다.
- 창 기록: 5절
- 라벨: 6절
- 도전자와 판정: 7절
- 보고서: 8절
- 2단계 대비: 9.7
- 검증: 10절
- api와 요청: 11절

의존: Task 1 · 2는 따로 해도 된다. 나머지는 3 → 4 → 5 → 6 → 7 → 8 순서다.

## Global Constraints

작업 범위:
- main 작업 트리에서 바로 고친다. 커밋, 푸시, 브랜치, PR, worktree는 만들지 않는다. 사용자가 커밋한다.
- `ui/`, `rag/`는 바꾸지 않는다. `api/`는 11.1의 1~3만 한다(4는 2단계).
- 바꾸지 않는 것: `ml/.py/train.py`, 서빙 번들, ML API 응답과 `/health`, 핸드오프 v1.4, 서빙 판정, outbox.
- 유료 API를 부르지 않는다. `rag/.env`를 읽지 않는다.

환경:
- 8 GB PC다. 서버를 띄우지 않는다(TestClient만 쓴다).
- 실데이터 실행은 10분 안에 끝나는 전경 실행으로 나눈다. 실행이 죽으면 자기가 띄운 python 프로세스만 찾아 끈다.
- 고친 파일은 원래 줄 끝(CRLF)을 지킨다. 새 파일은 LF여도 된다(`core.autocrlf=true`).
- 줄 길이는 120이다. 기존 ruff 오류는 `ml/tests/test_sensor_features.py:14`의 E402 하나뿐이다.
- 경로는 `PROJECT_ROOT`(= `ml/`) 기준이다(`ml/CLAUDE.md`).
- 시각은 모두 naive UTC로 바꿔 비교한다. 파일에 쓸 때는 끝에 `Z`를 붙인다.

| 이름 | 명령 | 지금 기준 |
|---|---|---|
| ML | `ml/`에서 `py -3.11 -m pytest -q` | 130 passed |
| API | 저장소 루트에서 `.venv\Scripts\python.exe -m pytest api/tests -q` | 84 passed, 1 skipped |

계획 결정(스펙이 정하지 않은 것. 사용자가 계획을 검토할 때 바꿀 수 있다):
1. **시험 위치.** 창 기록 서버 시험은 `ml/tests/test_pipeline.py` 끝에 둔다. 합성 번들을 그 파일의 test_1~test_3이 모듈 범위 `workspace`에 만들기 때문이다. 평가 · 루프 시험은 `ml/tests/test_feedback_loop.py`에 둔다.
2. **창 기록 쓰는 곳.** judge의 trace 점에 `early_warning_onset`·`warmup`을 더한다. `score_request`가 이번 요청에 새로 붙은 trace 점(`new_minutes`, `main.py:1618`)을 쓴다. trace 추가 조건을 그대로 쓰므로 5.3과 같다.
3. **창 시작 정리.** 루프는 창 시작을 분 단위로 내린다(`floor("60s")`, B-11 캐시 `time`과 같은 버킷 이름). 중복은 (설비, 내린 창 시작)마다 마지막 줄로 정리한다. 재시작 뒤 첫 초가 빠진 같은 분도 하나로 센다.
4. **모델이 여럿일 때.** 가장 늦은 창의 모델만 평가하고, 뺀 창 수를 경고에 적는다. 모델마다 점수 축이 달라 섞을 수 없다.
5. **기록 끝 규칙.** 기록 끝에 기대는 규칙은 다음 고장이 없는 창(lead = 무한대)에만 적용한다. 뒤에 고장이 있으면 결과를 이미 알기 때문이다. 그래서 실험실에서 B-18과 같은 답이 나온다. 해당하는 규칙:
   - 시작의 "대기"
   - 감시 일수에서 기록 끝 60분 빼기
   - 현장 정상 창에서 기록 끝 하루 빼기
6. **조치함(6.3).** 해결 종료로 닫힌 조기 경보 케이스에는 지금 규칙의 시작이 이어져 있다. 그 시작이 든 연속 구간(지금 규칙 기준)을 `acted`로 표시한다. 모든 규칙에서 `acted` 창의 오알람 시작은 오알람 대신 조치함으로 센다.
7. **사유 없음.** 사유 없이 오알람으로 닫힌 케이스(`FIELD_MEMORY`가 꺼진 때 등)는 "기타"가 아니라 "사유 없음"으로 따로 센다.
8. **판정 표기.** json의 판정은 코드와 한국어를 함께 싣는다(아래 표).
9. **outbox 보존.** outbox는 30일만 보존한다(`main.py:120`). 창 기록이 30일보다 긴 설비가 있으면 경고한다.
10. **확인된 고장만.** 확인된 고장만으로 한 결과(8.2의 4)는 수치만 싣고 판정하지 않는다.

글자 그대로 쓸 값:
- 상수(스펙 7.8, `feedback_eval.py`):

```python
MARGIN, CHEAPER = 0.120, 0.843
Q_MINUS, Q_PLUS, Q_REFERENCE = 0.960, 0.980, 0.975
MIN_FAILURES, MIN_GUNS = 10, 5
PROFILE_CASES, PROFILE_DAYS = 2, 2
FA_CAP = 24
REFERENCE = "model_ng_k1 @ 0.975"
GATE_NON_WELDING = 0.5
ZONE_S, MIN_ZONE_WINDOWS, PLACEBO_DAYS = (600, 3600), 25, range(1, 7)
CALIBRATION_LEAD_S = 86400
N_BOOT, SEED_SINGLE, SEED_PAIRED = 2000, 42, 7
POST_FAILURE_S = 3600
WINDOW_S = 60
OUTBOX_RETENTION_DAYS = 30
CLASSES = ("E01", "E02", "E03", "E04")
REASONS = ("설비 정지 · 전원 꺼짐", "정상 작업 중 변동", "센서 · 신호 문제", "기타")
REASON_NONE = "사유 없음"
```

- 판정 코드:

| 출처별 코드 | 한국어 | 종합 코드 | 한국어 |
|---|---|---|---|
| `pass` | 통과 | `propose` | 반영 제안 |
| `insufficient` | 증거 부족 | `review` | 검토 필요(실험실과 현장이 다름) / 검토 필요(실험실 확인 없음) |
| `na` | 해당 없음 | `hold` | 보류(현장 증거 쌓는 중) / 보류 |
| `fail` | 탈락 | `reject` | 반영 안 함 |
| | | `not_evaluated` | 평가 안 함 |

  종합의 괄호 문구는 스펙 7.5 표의 그 줄 그대로다.
- 보고서 문구:
  - 제목: `# ML 피드백 루프 보고서 (D-9)`
  - 시뮬레이션 라벨이 섞이면 제목 다음 첫 줄: `> 시뮬레이션 라벨 포함 — 성능 아님`
  - 절 제목: `## 1. 머리말`, `## 2. 한눈에 보기`, `## 3. 라벨 현황`, `## 4. 조기 경보`, `## 5. critical 프로필`, `## 6. 사유별 메모`, `## 7. 고장 뒤 조기 경보 진단`, `## 8. 한계`
  - 머리말 문장: `이 보고서는 제안만 하며, 루프는 아무것도 바꾸지 않음`
  - 프로필 위험 문장: 스펙 7.6 그대로
  - 지켜보는 중: `지켜보는 중({PROFILE_CASES}건 중 {n}건, {PROFILE_DAYS}일 중 {m}일)`
  - 적용 방법: 스펙 7.2 표의 "적용 방법" 칸 그대로(Q0.975 줄은 11-08에 고친 문장)
- 경고(`warnings`):
  - `창 기록 없음: 조기 경보 현장 평가는 해당 없음`
  - `창 기록에 모델이 여럿이라 가장 늦은 모델({id})의 창만 평가함 (뺀 창: {id} {n}개, …)`
  - `읽지 못한 줄 {n}개를 건너뜀 ({곳})`
  - `창 기록이 30일보다 긴 설비({설비들}): outbox는 30일만 보존하므로 오래된 고장이 빠졌을 수 있음`
  - `창 기록이 없는 설비의 고장 {n}건은 평가에서 뺌`
  - `창 기록에서 시작 창을 찾지 못한 조기 경보 핸드오프 {n}건`
  - `서빙 판정과 다시 계산한 지금 규칙이 다른 창 {n}개`
  - `현장 정상 창이 없어 {이름들}의 현장 평가는 해당 없음`

## Review Focus

1. **번들을 바꾼 뒤의 창 기록.** 모델이 둘 섞이면 늦은 모델의 창만 평가하고 경고한다. 섞어서 분위수를 내면 안 된다. Task 5 `test_read_windows_keeps_newest_model`.
2. **재시작 뒤 같은 분이 두 번, 첫 초가 다름.** 한 창으로 세고 나중 줄을 쓴다. Task 5 `test_read_windows_dedup_floor`.
3. **서버가 죽어 끊긴 마지막 줄.** 그 줄을 건너뛰고 경고하며, 보고서는 끝까지 나온다. Task 5 `test_read_windows_skips_broken_lines`.
4. **11.1 이전 api의 `/api/history`.** `repeat_event_ids`가 없으면 시간 범위로 잇고, `event_type`이 없으면 critical로 본다. Task 5 `test_case_index_paths`, `test_failures_from_status`.
5. **outbox 30일보다 긴 창 기록.** 오래된 고장이 빠졌을 수 있다고 경고한다. Task 6 `test_report_warns_long_log`.

---

### Task 1: api 변경 (스펙 11.1의 1~3)

**Files:**
- Modify: `api/store.py`
  - `put` 56-63
  - `open_case` 설명 91행
  - 새 함수 `group_key`
- Modify: `api/main.py`
  - `ingest` 102-137: 묶음 키, 124-125행의 주석과 `severity`
  - `summary` 174-191
  - `history` 425-444
- Create: `api/tests/test_feedback_api.py`
- Modify: `api/README.md` ("## 연결과 저장" 절 끝에 한 문단)

**Interfaces:**
- Produces:
  - `store.group_key(e: dict) -> str | None`
    - 값: `e["trigger"].get("rule_code")`. 그게 없고 `e.get("event_type") == "early_warning"`이면 `"early_warning"`, 아니면 None.
    - `put`은 이 값을 `cases.code`에 쓰고, `ingest`는 이 값을 `open_case`에 넘긴다.
  - 케이스 `severity`: `event_type`이 `early_warning`이면 `"warning"`, 아니면 `"critical"`.
  - `summary(d)["event"]["event_type"]`: `e.get("event_type") or "critical"`.
  - `/api/history` 항목의 `repeat_event_ids: list[str]`: 케이스 `repeats`의 `event_id`를 순서대로.

- [ ] **Step 1: 시험을 쓴다.**

  도우미:
  - `_ew(make_handoff, eid, t)`: `make_handoff(eid, code=None, t=t, source="early_warning")`에 `schema_version "1.3"`과 `event_type "early_warning"`을 넣는다.
  - 모델 단독 critical: `make_handoff(eid, code=None, t=t, source="model")`.
  - 설비는 모두 `test_0`이고, 시각은 늘어나기만 하면 된다.

  시험:
  1. `test_ew_severity_warning`: 조기 경보 → `severity == "warning"`. E029 규칙과 모델 단독은 `"critical"`.
  2. `test_ew_and_model_critical_grouped_apart`
     - w1(조기 경보), m1(모델 단독), w2(조기 경보), m2(모델 단독) 순서로 보낸다 → `["new", "new", "repeat", "repeat"]`.
     - CASE-0001의 `repeats` event_id는 `["w2"]`, CASE-0002는 `["m2"]`다.
     - `sqlite3`로 `SELECT case_id, code FROM cases ORDER BY case_id` → `[("CASE-0001", "early_warning"), ("CASE-0002", None)]`.
  3. `test_list_and_history_fields`
     - w1, w2(반복), r1(E029, `event_type`이 없는 v1.2)을 보낸다.
     - `/api/cases`의 `event_type`: CASE-0001 `"early_warning"`, CASE-0002 `"critical"`.
     - 둘 다 `false_alarm`으로 닫은 뒤 `/api/history`
       - `repeat_event_ids`: CASE-0001 `["w2"]`, CASE-0002 `[]`
       - `event.event_type`은 위와 같다.
  4. `test_default_grouping_unchanged`
     - 조기 경보 없이 E029 두 번 → new, repeat. 모델 단독 두 번 → new, repeat.
     - `severity`는 모두 critical이다. `code` 칸은 `"E029"`와 None이다.
- [ ] **Step 2: 실패를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests/test_feedback_api.py -q`
- [ ] **Step 3: 구현한다.**
  - `ingest` 124행 주석 "알림은 ML이 critical일 때만 보낸다"를 고친다: "조기 경보 핸드오프(event_type early_warning, ML `RSW_EARLY_WARNING_HANDOFFS=1`)는 주의, 나머지는 긴급".
  - `open_case` 설명은 "같은 설비 · 같은 묶음 키(`group_key`)"로 고친다.
  - README 문단에 적을 것:
    - 세 가지 변경
    - 조기 경보 핸드오프를 켜지 않으면 응답에 필드 둘이 늘 뿐이다.
    - 옛 DB의 조기 경보 케이스는 `code`가 비어 있어 예전 묶음에 남는다(스펙 13절).
- [ ] **Step 4: 통과를 확인한다.** `.venv\Scripts\python.exe -m pytest api/tests -q` → 88 passed, 1 skipped.
- [ ] **Step 5:** 커밋하지 않는다. `git status -- api`에 고친 파일 셋과 새 시험 하나만 보인다.

### Task 2: 창 기록 (스펙 5절)

**Files:**
- Modify: `ml/main.py`
  - 맨 위 설명 41-45: 상태 문단 끝에 한 문장
  - judge의 trace 점 1102-1107: 칸 둘
  - `Outbox` 뒤(1475행 근처): `window_log_line`, `WindowLog`
  - lifespan 1483-1500: 만들기, 경고, `models.json`
  - `score_request` 1618행 뒤: 쓰기
- Test: `ml/tests/test_pipeline.py` (끝에 `test_17_window_log`, `test_17b_window_log_memory_only`)

**Interfaces:**
- Produces:
  - 파일
    - `<상태 폴더>/windows/<quote(gun_id, safe="-_.")>/<YYYY-MM-DD>.jsonl`. 날짜는 창 시작의 UTC 날짜다.
    - `<상태 폴더>/windows/models.json`
  - 줄 키(이 순서): `window_start, window_end, model, score, threshold, alarm, held, sustained, critical, early_warning, early_warning_onset, ew_threshold, warmup, gun_norm, non_welding_share, error_codes, error_code, vec`
    - `window_start`, `window_end`: trace 점 값에 `rag_mapping.utc_iso(pd.Timestamp(x).isoformat())`
    - `model`: `f"{d.info.name}@{d.info.created}"`
    - `ew_threshold`: `float(d.early_warning["threshold"])`. 조기 경보가 없는 번들이면 None.
    - `vec`: `[float(x) for x in p["vec"]]`
  - `models.json`: `{모델 id: {"model_cols": [...], "threshold": float, "early_warning": {"rule", "threshold", "quantile"} | None}}`. 있던 항목은 그대로 둔다.
  - `window_log_line(p: dict, model_id: str, ew_threshold: float | None) -> dict`
  - `class WindowLog(root: str | None)`. root가 None이면 아무것도 하지 않는다.
    - `add_model(model_id: str, entry: dict) -> None`
    - `write(gun_id: str, points: list[dict], model_id: str, ew_threshold: float | None) -> None`
  - `app.state.window_log`

- [ ] **Step 1: 시험을 쓴다.**
  1. `test_17_window_log(workspace, tmp_path, monkeypatch, caplog)`
     - 준비: test_15처럼 `thr` = 합성 test_0의 오프라인 점수 중앙값. 그러면 조기 경보가 자주 켜지고 꺼진다.
     - 도우미 `_stream(state, log_on)`
       - `RSW_STATE_DIR`과 `RSW_WINDOW_LOG`를 정하고 main을 다시 읽는다.
       - TestClient 안에서 `detector.early_warning["threshold"] = thr`로 두고 `stream_chunks(client, "WL", raw)`를 돌린다.
       - 응답 목록과 `/guns/WL/trace?minutes=360`의 점과 events를 돌려준다.
     - 끔(`off`): `off/windows`가 없다.
     - 켬(`on`): 응답 목록이 끔과 같다(JSON 전체).
     - 줄(`on/windows/WL/*.jsonl` 모두)
       - 모든 줄의 키 목록이 위 순서와 같다.
       - 줄 수와 `window_start` 목록(Z 붙임)이 trace 점과 같고, `score`도 같다.
       - `early_warning_onset`이 참인 줄 수 = events의 `early_warning_start` 수 ≥ 1.
       - `warmup`이 참인 줄이 하나 이상 있고, 그 줄들의 `early_warning`은 모두 거짓이다.
     - `models.json`
       - 키는 `f"{info.name}@{info.created}"` 하나다.
       - `model_cols`가 번들과 같고, `len(vec)`이 그 길이다.
       - `early_warning.quantile`이 번들 값이다.
     - 쓰기 실패(`bad`)
       - 켜기 전에 `bad/windows/WL`을 **파일**로 만든다.
       - 응답 목록이 끔과 같다.
       - caplog에 `"window log"`가 든 경고가 정확히 한 번 있다.
  2. `test_17b_window_log_memory_only(workspace, monkeypatch, caplog)`
     - `RSW_STATE_DIR=""`, `RSW_WINDOW_LOG=1`로 기동하면 `"RSW_WINDOW_LOG"`가 든 경고가 한 번 나온다.
     - 요청은 200이고 `app.state.window_log.root is None`이다.
- [ ] **Step 2: 실패를 확인한다.** `ml/`에서 `py -3.11 -m pytest tests/test_pipeline.py -q -k "test_1_preprocess or test_2_preprocess or test_3_train or test_17"`. 앞 단계가 번들을 만들어야 하므로 `-k test_17`만으로는 돌지 않는다.
- [ ] **Step 3: 구현한다.**
  - trace 점에 `"early_warning_onset": ew_onset`, `"warmup": self.in_warmup(g, w.index[0])`를 더한다. `/trace` 응답은 칸을 골라 만들므로 바뀌지 않는다(`main.py:2038`).
  - `write`
    - 점을 날짜별로 나누고 `os.makedirs(..., exist_ok=True)` 뒤에 덧붙인다. 한 줄은 `json.dumps(line, ensure_ascii=False) + "\n"`.
    - 예외는 모두 잡는다.
    - 설비마다 처음 한 번만 경고한다: `log.warning("window log: could not write gun %s (%s); later errors for this gun are not logged", ...)`.
  - `add_model`: 있는 `models.json`을 읽고(읽지 못하면 빈 dict), 항목을 넣어 `write_json_atomic`으로 쓴다.
  - lifespan
    - `on = os.environ.get("RSW_WINDOW_LOG", "").strip().lower() in ("1", "true", "on", "yes")`
    - `on`인데 `state_dir`이 None이면 `log.warning("RSW_WINDOW_LOG is on, but RSW_STATE_DIR is empty (memory only): no window log is written")`
    - `app.state.window_log = WindowLog(state_dir if on else None)`. root가 있으면 `add_model`을 부른다.
  - `score_request`: `new_minutes`가 있으면 `request.app.state.window_log.write(req.gun_id, new_minutes, model_id, ew_threshold)`를 부른다.
  - 맨 위 설명에 더할 문장: "With RSW_WINDOW_LOG=1 (D-9) every newly scored window is also appended to <dir>/windows/<gun_id>/<YYYY-MM-DD>.jsonl (verdicts, codes, model input `vec`; windows/models.json maps the model id to its columns and thresholds) for the offline feedback loop (.py/feedback_loop.py). Off by default."
- [ ] **Step 4: 통과를 확인한다.** Step 2 명령 → 통과. 이어서 `py -3.11 -m pytest -q` → 132 passed.

### Task 3: 평가부 — 라벨과 지표 (스펙 6.2, 7.2, 7.3)

**Files:**
- Create: `ml/.py/feedback_eval.py`
- Create: `ml/tests/test_feedback_loop.py` (`sys.path`에 `.py/`를 넣는 방식은 test_pipeline과 같다)

**Interfaces:**
- Produces:
  - Global Constraints의 상수와 `CONSTANT_SOURCES: dict[str, str]`(상수 이름 → 스펙 7.8 "출처" 칸 그대로)
  - 창 표 열
    - 공통: `gun: str`, `window_start: datetime64[ns]`(naive UTC, 분 버킷), `score: float`, `warmup: bool`, `non_welding_share: float`, `error_active: bool`
    - 현장만: `early_warning: bool`, `ew_threshold: float`
  - 고장 표 열: `gun: str`, `T: datetime64[ns]`. 있으면 `status`, `event_id`, `rule_code`.
  - `@dataclass Frame`
    - `w: DataFrame`: 남은 창. `end`, `lead`(초, 다음 고장이 없으면 inf), `nxt`(고장 위치, 없으면 −1) 열이 붙는다.
    - `guns: list[str]`, `gun: ndarray[int]`
    - `brk`, `ok`, `normal`, `tail`: `ndarray[bool]`(창별)
    - `days: ndarray[float]`, `record_end: ndarray[datetime64]`(설비별)
    - `failures: DataFrame`(남은 고장, (설비 순서, T) 순), `fail_gun: ndarray[int]`
    - `zones: dict[str, list[ndarray[int]]]`: `"real"`, `"p1"`..`"p6"`마다 고장별 창 위치
    - `valid: dict[str, ndarray[bool]]`
    - `excluded_windows: int`, `dropped_failures: int`
  - `build_frame(windows: DataFrame, failures: DataFrame) -> Frame`
  - `@dataclass RuleResult`
    - `real`, `chance`: 고장별 float(판정 안 함은 nan)
    - `fa`, `pending`, `acted`, `days`: 설비별
    - `onset: ndarray[bool]`: 창별
  - `evaluate(fr: Frame, c: ndarray, acted: ndarray | None = None) -> RuleResult`
  - `@dataclass Evidence`: `real`, `chance`, `fail_gun`, `n_guns: int`, `fa_per_day: float`
  - `pool(results: list[RuleResult], fail_gun: ndarray, n_guns: int) -> Evidence`
  - `summary(ev: Evidence) -> dict`
    - 키: `fa_per_day, recall, chance, excess, ci, failures, guns`
    - 판정된 고장이 없으면 `recall`~`ci`는 None이다.
  - `paired(a: Evidence, b: Evidence) -> dict`: 키 `D, ci`
  - `normal_population(fr: Frame) -> ndarray[bool]`
  - `quantile_threshold(fr: Frame, q: float) -> float`: 모집단이 비면 nan
  - `condition(fr: Frame, kind: str, thr: float, scores: ndarray | None = None) -> ndarray[bool]`
    - `"model_ng"`: 점수 ≥ thr
    - `"model"`: 그리고 비용접 비율 ≤ `GATE_NON_WELDING`
    - 둘 다 `& ok`다. `scores`가 None이면 `w["score"]`를 쓴다(9.7의 끼울 자리).
  - `@dataclass(frozen=True) Challenger`: `name, kind, quantile, reason, apply_ko, lab: bool`
    - `quantile`이 None이면 지금 규칙의 임계값을 쓴다.
    - `CHALLENGERS`는 스펙 7.2 순서 `GATE, Q+, Q-, Q0.975`다. Q0.975만 `lab=False`다.

- [ ] **Step 1: 시험을 쓴다.** 도우미 `_wins(gun, start, minutes, **cols) -> DataFrame`은 1분 간격 창 표를 만든다.
  1. `test_frame_matches_b18` (핵심)
     - 설비 3대, 각 3일. 점수는 무작위(씨앗 고정), 첫 2시간은 워밍업, 한 설비에 30분 공백, 비용접 비율도 무작위다.
     - 고장은 설비마다 기록 끝 10분 전에 하나다.
     - 같은 표를 B-18 `Frame`(`results/B-18/early_warning.py`)에도 넣는다. 열 이름은 `file, time, lead, cls, fold, warmup, score_ens, non_welding`이다. B-18 `load`처럼 정렬하고 `lead > 0`인 행만 남긴다.
     - `model_ng`와 `model`, 임계값 q0.9에서 아래가 우리 결과와 정확히 같다(nan 위치 포함).
       - 설비별 `fa`, `days`
       - `real`, `p1`..`p6`
  2. `test_labels_hit_miss_undetermined`
     - 구역 안에 알람이 있으면 1.0, 없으면 0.0.
     - 공백 때문에 구역의 쓸 수 있는 창이 24개면 nan.
  3. `test_post_failure_exclusion_and_lead_to_next`: 한 설비에 고장 둘(2일 간격).
     - T1 뒤 60분의 창 60개가 빠진다(`excluded_windows == 60`).
     - 그 뒤 창의 lead는 T2 기준이다.
     - T1 + 30분의 알람은 어디에도 세지 않는다.
     - T1 + 90분(T2보다 60분 넘게 앞)의 시작은 오알람이다.
  4. `test_pending_and_tail`: 고장이 없는 설비.
     - 기록 끝 30분 전의 시작은 `pending`이고, 2시간 전의 시작은 오알람이다.
     - 마지막 60분의 창은 `days`에서 빠진다.
  5. `test_placebo_skips_overlap`: T2 = T1 + 1일이면 T2의 `p1`은 nan이다(T1의 앞 60분과 겹침). `p2`는 유효하다.
  6. `test_summary_and_paired_match_calibration`
     - 설비마다 고장 하나인 Evidence의 `summary`·`paired`가 `results/D-9/calibrate.py`의 `summary`·`paired`와 같다. 입력은 `[{"real", "chance", "fa", "days"}]`이고, calibrate.py는 importlib로 파일 경로에서 읽는다.
     - 고장 둘인 설비 하나뿐이면(`n_guns = 1`) CI = [excess, excess]다.
  7. `test_condition_and_population`
     - 비용접 0.6인 창은 `model`에서 거짓, `model_ng`에서 참이다.
     - 워밍업 창은 늘 거짓이다.
     - 모집단: lead > 86400, 워밍업 아님, 코드 없음. lead가 무한대면 기록 끝 하루 안의 창은 뺀다.
  8. `test_constants_match_calibration`: `results/D-9/calibrate.json`과 비교한다.
     - `round(noise.margin, 3) == MARGIN`, `round(noise.cheaper, 3) == CHEAPER`
     - `quantiles == {"q_minus": Q_MINUS, "q_plus": Q_PLUS}`
     - `rule_v2.min_failures == MIN_FAILURES`
     - `rule_v2.profile`의 `k`, `d` == `PROFILE_CASES`, `PROFILE_DAYS`
- [ ] **Step 2: 실패를 확인한다.** `ml/`에서 `py -3.11 -m pytest tests/test_feedback_loop.py -q` → ImportError.
- [ ] **Step 3: 구현한다.**
  - `build_frame` 규칙. 스펙 6.2를 풀어 쓴 것이고, 설비마다 고장이 기록 끝에 하나면 B-18 `load` + `Frame`과 같아야 한다.
    1. 창을 (gun, window_start)로 정렬한다. `end = window_start + 60 s`다. 기록의 `window_end`는 쓰지 않는다.
    2. 창마다 `nxt`는 같은 설비에서 `T > end`인 첫 고장이고, `lead = T[nxt] − end`다(없으면 inf). `T ≤ end`인 마지막 고장에 대해 `0 ≤ end − T < POST_FAILURE_S`이면 그 창을 지운다.
    3. `guns`는 남은 창의 설비 이름을 정렬한 것이다(B-18 `pd.factorize`와 같은 순서). 창이 없는 설비의 고장은 지운다.
    4. `brk`는 첫 창, 설비가 바뀐 창, 앞 창과 시작 차이가 60초가 아닌 창이다. `ok = ~warmup`.
    5. 기록 끝과 감시 일수
       - `record_end`(설비별) = 지우기 전 모든 창의 `end` 최댓값
       - `tail = isinf(lead) & (end > record_end − ZONE_S[1])`
       - `normal = (lead > ZONE_S[1]) & ok & ~tail`
       - `days = normal 수 / 1440`
    6. 실제 구역(고장 i): `nxt == i`, `ZONE_S[0] ≤ lead ≤ ZONE_S[1]`, `ok`인 창. 창이 `MIN_ZONE_WINDOWS` 이상이면 유효하다.
    7. 우연 구역(i, k): 같은 설비, `ok`, `ZONE_S[0] ≤ (T_i − k일) − end ≤ ZONE_S[1]`인 창. 아래 둘을 모두 만족하면 유효하다.
       - 창이 25개 이상
       - 닫힌 구간 [A − 3600, A − 600](A = T_i − k일)이 같은 설비 다른 고장 j의 [T_j − 3600, T_j]와 겹치지 않음
  - `evaluate`
    - 시작 = `c & (brk | ~앞 창 c)`(지속 1창)
    - 고장별 적중 = 구역 안에 `c`가 하나라도 있음(구역이 무효면 nan). 우연 = 유효한 k들의 평균(없으면 nan).
    - 설비별: `fa = 시작 & normal & ~acted`, `acted = 시작 & normal & acted`, `pending = 시작 & tail`
  - `pool`: 실행마다 고장별 `real`·`chance`를 `nanmean`으로 합친다(A-5 `stack`). `fa_per_day = Σfa / Σdays`.
  - `summary`·`paired`의 추출은 아래 순서 그대로 한다. 실험실(설비마다 고장 하나, 설비 순서)에서는 `f == gi`가 되어 A-5와 같은 수가 나온다.

```python
rng = np.random.default_rng(SEED_SINGLE)          # paired: SEED_PAIRED, 두 Evidence에 같은 f
fails_of = [np.flatnonzero(ev.fail_gun == g) for g in range(ev.n_guns)]
for _ in range(N_BOOT):
    gi = rng.integers(0, ev.n_guns, ev.n_guns)
    f = np.concatenate([fails_of[g] for g in gi])
    boot.append(np.nanmean(ev.real[f]) - np.nanmean(ev.chance[f]))
ci = [np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5)]
```

  - 빈 조각의 nanmean에서 나는 `RuntimeWarning`은 calibrate.py처럼 끈다.
- [ ] **Step 4: 통과를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q` → 8 passed.

### Task 4: 평가부 — 판정, 프로필, 사람 라벨, 평가 조립 (스펙 6.3, 6.4, 7.1, 7.4~7.7, 8.2의 7)

**Files:**
- Modify: `ml/.py/feedback_eval.py`
- Test: `ml/tests/test_feedback_loop.py`

**Interfaces:**
- Consumes: Task 3 전부
- Produces:
  - `source_verdict(ch: dict, d_champion: dict, d_reference: dict, fa_champion: float, fa_reference: float, field: bool) -> str`
    - 값: `"pass" | "insufficient" | "fail"`. `ch`는 `summary` 결과다.
    - `"na"`는 부르는 쪽이 정한다.
  - `combine(lab: str, field: str) -> tuple[str, str]`: (종합 코드, 한국어). 스펙 7.5 표의 11칸.
  - `VERDICT_KO: dict[str, str]`: 출처별 코드 → 한국어
  - `@dataclass CriticalCase`: `case_id: str, gun: str, cls: str | None, outcome: str, detected_at: pd.Timestamp`. `outcome`은 `"false_alarm" | "real" | "unknown"`.
  - `profile_proposals(cases: list[CriticalCase], profiles: dict[str, list[str] | None]) -> dict`: 키 `proposals`, `watching`, `review`
    - proposal: `{"gun", "exclude": [유형], "body": {"expected_fault_classes": [...]}, "cases": [{"case_id", "date"}], "risk_ko"}`
    - watching: `{"gun", "cls", "cases", "days", "text"}`
    - review: `{"gun", "exclude", "text": "검토 필요"}`
  - `reason_of(text: str | None) -> str`
  - `@dataclass EwLink`: `case_id: str, outcome: str, reason: str | None, note: str | None, gun: str, onset_start: pd.Timestamp | None`
  - `onset_index(fr: Frame, gun: str, start: pd.Timestamp) -> int | None`
  - `onset_label(fr: Frame, rr: RuleResult, i: int) -> str`: `"false_alarm" | "pending" | "hit_side" | "not_onset"`
  - `acted_mask(fr: Frame, champion_c: ndarray, links: list[EwLink]) -> ndarray[bool]`
  - `ew_human(fr: Frame, champion: RuleResult, links: list[EwLink]) -> dict`: 키 `false_alarm, pending, acted, reasons, notes, disagreements, unmatched`
  - `post_failure(windows: DataFrame, failures: DataFrame) -> list[dict]`: 0~120분을 10분씩. 키 `from_min, to_min, windows, ew_share`.
  - `champion_mismatch(windows: DataFrame) -> int`
  - `lab_evaluation(runs: Iterable[tuple[str, DataFrame, DataFrame]]) -> dict`: 키 `runs, champion, reference, challengers, thresholds`
  - `field_evaluation(windows: DataFrame, failures: DataFrame, model_entry: dict | None, links: list[EwLink]) -> dict | None`
    - 키: `failures, guns, mismatch, champion_threshold, reference_threshold, reference_basis, champion, reference, challengers, labels, post_failure, warnings`
    - 창이 없으면 None이다.
  - 도전자 결과의 모양은 실험실과 현장이 같다: `{"threshold", "summary", "d_champion", "d_reference", "verdict"}`. 평가할 수 없으면 `{"verdict": "na"}`.

- [ ] **Step 1: 시험을 쓴다.**
  1. `test_source_verdict_paths` (parametrize). 실험실은 `field=False`다. 현장은 판정된 고장 10건 · 설비 5대로 준다.
     - 더 잘 잡음
       - `d_champion.ci = [0.01, 0.2]`, 오알람 20 → pass
       - 오알람 25 → fail(상한 24)
     - 덜 울림
       - `d_reference.ci = [-0.10, 0.01]`, 오알람 13.0, 기준점 16.4, 지금 규칙 16.4 → pass
       - CI 상한 −0.001 → fail
       - CI 하한 −0.120 → fail(초과해야 통과)
       - 오알람 13.9(> 0.843 × 16.4) → fail
       - 기준점 20, 지금 규칙 12, 오알람 12 → fail(지금 규칙보다 적어야 함)
     - 증거 부족
       - 현장, 판정된 고장 9 → insufficient
       - 현장, 고장 10 · 설비 4 → insufficient
       - 실험실은 고장 수를 보지 않는다.
  2. `test_combine_table`: 7.5의 11칸 모두. 특히 `combine("na", "pass") == ("review", "검토 필요(실험실 확인 없음)")`.
  3. `test_profile_boundaries`. 기본은 프로필 None, 유형 E04다. 바뀌는 곳만 적는다.
     - g1: 오알람 1건 → watching `지켜보는 중(2건 중 1건, 2일 중 1일)`
     - g2: 2건, 같은 UTC 날 → watching(2건 중 2건, 2일 중 1일)
     - g3: 2건, 23:59Z와 다음 날 00:01Z → proposal. body는 `["E01","E02","E03"]`이고 위험 문장이 붙는다.
     - g4: g3에 실제 E04 1건을 더함 → 아무것도 없음
     - g5: 프로필 `["E04"]`에 2건 · 2일 → review, 제안 없음
     - g6: 프로필 `["E01","E02"]`에 E04 2건 · 2일 → 아무것도 없음
     - g7: E03과 E04가 모두 2건 · 2일 → 제안 하나, body `["E01","E02"]`
     - `cls`가 None인 케이스와 `unknown` 결과는 세지 않는다.
  4. `test_reason_of`
     - 칩 넷은 그대로다.
     - `"아무 글"` → `"기타"`
     - None, `""`, 공백 → `"사유 없음"`
  5. `test_ew_human_and_acted`
     - 설비 하나, 기록 끝에 고장 하나.
     - 지금 규칙의 시작 셋: A(멀리, 오알람), B(고장 30분 전, 적중 쪽), C(멀리, 오알람).
     - 링크
       - A: 오알람, 사유 "설비 정지 · 전원 꺼짐"
       - B: 오알람
       - C: 해결 종료
       - 기록에 없는 시작 하나
     - `acted_mask`로 평가하면 오알람 1(A), acted 1(C)이다.
     - `reasons == {"설비 정지 · 전원 꺼짐": 1}`, disagreements는 B의 케이스 하나, `unmatched == 1`.
  6. `test_post_failure_bins`: 고장 뒤 120분 중 앞 30분만 조기 경보 → 앞 세 칸 `ew_share`는 1.0, 나머지는 0.0, 칸마다 창 10개.
  7. `test_field_evaluation`
     - 준비: 설비 6대 × 3일, 설비마다 고장 2건(모두 12건). `early_warning = (점수 ≥ q0.95 임계값) & ~warmup`이고 `ew_threshold` 열은 그 임계값이다.
     - `model_entry.early_warning.quantile == 0.975`일 때
       - `reference_basis == "ew_threshold"`, `mismatch == 0`
       - reference summary == champion summary
       - 도전자 넷의 verdict ∈ {pass, fail}
     - quantile이 0.98이면 `reference_basis == "field_q0.975"`.
     - 한 창의 `early_warning`을 뒤집으면 `mismatch == 1`.
     - 창 표가 비면 None.
  8. `test_lab_evaluation_plumbing`: 같은 설비들의 합성 실행 둘.
     - `runs`에 두 이름이 있다.
     - `challengers["Q0.975"] == {"verdict": "na"}`
     - champion `fa_per_day` = 두 실행의 Σfa / Σdays
     - 실행마다 임계값 셋(`champion`, `Q+`, `Q-`)
- [ ] **Step 2: 실패를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q -k "verdict or combine or profile or reason or human or post_failure or field_evaluation or lab_evaluation"`
- [ ] **Step 3: 구현한다.**
  - `source_verdict`
    - `field`이고 (`ch["failures"] < MIN_FAILURES` 또는 `ch["guns"] < MIN_GUNS`)이면 insufficient
    - 더 잘 잡음: `d_champion["ci"][0] > 0`이고 오알람 ≤ `FA_CAP`
    - 덜 울림: 아래 넷을 모두 만족
      - `d_reference["ci"][0] > -MARGIN`
      - `d_reference["ci"][1] >= 0`
      - 오알람 ≤ `CHEAPER * fa_reference`
      - 오알람 < `fa_champion`
  - 실험실
    - 실행마다 `build_frame` 뒤 `thr_c = quantile_threshold(fr, Q_REFERENCE)`
    - 지금 규칙 = `model_ng`@thr_c, GATE = `model`@thr_c, Q+ · Q− = 각 분위수의 `model_ng`
    - 실행 사이 `guns`와 고장 순서가 같은지 assert한다(calibrate.py와 같음).
    - 기준점이 지금 규칙이므로 `d_reference = d_champion`이다.
  - 현장
    - `champion_c = w["early_warning"]`
    - 지금 규칙 임계값 = `model_entry["early_warning"]["threshold"]`. 없으면 첫 non-NaN `ew_threshold`. 둘 다 없으면 None을 돌려주고 경고한다.
    - 기준점 임계값(7.1): 번들 quantile이 0.975면 지금 규칙 임계값, 아니면 `quantile_threshold(fr, 0.975)`
    - GATE는 지금 규칙 임계값을 쓴다. Q+ · Q− · Q0.975는 현장 분위수를 쓴다. 분위수가 nan이면 그 도전자는 `na`이고 경고한다.
    - 모든 규칙을 같은 `acted_mask`로 평가한다(계획 결정 6).
    - `labels`에는 지금 규칙의 고장별 적중 · 미탐 · 판정 안 함 수와 `ew_human`을 넣는다.
    - `warnings`에는 `dropped_failures`, `unmatched`, `mismatch`, 빈 모집단을 Global Constraints 문장으로 넣는다.
  - `onset_label`: 시작이 아니면 `not_onset`, `tail`이면 `pending`, `normal`이면 `false_alarm`, 나머지(lead ≤ 3600)는 `hit_side`.
  - `ew_human`
    - 오알람 링크의 시작이 `false_alarm`이면 사유를 센다(`reason_of`). 기타와 사유 없음의 메모는 `notes`에 넣는다.
    - 오알람 링크인데 `hit_side`면 disagreements에 넣는다: `{"case_id", "gun", "onset_at", "auto"}`.
    - 해결 종료 링크는 acted로 센다. 시작 창을 못 찾으면 `unmatched`다.
  - `profile_proposals`: 스펙 7.6. 날짜는 `detected_at`의 UTC 날짜다. 순서는 설비 이름, 그다음 `CLASSES`.
  - `champion_mismatch`: `ew_threshold`가 있는 창 중 `early_warning != (score >= ew_threshold) & ~warmup`인 창의 수.
- [ ] **Step 4: 통과를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q` → 16 passed.

### Task 5: 루프 — 입력 읽기와 실험실 대조 (스펙 4, 6.1, 6.5, 6.6, 성공 기준 3)

**Files:**
- Create: `ml/.py/feedback_loop.py` (읽기 부분)
- Test: `ml/tests/test_feedback_loop.py`

**Interfaces:**
- Consumes: `feedback_eval`(Task 3 · 4), `preprocess.TERMINAL_TO_CLASS`
- Produces:
  - `class InputError(Exception)`: 준 입력이 없거나 열리지 않음(끝 상태 2)
  - `utc_naive(s: str | None) -> pd.Timestamp | None`
  - `read_windows(state: str) -> tuple[DataFrame, dict, list[str]]`: (창 표 + `model` 열, `models.json`, 경고)
  - `read_outbox(state: str) -> tuple[list[dict], list[str]]`: (handoff 문서, 경고)
  - `read_profiles(state: str) -> dict[str, list[str] | None]`: `guns/*.json`의 `gun_id` → `profile`
  - `read_cases(cases: str | None = None, history: str | None = None, api: str | None = None) -> tuple[list[dict], dict[str, str] | None]`
    - 케이스 하나: `{"case_id", "event": {"event_id", "gun_id", "detected_at", "trigger", "event_type"}, "closure": dict | None, "repeat_event_ids": list[str] | None, "simulated": bool}`
    - 둘째 값: DB면 `events` 표(event_id → case_id), 아니면 None
  - `case_index(cases: list[dict], events: dict | None, outbox: list[dict]) -> dict[str, dict]`: event_id → 케이스
  - `failures_from(outbox: list[dict], index: dict) -> DataFrame`: 열 `gun, T, status, event_id, rule_code`. `status`는 `"confirmed" | "presumed" | "rejected"`.
  - `critical_cases(cases: list[dict]) -> list[CriticalCase]`
  - `ew_links(cases: list[dict], index: dict, outbox: list[dict]) -> list[EwLink]`
  - `iter_lab(lab_dir: str, meta_csv: str) -> Iterator[tuple[str, DataFrame, DataFrame]]`

- [ ] **Step 1: 시험을 쓴다.** 도우미:
  - `_line(start, **kw) -> dict`: 5.4의 키를 가진 줄. `vec`은 빈 목록.
  - `_state(tmp_path, lines_by_gun, outbox=(), profiles=None)`: 상태 폴더를 만든다.

  시험:
  1. `test_read_windows_dedup_floor`: 같은 분에 `"…10:05:00Z"`(점수 0.1)와 `"…10:05:03Z"`(점수 0.2) → 한 줄, `window_start` 10:05:00, 점수 0.2.
  2. `test_read_windows_skips_broken_lines`: 끝 줄이 잘린 파일 → 그 줄만 빠지고 경고 `읽지 못한 줄 1개를 건너뜀 (창 기록)`.
  3. `test_read_windows_keeps_newest_model`: 설비 A에 m1 줄(앞 날짜) 5개와 m2 줄(뒤 날짜) 3개 → m2의 3줄만 남고, 경고에 `m1 5개`가 있다.
  4. `test_read_windows_gun_dir_unquoted`: 폴더 `line%2F1` → 설비 `line/1`. `error_codes`가 비면 `error_active`는 거짓이다.
  5. `test_read_cases_inputs_and_errors`
     - `api/store.py`의 두 `CREATE TABLE` 문으로 DB를 만든다(케이스 2, events 3) → 케이스 2개와 events dict.
     - `{"items": [...]}` JSON → 그 항목들.
     - 다음은 모두 `InputError`: 없는 파일, DB가 아닌 파일, `items`가 없는 JSON, 닫힌 포트 `http://127.0.0.1:9`.
  6. `test_case_index_paths`
     - events 표가 있으면 그것만 쓴다.
     - `repeat_event_ids`가 있으면 대표와 그 목록을 잇는다.
     - 키가 없으면(11.1 이전) 아래를 모두 만족하는 알림을 잇는다: 같은 설비, 같은 `rule_code`(None끼리 같음), `detected_at` ∈ [케이스 감지, `closure.at`].
     - 범위 밖의 알림과 다른 케이스의 대표는 잇지 않는다.
  7. `test_failures_from_status`
     - v1.2(`event_type` 없음) E029, 케이스가 오알람 → rejected
     - E012, 케이스가 해결 종료 → confirmed
     - E016, 케이스 없음 → presumed
     - 조기 경보와 모델 단독 critical은 후보가 아니다.
     - `T`는 `rule_trigger_time`이고, 없으면 `detected_at`이다.
  8. `test_ew_links_and_critical_cases`
     - 반복이 하나 붙은 조기 경보 케이스 → 링크 둘. `onset_start`는 `onset_at`을 분으로 내린 값이다.
     - outbox에 없는 이벤트 → `onset_start` None
     - critical E029 → `cls` E04, 모델 단독 → None
     - 결과 매핑: false_alarm → false_alarm, resolved → real, 그 밖 → unknown
  9. `test_lab_reproduces_b20`. `results/B-19/cache`나 `results/B-17/cache/meta.csv`가 없으면 skip한다. `fe.lab_evaluation(fl.iter_lab(...))`의 결과를 비교한다(비율은 셋째 자리, 오알람은 둘째 자리에서 반올림).
     - 지금 규칙: `fa_per_day` 16.40, recall 0.488, chance 0.343, excess 0.146, ci (0.059, 0.221)
     - GATE: 15.10, excess 0.080, ci (−0.009, 0.162), `d_champion` −0.066 (−0.128, −0.013)
     - Q+: 13.59, `d_champion` −0.022 (−0.050, 0.002)
     - verdict: GATE fail, Q- fail, Q+ pass, Q0.975 na
- [ ] **Step 2: 실패를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q -k "read_ or case_index or failures_from or ew_links or lab_reproduces"`
- [ ] **Step 3: 구현한다.**
  - `read_windows`
    - `windows/<폴더>/*.jsonl`을 이름순으로, 줄 순서대로 읽는다. 설비는 `unquote(폴더)`다. `vec`은 버린다(1단계에서는 쓰지 않음).
    - `window_start = utc_naive(...).floor("60s")`, `error_active = bool(error_codes)`
    - 중복을 정리한 뒤 모델이 여럿이면 `window_start`가 가장 늦은 줄의 모델만 남긴다. 동률이면 id가 큰 모델이다.
  - `read_cases`
    - DB: `sqlite3.connect(Path(p).resolve().as_uri() + "?mode=ro", uri=True)`로 열고 `SELECT data FROM cases`, `SELECT event_id, case_id FROM events`를 읽는다.
    - DB 케이스의 `closure`는 `records.closure`이고, `event`는 저장된 핸드오프 전체다.
    - API: `httpx.get(url.rstrip("/") + "/api/history", timeout=10)`. 200이 아니거나 연결이 안 되면 `InputError`다.
  - `case_index`의 시간 범위 잇기: 맞는 케이스가 여럿이면, 감지 시각이 알림보다 늦지 않은 것 중 가장 늦은 케이스로 잇는다.
  - `iter_lab`
    - `scores_cv_*.parquet`을 이름순으로 읽는다. 열은 `file, time, error_active, non_welding, warmup, score_ens`만 읽는다.
    - 바꾸기: `gun=file`, `window_start=time`, `score=score_ens`, `non_welding_share=non_welding`, `error_active = error_active > 0`, `warmup = warmup.astype(bool)`
    - 고장은 `meta.csv`(`parse_dates=["fault_time"]`)에서 그 실행의 설비만 골라 `gun=file`, `T=fault_time`으로 만든다.
    - 메모리를 아끼려고 실행을 하나씩 내보낸다.
- [ ] **Step 4: 통과를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q` → 25 passed. 실험실 대조는 1분 안팎 걸린다.

### Task 6: 루프 — 보고서와 명령줄 (스펙 4, 7.7, 8)

**Files:**
- Modify: `ml/.py/feedback_loop.py`
- Test: `ml/tests/test_feedback_loop.py`

**Interfaces:**
- Consumes: Task 4 · 5
- Produces:
  - `build_report(args: argparse.Namespace, now: datetime) -> dict`
  - `render_md(report: dict) -> str`
  - `main(argv: list[str] | None = None) -> int`: 보고서를 쓰면 0, `InputError`면 2(stderr에 한 줄)
  - 명령줄: 스펙 4절 그대로.
    - `--cases` · `--history` · `--api`는 argparse 상호 배타 묶음이다.
    - 기본값: `--lab` 없음, `--lab-meta results/B-17/cache/meta.csv`, `--out results/D-9/runs`, `--tag` = UTC `%Y%m%d-%H%M`
  - 출력: `<out>/<tag>/report.json`과 `report.md`
  - json(8.3)
    - `schema: "feedback_report/1"`
    - `meta`: `generated_at`(UTC, Z), `tag`, `model`, `guns: {설비: {"first", "last", "windows"}}`, `simulated`, `note_ko`
    - `criteria`: `{상수 이름: {"value", "source"}}`(7.8 순서)
    - `inputs`: 준 문자열 그대로(`state, cases, history, api, lab, lab_meta`)와 `counts: {"windows", "outbox", "cases", "lab_runs"}`
    - `labels`
      - `failures: {"confirmed", "presumed", "rejected", "hit", "miss", "undetermined"}`
      - `ew_onsets: {"false_alarm", "pending", "acted"}`
      - `critical_cases: {"false_alarm", "real", "unknown"}`
      - `reasons`, `disagreements`
    - `early_warning`
      - `champion: {"lab", "field", "mismatch_windows"}`
      - `reference: {"definition": REFERENCE, "field_threshold", "field_basis", "lab", "field"}`
      - `challengers: [{"name", "reason", "apply_ko", "lab", "field", "lab_verdict", "field_verdict", "verdict", "verdict_ko"}]`
      - `confirmed_only: {"champion", "challengers": {이름: summary}}`
    - `critical`: `profile_proposals` 결과
    - `reasons`: `{사유: {"count", "linked": {"challenger", "verdict_ko"} | None, "guns", "notes"}}`(7.7 표)
    - `post_failure`, `warnings`
  - md
    - 스펙 8.2 순서다. 모든 숫자는 json에서 온다.
    - 오알람/건·일은 소수 둘째 자리, 비율과 D는 셋째 자리, CI는 `[+0.059, +0.221]` 꼴이다.
    - 판정마다 적용 방법을 한 줄 붙인다.
  - json은 NaN · inf를 None으로 바꾼 뒤 `allow_nan=False`, `ensure_ascii=False`, `indent=1`로 쓴다. 목록 순서는 정렬로 고정한다(8.5).

- [ ] **Step 1: 시험을 쓴다.** 도우미 `_synthetic_state(tmp_path) -> (state, history_json)`
  - 창 기록: Task 4 `test_field_evaluation`의 표를 5.4 줄로 쓴다.
  - outbox: 고장마다 critical E029/E012 핸드오프, 시작 몇 개에 조기 경보 핸드오프
  - history
    - 오알람 닫기(사유 두 가지)
    - 해결 종료된 조기 경보 하나
    - 한 설비에 2일에 걸친 critical 오알람 2건
    - 한 항목에 `"simulated": true`

  시험:
  1. `test_report_end_to_end`
     - `main([...])`가 0을 돌려준다.
     - json 최상위 키 = `schema` + 8.3의 열 개
     - md에서 제목 다음 첫 줄이 시뮬레이션 경고이고, 절 제목 여덟 개가 순서대로 있다.
     - 도전자 넷의 `verdict`가 종합 코드 안에 있다.
     - 설계한 설비에 프로필 제안이 있다. `reasons`에 넣은 칩이 있다. `post_failure`가 12칸이다.
  2. `test_report_reproducible`: 같은 인자(같은 `--tag`, 다른 `--out`)로 두 번 돌리면 json이 `meta.generated_at` 빼고 같다.
  3. `test_exit_codes`
     - 없는 `--state`, 없는 `--cases`, 깨진 `--history` JSON, 없는 `--lab` → 2
     - `--cases`와 `--history`를 같이 주면 `SystemExit(2)`
     - 빈 상태 폴더만 주면 0이다. 조기 경보 도전자는 모두 `not_evaluated`이고, 경고에 `창 기록 없음: …`이 있다.
  4. `test_report_warns_long_log`: 한 설비의 창 기록이 31일에 걸치면(짧은 두 묶음) 30일 경고에 그 설비 이름이 든다. 다른 설비 이름은 들지 않는다.
  5. `test_report_run2_shape`: 실행 ②를 합성으로 따라 한다.
     - 준비: v1.2 critical 6건(E029 5건이 5일에 걸침, E012 1건), `guns/test_0.json`의 프로필 None, DB 케이스 6개(오알람 5, 해결 종료 1)와 events
     - 프로필 제안이 `{"expected_fault_classes": ["E01", "E02", "E03"]}` 하나다.
     - 조기 경보 도전자는 모두 `not_evaluated`다.
- [ ] **Step 2: 실패를 확인한다.** `py -3.11 -m pytest tests/test_feedback_loop.py -q -k "report_ or exit_codes"`
- [ ] **Step 3: 구현한다.**
  - 순서: 입력 읽기 → 실험실(있으면) → 현장(창 기록이 있으면).
    - 판정에 쓰는 고장은 확인 + 추정이다.
    - 확인만으로 한 번 더 평가해 `confirmed_only`에 넣는다.
  - 출처별 판정: 실험실이 없거나 도전자가 `lab=False`면 실험실은 `na`, 현장 창 기록이 없으면 현장은 `na`다.
  - `combine` 뒤 7.7 표로 `reasons`를 채운다.
    - 설비 정지 · 전원 꺼짐 → GATE, 정상 작업 중 변동 → Q+(`linked`)
    - 센서 · 신호 문제 → 그 사유가 달린 설비 목록
    - 기타, 사유 없음 → 메모 목록
  - 경고는 Global Constraints 문장으로 쓴다.
    - 30일 경고: 설비별로 `last − first > OUTBOX_RETENTION_DAYS`일 때
    - 케이스 하나라도 `simulated`이면 `meta.simulated = True`
  - md 절의 내용
    - 1 머리말: 만든 시각, 설비별 기록 기간, 모델 id, 입력 경로, 7.8 상수와 출처, 머리말 문장
    - 2 한눈에 보기: 종합 판정별 도전자 목록과 프로필 제안 수
    - 5 critical 프로필: 제안마다 근거 케이스(번호, 날짜), 위험 문장, `PUT /guns/<id>/profile`과 본문 JSON. 이어서 지켜보는 중 목록과 검토 필요 목록
    - 8 한계에 적을 것:
      - 판정된 고장 수, 조치함으로 뺀 수
      - 임계값을 정한 데이터로 평가도 했다는 것
      - 낮은 현장 판정력(3.2: 64건에 GATE만 한 차이를 39%)
      - 6.7의 한계 넷
      - 조치함을 모든 규칙에 같게 뺐다는 것(계획 결정 6)
- [ ] **Step 4: 통과를 확인한다.**
  - `py -3.11 -m pytest tests/test_feedback_loop.py -q` → 30 passed
  - `py -3.11 -m ruff check .` → 기존 E402 하나

### Task 7: 오프라인 실행 (스펙 10.2, 성공 기준 3 · 4 · 5)

**Files:**
- Create: `ml/results/D-9/offline_check.py`
- Modify: `ml/.gitignore` (46행 `results/D-8/runs/` 다음 줄에 `results/D-9/runs/`)

**Interfaces:**
- Consumes: `replay.replay_file`, `main.app`, `feedback_eval`, `feedback_loop`
- Produces: `ml/`에서 `py -3.11 results/D-9/offline_check.py <명령>`. 상태 폴더는 `results/D-9/runs/state-test8`의 절대 경로다.
  - `replay <csv>... [--start-hours H] [--hours H] [--no-reset]`
    - 환경: `RSW_STATE_DIR`, `RSW_WINDOW_LOG=1`, `RSW_EARLY_WARNING_HANDOFFS=1`. `RSW_RAG_URL`은 지운다.
    - TestClient 하나로 파일마다 `replay_file(client, path, chunk_s=60, ...)`을 부른다. 요약과 걸린 초를 찍는다.
  - `simulate`: 스펙 10.2 실행 ①의 2를 `runs/history_sim.json`(`{"items": [...]}`)으로 쓴다.
    - 파일 끝은 `meta.csv`의 `end`다.
    - 케이스 번호는 감지 순서로 `SIM-0001`…이다. `closure.at`은 감지 + 10분(Z), `repeat_event_ids`는 `[]`, 모든 항목에 `"simulated": true`.
    - 조기 경보 라벨은 `onset_label`로 정한다. 시작 창은 `onset_index(fr, 설비, floor(onset_at))`로 찾는다. 이때 고장은 해결 종료로 닫은 critical이다.
  - `loop1`: `--state <상태> --history runs/history_sim.json --lab results/B-19/cache --tag offline-1`
  - `loop2 [--tag T]`: `--state state/d8-demo1029 --cases ../api/state/d8-demo1029.db --tag offline-2`
  - `scores`: 설비마다 창 기록 점수를 `models/scores/<gun>_ensemble.csv`의 `score`와 맞춘다.
    - 맞추는 키: 내린 창 시작 ↔ `time`
    - 창 수, 최대 |차이|, 1e-4 안의 창 수를 `runs/scores.json`에 쓰고 찍는다.

- [ ] **Step 1:** `.gitignore`에 줄을 더하고 `git check-ignore -v results/D-9/runs/x`로 확인한다.
- [ ] **Step 2:** `offline_check.py`를 쓴다. 맨 위 설명에 명령마다 입력과 출력을 적고, "시뮬레이션 라벨 — 성능 아님"을 적는다.
- [ ] **Step 3: 재생한다.**
  - `results/D-9/runs/state-test8`이 없는지 먼저 본다. 있으면 멈추고 사용자에게 묻는다.
  - `test/test_0.csv`부터 `test_7.csv`까지 명령 하나에 파일 하나씩 돌린다.
  - 첫 파일의 시간을 보고, 9분을 넘으면 `--hours`로 나눈다. 이어지는 조각에는 `--no-reset`을 준다.
  - 끝나면 `windows/` 아래에 설비 8개가 있고 `outbox/`가 있다.
- [ ] **Step 4: 점수를 대조한다.** `scores` → 한 파일 이상에서 모든 창이 1e-4 안이다(성공 기준 5).
  - 아니면 코드를 고치기 전에 CSV의 `threshold` 열이 번들 `threshold`(0.9457997…)와 같은지부터 본다. 같은 번들의 산출물인지 확인하는 것이다.
- [ ] **Step 5: 실행 ①.** `simulate` 뒤 `loop1`을 돌리고 `runs/offline-1/report.md`를 확인한다.
  - 맨 위에 `> 시뮬레이션 라벨 포함 — 성능 아님`이 있다.
  - 실험실 수치가 Task 5 시험 9와 같다.
  - 현장 판정은 모두 증거 부족이다(판정된 고장 ≤ 8).
  - 종합: GATE · Q- `반영 안 함`, Q+ `보류(현장 증거 쌓는 중)`, Q0.975 `보류`.
  - 참고(다르면 실패로 보지 않고 이유를 적는다): critical 프로필 제안은 test_0 · test_4의 E04다. `calibrate.json` rule_v2 test_check와 같은 설비다.
- [ ] **Step 6: 실행 ②.** `loop2`를 돌리고 `runs/offline-2/report.json`을 확인한다.
  - test_0 제안 body는 `{"expected_fault_classes": ["E01", "E02", "E03"]}`다.
  - 근거 케이스는 CASE-0001~0005(09-04 · 06 · 07 · 08 · 10)다.
  - 조기 경보 도전자는 모두 `평가 안 함`이다.
- [ ] **Step 7: 재현성.** `loop2 --tag offline-2b`를 돌리고, 두 json이 `meta.generated_at`·`meta.tag` 빼고 같은지 파이썬 한 줄로 비교한다.
- [ ] **Step 8:** `git status`에 `results/D-9/runs/`가 보이지 않는지 본다. 남길 숫자는 Task 8의 history에 쓴다.
  - 실행 ①의 판정과 실험실 수치
  - 실행 ②의 제안
  - 점수 대조의 최대 차이
  - 재생 시간

### Task 8: 문서와 마무리 확인 (스펙 10.3, 10.4, 11.2)

**Files:**
- Create: `ml/docs/feedback-loop-requests.md`
- Modify:
  - `ml/history.md`: 끝에 `### 49. D-9 ML 피드백 루프 — 현장 라벨로 운영 규칙을 판정하는 보고서 (2026-11-08)`
  - `ml/MEMORY.md`: 5절 서빙 계약에 창 기록 한 줄, 변경 이력 끝에 한 줄
  - `ml/README.md`: 파일 구성 269 · 270 · 275행, 5.10의 환경 변수 문단(758행 근처), 시험 목록 731행, 사용법 한 단락
  - `ml/techspec.md`: 35행 D-8 줄 다음에 D-9 줄
  - `ml/CLAUDE.md`: Layout 줄, Run Tests의 시험 수와 목록
  - `ml/docs/hitl-loop.md`: 요약에 한 줄

- [ ] **Step 1: 요청 문서.** `ml/docs/feedback-loop-requests.md`
  - 맨 위: "1단계를 막는 요청은 없음", 우리가 한 api 변경(11.1) 요약 세 줄
  - 스펙 11.2의 FE 요청 셋을 항목마다 무엇 · 왜 · 어디(file:line)로 쓴다.
- [ ] **Step 2: history 49절.**
  - 사용자 결정: 스펙 2절의 11-08 줄들, 사용자 말 포함
  - 점검(3.1)과 보정(3.2)의 수치
  - 구현: 파일, 계획 결정 10개
  - 시험 수
  - Task 7 결과. "시뮬레이션 라벨 — 성능 아님"을 같이 적는다.
  - 남은 일: FE 요청, 2단계, 실서버 확인
  - 스펙과 계획 경로
- [ ] **Step 3: 나머지 문서.**
  - MEMORY 5절: `RSW_WINDOW_LOG=1`이고 상태 폴더가 있으면 `windows/<gun>/<날짜>.jsonl`(5.4 필드, `vec` 포함)을 쓴다. 기본 끔이고 응답은 같다.
  - MEMORY 변경 이력: 11-07 줄과 같은 꼴로 한 줄
  - README
    - 위 세 곳을 고친다.
    - 사용법 예: `python .py/feedback_loop.py --state state --cases ../api/state/api.db --lab results/B-19/cache`와 출력 위치
  - techspec D-9 줄: 상태는 "1단계 구현 2026-11-08(보고서만), 2단계 설계만", `history.md` 49절
  - CLAUDE.md
    - `.py/` 목록에 `feedback_eval.py` · `feedback_loop.py`(D-9 피드백 루프 보고서)
    - `tests/` 목록에 `test_feedback_loop.py`
    - 시험 수
  - hitl-loop.md 요약: "ML 쪽 고리(현장 라벨로 ML 운영 규칙을 판정하는 보고서)는 D-9: `docs/superpowers/specs/2026-11-08-ml-feedback-loop-design.md`"
- [ ] **Step 4: 시험 전체.**
  - ML: `ml/`에서 `py -3.11 -m pytest -q` → 162 passed(130 + Task 2의 2 + `test_feedback_loop.py`의 30)
  - API: `.venv\Scripts\python.exe -m pytest api/tests -q` → 88 passed, 1 skipped
- [ ] **Step 5: ruff.**
  - ML: `ml/`에서 `py -3.11 -m ruff check .` → 기존 E402 하나
  - api: `py -3.11 -m ruff check api/main.py api/store.py api/tests/test_feedback_api.py --line-length 120`
    - 결과를 HEAD 판과 비교한다. HEAD 판은 `git show HEAD:api/main.py` 등으로 스크래치 폴더에 꺼낸다.
    - 새 오류가 없어야 한다.
- [ ] **Step 6: 작업 트리.**
  - `git status`와 `git diff --stat`: 계획의 파일만 바뀌었다. `ui/` · `rag/` 변경이 없고 `results/D-9/runs/`는 보이지 않는다.
  - 고친 파일의 줄 끝이 그대로다(파일 전체가 바뀐 diff가 없다).
- [ ] **Step 7: 보고한다.**
  - 스테이징 목록(파일별 한 줄)과 시험 수
  - "reset이나 checkout을 하면 커밋 안 된 변경이 사라진다"는 경고
  - 남은 일: FE 요청 전달, 2단계, 실서버 확인
