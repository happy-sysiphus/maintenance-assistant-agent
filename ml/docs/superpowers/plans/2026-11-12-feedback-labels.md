# D-12 사람 판정 반영 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 작업자가 닫은 조기 경보 케이스의 판정(해결 · 오알람)을 api에서 가져와, ML 루프의 다음 재학습과 A/B 판정에 넣고 기록으로 남긴다.

**Architecture:** api의 새 읽기 주소가 닫힌 케이스의 사실만 내준다. ML의 새 모듈(`model_feedback.py`)이 이를 등록부 폴더의 `feedback.json`에 쌓고, 케이스마다 사람 판정과 판정 구간을 낸다. `tick`은 그 구간을 `train.py --feedback`으로 넘기고, `train.py`는 구간 안의 창을 "정상"에서 뺀다. 판정이 없으면 모든 결과가 지금과 같다.

**Tech Stack:** Python 3.11, pandas, scikit-learn, LightGBM, FastAPI, httpx, pytest, ruff

**상태(2026-11-13):** 작업 1 ~ 6 완료. 시험 수는 계획과 조금 다르다: ml은 겹치지 않게 세어 224개(계획의 227은 겹쳐 센 값), api는 50개. 최종 검토를 반영한 뒤에는 ml 227개(시험 3개 추가), api 51개다. 미뤘던 사소한 지적까지 고친 뒤에는 ml 237개, api 56개다(`api/store.py`에 읽기 함수 하나 추가). 계획과 달라진 점은 `ml/history.md` 51절과 실행 기록에 있다.

**Spec:** `ml/docs/superpowers/specs/2026-11-12-feedback-labels-design.md`. 사람 판정 규칙은 4절, `train.py`가 바뀌는 네 곳은 5절, 루프와 기록은 6절, 실패 처리는 8절, 테스트 목록은 9절이다. 이 계획은 스펙을 되풀이하지 않고 파일 · 이름 · 시그니처 · 테스트만 정한다. 실행자는 둘 다 읽는다. 바탕이 되는 D-11 스펙은 `2026-11-11-model-loop-design.md`다.

## Global Constraints

- 커밋, 푸시, 머지, PR 금지. 사용자가 직접 커밋한다. 변경은 `main` 작업 트리에 커밋 없이 두고, 작업이 끝날 때마다 스테이징할 파일 목록을 보고한다. 다른 브랜치로 옮기거나 worktree를 만들지 않는다.
- 작업 트리에 커밋 안 된 변경이 이미 있다(11-12 교체 기준 완화와 잠정 교체: `ml/` 15개 항목, 이 스펙과 계획). 되돌리거나 덮어쓰지 않는다. `git stash`, `git checkout --`, `git reset`을 쓰지 않는다.
- `rag/`, `ui/` 코드를 바꾸지 않는다. `api/`에서는 새 파일 `api/ml_feedback.py`, 새 테스트, 새 문서, 그리고 `api/main.py`의 두 줄(가져오기와 등록)만 더한다. api의 기존 응답은 하나도 바꾸지 않는다.
- 유료 API 호출 금지(Claude, OpenAI). `.env`를 읽거나 출력하지 않는다. 팀이 함께 쓰는 Aura에 쓰지 않는다: 실제 확인에서 `/warning/resolve`, `/warning/candidates`, `/warning/manual`을 부르지 않는다.
- 실제 데이터 폴더 `train/`, `test/`, `preprocessed/`를 바꾸지 않는다. 테스트는 임시 폴더의 합성 데이터만 쓴다. 기존 등록부 `models/registry/`, `models/registry-replay-1/`도 바꾸지 않는다. 실제 확인은 `models/registry-d12/`를 쓴다(git이 무시한다).
- 8 GB PC다. 오래 도는 명령은 포그라운드에서 10분 이내로 끊어 돌린다. 백그라운드로 돌리지 않는다. 남은 메모리가 1 GB 아래면 무거운 단계를 시작하지 않고 보고한다. 자기가 띄운 프로세스만 끝낸다.
- 작업 트리의 파일은 CRLF다(`core.autocrlf=true`). 고치는 파일의 줄 끝을 유지한다(`model_loop.py`, `model_registry.py`, `test_model_loop.py`, 스펙과 계획은 LF다). 새 파일은 LF로 써도 된다.
- 계약은 추가만 한다. `/predict`, `/model`, `AnomalyResult`, `RagHandoff`, `/model/loop`, 등록부의 기존 필드를 바꾸지 않는다. `RSW_MODEL_REGISTRY`가 없는 서버, `feedback.json`이 없는 등록부, `--feedback`이 없는 `train.py`는 지금과 똑같이 동작한다.
- 스펙의 값: 판정 구간 `3600`초, 이상 확인 = `outcome == "resolved"`이고 `checks.abnormal >= 1`, 케이스당 경보 하나(고른 경보, 없으면 대표 경보), 적중 구간 = 고장 600~3600초 전(`train.EW_ZONE_S`), api 응답의 `schema_version` = `"1.0"`.
- 줄 길이 120. `ruff check .`(ml)의 기존 오류는 6개(`results/B-22/summarize.py` 5개, `tests/test_sensor_features.py:14` 1개)이고 새 오류가 없어야 한다.
- ml 명령은 `ml/`에서 `py -3.11`로, api 명령은 `api/`에서 `../.venv/Scripts/python.exe`로 실행한다.
  - `UNIT`: `py -3.11 -m pytest tests/test_model_loop.py tests/test_feedback.py -q -p no:cacheprovider` (학습 없음, 수 초)
  - `PIPE`: `py -3.11 -m pytest tests/test_pipeline.py -q -p no:cacheprovider -k "test_1_ or test_2_ or test_3_ or test_3d or loop"` (약 3분). `test_pipeline.py`의 테스트는 앞 테스트가 만든 `workspace`에 기대므로 하나만 고르면 돌지 않는다. 새 테스트 이름에는 모두 `loop`를 넣는다.
  - `FULL`: (1) `py -3.11 -m pytest tests/test_pipeline.py -q -p no:cacheprovider -k "not loop"`(54개, 약 7분), (2) `PIPE`, (3) `py -3.11 -m pytest tests/test_model_loop.py tests/test_feedback.py tests/test_rag_mapping.py tests/test_sensor_features.py -q -p no:cacheprovider`. 한 명령으로 돌리면 10분을 넘는다.
  - `API`: `../.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider` (지금 42개 통과, 1개 건너뜀)
- Bash 도구의 heredoc에 작은따옴표 세 개나 홀수 개가 들어가면 실패한다. 긴 코드는 Write 도구로 파일에 쓴다.

## Review Focus

스펙이 말하지 않았지만 쓰다 보면 만나는 입력이다. 각 줄의 테스트는 해당 작업에 들어 있다.

1. **경보 시각의 표기.** `2021-09-01T10:00:00Z`, `2021-09-01T10:00:00+00:00`, `2021-09-01T10:00:00`이 같은 시각으로 읽힌다. 전처리 파일은 시간대 없는 UTC다. Task 2 `test_feedback_time_forms`.
2. **깨진 `feedback.json`.** 반쯤 쓰였거나 JSON이 아니면 `FeedbackError`(경로 포함)로 멈춘다. 점검은 등록부를 바꾸지 않고, 판정 없이 조용히 학습하지 않는다. Task 2 `test_feedback_store_rejects_broken_json`, Task 3 `test_loop_tick_stops_on_broken_feedback`.
3. **같은 케이스를 다시 가져옴, 다시 닫혀 결론이 바뀜.** 같은 내용은 세지 않고, 바뀐 내용은 나중 것이 이긴다. api가 더는 주지 않는 건은 남는다. Task 2 `test_feedback_store_merge`.
4. **끊긴 학습을 이어 할 때 판정이 달라져 있음.** 폴더에 남은 결과를 "이미 학습됨"으로 쓰지 않고 다시 학습한다. Task 4 `test_loop_can_reuse_needs_same_feedback`.
5. **구간이 파일 끝에 걸치거나 고장 전 1시간과 겹침.** 있는 창만 표시하고, 원래 정상이 아닌 창은 `windows_already_not_normal`로 센다. 적중 수는 달라지지 않는다. Task 1 `test_feedback_mark_partial_span`, `test_feedback_ew_hits_unchanged`.

## 파일 구조

| 파일 | 하는 일 | 작업 |
|---|---|---|
| `ml/.py/train.py` (수정) | `--feedback`, `confirmed` 열, 정상에서 빼는 네 곳, `feedback` 블록, `test_no_feedback` | 1 |
| `ml/.py/model_feedback.py` (새 파일) | 사람 판정 규칙, 파일 범위, 판정 저장 파일, api에서 가져오기, 파일 목록별 구간 모으기 | 2 |
| `ml/.py/model_registry.py` (수정) | 버전 기록의 `feedback`, 설정 기본값, `_brief`의 `feedback` | 3 |
| `ml/.py/model_loop.py` (수정) | `tick`의 `feedback_fn`, 기록, `train.py` 호출에 구간 넘기기, `feedback pull`, `status`, 보고서 | 3, 4 |
| `ml/tests/test_feedback.py` (새 파일) | Task 1의 함수 테스트, Task 2의 단위 테스트 | 1, 2 |
| `ml/tests/test_model_loop.py`, `ml/tests/test_pipeline.py` (수정) | 점검과 CLI, 합성 데이터로 끝에서 끝까지 | 1, 3, 4 |
| `api/ml_feedback.py` (새 파일), `api/main.py` (두 줄) | `GET /api/ml/feedback` | 5 |
| `api/tests/test_ml_feedback.py`, `api/docs/ml-feedback-api-20261112.md` (새 파일) | api 테스트와 설명 | 5 |
| `ml/README.md`, `history.md`, `MEMORY.md`, `CLAUDE.md`, `techspec.md` | 문서 | 6 |

---

### Task 1: `train.py`: 판정 구간을 정상에서 빼기

**Files:**
- Modify: `ml/.py/train.py` (`fit_and_eval`의 `normal`, `early_warning_population`, `evaluate`의 `normal`, `early_warning_metrics`, `evaluate_test`, `main`)
- Create: `ml/tests/test_feedback.py`
- Modify: `ml/tests/test_pipeline.py` (`test_18d` 뒤에 새 테스트)

**Interfaces:**
- Consumes: 없음.
- Produces:

```python
# train.py
def load_feedback(path) -> dict
    # {"confirmed": {file: [(Timestamp, Timestamp), ...]}, "false_alarm": {...}}. 없는 키는 {}. 시각은 시간대 없는 UTC
def mark_confirmed(w, spans) -> pd.DataFrame
    # 한 파일의 창 w(index = 창 시작)에 float32 `confirmed` 열을 붙여 돌려줌. start <= 창 시작 < end 이면 1.0. spans가 비면 전부 0.0
def confirmed_mask(w) -> np.ndarray
    # bool. `confirmed` 열이 없으면 전부 False
def feedback_counts(w, fb) -> dict
    # 여러 파일을 이은 w에서 {"spans": int, "windows_excluded": int, "windows_already_not_normal": int, "false_alarm_conflicts": int}
```

- CLI: `--feedback PATH` (스펙 5절의 JSON). 학습과 `--evaluate` 양쪽.
- `confirmed` 열은 **구간이 하나라도 있을 때만**, 읽는 모든 파일에 붙인다(구간 없는 파일은 0.0). `--feedback`이 없거나 `confirmed`가 비면 열이 없고 결과가 지금과 같다.
- 정상에서 빼는 네 곳은 모두 `confirmed_mask`를 읽는다. `early_warning_metrics`에서는 `normal = (ld > EW_ZONE_S[1]) & k & ~confirmed`만 바꾼다. 적중 구간(`hit`)은 건드리지 않는다. `ew_cols`에 `"confirmed"`를 더한다.
- `windows_excluded` = `confirmed == 1`이고 `label == 0`이고 `error_active == 0`인 창 수. `windows_already_not_normal` = `confirmed == 1`인 나머지 창 수. `false_alarm_conflicts` = `false_alarm` 구간 중, 구간의 첫 창(경보가 난 창)의 `early_warning_leads`가 600~3600초인 구간 수(실행 중 고침: 처음 계획은 "구간 안에 그런 창이 하나라도"였고 설계 4절과 어긋났다). `spans` = `confirmed` 구간 수.
- 학습 결과: `metrics["feedback"]`(번들과 `_metrics.json`의 `metrics` 안). `--feedback`이 없으면 키가 없다.
- `--evaluate` 결과 JSON: 최상위에 `"feedback"`. `confirmed` 창이 하나라도 있으면 최상위에 `"test_no_feedback"`(같은 점수, `confirmed` 열 없이 낸 `evaluate`)도 쓴다. 채점은 한 번만 한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`tests/test_feedback.py`. `sys.path`에 `.py`를 넣는 방식은 `tests/test_model_loop.py`와 같다. `import train`)

| 테스트 | 확인하는 것 |
|---|---|
| `test_feedback_load` | 스펙 5절의 예시 JSON을 읽으면 `confirmed["E01_3"] == [(Timestamp("2021-09-01 10:00"), Timestamp("2021-09-01 11:00"))]`. `"…Z"`로 쓴 시각도 같은 값. `false_alarm` 키가 없는 파일은 `{}` |
| `test_feedback_mark_and_mask` | 1분 창 180개(10:00부터)에 구간 10:30~11:30 → `confirmed` 합 60, 10:30 창은 1, 11:30 창은 0. `confirmed_mask`가 같은 위치에 True. 열이 없는 프레임의 `confirmed_mask`는 전부 False |
| `test_feedback_mark_partial_span` | 구간이 마지막 창 뒤로 30분 넘어가면 있는 30개만 표시. 구간이 파일 밖이면 0개이고 오류 없음 |
| `test_feedback_counts` | 구간 60창 중 2창이 `label == 1` → `windows_excluded == 58`, `windows_already_not_normal == 2`, `spans == 1`. `false_alarm` 구간이 고장 30분 전 창을 덮으면 `false_alarm_conflicts == 1`, 고장 3시간 전이면 0 |
| `test_feedback_population_excludes_confirmed` | `early_warning_population(w)`의 True 수가 `confirmed` 창 수만큼 준다 |
| `test_feedback_ew_false_alarm_removed` | 고장 5시간 전 한 창만 임계값을 넘는 점수(`cfg = {"enabled": True, "kind": "model_ng", "threshold": 0.5, "sustain": 1, "window_s": 60, "cooldown_s": 3600}`) → 열 없이 `per_file[f]["false_alarms"] == 1`. 그 창에 `confirmed = 1` → 0이고 `days`가 `1 / 1440` 줄어듦 |
| `test_feedback_ew_hits_unchanged` | 고장 30분 전 창이 임계값을 넘고 그 창에 `confirmed = 1` → `per_file[f]["hit"]`가 열이 없을 때와 같은 1.0 |
| `test_feedback_evaluate_normal` | `evaluate(val, scores, …)`의 `n_normal`이 `confirmed`이면서 정상이던 창 수만큼 준다. `n_pre_failure`는 같다 |

`test_pipeline.py`(합성 데이터, 학습 포함):

| 테스트 | 확인하는 것 |
|---|---|
| `test_18e_loop_feedback_evaluate` | `workspace["pre"]/manifest.csv`에서 `E04_0`의 `start`를 읽어 구간 `[start + 2 h, start + 3 h)`를 만든다. `test_18c`와 같은 `--evaluate`에 `--feedback`을 주면 `e["feedback"]["spans"] == 1`, `1 <= windows_excluded <= 60`, `e["test"]["n_normal"] == e["test_no_feedback"]["n_normal"] - e["feedback"]["windows_excluded"]`, `e["test"]["n_pre_failure"] == e["test_no_feedback"]["n_pre_failure"]` |
| `test_18f_loop_feedback_empty_is_same` | `{"confirmed": {}, "false_alarm": {}}`를 주면 `e["test"]`가 `--feedback` 없이 낸 것과 같고(`evaluated` 시각 제외), `"test_no_feedback"` 키가 없고, `e["feedback"]["spans"] == 0` |
| `test_18g_loop_feedback_train` | `test_18`과 같은 iforest 학습에 같은 구간의 `--feedback` → `_metrics.json`의 `metrics["feedback"]["windows_excluded"] >= 1`. `--feedback` 없이 학습한 `test_18`의 결과에는 `metrics`에 `feedback` 키가 없음 |

- [ ] **Step 2: 실패를 확인한다**

Run: `py -3.11 -m pytest tests/test_feedback.py -q -p no:cacheprovider`
Expected: FAIL, `AttributeError: module 'train' has no attribute 'load_feedback'`

- [ ] **Step 3: `train.py`에 구현한다**

- 네 함수를 `early_warning_leads` 뒤에 둔다(`feedback_counts`가 그것을 쓴다).
- 학습 루프(`for i, f in enumerate(files, 1)`)와 `evaluate_test`의 파일 루프에서, `window_file`과 `relabel` 뒤에 `mark_confirmed`를 부른다. 파일 이름은 `os.path.splitext(os.path.basename(f))[0]`이다.
- `evaluate_test`에 `feedback=None` 인자를 더한다. `confirmed` 창이 있으면 같은 `scores`로 `evaluate(test_w.drop(columns=["confirmed"]), …)`를 한 번 더 불러 `m["before_feedback"]`에 둔다. `main`은 이를 꺼내 최상위 `"test_no_feedback"`으로 쓰고, `"feedback": feedback_counts(…)`를 최상위에 쓴다.

- [ ] **Step 4: 통과를 확인한다**

Run: `py -3.11 -m pytest tests/test_feedback.py -q -p no:cacheprovider` → 8 passed
Run: `PIPE` → 34 passed (기존 31 + 3)

- [ ] **Step 5: 기존 결과가 그대로인지 확인한다**

Run: `FULL`의 (1) → 54 passed. `ruff check .` → 6 errors(기존 그대로).

---

### Task 2: `model_feedback.py`: 사람 판정 규칙과 저장 파일

**Files:**
- Create: `ml/.py/model_feedback.py`
- Modify: `ml/tests/test_feedback.py`

**Interfaces:**
- Consumes: api 응답의 항목 모양(스펙 3절). Task 5가 만들지만 이 작업은 dict만 쓴다.
- Produces:

```python
SPAN_S = 3600
FILE = "feedback.json"
VERDICTS = ("confirmed", "false_alarm", "unknown", "unmatched")

class FeedbackError(ValueError): ...            # 저장 파일을 읽을 수 없음, api가 응답하지 않음. 메시지에 경로나 주소

def naive_utc(s) -> pd.Timestamp                 # "…Z", "+00:00", 시간대 없음 → 시간대 없는 UTC
def load_ranges(data_dir: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    # manifest.csv의 남은 파일(dropped가 빈 것)마다 (start, end). manifest가 없으면 ValueError
def verdict(item: dict, ranges: dict, span_s: int = SPAN_S) -> dict
    # {"verdict": VERDICTS 중 하나, "event_id": str | None, "file": str | None, "start": iso | None, "end": iso | None}
class Store:
    def __init__(self, root: str) -> None        # path = <root>/feedback.json
    def load(self) -> dict | None                # 없으면 None. JSON이 아니거나 schema_version != 1이면 FeedbackError
    def merge(self, items: list[dict], source: str, at: str) -> dict[str, int]
        # {"new", "changed", "same", "skipped"}. 열쇠 = 대표 경보의 event_id. 임시 파일 + os.replace
def fetch(api_url: str, timeout: float = 30.0) -> list[dict]
    # GET <api_url>/api/ml/feedback 의 items. 연결 실패 · 200 아님 · schema_version이 "1."로 시작하지 않음 → FeedbackError
def collect(data: dict | None, ranges: dict, names: list[str], span_s: int = SPAN_S) -> dict | None
    # data가 None이면 None. 아니면
    # {"spans": {"confirmed": {file: [[start, end]]}, "false_alarm": {file: [[start, end]]}},
    #  "counts": {"confirmed": int, "false_alarm": int, "unknown": int, "unmatched": int}, "event_ids": [str]}
FeedbackFn = Callable[[list[str]], "dict | None"]
def make_feedback_fn(root: str, data_dir: str, span_s: int = SPAN_S) -> FeedbackFn
    # 부를 때마다 Store(root).load()와 load_ranges(data_dir)를 읽어 collect
```

- `verdict`의 순서(스펙 4절): 경보 고르기(`selected_event_id`가 `events`에 있으면 그것, 아니면 `representative`가 참인 것) → 파일 맞추기(`gun_id`가 `ranges`에 있고 `start <= window_start <= end`) → 결론. 고른 경보에 `gun_id`나 `window_start`가 없으면 `unmatched`.
- `collect`의 `counts`: `confirmed` · `false_alarm` · `unknown`은 파일이 `names`에 든 판정만 센다. `unmatched`는 저장 파일 전체에서 센다. `spans`의 구간은 시작 시각 순, `event_ids`는 `confirmed` 구간에 쓴 경보다. 구간의 시각은 시간대 없는 ISO 문자열이다.
- `merge`는 항목에 `received_at = at`을 붙여 저장한다. 같은지 비교할 때는 `received_at`을 뺀다. `events`가 없거나 대표 경보가 없는 항목은 저장하지 않고 `skipped`로 센다.

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`tests/test_feedback.py`에 이어서. `import model_feedback as mf`)

항목을 만드는 도우미 `_item(outcome="resolved", abnormal=1, selected=None, events=None)`를 둔다. 기본 경보는 `{"event_id": "e1", "representative": True, "gun_id": "E01_3", "window_start": "2021-09-01T10:00:00Z", "detected_at": "2021-09-01T10:00:59Z", "detector_created": "2026-11-01T18:46:47"}`, 기본 범위는 `{"E01_3": (Timestamp("2021-08-30"), Timestamp("2021-09-06"))}`다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_feedback_verdict_confirmed` | `{"verdict": "confirmed", "event_id": "e1", "file": "E01_3", "start": "2021-09-01T10:00:00", "end": "2021-09-01T11:00:00"}` |
| `test_feedback_verdict_table` | `abnormal=0` → `unknown`. `outcome="false_alarm"` → `false_alarm`(이상 점검이 있어도). `resolution_status`가 `"not_linked"`여도 `abnormal=1`이면 `confirmed` |
| `test_feedback_verdict_selected_event` | 대표 `e1`(10:00)과 반복 `e2`(14:00)에 `selected="e2"` → 구간 14:00~15:00, `event_id == "e2"`. `selected`가 `events`에 없는 ID면 대표 경보 |
| `test_feedback_verdict_unmatched` | `gun_id="test_0"`(범위에 없음), `window_start`가 파일 종료 뒤, 고른 경보에 `window_start` 없음 → 셋 다 `unmatched`이고 `file`은 None |
| `test_feedback_time_forms` | `window_start`가 `"2021-09-01T10:00:00Z"`, `"…+00:00"`, `"2021-09-01T10:00:00"`, `"2021-09-01T19:00:00+09:00"`일 때 `start`가 모두 `"2021-09-01T10:00:00"` |
| `test_feedback_span_setting` | `span_s=1800` → `end`가 10:30 |
| `test_feedback_load_ranges` | 임시 `manifest.csv`(한 줄은 `dropped`에 값) → 남은 파일만, `start`와 `end`가 Timestamp. 파일이 없으면 ValueError |
| `test_feedback_store_merge` | 빈 폴더에 항목 2개 → `{"new": 2, "changed": 0, "same": 0, "skipped": 0}`, `load()["cases"]`의 열쇠가 대표 `event_id`. 같은 2개를 다시 → `same` 2. 하나의 `outcome`을 바꿔 그 하나만 다시 → `changed` 1이고 다른 하나는 남아 있음. `source`와 `pulled_at`이 마지막 호출 값 |
| `test_feedback_store_rejects_broken_json` | 파일에 `{"schema_version": 1,`만 쓰고 `load()` → `FeedbackError`, 메시지에 경로, 파일 내용 불변. 파일이 없으면 None |
| `test_feedback_collect` | 저장 파일에 `confirmed` 1(E01_3), `false_alarm` 1(E01_3), `unknown` 1(E02_1), `unmatched` 1 → `names=["E01_3"]`이면 `counts == {"confirmed": 1, "false_alarm": 1, "unknown": 0, "unmatched": 1}`, `spans["confirmed"] == {"E01_3": [["2021-09-01T10:00:00", "2021-09-01T11:00:00"]]}`, `event_ids == ["e1"]`. `data=None`이면 None |
| `test_feedback_fetch` | `httpx.get`을 monkeypatch: 정상 응답 → `items`. `ConnectError` → `FeedbackError`(주소 포함). 500 → `FeedbackError`. `schema_version: "2.0"` → `FeedbackError` |

- [ ] **Step 2: 실패를 확인한다**

Run: `py -3.11 -m pytest tests/test_feedback.py -q -p no:cacheprovider`
Expected: FAIL, `ModuleNotFoundError: No module named 'model_feedback'`

- [ ] **Step 3: `ml/.py/model_feedback.py`를 쓴다**

`httpx`는 `fetch` 안에서 가져온다(`model_loop.make_notify_fn`과 같은 방식). 저장은 `model_registry.Registry.save`와 같이 임시 파일 + `os.replace`다. 모듈 맨 위 설명에 스펙 경로와 사람 판정 규칙 표를 한 번 적는다.

- [ ] **Step 4: 통과를 확인한다**

Run: `py -3.11 -m pytest tests/test_feedback.py -q -p no:cacheprovider` → 19 passed. `ruff check .` → 6 errors.

---

### Task 3: 점검(`tick`)과 등록부 기록

**Files:**
- Modify: `ml/.py/model_registry.py` (`Registry.init`, `add_version`, `_brief`)
- Modify: `ml/.py/model_loop.py` (`LoopConfig`, `TrainFn` · `EvalFn` 설명, `tick`, `replay`)
- Modify: `ml/tests/test_model_loop.py`

**Interfaces:**
- Consumes: Task 2의 `collect`가 돌려주는 dict 모양(`spans`, `counts`, `event_ids`). `tick`은 `model_feedback`을 가져오지 않고 `feedback_fn`만 받는다.
- Produces:

```python
# model_registry.py
CONFIG_DEFAULTS = {"feedback_span_s": 3600}     # init: 어느 쪽에 이 키가 없으면 이 값으로 보고 비교한다. 저장된 config는 고치지 않는다
def add_version(data, version, *, cutoff, train_files, seed, created, bundle, status, feedback: dict | None = None) -> None
    # 버전 기록에 "feedback": feedback (없으면 None)
def _brief(...)                                  # "feedback": 버전 기록의 feedback 을 더한다

# model_loop.py
class LoopConfig: feedback_span_s: int = 3600   # as_dict()에 들어간다. CLI 옵션은 만들지 않는다
def tick(reg, files, now, cfg, train_fn, eval_fn, notify_fn=None, feedback_fn=None) -> list[str]
def replay(reg, files, start, end, interval_days, cfg, train_fn, eval_fn, notify_fn=None, feedback_fn=None)
```

- `fb = feedback_fn(파일 목록)`이 None이 아니면 학습 · 채점 함수를 네 번째 인자로 부른다: `train_fn(v, names, seeds, fb["spans"])`, `eval_fn(version, seed, names, fb["spans"])`. `feedback_fn`이 없거나 None을 돌려주면 지금처럼 세 인자로 부른다.
- 학습 함수의 반환에 `"feedback"`(Task 1의 `metrics["feedback"]`)이 올 수 있다. 채점 함수의 반환에 `"before"`(반영 전의 `metrics_summary`, 없으면 None)와 `"feedback"`이 올 수 있다(Task 4가 만든다).
- 버전 기록: `feedback = None`(`fb`가 None) 또는 `{**fb["counts"], "event_ids": fb["event_ids"], **(out.get("feedback") or {})}`. 학습 파일 기준으로 `feedback_fn(names)`를 부른다.
- A/B 판정 기록에 세 키를 더한다. `feedback`: 섀도 파일 기준, 버전 기록과 같은 모양(`ma.get("feedback")`을 합친다). `values_before`와 `swap_before`: `ma.get("before")`가 None이 아니면 `judge(ma["before"], mb["before"], 시드들의 before auroc, cfg)`의 `values`와 `swap`, 아니면 둘 다 None. 기존 `values` · `checks` · `swap`은 반영 후 수치다.
- 잠정 확인 기록에도 같은 방식으로 `feedback`, `values_before`, `revert_before`(`probe`의 `revert`)를 더한다.
- 사람 판정만으로는 학습하지 않는다: 학습 조건(`train_min_new`)은 그대로다.
- `feedback_fn`이 예외(`FeedbackError`)를 내면 그대로 올린다. 그때까지 등록부를 저장하지 않는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`tests/test_model_loop.py`)

`Fakes`에 더한다: `train(self, version, files, seeds, spans=None)`과 `eval(self, version, seed, files, spans=None)`이 `spans`를 `self.train_spans[version]`, `self.eval_spans`(목록)에 적는다. `self.before`(버전 → auroc)가 있으면 `eval`이 `"before": _m(그 값, …)`을 붙이고, `self.fb_block`이 있으면 `"feedback": self.fb_block`을 붙인다. 도우미 `_fb(confirmed_files)`는 주어진 파일마다 구간 하나를 가진 `collect` 모양의 dict를 `names`에 든 파일로만 걸러 돌려주는 `feedback_fn`을 만든다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_loop_config_has_feedback_span` | `LoopConfig().as_dict()["feedback_span_s"] == 3600` |
| `test_loop_registry_old_config_still_opens` | `feedback_span_s`가 없는 config로 만든 등록부를 `reg.init(LoopConfig().as_dict()와 같은 값)`으로 열 수 있다. 저장된 config에는 키가 여전히 없다. `feedback_span_s`가 1800이면 `RegistryError`(메시지에 `feedback_span_s`) |
| `test_loop_tick_without_feedback_fn` | `feedback_fn` 없이 `_swap_run` → 학습 · 채점 함수가 세 인자로만 불림(`train_spans`가 비어 있음). 버전 기록의 `feedback`은 None, 판정 기록의 `feedback` · `values_before` · `swap_before`는 None |
| `test_loop_tick_feedback_fn_returns_none` | `feedback_fn=lambda names: None` → 위와 같은 결과 |
| `test_loop_tick_passes_spans_to_training` | 첫 16건 중 `w00_0`에 구간 → `fk.train_spans["v001"]["confirmed"] == {"w00_0": [[…]]}`. `versions["v001"]["feedback"]`에 `confirmed == 1`, `event_ids`, 그리고 학습 함수가 돌려준 `windows_excluded` |
| `test_loop_tick_passes_spans_to_scoring` | 섀도 파일 하나에 구간 → 그 판정의 채점 호출 전부(A, B, 시드)가 같은 `spans`를 받음. 판정 기록의 `feedback["confirmed"] == 1` |
| `test_loop_decision_records_before_and_after` | `aurocs = {"v002": 0.80}`, `before = {"v001": 0.70, "v002": 0.69}` → `swap is True`, `swap_before is False`, `values_before["auroc_b"] == 0.69`, `values["auroc_b"] == 0.80` |
| `test_loop_feedback_alone_does_not_train` | 새로 끝난 파일이 `train_min_new`보다 적은 점검에 `feedback_fn`을 줘도 `train:`이 없음 |
| `test_loop_tick_stops_on_broken_feedback` | 이미 점검이 한 번 기록된 등록부에서, `feedback_fn`이 `ValueError("cannot read …/feedback.json")`를 내면 `tick`이 그 예외를 올리고, `registry.json` 내용이 호출 전과 같고, 잠금 파일이 없음 |
| `test_loop_probation_records_feedback` | `_provisional_run`에 `feedback_fn`과 `before` → 잠정 확인 기록에 `feedback`, `values_before`, `revert_before` |
| `test_loop_registry_brief_has_feedback` | `mr.summary(data)["champion"]["feedback"]`가 버전 기록의 값 |

기존 테스트 중 버전 기록 · 판정 기록 · `summary`를 dict 전체로 비교하는 것에는 새 키(`feedback` 등, 값 None)를 더한다. `tests/test_pipeline.py`에서 `/model/loop` 응답을 dict 전체로 비교하는 테스트도 같다(`champion` · `challenger`에 `"feedback": None`). `ml/main.py`는 `/model/loop`의 설명 문장에 새 필드를 한 줄 적는 것 말고는 바꾸지 않는다.

- [ ] **Step 2: 실패를 확인한다**

Run: `UNIT`
Expected: 새 11개 FAIL(`tick() got an unexpected keyword argument 'feedback_fn'` 등), 나머지는 통과

- [ ] **Step 3: 구현한다**

`tick` 안에 작은 도우미 둘을 둔다: 파일 목록으로 `fb`를 얻는 것, `fb`가 있을 때만 네 번째 인자를 붙여 부르는 것. `TrainFn` · `EvalFn`은 `Callable[..., …]`로 바꾸고 주석에 선택 인자를 적는다.

- [ ] **Step 4: 통과를 확인한다**

Run: `UNIT` → 101 passed(`test_model_loop.py` 82 + `test_feedback.py` 19). `ruff check .` → 6 errors.

---

### Task 4: `train.py` 호출, `feedback pull`, `status`, 보고서

**Files:**
- Modify: `ml/.py/model_loop.py` (`can_reuse`, `make_train_fn`, `_evaluate_to`, `make_eval_fn`, `build_report`, `report_markdown`, `_status`, `main`)
- Modify: `ml/tests/test_model_loop.py`, `ml/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1의 `--feedback`, `metrics["feedback"]`, 최상위 `"feedback"` · `"test_no_feedback"`. Task 2의 `Store`, `fetch`, `load_ranges`, `collect`, `make_feedback_fn`, `FeedbackError`. Task 3의 `tick(…, feedback_fn)`.
- Produces:

```python
def can_reuse(folder: str, names: list[str], model: str, spans: dict | None = None) -> bool
    # 파일 목록이 같고, <folder>/feedback.json 의 내용이 spans와 같을 때만 True. 구간이 하나도 없는 spans는 None과 같고, 그때는 파일이 없어야 한다
# make_train_fn(...)이 돌려주는 함수: train(version, names, seeds, spans=None) -> {"created", "bundle", "feedback": dict | None}
# make_eval_fn(...)이 돌려주는 함수:  evaluate(version, seed, names, spans=None) -> {**metrics_summary(test), "before": dict | None, "feedback": dict | None}
```

- 학습: 구간이 있으면 `<폴더>/feedback.json`에 쓰고 `--feedback`으로 넘긴다. 구간이 없으면 남아 있던 그 파일을 지운다. 시드 모델 폴더도 같다.
- 채점: 캐시 열쇠 = `sha1`(정렬한 파일 이름 + 구간이 있으면 `"\n" + json.dumps(spans, sort_keys=True)`)의 앞 12자. 구간이 없으면 열쇠가 지금과 같다(기존 캐시를 그대로 쓴다). 구간은 `<열쇠>.feedback.json`에 쓰고 `--feedback`으로 넘긴다.
- CLI: `model_loop.py feedback pull --api-url URL [--registry …]`. 등록부 잠금을 잡고 `fetch` → `Store.merge`. 출력 한 줄: `feedback: 2 new, 1 changed, 5 same, 0 skipped (8 cases) <- URL`. `FeedbackError`는 `ValueError`라서 `main`의 기존 처리로 exit 2가 된다.
- `tick` · `replay` 명령은 항상 `make_feedback_fn(reg.root, args.data_dir, cfg.feedback_span_s)`를 넘긴다.
- `status`: 저장 파일이 있으면 한 줄을 더한다: `feedback    8 cases, pulled <pulled_at>: confirmed 1, false_alarm 3, unknown 2, unmatched 2`. 판정별 건수는 `collect(data, ranges, 전체 파일 이름)`으로 낸다. `manifest.csv`를 읽을 수 없으면 건수 부분을 빼고 케이스 수만 보인다.
- 보고서: `build_report`에 `"feedback": {"versions": {버전: 기록의 feedback}, "decisions": [{"at", "champion", "challenger", "feedback", "values", "values_before", "swap", "swap_before"}]}`. `report_markdown`에 `## 사람 판정` 절: 반영한 버전의 표(버전, 이상 확인, 오알람, 모름, 못 맞춤, 뺀 창 수), 반영 전후가 있는 A/B 판정의 표(AUROC 차와 오알람의 전후, 결론의 전후). 반영된 것이 없으면 `반영된 사람 판정이 없습니다.` 한 줄.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_model_loop.py`:

| 테스트 | 확인하는 것 |
|---|---|
| `test_loop_can_reuse_needs_same_feedback` | 학습이 끝난 폴더(파일 셋과 `feedback.json` 없음)는 `spans=None`과 빈 구간에 True, 구간이 있으면 False. `feedback.json`이 구간 A인 폴더는 A에 True, 다른 구간과 None에 False |
| `test_loop_cli_feedback_pull` | `mf.fetch`를 monkeypatch해 항목 2개 → exit 0, 출력에 `2 new`, `<등록부>/feedback.json`이 생김. 다시 부르면 `2 same`. 잠금 파일이 남지 않음 |
| `test_loop_cli_feedback_pull_api_down` | `fetch`가 `FeedbackError` → exit 2, stderr에 주소, 이미 있던 `feedback.json` 내용 불변 |
| `test_loop_cli_status_shows_feedback` | 저장 파일이 있는 등록부의 `status` 출력에 `feedback` 줄과 `confirmed 1`. 저장 파일이 없으면 그 줄이 없음. `--data-dir`에 `manifest.csv`가 없어도 exit 0이고 케이스 수가 보임 |
| `test_loop_report_feedback_section` | 버전과 판정에 `feedback` · `values_before`가 있는 등록부 → `report.md`에 `## 사람 판정`, 버전 표와 전후 표. 아무 판정도 없으면 `반영된 사람 판정이 없습니다.` |

`tests/test_pipeline.py`(실제 학습, 합성 데이터). `test_20c` 뒤에 둔다:

| 테스트 | 확인하는 것 |
|---|---|
| `test_20d_loop_feedback_end_to_end` | 새 등록부 폴더(`tmp_path`)에 `feedback.json`을 직접 쓴다: 가장 먼저 끝나는 합성 파일에 `confirmed` 하나(`[start + 2 h, start + 3 h)`, `gun_id` = 파일 이름), 같은 파일에 `false_alarm` 하나. `loop_ws["settings"]`로 `replay`를 첫 모델까지 돌린다. `versions["v001"]["feedback"]`의 `confirmed == 1`, `false_alarm == 1`, `spans == 1`, `windows_excluded >= 1`, `event_ids`가 그 경보. `v001/feedback.json`이 있고 `v001/baseline_ensemble_metrics.json`의 `metrics["feedback"]["spans"] == 1` |
| `test_20e_loop_feedback_resume_and_judge` | 같은 등록부로 `replay`를 끝까지 → 첫 모델을 다시 학습하지 않음(출력에 `already trained`). 판정 기록에 `feedback` 키가 있고, 섀도 파일에 구간이 없으므로 `values_before`는 None |

- [ ] **Step 2: 실패를 확인한다**

Run: `UNIT`
Expected: 새 5개 FAIL(`can_reuse() takes 3 positional arguments`, `invalid choice: 'feedback'` 등)

- [ ] **Step 3: 구현한다**

`_evaluate_to`는 `"test"`만이 아니라 JSON 전체를 돌려주게 바꾸고, 부르는 쪽 둘(`make_eval_fn`, `build_report`)을 맞춘다. `build_report`의 테스트 8건 평가에는 구간을 넘기지 않는다(test 파일에는 사람 판정이 없다).

- [ ] **Step 4: 통과를 확인한다**

Run: `UNIT` → 106 passed. Run: `PIPE` → 36 passed.

- [ ] **Step 5: 전체를 확인한다**

Run: `FULL` 셋 → (1) 54 passed, (2) 36 passed, (3) 137 passed(`test_model_loop.py` 87 + `test_feedback.py` 19 + 나머지 31). `ruff check .` → 6 errors.

---

### Task 5: api `GET /api/ml/feedback`

**Files:**
- Create: `api/ml_feedback.py`, `api/tests/test_ml_feedback.py`, `api/docs/ml-feedback-api-20261112.md`
- Modify: `api/main.py` (가져오기 한 줄, `app.include_router(ml_feedback.router)` 한 줄)

**Interfaces:**
- Consumes: `store.all_cases()`, `store.warning_delivery_for_case(case_id)`(`{"event_id", "status"}` 또는 None).
- Produces:

```python
router = APIRouter()
def item(case: dict, delivery: dict | None) -> dict | None    # 닫힌 조기 경보 케이스가 아니면 None. 모양은 스펙 3절
def items(cases: list[dict], delivery_for) -> list[dict]       # closed_at 순
@router.get("/api/ml/feedback")                                # {"schema_version": "1.0", "items": [...]}
```

- 대상: `case["event"].get("event_type") == "early_warning"`이고 `records.closure.outcome`이 `"resolved"` 또는 `"false_alarm"`.
- `selected_event_id` = `delivery["event_id"]`(없으면 None). `resolution_status` = `delivery["status"]`, 없으면 `case.get("warning_resolution_status")`.
- `checks`: `records.checks`와 `warning_events[고른 경보 또는 대표 경보].draft.checks`를 점검 ID로 합친다(같은 ID는 초안이 이긴다). 값은 `"abnormal"` 또는 `{"result": "abnormal"}` 두 형식을 다 읽는다. `abnormal` · `normal` · `unchecked` 개수를 내고, 다른 값은 세지 않는다.
- `events`: 대표 경보가 먼저(`representative: true`), 그 뒤 `repeats`. `handoff`가 있으면 `gun_id`, `window_start`, `detected_at`, `detector_created`(= `handoff["detector"]["created"]`)를 채운다. 없으면 `event_id`와 `detected_at`만 채우고 나머지는 None이다.

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`api/tests/test_ml_feedback.py`. `conftest.py`의 `make_handoff` · `make_case` · `api`를 쓴다. 조기 경보 핸드오프는 `make_handoff(…, code=None, source="model")`에 `event_type="early_warning"`과 `detector={"created": "2026-11-01T18:46:47"}`를 더해 만든다)

| 테스트 | 확인하는 것 |
|---|---|
| `test_item_resolved_legacy_checks` | `closure="resolved"`, `checks={"K1": {"result": "abnormal"}, "K2": {"result": "normal"}}`, `warning_resolution_status="not_linked"`, `delivery=None` → `outcome == "resolved"`, `checks == {"abnormal": 1, "normal": 1, "unchecked": 0}`, `selected_event_id is None`, `resolution_status == "not_linked"`, `closed_at`이 closure의 `at` |
| `test_item_resolved_selected_draft` | 반복 `w2`를 고름(`delivery={"event_id": "w2", "status": "saved"}`), `warning_events["w2"]["draft"]["checks"] == {"c1": "abnormal", "c2": "unchecked"}` → `selected_event_id == "w2"`, `resolution_status == "saved"`, `checks == {"abnormal": 1, "normal": 0, "unchecked": 1}` |
| `test_item_same_check_counted_once` | 같은 ID가 `records.checks`에 normal, 초안에 abnormal → `abnormal` 1, `normal` 0 |
| `test_item_false_alarm` | `closure="false_alarm"` → 항목이 나오고 `outcome == "false_alarm"` |
| `test_item_excluded` | E 코드 케이스(닫힘), 열린 조기 경보 케이스, `outcome="handed_over"`인 조기 경보 케이스 → 셋 다 None |
| `test_item_events` | 대표 + `handoff` 있는 반복 + `handoff` 없는 반복 → 길이 3, 첫째만 `representative`가 참, 둘째는 `window_start`와 `detector_created`가 채워짐, 셋째는 `event_id` · `detected_at`만 있고 `gun_id is None` |
| `test_route` | `store.put`으로 케이스 둘(닫힌 조기 경보, 닫힌 E 코드)을 넣고 `GET /api/ml/feedback` → 200, `schema_version == "1.0"`, `items` 길이 1 |

- [ ] **Step 2: 실패를 확인한다**

Run(`api/`에서): `../.venv/Scripts/python.exe -m pytest tests/test_ml_feedback.py -q -p no:cacheprovider`
Expected: FAIL, `ModuleNotFoundError: No module named 'ml_feedback'`

- [ ] **Step 3: `api/ml_feedback.py`를 쓰고 `api/main.py`에 두 줄을 더한다**

`main.py`의 줄 끝(CRLF)을 유지한다. 가져오기는 `import store` 근처에, 등록은 `app = FastAPI(…)` 뒤에 둔다.

- [ ] **Step 4: 통과를 확인한다**

Run: `API` → 49 passed, 1 skipped (기존 42 + 7)

- [ ] **Step 5: `api/docs/ml-feedback-api-20261112.md`를 쓴다**

주소, 응답 예(스펙 3절), 필드별 출처, "사실만 내주고 정답 여부는 ML이 정한다", 기존 주소는 바뀌지 않는다는 점. 한 쪽 이내.

---

### Task 6: 실제 확인 한 번과 문서

**Files:**
- Modify: `ml/README.md`, `ml/history.md`, `ml/MEMORY.md`, `ml/CLAUDE.md`, `ml/techspec.md`, 스펙의 상태 줄
- 실행 결과: `ml/results/D-12/runs/check-1/` (`run.log`, `status.txt`, 요약 `README.md`)

**Interfaces:**
- Consumes: Task 1 ~ 5 전부.
- Produces: 없음.

스펙 9절의 "실제 확인 한 번"이다. 성공 기준 1(학습 쪽)만 본다.

- [ ] **Step 1: api를 외부 서비스 없이 띄울 수 있는지 먼저 본다**

`API_DB`를 scratch 폴더의 새 파일로 두고 `api/`에서 `../.venv/Scripts/python.exe -m uvicorn main:app --port 9011`. `GET /api/ml/feedback`이 `{"schema_version": "1.0", "items": []}`이면 된다. 시작하지 못하거나 케이스 저장 때 Neo4j · Claude를 부르려 하면 `api/tests/warning_demo_server.py`로 바꾸고, 그 사실을 결과 요약에 적는다. 둘 다 안 되면 멈추고 사용자에게 알린다.

- [ ] **Step 2: 경보를 만든다**

- 스트림할 파일: `models/registry-replay-1/registry.json`의 `versions.v001.train_files` 중 하나. 원본 CSV는 `train/` 아래에 있다(읽기만).
- api가 ML 서버 주소를 읽는 환경 변수를 `api/main.py`의 `ml_get`에서 확인해 8012 포트로 맞춘다(점수 추이를 ML 서버에서 가져온다).
- ML 서버: `RSW_STATE_DIR`를 scratch 폴더로, `RSW_EARLY_WARNING_HANDOFFS=1`, `RSW_RAG_URL=http://127.0.0.1:9011/api/ml/handoffs`로 두고 `py -3.11 -m uvicorn main:app --port 8012`. 기본 서빙 번들을 쓴다(`RSW_MODEL_REGISTRY` 없음).
- `py -3.11 .py/replay.py <CSV> --url http://127.0.0.1:8012 --hours 12`로 시작해, api에 조기 경보 케이스가 생길 때까지 `--start-hours`와 `--no-reset`으로 12시간씩 이어 흘린다. 데이터 48시간 안에 경보가 없으면 다른 파일로 바꾼다.
- Expected: `GET http://127.0.0.1:9011/api/cases`(또는 임시 DB)에 `event_type == "early_warning"`이고 `gun_id`가 그 파일 이름인 케이스가 있다.

- [ ] **Step 3: 이상을 남기고 예전 경로로 닫는다**

`PUT /api/cases/<id>/checks` (`{"checks": {"D12-CHECK": {"result": "abnormal"}}}`) → `PUT /api/cases/<id>/log` (`{"action": "D-12 경로 확인용", "approved": true}`) → `POST /api/cases/<id>/close` (`{"outcome": "resolved"}`).
Expected: `GET /api/ml/feedback`의 항목에 `outcome == "resolved"`, `checks.abnormal == 1`, `resolution_status == "not_linked"`. Aura와 Claude로 나간 요청이 없다(api 로그 확인).

- [ ] **Step 4: 가져와서 첫 모델을 학습한다**

api가 떠 있는 동안 먼저 가져온다:

Run: `py -3.11 .py/model_loop.py feedback pull --api-url http://127.0.0.1:9011 --registry models/registry-d12`
Expected: `feedback: 1 new, …`

그 뒤 자기가 띄운 두 서버를 끝내고(학습에 메모리가 필요하다) 첫 모델까지 재생한다:

Run: `py -3.11 .py/model_loop.py replay --start 2021-08-16 --end 2021-09-07 --registry models/registry-d12`
Expected: `2021-09-06T00:00:00  first:v001`. 10분을 넘으면 같은 명령을 다시 부르면 이어진다.

- [ ] **Step 5: 기록을 확인하고 남긴다**

Run: `py -3.11 .py/model_loop.py status --registry models/registry-d12`
Expected: `feedback` 줄에 `confirmed 1`. `registry.json`의 `versions.v001.feedback`에 `confirmed == 1`, `windows_excluded >= 1`, `event_ids`가 그 경보.

`models/registry-replay-1/v001`(같은 16건, 같은 시드, 판정 없음)과 `threshold`, 조기 경보 `threshold`, `n_fit_windows`를 나란히 적는다. 이 차이를 성능 향상으로 해석하지 않는다.

`results/D-12/runs/check-1/`에 명령과 출력(`run.log`), `status` 출력, 위 비교와 한계(스펙 7절)를 담은 `README.md`를 쓴다.

- [ ] **Step 6: 문서를 고친다**

| 파일 | 내용 |
|---|---|
| `ml/README.md` | 5.12 뒤에 5.13 "사람 판정 반영(D-12)": 흐름, 사람 판정 규칙 표, 명령(`feedback pull`), 실제 확인 결과, 한계(지금 모델 쪽으로 기움, 성능 향상은 보이지 않음). 테스트 수 갱신 |
| `ml/history.md` | 51절: 결정 네 가지(완료 기준, 정상에서 빼기, 반영 후 수치로 결정, 고른 경보만)와 근거, 포크 세션 검토에서 고친 점, 실제 확인 결과 |
| `ml/MEMORY.md` | 해당 절과 변경 이력 한 줄 |
| `ml/CLAUDE.md` | 명령 줄에 `feedback pull`, 테스트 수, 레이아웃에 `model_feedback.py`와 `tests/test_feedback.py` |
| `ml/techspec.md` | D-12 행 |
| 스펙 | 상태 줄을 "구현 2026-11-12"로 |

- [ ] **Step 7: 마지막 확인**

Run: `FULL` 셋과 `API`, `ruff check .`
Expected: ml 227개 통과(54 + 36 + 137), api 49 passed 1 skipped, ruff 6 errors. 사용자에게 스테이징할 파일 목록(`git add -A ml api/ml_feedback.py api/main.py api/tests/test_ml_feedback.py api/docs/ml-feedback-api-20261112.md`)과, 커밋 전에는 reset으로 사라진다는 점을 알린다.

---

## 스펙 대조

| 스펙 | 작업 |
|---|---|
| 2절 흐름, 시각 맞추기 | 2 (`naive_utc`, `load_ranges`), 4 (`feedback pull`) |
| 3절 api 주소 | 5 |
| 4절 사람 판정 규칙 | 2 (`verdict`, `collect`) |
| 5절 `train.py` 네 곳, `feedback` 블록, `test_no_feedback` | 1 |
| 6.1절 저장 파일과 명령 | 2 (`Store`), 4 (CLI, `status`) |
| 6.2절 점검 | 3 (`tick`), 4 (학습 · 채점 호출, 캐시 열쇠, 이어 하기) |
| 6.3절 기록, 보고서, `/model/loop`, 설정 | 3 (기록, `_brief`, `CONFIG_DEFAULTS`), 4 (보고서) |
| 7절 한계 | 3 (`values_before`, `swap_before`), 6 (문서) |
| 8절 실패 처리 | 2 (`FeedbackError`), 3 (`test_loop_tick_stops_on_broken_feedback`), 4 (`test_loop_cli_feedback_pull_api_down`), 1 (`test_feedback_mark_partial_span`) |
| 9절 테스트, 실제 확인 | 1 ~ 5, 6 |
| 10절 문서 | 5 (api 문서), 6 |
