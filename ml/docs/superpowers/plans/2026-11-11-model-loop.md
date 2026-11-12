# D-11 모델 루프 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새로 끝난 파일로 모델을 다시 학습하고, 현재 모델과 섀도 비교해, 다섯 조건을 모두 만족하면 서빙 모델을 재시작 없이 바꾼다.

**Architecture:** 등록부(`model_registry.py`)가 버전 · 현재 모델 · 도전자 · 판정을 `registry.json` 하나에 담는다. 루프 실행기(`model_loop.py`)의 `tick`이 점검 한 번을 하고, 학습과 채점은 `train.py`를 별도 프로세스로 부른다. 서버(`main.py`)는 등록부의 현재 모델을 읽고 `POST /model/reload`로 바꿔 탄다.

**Tech Stack:** Python 3.11, pandas, scikit-learn, LightGBM, FastAPI, Pydantic v2, pytest, ruff

**상태(2026-11-12):** 작업 1 ~ 8 완료, 새 검토자 한 명의 최종 검토 반영. 실제 64건 재생 · 보고서 · 서버 교체 확인을 11-12에 돌렸고(`results/D-11/runs/replay-1/`) 결과를 문서에 넣었다(`history.md` 50절). 점검 하나가 10분을 넘어, Step 3은 같은 재생을 남은 시간에 맞춰 나눠 부르는 보조 스크립트로 돌렸다.

**Spec:** `ml/docs/superpowers/specs/2026-11-11-model-loop-design.md`. 규칙은 3절, 판정은 5절, 구성 요소는 8절, 실패 처리는 9절, 테스트 목록은 11절이다. 이 계획은 스펙을 되풀이하지 않고 파일 · 이름 · 시그니처 · 테스트만 정한다. 실행자는 둘 다 읽는다.

## Global Constraints

- 커밋, 푸시, 머지, PR 금지. 사용자가 직접 커밋한다. 변경은 `main` 작업 트리에 커밋 없이 두고, 작업이 끝날 때마다 스테이징할 파일 목록을 보고한다. 다른 브랜치로 옮기거나 worktree를 만들지 않는다.
- 작업 트리에 커밋 안 된 변경이 이미 있다(11-11 피드백 루프 정리: `api/` 8개, `ml/` 14개, 이 스펙과 계획). 되돌리거나 덮어쓰지 않는다. `git stash`, `git checkout --`, `git reset`을 쓰지 않는다.
- `api/`, `rag/`, `ui/` 코드를 바꾸지 않는다.
- 유료 API 호출 금지. `.env`를 읽거나 출력하지 않는다.
- 실제 데이터 폴더 `train/`, `test/`, `preprocessed/`를 바꾸지 않는다. 테스트는 임시 폴더의 합성 데이터만 쓴다. Task 8의 실제 실행도 `preprocessed/`를 읽기만 한다.
- 8 GB PC다. 오래 도는 명령은 포그라운드에서 10분 이내로 끊어 돌린다. 백그라운드로 돌리지 않는다. 자기가 띄운 프로세스만 끝낸다.
- 작업 트리의 파일은 CRLF다(`core.autocrlf=true`). 고치는 파일의 줄 끝을 유지한다. 새 파일은 LF로 써도 된다(git이 맞추고 경고만 나온다).
- 계약은 추가만 한다. `/predict`, `/model`, `AnomalyResult`, `RagHandoff`의 기존 필드를 바꾸지 않는다. `RSW_MODEL_REGISTRY`가 없으면 서버 동작은 지금과 같아야 한다.
- 경로는 `PROJECT_ROOT` 기준이다. 현재 폴더에 기대지 않는다. 줄 길이 120. `ruff check .`의 기존 오류는 `tests/test_sensor_features.py:14` E402 하나뿐이고 새 오류가 없어야 한다.
- 설정 기본값은 스펙 3절 표 그대로다: `initial_min_files=16`, `train_min_new=5`, `shadow_min_files=5`, `seeds=(42, 43, 44)`, `fa_ratio_max=1.2`, `fa_cap=24.0`, 재생 간격 7일.
- 명령은 모두 `ml/`에서 `py -3.11`로 실행한다. `UNIT`은 `py -3.11 -m pytest tests/test_model_loop.py -q -p no:cacheprovider`(학습 없음, 수 초)다.
- `tests/test_pipeline.py`의 테스트는 앞 테스트가 만든 `workspace`에 기대므로 `-k`로 하나만 고르면 돌지 않는다. `PIPE`는 `py -3.11 -m pytest tests/test_pipeline.py -q -p no:cacheprovider -k "test_1_ or test_2_ or test_3_ or test_3d or loop"`를 뜻한다. 새 테스트 이름에는 모두 `loop`를 넣는다.
- 전체 시험은 한 명령으로 돌리면 10분을 넘으므로 셋으로 나눈다. `FULL`은 이 셋을 차례로 돌리는 것이다: (1) `py -3.11 -m pytest tests/test_pipeline.py -q -p no:cacheprovider -k "not loop"`(기존 54개, 약 7분), (2) `PIPE`, (3) `py -3.11 -m pytest tests/test_model_loop.py tests/test_rag_mapping.py tests/test_sensor_features.py -q -p no:cacheprovider`.

## Review Focus

스펙이 말하지 않았지만 쓰다 보면 만나는 입력이다. 각 줄의 테스트는 해당 작업에 들어 있다.

1. **깨진 `registry.json`.** 반쯤 쓰였거나 손으로 고쳐 JSON이 아니면, 실행기는 파일 경로를 담은 `RegistryError`로 멈추고 파일을 건드리지 않는다. 서버의 reload는 500이고 옛 모델로 계속 돈다. Task 1 `test_loop_registry_rejects_broken_json`, Task 6 `test_loop_cli_broken_registry`, Task 7 `test_21g_loop_reload_broken_registry_keeps_old_model`.
2. **등록된 번들 파일이 없다.** 현재 모델의 번들이 지워졌으면 서버는 경로를 담은 오류로 시작하지 않는다. reload는 500이고 옛 모델을 유지한다. Task 7 `test_21h_loop_registry_missing_bundle`.
3. **`--now`의 시각 표기.** `2021-09-20`, `2021-09-20T00:00:00`, `2021-09-20T00:00:00Z`가 같은 시각으로 읽힌다. `manifest.csv`의 `end`는 시간대 없는 UTC다. Task 3 `test_loop_now_forms`.
4. **학습 중 Ctrl+C.** 잠금 파일이 풀리고 등록부는 바뀌지 않는다. 다음 실행이 이어 간다. Task 1 `test_loop_lock_released_on_interrupt`, Task 3 `test_loop_train_failure_leaves_registry`.
5. **`manifest.csv`에는 있는데 parquet가 없다.** `train.py --files-from`이 없는 파일 이름을 알리며 멈추고, 점검은 오류로 끝나고 등록부는 바뀌지 않는다. Task 4 `test_18b_loop_files_from_missing_file`.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `.py/model_registry.py` (새로) | `registry.json` 읽기 · 쓰기 · 잠금, 등록부 dict를 바꾸는 순수 함수 |
| `.py/model_loop.py` (새로) | 설정, 판정 함수, 점검 · 재생 로직, `train.py` 호출 래퍼, 보고서, CLI |
| `.py/train.py` (수정) | `--files-from`, `--eval-files-from`, `--eval-out` |
| `.py/preprocess.py` (수정) | `--split append` |
| `main.py` (수정) | 등록부에서 읽기, `ModelGate`, `POST /model/reload`, `GET /model/loop`, `/health`의 `model_version` |
| `tests/test_model_loop.py` (새로) | 등록부 · 판정 · 점검 로직 단위 테스트(학습 없음) |
| `tests/test_pipeline.py` (수정) | 합성 데이터가 필요한 테스트(`train.py` 옵션, 추가 전처리, 루프 전체, 서버) |
| `.gitignore` (수정) | `models/registry/` |

등록부 dict의 형식은 스펙 8.1절의 예시에 두 가지를 더한 것이다. 버전마다 `"bundle"`(등록부 폴더 기준 상대 경로, 예 `"v001/baseline_ensemble.joblib"`), 최상위에 `"ticks"`(점검마다 `{"at", "actions"}`). 보고서의 '점검마다 한 일'이 `ticks`에서 나온다.

---

### Task 1: 모델 등록부

**Files:**
- Create: `.py/model_registry.py`
- Test: `tests/test_model_loop.py` (새로)

**Interfaces:**
- Produces:

```python
SCHEMA = 1
STATUSES = ("champion", "challenger", "retired", "rejected")

class RegistryError(Exception): ...

class Registry:
    def __init__(self, root: str) -> None            # 폴더. registry.json은 그 안
    path: str                                         # <root>/registry.json
    def load(self) -> dict | None                     # 없으면 None. JSON이 아니면 RegistryError(경로 포함)
    def init(self, config: dict) -> dict              # 없으면 empty(config)를 저장하고 돌려줌. 있으면 config가 같을 때만 돌려줌, 다르면 RegistryError
    def save(self, data: dict) -> None                # 임시 파일 + os.replace (main.write_json_atomic과 같은 재시도)
    def bundle_path(self, data: dict, version: str) -> str
    def version_dir(self, version: str) -> str
    def lock(self) -> ContextManager[None]            # <root>/registry.lock 을 O_EXCL로. 이미 있으면 RegistryError(경로 포함). 예외 · KeyboardInterrupt에도 지움

def empty(config: dict) -> dict                       # schema, config, last_tick None, champion None, challenger None, versions {}, decisions [], history [], ticks []
def next_version(data: dict) -> str                   # "v001", "v002", ...
def latest_version(data: dict) -> str | None          # 번호가 가장 큰 버전
def add_version(data, version, *, cutoff: str, train_files: list[str], seed: int, created: str, bundle: str, status: str) -> None
def set_champion(data, version, *, at: str, by: str) -> None   # 앞 현재 모델은 "retired", history에 {"at","champion","by"} 추가. by는 "first" | "decision" | "rollback"
def set_challenger(data, version: str | None) -> None
def reject(data, version: str) -> None                # status "rejected", challenger None
def record_decision(data, decision: dict) -> None
def record_tick(data, at: str, actions: list[str]) -> None     # ticks에 추가하고 last_tick 갱신
def previous_champion(data) -> str | None             # history의 끝에서 두 번째 champion
def summary(data, limit: int = 10) -> dict            # 스펙 8.4절 GET /model/loop의 champion · challenger · last_tick · decisions 부분
```

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`tests/test_model_loop.py`. 맨 위에서 `sys.path`에 `.py`를 넣는 방식은 `tests/test_rag_mapping.py`와 같게 한다)

| 테스트 | 확인하는 것 |
|---|---|
| `test_loop_registry_init_and_reload` | `init(cfg)` 뒤 `load()`가 같은 dict. `registry.json`이 생긴다 |
| `test_loop_registry_config_mismatch` | 다른 `config`로 `init`하면 `RegistryError` |
| `test_loop_registry_versions_and_champion` | `next_version`이 v001 → v002. `set_champion` 두 번 뒤 첫 버전은 `retired`, `history` 길이 2, `previous_champion` == v001 |
| `test_loop_registry_reject_and_latest` | `reject` 뒤 상태 `rejected` · `challenger` None. `latest_version`은 떨어진 버전도 센다 |
| `test_loop_registry_atomic_save` | `save` 도중 `json.dump`가 예외를 내도(monkeypatch) 원래 파일 내용이 그대로다 |
| `test_loop_registry_rejects_broken_json` | 파일에 `{"schema": 1,`만 쓰고 `load()` → `RegistryError`, 메시지에 경로. 파일 내용 불변 |
| `test_loop_registry_lock` | `lock()` 안에서 다시 `lock()` → `RegistryError`. 블록을 나오면 잠금 파일이 없다 |
| `test_loop_lock_released_on_interrupt` | `with reg.lock(): raise KeyboardInterrupt` 뒤 잠금 파일이 없다 |
| `test_loop_registry_summary_limit` | 판정 12건 중 `summary(data, 10)["decisions"]`가 최근 10건, 최신이 마지막 |

- [ ] **Step 2: `UNIT`을 돌려 실패를 확인한다.** 기대: `ModuleNotFoundError: model_registry`.
- [ ] **Step 3: `.py/model_registry.py`를 쓴다.** 표준 라이브러리만 쓴다(`main.py`가 import하므로 pandas를 끌어오지 않는다).
- [ ] **Step 4: `UNIT`이 통과하는지 본다.** 기대: 9 passed.
- [ ] **Step 5: `ruff check .py/model_registry.py tests/test_model_loop.py`에 오류가 없는지 보고, 스테이징할 파일을 보고한다.**

---

### Task 2: 판정 함수

**Files:**
- Create: `.py/model_loop.py` (이 작업에서는 설정과 판정만)
- Test: `tests/test_model_loop.py`

**Interfaces:**
- Produces:

```python
DEFAULT_RECIPE: tuple[str, ...]      # 스펙 4절의 인자 그대로. "--seed"는 넣지 않는다(실행기가 붙인다)

@dataclass(frozen=True)
class LoopConfig:
    initial_min_files: int = 16
    train_min_new: int = 5
    shadow_min_files: int = 5
    seeds: tuple[int, ...] = (42, 43, 44)
    fa_ratio_max: float = 1.2
    fa_cap: float = 24.0
    recipe: tuple[str, ...] = DEFAULT_RECIPE
    def as_dict(self) -> dict        # registry config에 들어가는 모양 (튜플은 리스트로)

def metrics_summary(m: dict) -> dict
    # train.py --evaluate가 쓴 JSON의 "test" 블록 → 판정 입력
    # {"auroc": float | None,                 m["auroc_gun_mean"], NaN은 None
    #  "per_file": {file: float | None},      m["per_file"][f]["auroc"], NaN은 None
    #  "ew": bool,                            "early_warning" 블록이 있는가
    #  "hits": int,                           early_warning.per_file[*]["hit"]의 합(NaN 제외). ew False면 0
    #  "fa": float}                           early_warning.false_alarms_per_gun_day. ew False거나 NaN이면 0.0

def judge(a: dict, b: dict, seed_aurocs: list[float | None], cfg: LoopConfig) -> dict
    # a, b: metrics_summary 형식. seed_aurocs: B(시드 42)와 시드 모델들의 "auroc"
    # → {"values": {"auroc_a","auroc_b","noise","files_b_ge_a","files_compared","hits_a","hits_b","fa_a","fa_b"},
    #    "checks": {"auroc_margin","file_majority","ew_hits","fa_ratio","fa_cap"},   # 각각 bool
    #    "swap": bool, "reason": str}
```

판정 규칙은 스펙 5절이다. 스펙이 정하지 않은 세부는 이렇게 한다.

- 우연 폭: `seed_aurocs`에서 None을 뺀 값이 2개 미만이면 `noise`는 None이다.
- `auroc_a`, `auroc_b`, `noise` 중 하나라도 None이면 `swap=False`, `checks`는 모두 False, `reason`은 `"판정 불가: "`로 시작하고 빠진 값을 적는다.
- `file_majority`: 둘 다 값이 있는 파일만 센다. `files_b_ge_a * 2 > files_compared`. 비교한 파일이 0이면 False.
- `fa_ratio`: `a["ew"]`가 False면 True이고 `reason`에 `"A에 조기 경보가 없어 오알람 비율 조건을 건너뜀"`을 덧붙인다.
- `reason`: 교체면 `"다섯 조건을 모두 만족"`. 아니면 어긴 조건의 이름과 수치를 쉼표로 잇는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.** 입력을 만드는 도우미 `_m(auroc, per_file, hits=1, fa=10.0, ew=True)`를 둔다.

| 테스트 | 입력 → 기대 |
|---|---|
| `test_loop_judge_swap` | A 0.70, B 0.75, 시드 [0.75, 0.74, 0.76](폭 0.02), 5파일 중 4개에서 B ≥ A, 적중 2 대 2, 오알람 10 대 11 → `swap` True, 다섯 조건 모두 True, `noise` ≈ 0.02 |
| `test_loop_judge_margin_fails` | 위에서 B 0.71(차 0.01 < 폭 0.02) → `auroc_margin` False, `swap` False, `reason`에 `auroc_margin` |
| `test_loop_judge_margin_equal_fails` | 차가 폭과 정확히 같으면 False (초과여야 한다) |
| `test_loop_judge_majority` | 5파일 중 2개만 B ≥ A → `file_majority` False. 4파일 중 2개(동률 아님, 절반)도 False. 한 파일의 AUROC가 같으면 B ≥ A로 센다 |
| `test_loop_judge_majority_skips_missing` | 파일 5개 중 2개가 한쪽 None → `files_compared` 3 |
| `test_loop_judge_hits` | 적중 A 3, B 2 → `ew_hits` False |
| `test_loop_judge_fa_ratio` | 오알람 A 10.0, B 12.0 → True(1.2배는 통과). B 12.1 → False |
| `test_loop_judge_fa_cap` | 오알람 A 23.0, B 24.0 → True. A 23.0, B 24.1 → `fa_cap` False |
| `test_loop_judge_no_auroc` | A의 `auroc` None → `swap` False, `reason`이 `판정 불가`로 시작 |
| `test_loop_judge_one_seed` | `seed_aurocs`가 `[0.75, None, None]` → `noise` None, `판정 불가` |
| `test_loop_judge_a_without_ew` | A `ew=False`, B `ew=True` 적중 1 · 오알람 5 → `fa_ratio` True, `ew_hits` True |
| `test_loop_judge_b_without_ew` | A 적중 1, B `ew=False` → `ew_hits` False |
| `test_loop_metrics_summary` | 손으로 만든 `"test"` 블록(파일 3개, 한 파일 `auroc` NaN, `hit` 1 · 0 · NaN, `false_alarms_per_gun_day` 8.3) → `per_file`의 NaN은 None, `hits` 1, `fa` 8.3, `ew` True. `early_warning` 키가 없으면 `ew` False · `hits` 0 · `fa` 0.0 |
| `test_loop_default_recipe` | `DEFAULT_RECIPE`에 `--cv`, `4`, `--early-warning-q`, `0.975`, `--exclude-non-welding`, `--drop-features`, `c19`가 있고 `--seed`는 없다 |

- [ ] **Step 2: `UNIT`으로 실패를 확인한다.**
- [ ] **Step 3: `LoopConfig`, `DEFAULT_RECIPE`, `metrics_summary`, `judge`를 `.py/model_loop.py`에 쓴다.** 이 파일은 맨 위에서 `train`을 import하지 않는다(단위 테스트가 가볍게 돌도록). pandas는 써도 된다.
- [ ] **Step 4: `UNIT` 통과를 확인한다.**
- [ ] **Step 5: `ruff` 확인, 스테이징 목록 보고.**

---

### Task 3: 점검 · 재생 로직

**Files:**
- Modify: `.py/model_loop.py`
- Test: `tests/test_model_loop.py`

**Interfaces:**
- Consumes: Task 1의 `Registry`와 순수 함수, Task 2의 `LoopConfig` · `judge`.
- Produces:

```python
def parse_time(s: str) -> pd.Timestamp               # "2021-09-20", "...T00:00:00", "...T00:00:00Z" → 시간대 없는 UTC
def load_manifest(data_dir: str) -> list[tuple[str, pd.Timestamp]]
    # manifest.csv에서 dropped가 비었고 end가 있는 줄만, (file, end)를 end 순으로. 같은 file이 두 번 나오면 ValueError

TrainFn = Callable[[str, list[str], list[int]], dict]
    # (version, train_files, seeds) → {"created": str, "bundle": str}. seeds[0]으로 버전 번들, seeds[1:]로 시드 모델
EvalFn = Callable[[str, int | None, list[str]], dict]
    # (version, seed 또는 None=버전 번들, files) → metrics_summary 형식
NotifyFn = Callable[[], None]                         # 현재 모델이 바뀐 뒤 한 번 부른다

def tick(reg: Registry, files: list[tuple[str, pd.Timestamp]], now: pd.Timestamp, cfg: LoopConfig,
         train_fn: TrainFn, eval_fn: EvalFn, notify_fn: NotifyFn | None = None) -> list[str]
def replay(reg, files, start: pd.Timestamp, end: pd.Timestamp | None, interval_days: int, cfg,
           train_fn, eval_fn, notify_fn=None) -> list[tuple[str, list[str]]]
```

`tick`이 돌려주는 행동 문자열(등록부 `ticks`에도 같은 값이 들어간다):

| 문자열 | 뜻 |
|---|---|
| `"wait_first:<끝난 파일 수>"` | 현재 모델이 없고 파일이 모자람 |
| `"first:v001"` | 첫 모델 학습 |
| `"defer:<섀도 파일 수>"` | 섀도 파일이 모자라 판정을 미룸 |
| `"swap:v001>v002"` / `"keep:v001|v002"` | 판정 결과 |
| `"train:v003"` | 도전자 학습 |
| `"idle"` | 할 일이 없음 |

규칙은 스펙 3절이다. 구현에서 지킬 순서:

- `tick` 전체를 `reg.lock()` 안에서 한다. `reg.init(cfg.as_dict())`로 등록부를 얻는다.
- `now < last_tick`이면 `ValueError`. 같으면 허용한다(재실행).
- 학습이나 채점 함수가 예외를 내면 등록부를 저장하지 않고 그대로 올린다. 저장은 단계가 끝난 뒤에만 한다: 첫 모델 등록, 판정 기록 + 현재 모델 변경(한 번의 `save`), 도전자 등록, 마지막에 `record_tick`.
- 판정 뒤 같은 점검에서 3번(새 도전자)으로 이어 간다. 미룬 경우는 이어 가지 않는다.
- 판정의 채점 호출: `eval_fn(A, None, shadow)`, `eval_fn(B, None, shadow)`, 그리고 `cfg.seeds[1:]`마다 `eval_fn(B, seed, shadow)`. `seed_aurocs`는 B의 값을 맨 앞에 둔다.
- 판정 dict에는 `judge`의 결과에 `at`, `champion`, `challenger`, `shadow_files`를 더한다(스펙 8.1절).
- `notify_fn`은 교체가 저장된 뒤에 부른다. 예외가 나면 경고를 출력하고 넘어간다.
- `replay`: `start + k * interval_days`를 `k = 1, 2, ...`로 돈다. `end`가 None이면 마지막 파일의 `end`를 지난 첫 점검까지. `last_tick`보다 앞의 점검은 건너뛴다. `last_tick`의 점검은 다시 부른다(도중에 끊겼을 수 있다. 끝난 점검이면 아무것도 쓰지 않는다). 점검 기록은 저장 지점마다 그 점검의 항목에 이어 붙인다(구현 중 결정, 11-11).

- [ ] **Step 1: 실패하는 테스트를 쓴다.** 가짜 함수를 쓴다: `fake_train`은 호출을 기록하고 `{"created": f"c-{version}", "bundle": f"{version}/b.joblib"}`을 돌려준다. `fake_eval`은 버전별 AUROC 표를 받아 `metrics_summary` 형식을 만든다. 달력은 도우미 `_files(counts, start="2021-08-16")`로 만든다: 주마다 `counts[i]`개 파일이 그 주의 화요일 12:00에 끝난다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_loop_spec_timeline` | 2019-12-25에 끝나는 파일 1개 + `_files([2, 2, 11, 17, 7, 1, 4, 3, 12, 3, 1])`, 기본 설정, 모든 판정이 유지가 되게 한 `fake_eval`, `replay(start=2021-08-16, interval_days=7)`. 점검별 행동이 스펙 3절 표와 같다: 08-23 · 08-30 `wait_first:3` · `wait_first:5`, 09-06 `first:v001`, 09-13 `train:v002`, 09-20 `keep:v001|v002` + `train:v003`, 09-27 `defer:1`, 10-04 `keep:v001|v003` + `train:v004`, 10-11 `defer:3`, 10-18 `keep:v001|v004` + `train:v005`, 10-25 `defer:3`, 11-01 `defer:4`. 학습 파일 수는 16 · 33 · 40 · 45 · 60, 섀도 파일 수는 7 · 5 · 15 |
| `test_loop_swap_changes_champion` | `fake_eval`이 B를 이기게 하면 `swap:v001>v002`, 등록부 `champion` v002, v001 `retired`, `history` 2건, `notify_fn`이 한 번 불림 |
| `test_loop_keep_rejects_challenger` | 유지면 v002 `rejected`, `challenger` None, `notify_fn`은 안 불림 |
| `test_loop_seed_models_only_for_challengers` | `fake_train` 호출 기록: v001은 `seeds == [42]`, v002는 `[42, 43, 44]` |
| `test_loop_eval_calls` | 판정 한 번에 `fake_eval` 호출이 `(A, None)`, `(B, None)`, `(B, 43)`, `(B, 44)`이고 파일 목록이 섀도 파일과 같다 |
| `test_loop_train_min_new` | 현재 모델만 있고 새 파일이 4건이면 `idle`, 5건이면 `train:` |
| `test_loop_same_now_is_noop` | 같은 `now`로 `tick`을 두 번 → 두 번째는 `defer:0` 또는 `idle`, `fake_train` 호출 수가 늘지 않는다 |
| `test_loop_time_backwards` | `now`가 `last_tick`보다 앞이면 `ValueError`, 등록부 불변 |
| `test_loop_train_failure_leaves_registry` | `fake_train`이 예외 → `tick`이 올리고, `registry.json` 내용이 호출 전과 같고, 잠금 파일이 없다. 다음 `tick`은 같은 버전 번호로 다시 시도한다 |
| `test_loop_eval_failure_keeps_challenger` | `fake_eval`이 예외 → 판정 기록 없음, `challenger`는 그대로 |
| `test_loop_notify_failure_is_warning` | `notify_fn`이 예외 → `tick`은 성공하고 교체는 저장돼 있다 |
| `test_loop_replay_resumes` | 재생을 09-13까지 한 뒤 처음부터 다시 부르면, 이미 한 점검은 건너뛰고 `fake_train` 호출이 겹치지 않는다. 전체를 한 번에 돈 것과 등록부가 같다(`created`는 가짜라 같다) |
| `test_loop_now_forms` | `parse_time`의 세 표기가 같은 `Timestamp`이고 `tzinfo`가 없다 |
| `test_loop_manifest_skips_dropped` | `dropped`가 채워졌거나 `end`가 빈 줄은 빠지고, 순서가 뒤섞인 입력도 `end` 순으로 나온다. 같은 `file`이 두 줄이면 `ValueError` |

- [ ] **Step 2: `UNIT`으로 실패를 확인한다.**
- [ ] **Step 3: `parse_time`, `load_manifest`, `tick`, `replay`를 쓴다.**
- [ ] **Step 4: `UNIT` 통과를 확인한다.**
- [ ] **Step 5: `ruff` 확인, 스테이징 목록 보고.**

---

### Task 4: `train.py` 파일 목록 옵션

**Files:**
- Modify: `.py/train.py` (`main()`의 인자 정의, `--evaluate` 분기, `files = sorted(glob...)` 부근)
- Test: `tests/test_pipeline.py` (`test_17_...` 뒤에 추가)

**Interfaces:**
- Produces (명령 인자): `--files-from PATH`, `--eval-files-from PATH`, `--eval-out PATH`. 뜻은 스펙 8.3절.
  - 목록 파일은 UTF-8, 한 줄에 이름 하나(확장자 없음, 예 `E01_0`). 빈 줄은 무시한다.
  - 없는 파일이 있으면 `SystemExit`이고 메시지에 그 이름들이 있다.
  - `--eval-files-from`을 쓰면 파일별 점수 CSV를 쓰지 않는다(`evaluate_test`에 `score_dir=None`). JSON의 지표 키는 지금처럼 `"test"`다.
  - `--evaluate`는 `--eval-files-from`이 있으면 `test_*.parquet`가 없어도 된다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.** `workspace`의 전처리 결과(8개: `E01_0` ~ `E04_1`)와 `run()`을 쓴다. 학습은 `--model iforest --val-frac 0.5 --warmup-hours 0.5 --no-test`로 가볍게 한다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_18_loop_files_from` | 목록 `E01_0, E01_1, E02_0, E02_1, E03_0, E03_1`로 학습 → 번들 지표의 `train_files` + `val_files`가 정확히 그 6개(`.parquet` 이름)이고 `E04_*`가 없다 |
| `test_18b_loop_files_from_missing_file` | 목록에 `E09_9` → 종료 코드가 0이 아니고 출력에 `E09_9` (이 테스트는 `subprocess.run`을 직접 쓴다) |
| `test_18c_loop_eval_files_from` | 위 번들을 `--evaluate --eval-files-from`(목록 `E04_0, E04_1`) `--eval-out <tmp>/e.json`으로 평가 → `e.json`의 `test.files`가 그 2개, `test.per_file`의 키가 `E04_0`, `E04_1`, `test.auroc_gun_mean`이 있다. `<model-dir>/scores`에 `E04_0`으로 시작하는 파일이 없다 |
| `test_18d_loop_options_absent` | 옵션 없이 `--evaluate` → 지금처럼 `baseline_iforest_test_metrics.json`이 갱신된다 |

- [ ] **Step 2: `PIPE`로 실패를 확인한다.** 기대: `unrecognized arguments: --files-from`.
- [ ] **Step 3: 세 옵션을 구현한다.** 학습 · 평가 로직은 바꾸지 않는다.
- [ ] **Step 4: `PIPE` 통과를 확인한다.**
- [ ] **Step 5: `ruff` 확인, 스테이징 목록 보고.**

---

### Task 5: 전처리 `--split append`

**Files:**
- Modify: `.py/preprocess.py` (`main()`, 모듈 docstring의 사용법)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces (명령): `preprocess.py --split append --input-dir <폴더> [--train-out-dir <전처리 폴더>]`. 뜻은 스펙 8.5절.

스펙이 정하지 않은 세부:

- `--input-dir`은 필수다. 없으면 `SystemExit`. 출력 폴더는 `--train-out-dir`이다(`--out-dir`을 주면 `SystemExit`).
- 설정과 표준화 값은 `--split test`처럼 `inherit_train_config` · `load_train_scaler`로 읽는다. `scaler.json`과 `preprocess_config.json`은 쓰지 않는다.
- 입력 이름은 지금 규칙대로 `<앞부분>_<정수>.csv`여야 한다. 아니면 그 파일을 건너뛰고 이유를 출력한다.
- `manifest.csv`에 `source` 열을 둔다(추가된 줄에만 값이 있고 기존 줄은 빈 값). 입력 파일의 원래 이름(확장자 없음)이다. 원래 이름이 기존 `file`이나 `source`에 있으면 건너뛴다.
- 앞부분이 `E01` ~ `E04`면 이름을 그대로 쓴다. 아니면 `infer_class`로 유형을 정한다.
  - 유형이 `unknown`이면 건너뛰고 `no terminal code in the last 10 min`을 출력한다. `manifest.csv`에 넣지 않는다.
  - 아니면 새 이름은 `<유형>_<N>`이고 N은 `manifest.csv`에서 그 유형의 가장 큰 `gun` 번호 + 1이다. parquet 안의 `file` · `gun` 열도 새 값으로 쓴다.
- 결측률로 빠진 파일은 `dropped` 사유와 함께 `manifest.csv`에 넣는다(다시 시도하지 않도록).
- 마지막에 `appended N, skipped M, dropped K`를 한 줄로 출력한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.** `workspace["pre"]`를 직접 바꾸면 뒤 테스트에 영향이 있다. 모듈 범위 픽스처 `append_ws(workspace, tmp_path_factory)`가 `workspace["pre"]`를 한 번 복사하고(`pre`) 입력 폴더(`incoming`)를 만든다. 19번 테스트들은 차례로 그 사본을 쓴다(`--train-out-dir`). 입력 파일은 `synth_file`로 만들고, 각 테스트가 자기 파일을 `incoming`에 더한다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_19_loop_append_named_file` | 입력 `E01_7.csv`(시작 2021-11-01) → 사본에 `E01_7.parquet`, `manifest.csv`에 `file == E01_7` · `source == E01_7` · `end`가 2021-11-01 01:59:59. `scaler.json`과 `preprocess_config.json`의 바이트가 실행 전과 같다. 정상 행의 `c5` 평균이 기존 파일과 같은 척도다(절댓값 0.5 미만) |
| `test_19b_loop_append_infers_class` | 입력 `line_3.csv`(E03으로 끝남) → `E03_2.parquet`(기존 E03의 최대 번호 1 + 1), parquet의 `file` 열이 모두 `E03_2`, `class`가 `E03`, `manifest.csv`의 `source == line_3`, `class_source`가 `terminal code E028` |
| `test_19c_loop_append_skips_no_fault` | 종료 코드가 없는 파일(`synth_file` 결과에서 `error` 열을 `"0"`으로 덮어쓴 것) → parquet가 생기지 않고 `manifest.csv` 줄 수가 그대로, 출력에 `no terminal code` |
| `test_19d_loop_append_is_idempotent` | 같은 명령을 다시(입력 폴더에 앞의 세 파일이 그대로 있다) → `appended 0`, `manifest.csv`가 그대로 |
| `test_19e_loop_append_then_train` | 추가된 `E03_2`를 `--files-from` 목록에 넣어 학습이 끝난다(Task 4와 이어짐) |

- [ ] **Step 2: `PIPE`로 실패를 확인한다.** 기대: `invalid choice: 'append'`.
- [ ] **Step 3: `--split append`를 구현한다.** `--split train` · `test`의 동작은 바꾸지 않는다. `manifest.csv`를 읽을 때 `source` 열이 없으면 빈 열로 본다.
- [ ] **Step 4: `PIPE` 통과를 확인한다.** `test_1_` · `test_2_`가 그대로 통과하는지도 본다(기존 분할 동작 불변).
- [ ] **Step 5: `ruff` 확인, 스테이징 목록 보고.**

---

### Task 6: 실행기 CLI, `train.py` 호출, 보고서

**Files:**
- Modify: `.py/model_loop.py`, `.gitignore`
- Test: `tests/test_pipeline.py`, `tests/test_model_loop.py`

**Interfaces:**
- Consumes: Task 1 ~ 5 전부.
- Produces:

```python
def seed_recipe(recipe: tuple[str, ...]) -> list[str]
    # "--cv N", "--early-warning-rule X", "--early-warning-q X", "--early-warning-cooldown-min X"를 빼고 "--no-early-warning"을 더한다
def model_name(recipe) -> str                         # "--model X"의 X, 없으면 "ensemble"
def make_train_fn(reg: Registry, data_dir: str, cfg: LoopConfig) -> TrainFn
def make_eval_fn(reg: Registry, data_dir: str, cfg: LoopConfig) -> EvalFn
def make_notify_fn(server_url: str | None) -> NotifyFn | None   # POST <url>/model/reload, timeout 60초
def build_report(reg: Registry, data_dir: str, cfg: LoopConfig) -> dict
def write_report(out_dir: str, report: dict) -> None  # report.json, report.md
def main(argv: list[str] | None = None) -> int        # tick | replay | report | status | rollback
```

파일 배치(등록부 폴더 기준):

| 경로 | 내용 |
|---|---|
| `v001/train_files.txt` | `--files-from` 목록 |
| `v001/baseline_<model>.joblib`, `baseline_<model>_metrics.json` | 버전 번들(`train.py`가 쓴다) |
| `v002/seeds/s43/` | 시드 모델(같은 파일 이름) |
| `v001/eval/<sha1 앞 12자>.json`, `v002/seeds/s43/eval/<...>.json` | 채점 결과. 키는 정렬한 파일 이름을 줄바꿈으로 이은 문자열의 sha1 |
| `v001/test_metrics.json` | 테스트 8건 채점(보고서용) |

동작:

- `make_train_fn`: 폴더에 번들과 지표 JSON이 있고 `train_files.txt`의 내용이 이번 목록과 같으면 그 학습을 건너뛴다(이어 하기). 목록이 다르면 다시 학습해 덮어쓴다. 이 판단은 `can_reuse(folder: str, names: list[str], model: str) -> bool`로 둔다. 학습 명령은 `[sys.executable, train.py, *recipe, "--seed", s, "--data-dir", D, "--model-dir", 폴더, "--files-from", 목록, "--no-test"]`이고, 시드 모델은 `seed_recipe(recipe)`를 쓴다. 종료 코드가 0이 아니면 표준 오류의 끝 20줄을 담아 `RuntimeError`. 끝나면 `joblib.load`로 번들이 읽히는지 확인하고, `created`는 지표 JSON에서 읽는다. 버전 번들과 시드 모델이 모두 생긴 뒤에 돌려준다.
- `make_eval_fn`: 결과 JSON이 있으면 읽기만 한다. 없으면 `train.py --evaluate --model <이름> --data-dir D --model-dir 폴더 --eval-files-from <임시 목록> --eval-out <경로>`. `metrics_summary(json["test"])`를 돌려준다.
- 하위 프로세스의 출력은 그대로 흘려보낸다(진행 상황이 보이도록).
- `rollback`: `--to`가 없으면 `previous_champion`. 대상이 없거나, 등록부에 없는 버전이거나, 지금의 현재 모델 · 도전자면 오류 메시지와 종료 코드 2(도전자는 판정을 거치지 않았다). 물러났거나 떨어진 버전은 받는다. `set_champion(..., by="rollback")` 뒤 `notify_fn`.
- `status`: 현재 모델, 도전자, 마지막 점검, 마지막 판정의 `swap` · `reason`을 출력한다. 등록부가 없으면 `no registry at <경로>`.
- `report`: 모든 버전을 `train.py --evaluate --model <이름> --model-dir <버전 폴더> --data-dir D --eval-out <버전 폴더>/test_metrics.json`으로 채점한다(있으면 건너뜀). `<D>/test`에 `test_*.parquet`가 없으면 테스트 표를 비우고 그 사실을 보고서에 적는다. `report.json`의 최상위 키: `config`, `ticks`, `versions`(버전별 `cutoff`, `status`, `n_train_files`, `by_class`, 테스트의 `auroc`, `auroc_gun_mean`, 조기 경보 `recall` · `chance` · `false_alarms_per_gun_day`), `decisions`, `first_vs_final`(`first`, `final`, 두 버전의 테스트 `auroc_gun_mean`과 그 차). `report.md`는 같은 내용을 표 넷(점검, 버전, 판정, 테스트)으로 쓴다.
- CLI 인자: 스펙 8.2절 + 설정값(`--initial-min-files`, `--train-min-new`, `--shadow-min-files`, `--seeds 42,43,44`, `--fa-ratio-max`, `--fa-cap`, `--recipe-file`). `--recipe-file`은 `train.py` 인자를 담은 JSON 배열이다.
- `RegistryError` · `ValueError`는 메시지를 출력하고 종료 코드 2로 끝낸다(스택 추적 없이).
- `.gitignore`에 `models/registry/` 한 줄을 `# 서빙 상태` 블록에 더한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/test_model_loop.py`(학습 없음):

| 테스트 | 확인하는 것 |
|---|---|
| `test_loop_seed_recipe` | `DEFAULT_RECIPE` → `--cv` · `4` · `--early-warning-q` · `0.975` 등이 없고 `--no-early-warning`이 있다. 나머지 인자는 순서대로 남는다 |
| `test_loop_can_reuse` | 번들 · 지표 JSON · 같은 목록이면 True. 목록이 다르거나 번들이 없으면 False |
| `test_loop_cli_status_without_registry` | `main(["status", "--registry", tmp])` → 0, 출력에 `no registry` |
| `test_loop_cli_rollback_without_history` | 현재 모델 하나뿐인 등록부에서 `rollback` → 2, 등록부 불변 |
| `test_loop_cli_rollback` | 손으로 만든 등록부(v001 → v002 교체 이력, 도전자 v003) → `rollback` 뒤 `champion` v001, `history`의 마지막 `by`가 `rollback`, v002는 `retired`, 도전자는 v003 그대로 |
| `test_loop_cli_rollback_to` | 같은 등록부에서 `--to v003`(도전자) → 2, `--to v009`(없음) → 2, `--to v002`(현재 모델) → 2. 셋 다 등록부 불변 |
| `test_loop_cli_broken_registry` | 깨진 `registry.json`에서 `tick` → 2, 출력에 경로, 파일 불변 |

`tests/test_pipeline.py`(합성 데이터, `lightgbm`이 없으면 건너뜀):

| 테스트 | 확인하는 것 |
|---|---|
| `test_20_loop_replay_end_to_end` | `workspace["pre"]`의 8개(끝나는 날 09-01 ~ 09-08, 각 02:00 직전)로 `replay --start 2021-09-01 --interval-days 1 --initial-min-files 6 --train-min-new 1 --shadow-min-files 1 --seeds 42,43 --recipe-file <r.json>`을 `tmp_path`의 등록부에 돌린다. `r.json`은 `["--model", "ensemble", "--val-frac", "0.5", "--warmup-hours", "0.5", "--exclude-non-welding"]`. 기대: 09-07 `first:v001`(6건), 09-08 `train:v002`(7건), 09-09 판정(`swap:` 또는 `keep:`) + `train:v003`(8건). 등록부에 판정 1건이 있고 `values`에 아홉 키, `checks`에 다섯 키가 있으며 `auroc_a` · `auroc_b` · `noise`가 숫자다. `v002/seeds/s43`에 번들이 있다 |
| `test_20b_loop_replay_resume_skips_training` | 같은 명령을 다시 → 출력에 새 학습이 없고(`train.py`가 불리지 않음: 번들의 수정 시각이 그대로) 등록부가 같다 |
| `test_20c_loop_report` | `report --out <tmp>` → `report.json`의 `versions`가 3개이고 각각 테스트 `auroc_gun_mean`이 있다. `first_vs_final.first == "v001"`. `report.md`에 표 머리 넷이 있다 |

- [ ] **Step 2: `UNIT`과 `PIPE`로 실패를 확인한다.**
- [ ] **Step 3: 구현한다.**
- [ ] **Step 4: `UNIT`과 `PIPE` 통과를 확인한다.**
- [ ] **Step 5: `ruff` 확인, 스테이징 목록 보고.**

---

### Task 7: 서빙 서버

**Files:**
- Modify: `main.py` (모듈 docstring의 주소 목록, `lifespan`, `/health`, `predict` · `put_profile` · `forget_gun`, 새 주소 둘)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1의 `Registry`, `summary`.
- Produces:

```python
REGISTRY_ENV = "RSW_MODEL_REGISTRY"

class ModelGate:
    """/predict는 여럿이 같이 들어가고(shared), 교체는 혼자 들어간다(exclusive). 교체가 기다리는 동안 새 shared는 기다린다."""
    def shared(self) -> ContextManager[None]
    def exclusive(self) -> ContextManager[None]

def resolve_model(registry_dir: str | None, model_path: str) -> tuple[str, str | None]
    # (번들 경로, 버전 또는 None). 등록부가 있는데 현재 모델이 없거나 번들 파일이 없으면 경로를 담은 RuntimeError
def swap_model(app: FastAPI, new: Detector, version: str) -> None   # exclusive 안에서 부른다

@app.post("/model/reload")   def model_reload(request: Request) -> dict[str, Any]      # 동기 함수(스레드풀에서 돈다)
@app.get("/model/loop")      async def model_loop(request: Request, limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]
```

`app.state`에 더하는 것: `registry`(`Registry` 또는 None), `model_version`(str 또는 None), `model_gate`.

동작은 스펙 8.4절이다. 구현에서 지킬 것:

- `lifespan`: `RSW_MODEL_REGISTRY`가 있으면 `resolve_model`로 경로와 버전을 얻는다. `RSW_MODEL_PATH`도 있으면 경고를 남기고 등록부를 쓴다.
- `predict`, `put_profile`, `forget_gun`(설비 상태를 바꾸는 셋)의 본문을 `with request.app.state.model_gate.shared():`로 감싼다. 읽기만 하는 주소는 그대로 둔다. 교체 순간에 걸린 읽기 요청 하나가 옛 모델 정보와 새 설비 상태를 섞어 볼 수 있지만 상태를 바꾸지 않는다.
- `model_reload`: 등록부가 없으면 409. 등록부를 읽어 현재 모델이 `app.state.model_version`과 같으면 `{"changed": False, "version": ...}`. 다르면 gate 밖에서 `Detector(새 경로)`를 만든다. 등록부 읽기나 번들 읽기에서 예외가 나면 500이고 `detail`에 이유를 넣는다(옛 모델 유지). 그 뒤 `with gate.exclusive(): swap_model(...)`.
- `swap_model`: 설비마다 `GunState.restore(g.snapshot(gun_id, 옛 created), gun_norm=새 gun_norm 여부, model_created=새 created)`로 새 상태를 만든다. 새 `GunStore(state_dir, 새 created)`로 모두 저장한다. 새 상태를 다 만든 뒤 마지막에 `app.state.detector`, `store`, `guns`, `model_version`을 연달아 바꾼다(중간에 예외가 나면 아무것도 바뀌지 않게). `outbox`는 건드리지 않는다. 새 번들에 조기 경보가 없는데 `ew_handoffs`가 켜져 있으면 `lifespan`과 같은 경고를 남긴다.
- `GET /model/loop`: 등록부가 없으면 `{"registry": None}`. 있으면 `{"registry": 경로, "loaded": {"version", "created"}, "in_sync": loaded.version == champion.version, **summary(data, limit)}`. 등록부를 못 읽으면 500.
- `/health`에 `"model_version"`을 더한다.
- 두 주소를 모듈 docstring의 주소 목록에 더한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**

`tests/test_pipeline.py`. `ModelGate`는 `main.py`에 두고, 그 시험도 여기에 둔다(`load_main(workspace)`로 얻은 모듈의 `ModelGate`). 도우미 `loop_registry(workspace, tmp_path)`: `tmp_path/reg`에 등록부를 만들고, `workspace["models"]/baseline_iforest.joblib`을 `v001/b.joblib`으로 복사해 현재 모델로 등록한다. `v002/b.joblib`은 같은 번들을 `joblib.load`해 `created`만 `"2099-01-01T00:00:00"`으로 바꿔 저장하고 도전자로 등록한다. 도우미 `loop_main(workspace, reg_dir, monkeypatch)`: `RSW_MODEL_REGISTRY`를 설정하고 `load_main`처럼 `main`을 다시 불러온다.

| 테스트 | 확인하는 것 |
|---|---|
| `test_21_loop_gate_exclusive_waits_for_shared` | 스레드 A가 `shared()` 안에 있는 동안 스레드 B의 `exclusive()`가 들어가지 못하고(0.2초 뒤에도 플래그가 안 섬), A가 나오면 들어간다 |
| `test_21_loop_gate_shared_waits_for_exclusive` | `exclusive()` 안에서는 다른 스레드의 `shared()`가 기다린다. 두 `shared()`는 동시에 들어간다 |
| `test_21a_loop_server_reads_registry` | `/health`의 `model_version == "v001"`, `/model/loop`의 `loaded.version == champion.version == "v001"`, `in_sync` True, `challenger.version == "v002"`, `decisions == []` |
| `test_21b_loop_server_without_registry` | 환경 변수 없이 `load_main` → `/health`의 `model_version` None, `/model/loop == {"registry": None}`, `POST /model/reload` 409 |
| `test_21c_loop_reload_noop` | 현재 모델이 그대로일 때 reload → `{"changed": False, "version": "v001"}` |
| `test_21d_loop_reload_swaps` | 설비 하나에 합성 테스트 파일의 앞부분을 `stream_chunks`로 보내 워밍업을 끝낸다(`/guns`의 `gun_norm == "gun"`). 등록부의 현재 모델을 v002로 바꾸고 reload → `changed` True, `previous == "v001"`, `/health`의 `model.created == "2099-01-01T00:00:00"` · `model_version == "v002"`, `/model/loop`의 `in_sync` True. 그 설비의 `/guns` `gun_norm == "warming_up"`이고 `n_samples` 등 나머지는 재시작한 것과 같다: `main.app.state.guns[g].snapshot(g, new)`가 `GunState.restore(교체 전 snapshot, gun_norm, new).snapshot(g, new)`와 같다. 이어서 보낸 `/predict`가 200 또는 202다 |
| `test_21e_loop_reload_keeps_stats_and_rules` | 교체 전 `/guns/{id}/stats`의 합계와 프로파일(`PUT /guns/{id}/profile`)이 교체 뒤에도 같다 |
| `test_21f_loop_reload_bad_bundle_keeps_old` | v002의 번들을 0바이트 파일로 만들고 현재 모델로 바꾼 뒤 reload → 500. `/health`는 여전히 v001이고 `/predict`가 동작한다 |
| `test_21g_loop_reload_broken_registry_keeps_old_model` | `registry.json`을 깨뜨리고 reload → 500, `/health`는 v001, `/model/loop`는 500 |
| `test_21h_loop_registry_missing_bundle` | 현재 모델의 번들 파일을 지운 등록부로 서버를 시작하면 `RuntimeError`이고 메시지에 번들 경로 |
| `test_21i_loop_reload_under_load` | `with TestClient(main.app)` 하나를 스레드들이 같이 쓴다(이 환경에서 4개 스레드가 동시에 처리되는 것을 확인했다). 스레드 4개가 서로 다른 설비로 `/predict`를 60초치씩 20번 보내는 동안, 주 스레드가 등록부를 v002로 바꾸고 reload한다. 모든 응답이 200 또는 202이고 5xx가 없다. 끝난 뒤 `/health`가 v002이고 설비 4개가 모두 `app.state.guns`에 있다 |
| `test_21j_loop_reload_keeps_pending_handoffs` | `test_8_outbox_retry_restart_idempotent`가 하는 방식으로 치명 이벤트 하나를 만들어 `/handoffs`에 1건이 있게 한 뒤 교체 → `/handoffs`가 같은 `event_id` 1건을 그대로 돌려주고 `handoff.detector.created`는 옛 값이다 |

- [ ] **Step 2: `PIPE`로 실패를 확인한다.**
- [ ] **Step 3: 구현한다.**
- [ ] **Step 4: `PIPE` 통과를 확인한다.**
- [ ] **Step 5: `FULL`을 돌린다.** 기대: 기존 85개 + 새 테스트가 모두 통과. `ruff check .`의 오류가 기존 E402 하나뿐인지 본다. 스테이징 목록 보고.

---

### Task 8: 실제 데이터 실행과 문서

**Files:**
- Create: `results/D-11/runs/replay-1/report.json`, `report.md`, `run.log`
- Modify: `README.md`, `MEMORY.md`, `CLAUDE.md`, `techspec.md`, `history.md`

**Interfaces:**
- Consumes: Task 6의 CLI, Task 7의 서버.

실행은 포그라운드에서 한다. 한 명령이 10분에 가까워지면 끊고 같은 명령으로 이어 간다. 등록부는 `models/registry/`를 쓴다. 시작 전에 그 폴더가 없는지 확인한다(있으면 사용자에게 묻는다).

- [ ] **Step 1: 시간을 잰다.** `py -3.11 .py/model_loop.py replay --start 2021-08-16 --end 2021-09-07`(v001만 생긴다)의 걸린 시간을 적는다. 이어서 `--end 2021-09-14`(v002 + 시드 모델 둘)의 걸린 시간을 적는다. 버전 모델은 파일 수에 비례한다고 보고 전체 시간을 어림한다(버전 16 · 33 · 40 · 45 · 60건, 시드 모델 8개).
- [ ] **Step 2: 어림한 전체가 1시간 30분을 넘으면 멈추고 사용자에게 보고한다**(스펙 10절). 넘지 않으면 계속한다.
- [ ] **Step 3: 재생을 끝까지 돌린다.** `py -3.11 .py/model_loop.py replay --start 2021-08-16`. 끊기면 같은 명령을 다시 부른다. 출력은 `results/D-11/runs/replay-1/run.log`에 이어 붙인다. 점검별 행동이 스펙 3절 표의 날짜 · 파일 수와 같은지 본다. 다르면 고치지 말고 차이를 보고한다(표는 실행 전 예상이다).
- [ ] **Step 4: 보고서를 만든다.** `py -3.11 .py/model_loop.py report --out results/D-11/runs/replay-1`. 판정마다 아홉 값과 다섯 조건이 채워졌는지, 테스트 8건 표에 모든 버전이 있는지 확인한다.
- [ ] **Step 5: 서버 교체를 실제 번들로 확인한다(성공 기준 2).** 보고서를 만든 뒤에 한다. 실제 등록부의 이력을 건드리지 않도록 `models/registry/`를 `models/registry-check/`로 통째로 복사해 그 사본으로 한다.
  1. `RSW_MODEL_REGISTRY=models/registry-check`와 임시 `RSW_STATE_DIR`로 서버를 켠다. `/health`의 `model_version`이 사본의 현재 모델인지 본다.
  2. `py -3.11 .py/replay.py`로 테스트 파일 하나의 앞 1시간을 보낸다.
  3. 다른 버전으로 바꾼다: `py -3.11 .py/model_loop.py rollback --registry models/registry-check --to <현재 모델도 도전자도 아닌 버전> --server-url http://localhost:8000`.
  4. 서버를 다시 시작하지 않은 채 `/health`와 `/model/loop`가 새 버전을 보이고 `in_sync`가 true인지, 이어서 보낸 `/predict`가 동작하는지 본다.
  5. 서버를 끄고 `models/registry-check/`와 임시 상태 폴더를 지운다.

  판정이 일으키는 교체와 `rollback`은 같은 경로(`set_champion` → reload 호출)를 쓴다. 판정 쪽 호출은 Task 3의 `test_loop_swap_changes_champion`이 본다.
- [ ] **Step 6: 문서를 고친다.**

| 문서 | 내용 |
|---|---|
| `history.md` | 작업 ID 표에 D-11, 날짜 표에 한 줄, 새 절 `### 50. D-11 모델 루프`. 배경(사용자 결정 11-11), 한 일, 재생 결과(점검 표, 판정 세 건의 수치와 결과, 테스트 8건 표), 실행 시간, 한계(스펙 13절), 시험 수 |
| `README.md` | 파일 구조 표에 `model_registry.py` · `model_loop.py`, 실행 절에 `replay` · `tick` · `report` · `status` · `rollback`, `preprocess.py --split append`, `RSW_MODEL_REGISTRY`, `POST /model/reload`, `GET /model/loop`, 스케줄러 예시(스펙 8.5절의 두 명령), 시험 수 |
| `MEMORY.md` | 설계 결정 절에 D-11(섀도 비교, 다섯 조건, 테스트 8건은 최종 확인만, 표준화 값 고정)과 변경 이력 한 줄 |
| `CLAUDE.md` | 명령에 `model_loop.py`, 시험 수, 레이아웃 줄 |
| `techspec.md` | D-11 줄(구현됨)과 후속 과제 넷(스펙 13절) |

결과 수치는 `report.json`에서 그대로 옮긴다. 판정이 모두 유지로 끝났으면 그렇게 적는다. 좋아졌다고 쓰려면 테스트 8건 표의 수치를 같이 적는다.

- [ ] **Step 7: `FULL`과 `ruff check .`를 마지막으로 돌리고, 스테이징할 파일 전체 목록과 재생 결과 요약을 보고한다.**

---

## 스펙 대조

| 스펙 | 작업 |
|---|---|
| 3절 점검 규칙, 기본값, 예상 표 | Task 3 (`test_loop_spec_timeline`) |
| 4절 학습(레시피, 시드 모델) | Task 2 (`DEFAULT_RECIPE`), Task 6 (`seed_recipe`, `make_train_fn`) |
| 5절 판정 | Task 2 |
| 6절 되돌리기 | Task 6 (`rollback`) |
| 7절 테스트 8건 | Task 6 (`report`), Task 8 |
| 8.1절 등록부 | Task 1 |
| 8.2절 실행기 | Task 3, Task 6 |
| 8.3절 학습 스크립트 | Task 4 |
| 8.4절 서버 | Task 7 |
| 8.5절 새 파일 추가 | Task 5 |
| 9절 실패 처리 | Task 1 (잠금, 깨진 파일), Task 3 (학습 · 채점 · 알림 실패, 시간 역행), Task 6 (하위 프로세스 실패), Task 7 (번들 · 등록부 읽기 실패) |
| 10절 실행 시간 | Task 8 Step 1 ~ 2 |
| 11절 테스트 | 각 작업의 표, Task 8 Step 3 ~ 5 |
| 12절 문서 | Task 8 Step 6 |
| 성공 기준 1 ~ 4 | Task 8 Step 3 ~ 4, Step 5, Task 3 (`same_now`, `replay_resumes`), Task 7 (`without_registry`) + `FULL` |
