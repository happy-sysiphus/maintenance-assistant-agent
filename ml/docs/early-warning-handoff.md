# 조기 경보 핸드오프 안내 (api·RAG 담당용)

2026-11-01 · ML (C-8, 핸드오프 v1.3)

요약: ML이 몇십 분 전 **조기 경보**를 낸다.
- **critical 이벤트와 구분**: 핸드오프로 보낼 때는 `event_type: "early_warning"`을 단다. 종료 코드와 고장 유형이 없다.
- **ML 기본값은 보내지 않음**: 서버를 `RSW_EARLY_WARNING_HANDOFFS=1`로 켜야 나간다.
- **api를 고치기 전에는 켜지 말 것**: 지금 api는 받은 핸드오프를 모두 "긴급" 케이스로 만든다.

ML 쪽 코드만 바뀌었다. api·RAG·UI는 손대지 않았다.

**모델 파일**: 조기 경보는 2026-11-01 번들에만 있다(`GET /health`의 `model.created`가 `2026-11-01T18:46:47`, `early_warning.model`이 `true`). 10/06 번들도 그대로 돌아가지만 조기 경보가 없다. 고장 감지 결과는 두 번들이 같다.

| 항목 | 내용 | 절 |
|---|---|---|
| 무엇인가 | 정밀도가 낮은 "수십 분 안 고장 가능성" 표시 | 1 |
| 언제 오나 | 에피소드 시작에 1회, 건별 60분 쿨다운, critical·종료 코드가 있으면 안 옴 | 2 |
| 어떻게 구분하나 | `event_type`, `trigger.source`, `early_warning` 블록 | 3 |
| 지금 api가 그대로 받으면 | 긴급으로 표시되고, 코드 없는 케이스로 묶이며, RAG 안내가 없음 | 4 |
| 얼마나 오나 | 건당 하루 ~16회 켜짐, 보내는 것은 그중 일부(쿨다운) | 5 |

## 1. 무엇인가

- 종료 코드가 뜨기 10–60분 전에 경보를 켜는 것이 목표다. 오프라인 평가(B-20)는 아래와 같다.
  - 교차검증 8회 평균: 고장의 49%에서 그 구간에 켜졌다.
  - 같은 시각에 고장이 없는 날에도 34%에서 켜졌다.
  - 테스트 8건 중 4건에서 켜졌다.
- 사용자 결정으로 **미탐 우선**이다. 오경보를 많이 감수한다. 경보 하나하나를 믿을 신호가 아니라 "위험 상승" 표시로 써야 한다.
- 고장 유형은 모른다. 종료 코드가 아직 없기 때문이다.
  - 건 프로파일(`PUT /guns/{id}/profile`)이 있으면 그 유형을 **후보**로만 싣는다.
  - 데이터상 고장 유형별 전조는 없었다(B-17).

## 2. 언제 오나

- 조기 경보 에피소드가 시작된 `/predict` 요청에서 한 번 온다(`early_warning_onset=true`).
- 같은 건은 마지막 조기 경보 핸드오프 뒤 60분 동안 다시 보내지 않는다. 쿨다운은 번들 `early_warning.cooldown_s`이고 재시작해도 유지된다.
- 다음 경우에는 보내지 않는다. 사유는 응답의 `early_warning_handoff_suppressed`에 남는다.
  - `critical`: 같은 요청에 critical 이벤트(모델 10분 지속, 종료 코드 규칙)가 있거나, 직전 요청이 critical이었다.
  - `terminal_code`: 요청 안에 종료 코드가 있었다(반복·프로파일 밖 발화 포함).
  - `cooldown`: 60분 안에 이미 보냈다.
  - `handoffs_off`: 플래그가 꺼져 있다.
- 전달 방식은 critical과 같다: 응답의 `handoff`, `GET /handoffs`, `RSW_RAG_URL` push(`Idempotency-Key`).
  - 조기 경보만 받으려면 `GET /handoffs?event_type=early_warning`을 쓴다.
  - 멱등 키는 critical과 겹치지 않는다.

## 3. 문서 (`RagHandoff` v1.3, 예시 `docs/rag_handoff_early_warning.example.json`)

| 필드 | critical | early_warning |
|---|---|---|
| `schema_version` | `"1.3"` | `"1.3"` |
| `event_type` | `"critical"` | `"early_warning"` |
| `trigger.source` | `model` / `rule` / `model+rule` | `"early_warning"` |
| `trigger.rule_code`, `rule_trigger_time` | 규칙 발화면 코드 | 항상 `null` |
| `fault_class` | 규칙 또는 최근 코드의 유형 | 항상 `null` |
| `summary_ko` | 센서 소견 + 유형 | "조기 경보: 수십 분 안 고장 가능성(확정 아님) - …"로 시작 |
| `caveats` | critical용 | 조기 경보용(정밀도·후보 안내) |
| `early_warning` | `null` | 아래 표 |

`early_warning` 블록:

| 키 | 뜻 |
|---|---|
| `since` / `onset_at` | 조건이 처음 맞은 창 / 경보가 켜진 창 끝(UTC, Z) |
| `active_min` | `since`부터 `detected_at`까지 분 |
| `rule`, `sustain_windows`, `threshold`, `quantile`, `score` | 판정 규칙(`model_ng_k1`: 점수 ≥ 임계값인 창 하나)과 그 값 |
| `horizon_min` | `[10, 60]`: 무엇을 경고하나(종료 코드 10–60분 전) |
| `expected` | 오프라인 정확도: recall 0.488, chance 0.343, 오경보 16.4회/건·일(B-20) |
| `candidate_fault_classes`, `candidate_terminal_codes` | 건 프로파일이 있으면 그 유형과 종료 코드(예: `["E04"]`, `["E029"]`), 없으면 `null` |
| `note_ko` | 화면용 한 줄 |

## 4. 지금 api가 그대로 받으면 (플래그를 켰을 때)

`api/main.py ingest` 기준이다. 이 동작을 바꿀지는 api 담당이 정한다.
- `severity`가 `"critical"`로 고정이라 화면에 "긴급"으로 나온다.
- 케이스 묶음 키가 `(gun_id, trigger.rule_code)`다. 조기 경보는 `rule_code`가 `null`이라 같은 건의 모델 단독 critical 케이스(역시 `null`)와 같은 열린 케이스로 묶일 수 있다.
- RAG 키는 `trigger.rule_code` → `fault_class.terminal_code`다. 둘 다 `null`이라 `guidance`가 `no_evidence`가 된다.
- 인박스는 코드가 없는 행을 점선 "확정 아님"으로 보여 준다. 이 표시는 조기 경보에도 맞는다.

구분해서 처리하려면 `event_type`을 읽으면 된다. 예: 별도 등급("주의"), 별도 케이스 묶음, 후보 코드로 RAG 조회 여부. 어떻게 할지는 api·RAG 쪽 결정이다.

## 5. 얼마나 오나

- 오프라인 기준으로 조기 경보 에피소드는 건당 하루 ~16회 시작한다(교차검증 8회 평균). 테스트 8건에서는 7–9회였다.
- 실제로 나가는 핸드오프는 쿨다운(60분)과 종료 코드 억제를 거친 수다. 고장 1시간 전보다 앞선 평소 운전 기준으로 건당 하루 **~4.5–6건**이다.
  - 테스트 8건 4.5건, 검증 14건 6.0건, 교차검증 64건 5.7건이다(`models/baseline_ensemble_metrics.json`, `metrics.*.early_warning.handoff_onsets_per_gun_day`).
  - 건 사이 차이가 크다. 테스트에서 7일 동안 test_5는 4건, test_1은 60건이었다.
- 서버의 `GET /guns/{id}/stats`에 `early_warnings`(시작 수)와 `early_warning_handoffs`(보낸 수)가 날짜별로 쌓인다.

## 6. HITL 리허설

`docs/hitl-loop.md`의 ③ 표는 플래그를 끈 상태 기준이다. `.py/hitl_loop.py`는 조기 경보 핸드오프를 건너뛴다.
