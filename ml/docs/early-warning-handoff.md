# 조기 경보 핸드오프 안내 (api·RAG 담당용)

2026-11-01 · ML (C-8, 핸드오프 v1.4)

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

## 3. 문서 (`RagHandoff` v1.4, 예시 `docs/rag_handoff_early_warning.example.json`)

| 필드 | critical | early_warning |
|---|---|---|
| `schema_version` | `"1.4"` | `"1.4"` |
| `event_type` | `"critical"` | `"early_warning"` |
| `trigger.source` | `model` / `rule` / `model+rule` | `"early_warning"` |
| `trigger.rule_code`, `rule_trigger_time` | 규칙 발화면 코드 | 항상 `null` |
| `fault_class` | 규칙 또는 최근 코드의 유형 | 항상 `null` |
| `summary_ko` | 센서 소견 + 유형 | "조기 경보: 수십 분 안 고장 가능성(확정 아님) - …"로 시작 |
| `caveats` | critical용 | 조기 경보용(정밀도·후보 안내) |
| `early_warning` | `null` | 아래 표 |
| `contributing_features` | 이벤트 1분의 상위 피처 최대 3개(시각 피처 제외) | 같음 |
| `sensor_features` | 이벤트 1분의 초당 원시값 | 같음 |

`early_warning` 블록:

| 키 | 뜻 |
|---|---|
| `since` / `onset_at` | 조건이 처음 맞은 창 / 경보가 켜진 창 끝(UTC, Z). 한 요청이 윈도우를 둘 이상 채점하면 `since`는 최신 윈도우 기준이라 `onset_at`과 같거나 더 늦을 수 있다 |
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
- 인박스는 코드가 없는 행을 점선 "모델 신호만"으로 보여 준다. 이 표시는 조기 경보에도 맞는다.

구분해서 처리하려면 `event_type`을 읽으면 된다. 예: 별도 등급("주의"), 별도 케이스 묶음, 후보 코드로 RAG 조회 여부. 어떻게 할지는 api·RAG 쪽 결정이다.

v1.4의 두 필드(`contributing_features`, `sensor_features`)에 필요한 일이다. 조기경보와 치명 두 이벤트에 모두 해당한다(2026-11-06, 설계 `docs/superpowers/specs/2026-11-06-handoff-v1-4-design.md` 12절). ML 구현은 이 일을 기다리지 않는다. 다만 끝나기 전에는 두 필드를 읽는 곳이 없다.

- RAG
  - `handoff.sensor_features`를 읽는다. `/predict` 최상위의 v4 `sensor_features`와 이름만 같고 내용이 다르다.
  - 코드가 있는 치명(`trigger.rule_code` 또는 `fault_class.terminal_code`)은 지금도 RAG를 호출하므로 두 필드를 가장 먼저 쓸 수 있는 경로다. 둘 다 없는 모델 단독 치명은 조기경보처럼 RAG를 호출하지 않는다.
  - 상위 3개는 합의한 영향이 가장 큰 3개에서 시각 피처를 뺀 목록이다. 조기경보 244건 중 169건, 치명 20건 중 3건(`replay.py` 구간 기준)에서 모델의 실제 상위 3개와 다르다.
  - 별도 입력이 필요하다. 블록과 상위 3개를 합치면 경보 244건과 치명 20건 모두 질문 제한 2,000자(`rag/service.py:105`)를 넘는다.
  - 시스템 프롬프트의 "입력에는 센서가 없습니다"(`rag/service.py:24`)를 바꿔야 한다.
  - 캐시 키(`rag/service.py:86`)에 `event_id`나 입력 해시를 넣어야 한다. 없으면 다른 분의 답이 재사용된다.
  - 질문마다 `event_id`, 프롬프트 버전, 문서 fingerprint, 모델, 실제 입력을 남긴다. 피드백 쪽에서 LLM이 본 것을 확인하기 위해서다.
  - LLM에 어떤 필드를 어떤 형태로 넘길지 정한다. 읽기 시험에서 gpt-5.4-mini가 질문 20개 중 12개만 맞혔고, `summary_ko`의 c10 문구는 블록과 어긋날 수 있고, 청크 요청에서는 블록과 `sensor_findings`의 구간이 다르고, 치명에서는 코드가 뜬 초가 블록 밖일 수 있다.
  - 프롬프트 시험에는 오경보 분을 포함한다. test_5 02:00은 실제 고장 전 경보 4건 중 하나라서, 이것만으로 맞추면 안 된다.
  - 피드백 쪽 기계 검사 후보: LLM이 인용한 숫자는 핸드오프나 그것에서 코드로 계산한 값이어야 하고, "평소와 다름"은 `sensor_findings` 항목에만 쓸 수 있다.
  - 설명은 `docs/handoff_sensor_features_for_rag.md` 한 곳에 있다.
- api
  - 두 필드를 RAG로 넘긴다. 지금 `enrich`는 코드만 넘긴다(`api/guidance.py:91`). RAG가 별도 입력을 정하면 `case["event"]`의 두 필드를 그 입력으로 넘긴다.
  - `event_type`을 읽는다. 지금은 `severity`가 `critical`로 고정이다.
  - 조기경보에는 E 코드가 없어 `enrich`가 RAG를 호출하지 않는다(`api/guidance.py:19-20`, `:80-91`).
  - 조기경보는 코드가 `null`인 열린 케이스에 합쳐지고, 이때 `event_id`와 `detected_at`만 남아 블록이 버려진다(`api/main.py:110-112`). 같은 건, 같은 코드의 치명도 열린 케이스가 있으면 마찬가지다.
  - 케이스가 2시간, 4시간, 8시간, 24시간 뒤에 닫힌다고 가정하면 조기경보 블록이 남는 비율은 69%, 52%, 37%, 18%다. 치명 20건은 같은 가정에서 모두 남는다. 코드로 발생한 치명은 같은 코드가 24시간 안에 다시 뜨면 반복이라 치명이 아니고, 모델 단독 1건은 `rule_code`가 없어 코드 케이스에 합쳐지지 않는다.
  - 조기경보 핸드오프는 api가 처리할 때까지 꺼 둔다(`RSW_EARLY_WARNING_HANDOFFS`). 치명 핸드오프는 플래그와 상관없이 두 필드를 싣는다. api는 받은 문서를 그대로 저장하므로 크기만 늘고 동작은 바뀌지 않는다.
  - 줄 번호는 origin/main 기준이다.

## 5. 얼마나 오나

- 오프라인 기준으로 조기 경보 에피소드는 건당 하루 ~16회 시작한다(교차검증 8회 평균). 테스트 8건에서는 7–9회였다.
- 실제로 나가는 핸드오프는 쿨다운(60분)과 종료 코드 억제를 거친 수다. 고장 1시간 전보다 앞선 평소 운전 기준으로 건당 하루 **~4.5–6건**이다.
  - 테스트 8건 4.5건, 검증 14건 6.0건, 교차검증 64건 5.7건이다(`models/baseline_ensemble_metrics.json`, `metrics.*.early_warning.handoff_onsets_per_gun_day`).
  - 건 사이 차이가 크다. 테스트에서 7일 동안 test_5는 4건, test_1은 60건이었다.
- 서버의 `GET /guns/{id}/stats`에 `early_warnings`(시작 수)와 `early_warning_handoffs`(보낸 수)가 날짜별로 쌓인다.
