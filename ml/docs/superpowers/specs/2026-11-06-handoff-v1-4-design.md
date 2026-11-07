# 핸드오프 v1.4: 이벤트 1분 구간 설계

2026-11-06 · ML · 상태: 사용자 승인 (11-06)

## 1. 목표

11-04 회의와 RAG 담당자 합의(11-06)를 ML 핸드오프에 반영한다.

- 조기경보와 치명 핸드오프에 똑같이 **시각 피처를 뺀 상위 피처 3개**와 **그 1분의 19개 채널 초당 값 전부**를 싣는다.
- 이벤트 종류와 상관없이 같은 내용을 보낸다(11-06 사용자 확인).
- 추후 LLM이 하는 RAG 매핑의 입력으로 쓴다. 할루시네이션과 장비 불일치는 피드백 루프가 잡는다.
- v4 문서는 쓰지 않고 새 블록을 만든다.

성공 기준:

1. 조기경보 핸드오프에는 경보를 낸 1분, 치명 핸드오프에는 3절 표의 1분의 두 필드가 실린다. 코드가 뜬 분이 끝나기 전에 나가는 치명은 코드가 없는 직전 1분이다.
2. 기존 필드의 의미와 `event_id`는 바뀌지 않는다.
3. 새 코드가 실패해도 이벤트는 사라지지 않는다.
4. `pytest`가 통과하고 `ruff check .`에 새 오류가 없다. `tests/test_sensor_features.py:14`의 E402는 기존 오류다.

범위 밖:

- api, rag, ui 코드. 필요한 일은 12절에 요청으로 적는다.
- v4 삭제, c10 문구 수정, 조기경보 규칙 변경. 14절에 남긴다.

## 2. 핸드오프 변경

`schema_version`을 `"1.4"`로 올리고 `RagHandoff`에 최상위 필드 두 개를 추가한다. 조기경보와 치명이 같은 형식이다.

| 필드 | 내용 |
|---|---|
| `contributing_features` | 상위 피처 최대 3개 |
| `sensor_features` | 그 1분의 원시값 |

두 이벤트의 차이는 기존 필드에 있다. 코드로 발생한 치명에는 `trigger.rule_code`와 `fault_class`(`basis: rule_trigger`)가 있다. 모델 단독 치명은 `rule_code`가 없고, 최신 윈도우의 마지막 코드가 건 프로파일 밖이 아닌 종료 코드이면 `fault_class`만 있다(`basis: latest_error_code`). 8개 파일의 모델 단독 치명 1건이 이 경우다. 조기경보에는 둘 다 없고, 조기경보만 `early_warning` 블록을 갖는다.

`handoff.sensor_features`는 /predict 최상위의 `sensor_features`(v4)와 이름만 같고 내용이 다르다. v4 삭제는 14절에서 따로 정한다.

아래 `sensor_features`는 `ml/test/test_5.csv` 2021-10-03T02:00의 실제 값이다. c19만 줄였다. `contributing_features`의 항목은 형식을 보여 주는 실제 예시이며(`docs/anomaly_result.example.json`), 이 분의 값은 아니다.

```json
{
  "schema_version": "1.4",
  "event_type": "early_warning",
  "early_warning": {"onset_at": "2021-10-03T02:00:59Z", "...": "기존과 같음"},
  "contributing_features": [
    {"feature": "c10_std", "sensor": "c10", "sensor_name": "US2", "sensor_name_ko": "US2 작동 허용 신호",
     "statistic": "std", "value": 0.5030977725982666, "reference": 0.0,
     "contribution": 0.0005980723090450724, "share": 0.01494375985182729}
  ],
  "sensor_features": {
    "window_start": "2021-10-03T02:00:00Z",
    "window_end": "2021-10-03T02:00:59Z",
    "missing_s": [],
    "channels": {
      "c1": [[0,1.2]], "c2": [[0,0]], "c3": [[0,158.1]], "c4": [[0,0]],
      "c5": [[0,-1367],[31,-1360],[35,-1354],[40,-1360],[42,-1349],[43,-1107],[44,-1225],[45,-1247],[46,-1266],[47,-1276],[49,-1265],[50,-1273],[51,-1271],[52,-1262],[53,-1267],[54,-1259],[55,-1258],[56,-1263],[57,-1256],[58,-1255],[59,-1254]],
      "c6": [[0,-21]], "c7": [[0,157.8632812]], "c8": [[0,5916]], "c9": [[0,137]],
      "c10": [[0,0],[40,1]], "c11": [[0,34287]], "c12": [[0,23328014],[41,23328016]],
      "c13": [[0,14.991055]], "c14": [[0,1000]], "c15": [[0,155],[16,158.1]], "c16": [[0,3.4]], "c17": [[0,1]], "c18": [[0,0]],
      "c19": [[0,1155316089],[1,1155409046],[2,1155316188], "... 60개 항목"]
    },
    "error": [[0,"0"],[41,"E006"]]
  }
}
```

## 3. 어느 1분을 담나

두 필드는 아래 표의 윈도우 하나를 담는다.

| 이벤트 | 담는 윈도우 |
|---|---|
| 조기경보 | 요청 안에서 `ew_onset`이 처음 참인 판정의 윈도우 |
| 코드로 발생한 치명 | 코드가 뜬 초(`rule_trigger_time`)가 들어 있는 윈도우. 그 분이 요청 안에서 채점되지 않았으면 그 뒤 첫 윈도우, 그것도 없으면 최신 윈도우 |
| 모델 단독 치명 | 요청 안에서 `critical`이 처음 참인 판정의 윈도우 |

- 조기경보에서 `sensor_features.window_end`는 항상 `early_warning.onset_at`과 같다.
- `detected_at`, 최상위 `window_start`, `trigger`의 점수, `sensor_findings`, `sensor_trend`, `context`, `early_warning.score`는 지금처럼 최신 윈도우를 가리킨다. 기존 의미를 바꾸지 않기 위해서다.
- 요청이 윈도우 하나만 채점하면(`windows_scored == 1`) 두 구간이 같다. 1분 단위 클라이언트는 거의 항상 이 경우다.
- 여러 분을 한 번에 보내면(`--chunk-s 600`) 다를 수 있다. 1분 단위라도 빈 구간 뒤에는 윈도우 둘이 함께 채점될 수 있다. 이때 최신 윈도우는 이벤트의 분이 아닐 수 있고, `early_warning.since`는 더 늦은 경보 구간을 가리킬 수 있다.

**코드가 뜬 분이 아직 끝나지 않은 경우**

코드가 분 중간에 뜨고 요청이 그 분 안에서 끝나면, 그 분은 아직 채점되지 않았고 모델의 상위 피처도 없다. 치명은 그 요청에서 바로 나가므로 분이 끝나기를 기다리지 않는다. 이때 두 필드는 직전에 완성된 1분을 담고, 코드가 뜬 초는 블록 밖에 있다(`rule_trigger_time > window_end`). 코드와 시각은 `trigger`와 `error_timeline`에 있다. 지금도 치명 핸드오프의 `detected_at`은 같은 분을 가리킨다(`test_4b_handoff_timeline_keeps_the_triggering_code`). 이 처리는 사용자가 11-06에 승인했다. 얼마나 자주 생기는지는 전송 시점에 달렸다(13절).

**코드가 뜬 분이 끝내 채점되지 않는 경우**

행이 30개 미만인 분은 채점되지 않는다. 긴 공백 바로 옆의 분이나 재시작으로 행을 잃은 분이 여기에 든다. 이때 두 필드는 그 뒤 첫 윈도우를 담고, 코드가 뜬 초는 블록 앞에 있다(`rule_trigger_time < window_start`). 202로 보류된 코드라도 그 분에 행이 30개 이상 있으면 나중에 채점되어 블록 안에 있다. 8개 테스트 파일의 코드 치명 19건에서는 생기지 않았다.

## 4. `sensor_features` 규칙

| 필드 | 형식 | 뜻 |
|---|---|---|
| `window_start`, `window_end` | UTC `Z` 문자열 | 실제 윈도우 경계. 항상 :00~:59는 아니다 |
| `missing_s` | `list[int]` | 윈도우 안에서 실제 행이 없는 초 |
| `channels` | `{c1~c19: [[초, 값], ...]}` | 19개 채널. 키는 항상 19개다 |
| `error` | `[[초, 코드], ...]` | 컨트롤러 코드. `"0"`은 코드 없음. 19개 채널과 별도다 |

**행 선택**

- `g.rows`에서 `[window_start, window_end]` 안의 행을 시간순으로 정렬하고, 같은 초는 마지막 도착 행만 남긴다. `features()`와 같은 규칙이다.
- trace의 `raw`는 쓰지 않는다. trace는 윈도우를 건너뛸 수 있다.
- 채점 시점에 도착한 행만 담는다. 늦게 온 행이나 재시작으로 잃은 행은 들어갈 수 없다.
- 실제 행만 보낸다. 빈 초를 앞 값으로 채운 값은 보내지 않는다.
- 값은 받은 그대로라서 모델이 본 값과 다를 수 있다. 모델은 빈 초를 앞 값으로 채우고, `c16 <= 0`인 초는 직전 용접 행의 값으로 대체해서 본다.

**인코딩**

- 채널마다 첫 행과 값이 바뀐 초에만 항목을 만든다. 값은 다음 항목까지 유지되고, `missing_s`의 초에는 값이 없다.
- 초는 시계 초(0~59)다. 윈도우는 한 시계 분 안에 있으므로 유일하고, `error_timeline` 시각과 바로 맞출 수 있다.
- 값은 받은 그대로이고 반올림하지 않는다. 정수로 떨어지는 실수는 정수로 쓴다(`2500.0`은 `2500`). c10은 1/0이다.
- 모든 채널이 같은 형식이다. c8, c9, c13, c17, c19는 데이터가 채널 종류 구분과 어긋나므로 종류로 분기하지 않는다.
- 통계, 개수, 편차, 설명 문구는 넣지 않는다. 설명은 9절의 문서 한 곳에 둔다.
- 행이 0개인 윈도우는 19개 키 모두 `[]`, `error: []`, `missing_s`는 윈도우 전체다. 8개 테스트 파일에서는 생기지 않았다.

**이 형식을 고른 이유**

- 60개 배열은 채널만 중앙값 약 6.0 KB, 같은 봉투를 씌우면 약 6.4 KB다. 변화점 형식은 실제 경보 중앙값이 3.4 KB다.
- 변화점 형식은 8개 파일의 80,553분을 되돌려 봐도 불일치가 0건이었다. 빠진 초가 있는 9,971분을 포함한다. 서버 코드가 아니라 같은 로직의 복제본으로 확인했다.
- 11-06에 사용자가 초당 값 전부와 이 형식으로 진행하기로 했다.

## 5. `contributing_features` 규칙

- 타입은 `list[FeatureContribution]`이다. /predict와 같은 9개 필드, 같은 숫자이고 반올림하지 않는다.
- 3절 윈도우의 벡터로 부른 `d.score(vec)` 결과에서 `contribution > 0`이고 `sensor`가 `rag_mapping.NOT_A_SYMPTOM`(`hour_sin`, `hour_cos`)에 없는 항목의 앞 3개다. 3개보다 적을 수 있다.
- 시각 피처만 뺀다. `pos_delta`, `welds_*`, `weld_duty_10min`, `error_share_10min`은 남기고, 한 신호의 mean과 std가 둘 다 들어갈 수 있다. 따라서 `sensor`가 항상 `channels`의 키는 아니다.
- 요청이 윈도우 하나만 채점하면 /predict `contributing_features`에 같은 필터를 적용한 앞 3개와 같다. /predict의 목록은 지금처럼 최신 윈도우의 상위 8개다.

**시각 피처를 빼는 이유**

`hour_sin`, `hour_cos`는 센서가 아니라 타임스탬프로 계산한 값이다. 블록에 대응하는 채널이 없고, 매뉴얼에 연결할 수 없고, 언제인지를 말할 뿐 건의 상태를 말하지 않는다. `sensor_findings`와 trace도 같은 이유로 뺀다(`rag_mapping.py:39-40`, `main.py:108`).

다만 모델은 시각에 꽤 기댄다. 경보 244건 중 169건에서 시각 피처가 상위 3위 안에 들고 62건에서 1위다. 따라서 이 목록은 모델의 상위 3개 그대로가 아니라 시각을 뺀 상위 3개다. 세 피처의 `share` 합은 중앙값 0.585이고, 시각을 뺄 때 사라지는 몫은 이 합이 작아지는 것으로만 드러난다(빼지 않으면 0.730).

## 6. 실패 처리

| 상황 | 결과 |
|---|---|
| 윈도우 선택 실패 | 최신 윈도우로 대체, 이벤트 유지, 로그 |
| 블록 생성 실패 | `sensor_features = null`, 상위 피처는 유지, 이벤트 유지, 로그 |
| 상위 피처 계산 실패 | `contributing_features = []`, 블록은 유지, 이벤트 유지, 로그 |
| 상위 피처 숫자에 NaN 또는 inf | 위와 같음 |
| `error_timeline`, `sensor_trend` 실패 | 지금과 같음. 새 필드에 영향 없음 |

- 새 코드는 기존 try와 분리해 각자의 try에서 돈다. `critical_window`는 `attach_handoff`의 try보다 먼저 인자로 평가되므로 자기 안에 try를 둔다. 여기서 500이 나면 규칙 상태가 이미 넘어가서 재전송으로도 이벤트가 돌아오지 않는다. `RagHandoff(**doc)`는 바깥 try에 있어 잘못된 블록이 닿으면 이벤트를 잃으므로, `HandoffSensorFeatures`로 먼저 검증한 뒤 `doc`에 넣는다.
- 센서값은 입력 단계에서 NaN과 inf가 거부되고(`main.py:148`) 블록은 계산을 하지 않으므로 변환이 필요 없다.
- 상위 피처는 `json.dumps(..., allow_nan=False)`로 검사한다. `FeatureContribution.value`는 필수 float라 null이 될 수 없고, 아웃박스에 NaN이 저장되면 `GET /handoffs` 전체가 500이 된다. `contribution`만 NaN이면 필터에서 그 항목만 빠지고 나머지는 남는다.
- 이 검사는 새 필드만 보호한다. 최신 윈도우의 벡터에 NaN이 있으면 /predict 응답과 `sensor_findings`에도 들어가고, 그 경로는 지금과 같다. 입력이 float32 범위 안에서 유한한 동안 벡터에 NaN이 생기는 경로는 지금 없다. 범위를 넘는 값은 핸드오프 전에 실패하는 기존 동작이고, 바꾸지 않는다.
- 새 필드는 기본값이 있어야 한다. 없으면 `Outbox.load`가 저장된 v1.3 레코드를 건너뛴다.

읽는 쪽은 버전이 아니라 값으로 판단해야 한다. 저장된 v1.3 레코드는 `schema_version "1.3"`인 채 두 키가 기본값으로 붙고, 실패한 경우처럼 `sensor_features`가 `null`일 수 있다.

## 7. 구현 위치

추가되는 운영 코드는 약 60줄이다. `build_handoff`는 버전 외에 바뀌지 않고, `/handoffs/preview`는 기본값을 유지한다.

**`ml/.py/rag_mapping.py`**

```python
SCHEMA_VERSION = "1.4"


def _plain(v: float) -> int | float:
    return int(v) if float(v).is_integer() else v  # int.is_integer needs Python 3.12


def sensor_block(rows, start, end, channels) -> dict[str, Any]:
    """rows in g.rows form: time a naive UTC datetime, c1-c19 numbers (c10 1.0 / 0.0), error a string."""
    by_t = {r["time"]: r for r in rows if start <= r["time"] <= end}  # arrival order: a resend replaces, as features()
    ts = sorted(by_t)

    def changes(key, conv=lambda v: v):
        out = []
        for t in ts:
            v = conv(by_t[t][key])
            if not out or out[-1][1] != v:
                out.append([t.second, v])
        return out

    return {"window_start": utc_iso(start), "window_end": utc_iso(end),
            "missing_s": sorted(set(range(start.second, end.second + 1)) - {t.second for t in ts}),
            "channels": {c: changes(c, _plain) for c in channels}, "error": changes("error")}
```

**`ml/main.py`**

```python
HANDOFF_TOP_FEATURES = 3


class HandoffSensorFeatures(BaseModel):  # before RagHandoff
    window_start: str
    window_end: str
    missing_s: list[int]
    channels: dict[str, list[tuple[int, int | float]]]
    error: list[tuple[int, str]]


# RagHandoff: defaults are required (Outbox.load skips a record that fails validation)
contributing_features: list[FeatureContribution] = Field(default_factory=list, description=...)
sensor_features: HandoffSensorFeatures | None = Field(None, description=...)


# score_request. verdicts.index() must not be used: Verdict is an eq dataclass holding a DataFrame
attach_handoff(request, result, g, window=critical_window(verdicts, vecs, rule))
...
attach_handoff(request, build_result(d, g, req.gun_id, feats.f, f, verdicts, score, contribs, h, vecs[-1]), g,
               window=critical_window(verdicts, vecs, h))
...
onset, onset_vec = next(((v, x) for v, x in zip(verdicts, vecs) if v.ew_onset), (None, None))
...
attach_handoff(request, result, g, early_warning=early_warning_info(d, g, result, onset.w.index[-1]),
               window=(onset.w, onset_vec))


def critical_window(verdicts: list[Verdict], vecs: np.ndarray, rule: RuleHit | None) -> tuple[pd.DataFrame, np.ndarray]:
    """The window a critical event stands for: the first window ending at or after the code's trigger second (the one
    holding it, or the next one when its minute was never scored), or the first window the model called critical; the
    latest window when no window reaches the trigger yet. A failure here costs the choice of window, never the event."""
    try:
        by_rule = rule is not None and rule.critical
        for v, vec in zip(verdicts, vecs):
            if (rule.time <= v.w.index[-1]) if by_rule else v.critical:
                return v.w, vec
    except Exception:
        log.exception("handoff window failed, using the latest window")
    return verdicts[-1].w, vecs[-1]


def handoff_sensor_features(g: "GunState", w: pd.DataFrame) -> dict[str, Any]:
    start, end = w.index[0].to_pydatetime(), w.index[-1].to_pydatetime()
    return HandoffSensorFeatures(**rag_mapping.sensor_block(g.rows, start, end, SENSOR_COLS)).model_dump()


def handoff_top_features(d: Detector, vec: np.ndarray) -> list[dict[str, Any]]:
    feats = [c for c in d.score(vec)[1] if c.contribution > 0 and c.sensor not in rag_mapping.NOT_A_SYMPTOM]
    top = [c.model_dump() for c in feats[:HANDOFF_TOP_FEATURES]]
    json.dumps(top, allow_nan=False)  # ValueError on NaN / inf: costs this field, never the pull list
    return top


# attach_handoff: new parameter window: tuple[pd.DataFrame, np.ndarray] | None = None,
# after the existing inner try, still under `if g is not None`
if window is not None:
    try:
        doc["sensor_features"] = handoff_sensor_features(g, window[0])
    except Exception:
        log.exception("handoff sensor features failed for %s at %s", result.gun_id, result.window_end)
    try:
        doc["contributing_features"] = handoff_top_features(d, window[1])
    except Exception:
        log.exception("handoff top features failed for %s at %s", result.gun_id, result.window_end)
```

필드 설명(`description=...`)은 2~5절의 뜻을 그대로 옮긴다. 새 코드는 `.py/sensor_features.py`를 쓰지 않는다. `main.py`에서 `sensor_features`는 그 모듈의 이름이므로 같은 이름의 변수를 만들지 않고, 클래스 이름 `SensorFeaturesDoc`은 v4가 쓰고 있어 `HandoffSensorFeatures`로 둔다.

## 8. 크기와 비용

측정 범위: 경보 244건은 `models/scores/*.csv`와 test CSV로 재구성했다. test_2 12건, test_5 5건은 `results/C-8/replay_check.json`과 일치한다. 치명 억제는 넣지 않았지만, 요청 단위로 재현하면 60초씩 보낼 때 요청 경계의 시계 초가 0~13인 경우(시계 분 전송 포함)의 목록과 같다. 경계가 14초 이상이면 test_1의 08-28 14:46 경보가, 41초 이상이면 08-27 01:50 경보도 같은 요청에 들어온 코드 때문에 억제되고 14:57과 02:13이 대신 나간다. `replay.py`의 test_1은 44초다. 합계는 어느 초든 244건이고, 이 문서의 경보 수치는 개수 1~2건, `share` 중앙값 0.002 이내, 블록 p50 3바이트 이내로만 달라진다. test_1 앞 53시간을 경계 0, 20, 44초로 서버 경로에서 재생하니 경보 시작 53건의 판정이 모두 이 재현과 같았다.

치명 20건은 코드 19건(실제 `rule_check`, `replay.py`와 같은 60초 구간)과 모델 단독 1건(점수 파일의 알람 구간)이다. test_5는 C-8 기록의 2건보다 1건 많다. C-8 재생은 1200초 청크라 E029(06:05:04)와 모델 치명(06:14:59)이 한 요청에 묶여 1건으로 남았고, 60초 구간에서는 따로 나온다. 서버 경로로 구간 재생하면 E029, 모델 단독, E028 세 건이 모두 나온다. 상위 3개는 실제 모델로 계산했다(13절).

| 블록 크기 (바이트) | p10 | p50 | p95 | 최대 |
|---|---:|---:|---:|---:|
| 압축 JSON | 1,766 | 3,396 | 4,416 | 4,665 |

- 저장 형식(기본 구분자)은 p50 4,015, 최대 5,535바이트다.
- 치명 20건의 블록은 압축 기준 1,387~3,893바이트, 중앙값 1,722바이트로 경보보다 작다.
- c19는 분마다 약 951바이트로 전형적인 블록의 28%다.
- 상위 3개는 경보 651~894바이트(중앙값 743), 치명 713~896바이트(중앙값 825)다.
- 조기경보 핸드오프 전체는 p50 약 12.2 KB, 최대 약 13.8 KB가 된다. 치명은 p50 약 10.0 KB, 최대 약 12.2 KB가 된다. 기존 부분은 이벤트마다 약 7.0~8.8 KB이고(중앙값: 조기경보 7.9 KB, 치명 7.4 KB), 예시 문서의 7,387바이트와 7,321바이트는 그중 작은 쪽이다.
- 예시로 쓰는 test_5 02:00은 하위 8%의 작은 분이다. 244건 중 221건은 용접 중인 분이다.
- ML 메모리는 30일 보관 경보 1,046건 기준 약 9~28 MB 늘어난다(복제본 측정). 치명은 같은 기간 약 86건이라 합쳐서 약 10~30 MB다.
- `GET /handoffs` 50건은 경보와 치명이 섞이면 약 403 KB에서 약 599 KB가 된다. 기본 설정(플래그 끔)에서는 치명뿐이라 약 145 KB 는다.

RAG가 전부 프롬프트에 넣을 때의 입력 토큰 수다. OpenAI 공개 어휘(`o200k_base`)로 세었고 gpt-5.4-mini가 이 인코딩을 쓴다는 것은 추정이다. 단가는 입력 100만 토큰당 $0.75, 기준 질문은 $0.00492다.

| 항목 | 토큰 |
|---|---:|
| 전형적인 크기의 블록 | 1,694 |
| 상위 3개 | 266 |
| 설명 문서 | 1,554 |
| **합계** | **3,514 ($0.00264, +54%)** |

프롬프트에서 c19만 빼면 425토큰이 줄고, ML 변경은 필요 없다.

## 9. RAG용 설명 문서

새 파일 `ml/docs/handoff_sensor_features_for_rag.md`에 한국어로 쓴다. 핸드오프마다 설명을 싣지 않고 이 문서 한 곳에 둔다. 설명이 틀려도 한 곳만 고치면 되고, 저장된 이벤트는 사실만 담은 채로 남는다. 읽는 쪽은 RAG 담당자와 RAG 코드이고, 프롬프트에 넣을지는 RAG가 정한다.

반드시 담을 읽기 규칙:

1. **읽는 법.** `channels`의 각 채널과 `error`는 `[초, 값]` 목록이다. 초는 시계 초(0~59)다. 값은 다음 항목까지 유지되고, `missing_s`의 초에는 값이 없다. 예를 들어 `"c10": [[0,0],[40,1]]`은 0~39초에 0, 40~59초에 1이다. `error`의 `"0"`은 코드 없음이다.
2. **구간.** `sensor_features`와 `contributing_features`는 이벤트를 낸 1분이고, `sensor_findings`, `sensor_trend`, `context`는 최신 윈도우다. 1분 단위 전송이면 둘은 같다. 치명에서 코드가 뜬 분이 아직 끝나지 않았으면 두 필드는 직전에 완성된 1분이고, 코드는 블록에 없다(`trigger.rule_trigger_time`이 `window_end`보다 늦다). 그 분이 끝내 채점되지 않았으면 두 필드는 그 뒤 첫 1분이고, 코드가 뜬 초는 블록 앞에 있다(`trigger.rule_trigger_time`이 `window_start`보다 이르다). 윈도우는 항상 :00~:59는 아니다. `sensor_features`는 `null`일 수 있고 상위 피처는 3개보다 적을 수 있다.
3. **모델 입력과의 차이.** 모델은 빈 초를 앞 값으로 채우고, `c16 <= 0`인 초의 입력 채널(c1~c10, c13~c18)을 직전 용접 행의 값으로 대체해서 본다. 따라서 블록의 c16=0은 모델이 본 값이 아니고, 상위 피처의 `value`는 블록으로 재현되지 않는다.
4. 단위가 가려져 있으므로 매뉴얼의 수치와 비교하지 않는다.
5. `contributing_features`는 모델이 기댄 피처를 보여 줄 뿐, 평소와 다르다는 뜻이 아니다. "평소와 다름"은 `sensor_findings`에 있는 항목만 쓸 수 있다.
6. `value`와 `reference`는 모델 공간의 숫자이므로 블록 값과 비교하지 않는다. `value - reference`는 피처끼리 비교하지 않고, 부호의 기준도 피처마다 다르다. c10, 모든 `*_std`, `weld_duty_10min`, `error_share_10min`은 학습 데이터 전체가 기준이다. 나머지 평균은 `context.gun_norm`이 `gun`일 때만 그 건이 기준이고, `global`이면 학습 데이터 전체가 기준이다. 순서와 `share`는 기준이 같으므로 비교할 수 있고, `share` 합이 작으면 세 피처로 점수가 설명되지 않는다는 뜻이다.
7. `sensor`가 c 채널이 아닐 수 있다. `welds_delta`, `welds_10min`은 c11에서, `pos_delta`는 c12에서, `weld_duty_10min`은 대체 전 c2에서, `error_share_10min`은 `error`에서 온다. 10분 피처는 이 1분으로 설명되지 않는다. `pos_delta`는 모델 정밀도로 계산되므로 블록으로 다시 계산하지 않는다.
8. 누름 없음(c4, c18이 모두 0)과 `context.non_welding_share`(`c16 <= 0`)는 다른 것이다.
9. 1초 샘플이므로 힘이 오르는 과정이나 c18과 c4의 순서는 대부분 보이지 않는다.
10. 설정값(c14~c16)은 1분 안에서도 바뀔 수 있다. 바뀌었다는 것만으로 이상이라고 쓰지 않는다.
11. 센서만으로 고장을 확정할 수 없다. 조기경보의 신뢰도는 `early_warning.expected`와 `caveats`를, 치명의 신뢰도는 `caveats`를 따른다.
12. `summary_ko`와 `sensor_findings`의 c10 문구 "꺼짐(평소 켜짐)"은 고정 문장이라 블록과 어긋날 수 있다. c10의 기준은 이 건이 아니라 학습 데이터 전체다.
13. 영문 `sensor_name`은 c4, c18에서 오래된 이름이므로 이 둘은 `sensor_name_ko`를 쓴다. c13은 한글 이름과 달리 c5의 설정값이 아니다.

쓰지 않을 내용: 원인, 부품, 조치, 매뉴얼 절, 임계값, 정상 범위, 여러 건(gun)에 공통인 "보통" 수준, c10이 평소 켜져 있다는 말, c19의 뜻("알 수 없음, 해석하지 말 것"만 적는다). `c16 <= 0`은 비용접의 정의이므로 임계값 금지에서 제외한다.

채널 표는 세 열이다.

| 채널 | 이름 | 값의 모양 |
|---|---|---|
| c4 | 힘 도달 신호 | 0/1 |

- 이름은 `SENSOR_KO`를 쓰고, 여기 없는 c11, c12, c19는 이 문서에만 적는다. `SENSOR_KO`는 늘리지 않는다.
- 값의 모양에는 숫자를 쓰지 않는다. 계단형, 0/1, 건마다 고정, 누적 카운터, 알 수 없음처럼 적는다.

## 10. 테스트

수정: `schema_version` 고정값 3곳(`tests/test_rag_mapping.py:145`, `tests/test_pipeline.py:398`, `:1305`).

`tests/test_rag_mapping.py`에 추가 (순수 함수, 모델 없음):

- 인코딩을 되돌리면 입력 행과 같다.
- 재전송은 마지막 도착이 남는다.
- 순서가 바뀐 행도 시간순으로 처리한다.
- 빈 초는 `missing_s`에 들어가고 거짓 항목이 없다.
- 부분 윈도우는 시계 초를 쓴다.
- 다음 분의 행은 제외한다.
- 행이 0개면 19개 키가 `[]`, `error`가 `[]`, `missing_s`가 윈도우 전체이고 예외가 없다.
- 정수로 떨어지는 값은 int, 그 외는 값 그대로다.

`tests/test_pipeline.py`에 추가:

- **기본(조기경보):** `window_end == onset_at`, 채널 19개, 피처 3개 이하, 모두 양수, `hour_*` 없음, `windows_scored == 1`이면 /predict 목록에 같은 필터를 적용한 앞 3개와 같음.
- **값 대조:** `c16 <= 0`인 초와 빠진 초가 있는 경보 분에서 `sensor_features`를 풀어, 19개 채널과 `error`를 보낸 행과 비교한다. 모델이 대체한 값이 아니라 받은 값인지 확인한다.
- **청크:** onset이 마지막 윈도우가 아니면 두 필드는 onset 것이고, `detected_at`, `sensor_findings`, `sensor_trend`, `context`, `early_warning.score`는 최신 윈도우 것이다.
- **onset 두 번:** 한 청크에 onset이 둘이면 첫 번째를 담는다.
- **치명, 코드:** 코드가 뜬 분이 요청 안에서 채점되면 `window_end`가 그 분이고 `error`에 코드가 있다.
- **치명, 분 미완성:** `test_4b_handoff_timeline_keeps_the_triggering_code` 상황에서 두 필드는 최신 윈도우 것이고, `rule_trigger_time > window_end`이며 `error`에 코드가 없다.
- **치명, 모델 단독:** `d.threshold = -1e9`, `d.warmup_alarms = True`로 1170행 청크 하나를 보낸다. `trigger.source`가 `model`이고 `window_end`는 10번째 윈도우(00:09:59), `detected_at`은 마지막 윈도우다. 60행씩 보내는 `test_9_restart_hold_and_critical_sustain` 상황은 `windows_scored`가 1이라 구분되지 않는다.
- **치명, 보류된 코드:** `test_6_serving_edge_cases`의 P 상황에서 마지막 요청을 `raw.iloc[430:560]`으로 늘려 윈도우 둘을 채점한다. `rule_trigger_time < window_start`이고 두 필드는 00:07 윈도우 것이며 `detected_at`은 00:08:59다. 원래 P 상황은 `windows_scored`가 1이라 최신 윈도우와 구분되지 않는다.
- **치명, 청크:** 코드가 청크 가운데 분에 뜨면 두 필드는 그 분 것이고, `detected_at`과 `sensor_findings`는 최신 윈도우 것이다.
- **치명, 코드 둘:** `test_10_rule_per_code_and_profile`의 행은 두 코드가 모두 00:05분에 떠서 구분되지 않는다. E016을 300~339행, E029를 400~439행에 두고 900행 청크 하나로 보내, 핸드오프마다 자기 코드의 분을 담고 두 핸드오프의 `sensor_features.window_end`가 다른지 본다.
- **`critical_window` 단위:** 가짜 판정으로 일곱 경우를 확인한다. 코드가 가운데 윈도우에 있음, 코드가 모든 윈도우 뒤에 있음(최신 윈도우), 코드가 모든 윈도우 앞에 있음(첫 윈도우), 코드가 두 윈도우 사이에 있음(뒤 윈도우), 코드 없이 모델 치명, 반복 코드와 모델 치명(코드를 무시), 해당 없음(최신 윈도우).
- **`handoff_top_features` 단위:** 가짜 detector의 `score`가 `hour_sin` 0.5, `c5` 0.4, `hour_cos` 0.3, `c2` 0.2, `c3` 0.0, `c6` -0.1을 돌려주면 결과는 `c5`, `c2` 두 개다. 양수 항목이 많으면 앞 3개다. 합성 데이터의 8개 항목은 모두 양수라 끝에서 끝까지의 시험으로는 이 필터에 닿지 않는다.
- **윈도우 선택 실패:** 보류 규칙의 시각에 시간대를 붙여 `critical_window`를 실패시켜도 핸드오프가 남고 두 필드는 최신 윈도우 것이다. 함수 자체를 예외로 바꾸면 try가 그 안에 있어 500이 되므로 이 방법으로 시험한다.
- **조회:** `GET /handoffs`가 블록을 담고 200을 반환한다.
- **이전 레코드:** 저장된 v1.3 레코드가 기본값으로 로드된다. `created_at`은 최근 시각으로 둔다.
- **NaN, inf:** `contributing_features`만 비고, 블록과 이벤트는 남고, `GET /handoffs`는 200이다. `d.score`를 바꾸면 /predict 응답까지 실패하므로, `handoff_top_features`에 넘기는 detector만 감싸서 주입한다.
- **잘못된 블록:** 빌더가 잘못된 모양을 돌려주면 `HandoffSensorFeatures` 검증에서 걸려 `sensor_features`만 `null`이고, 상위 피처와 이벤트는 남는다.
- **기존 문맥 실패:** `error_timeline` 생성이 실패해도 두 필드는 채워진다.

그대로 통과해야 하는 테스트: `test_schema_keys_match_main`, `test_no_situation_doc_mapping`, `test_no_causes_or_manuals_in_handoff`, `test_c10_reads_as_dropout`.

## 11. 문서와 예시

새 파일:

- `ml/docs/handoff_sensor_features_for_rag.md`: v4 문서 `docs/sensor_features_for_rag.md`와 이름이 겹치지 않게 `handoff_`를 붙인다.
- `ml/docs/handoff_sensor_features_typical.example.json`: test_0 2021-09-05T07:55의 실제 경보 분(3,796바이트, 용접 있음, c16=0 10초)을 순수 빌더로 만든다. test_5 02:00은 드문 유형이라 이 예시를 대표로 쓴다. 모델을 돌리지 않고 만든 예시라는 점은 `handoff_sensor_features_for_rag.md`에 적는다.

수정:

- `ml/docs/make_examples.py`
  - 위 예시를 첫 `TestClient` 앞에서 만든다. CSV 행은 `main.SensorReading(**r).model_dump()`에 c10을 1.0/0.0으로 바꾼 `g.rows` 형태로 넘긴다.
  - `sensor_features`는 채널당 한 줄로 쓴다. `rag_handoff.example.json`, `rag_handoff_early_warning.example.json`, 새 예시에 적용되고, 블록이 없는 예시는 바뀌지 않는다. `indent=2`로 쓰면 새 예시가 1,000줄을 넘는다.
- `python docs/make_examples.py` 재실행. `rag_handoff*.example.json`의 버전과 필드가 바뀐다. `rag_handoff.example.json`은 test_3의 E029 이벤트로, 코드가 뜬 분이 끝나기 전에 나간 경우라 블록은 09:25이고 코드는 블록 밖이다.
- `ml/docs/early-warning-handoff.md`: v1.4 필드와 12절의 요청을 넣는다.
- `ml/README.md` 5.10의 제목(`:748`)과 필드 설명, 154행의 핸드오프 내용 설명.
- `ml/techspec.md:3`, `:11`의 v1.3 언급, `:72`의 필드 버전(RAG 입력은 12절의 일이 끝날 때까지 종료 코드 그대로라는 점을 유지한다), `:77`의 핸드오프 내용 설명.
- `ml/docs/sensor_features_for_rag.md` 맨 위: 핸드오프의 `sensor_features`(v1.4)는 이 문서의 v4가 아니며 `docs/handoff_sensor_features_for_rag.md`를 보라는 한 줄.
- `ml/main.py` 헤더와 `RagHandoff` 설명, `ml/.py/rag_mapping.py`의 버전 언급 전부(`:3`, `:28`, `:175`), `ml/.py/mock_rag.py`, `ml/tests/test_rag_mapping.py`의 버전 언급.
- `ml/MEMORY.md`: 125행의 문서 버전과 133행의 ML 범위(사실에 이벤트 1분의 원시값과 상위 피처를 더한다)를 고치고 변경 이력에 한 줄을 더한다. 133행의 RAG 입력 설명은 RAG가 두 필드를 쓰기 전까지 맞으므로 그대로 둔다. 커밋 안 된 사용자 수정(81행, 218행)은 그대로 둔다.
- `ml/history.md`: 결정과 이력을 덧붙인다.
- `ml/CLAUDE.md`: 테스트 수만 고친다.

## 12. ML 밖에서 필요한 일

ML 구현은 이 항목들을 기다리지 않는다. 다만 이 일들이 끝나기 전에는 두 필드를 읽는 곳이 없다. 조기경보 핸드오프는 api가 처리할 때까지 꺼 둔다(`RSW_EARLY_WARNING_HANDOFFS`). 치명 핸드오프는 플래그와 상관없이 구현 즉시 두 필드를 싣는다. api는 받은 문서를 그대로 저장하고 케이스 상세로 내려보내므로(`api/main.py:122`) 크기만 늘고 동작은 바뀌지 않는다.

RAG 담당자:

- 이름은 합의한 `sensor_features` 그대로다. 다만 내용은 피처가 아니라 초당 원시값이고, /predict 최상위의 v4와 이름만 같으므로 `handoff.sensor_features`를 읽어야 한다.
- `trigger.rule_code`나 `fault_class.terminal_code`가 있는 치명은 지금도 RAG를 호출하므로(`api/guidance.py:19-20`) 두 필드를 가장 먼저 쓸 수 있는 경로다. 둘 다 없는 모델 단독 치명은 조기경보처럼 RAG를 호출하지 않는다.
- 상위 3개는 합의한 영향이 가장 큰 3개에서 시각 피처를 뺀 목록이다. 경보 244건 중 169건, 치명 20건 중 3건(`replay.py` 구간 기준)에서 모델의 실제 상위 3개와 다르다.
- 별도 입력이 필요하다. 블록과 상위 3개를 합치면 경보 244건과 치명 20건 모두 질문 제한 2,000자(`rag/service.py:105`)를 넘는다.
- 시스템 프롬프트의 "입력에는 센서가 없습니다"(`rag/service.py:24`)를 바꿔야 한다.
- 캐시 키(`rag/service.py:86`)에 `event_id`나 입력 해시를 넣어야 한다. 없으면 다른 분의 답이 재사용된다.
- 질문마다 `event_id`, 프롬프트 버전, 문서 fingerprint, 모델, 실제 입력을 남겨야 피드백 쪽에서 LLM이 본 것을 확인할 수 있다.
- LLM에 어떤 필드를 어떤 형태로 넘길지 정해야 한다. 볼 점이 넷 있다. 13절의 읽기 시험에서 원문을 값을 틀리게 읽었고, `summary_ko`의 c10 문구는 블록과 어긋날 수 있고, 청크 요청에서는 블록과 `sensor_findings`의 구간이 다르고, 치명에서는 코드가 뜬 초가 블록 밖일 수 있다.
- 프롬프트 시험에는 오경보 분을 포함한다. test_5 02:00은 실제 고장 전 경보 4건 중 하나라서, 이것만으로 맞추면 안 된다.
- 피드백 쪽 기계 검사 후보: LLM이 인용한 숫자는 핸드오프나 그것에서 코드로 계산한 값이어야 하고, "평소와 다름"은 `sensor_findings` 항목에만 쓸 수 있다.

api 담당자:

- 두 필드를 RAG로 넘겨야 한다. 지금 `enrich`는 코드만 넘긴다(`api/guidance.py:91`). RAG가 별도 입력을 정하면 `case["event"]`의 두 필드를 그 입력으로 넘긴다.
- `event_type`을 읽어야 한다. 지금은 `severity`가 `critical`로 고정이다.
- 조기경보에는 E 코드가 없어 `enrich`가 RAG를 호출하지 않는다(`api/guidance.py:19-20`, `:80-91`).
- 조기경보는 코드가 `null`인 열린 케이스에 합쳐지고, 이때 `event_id`와 `detected_at`만 남아 블록이 버려진다(`api/main.py:110-112`). 같은 건, 같은 코드의 치명도 열린 케이스가 있으면 마찬가지다.

| 케이스가 닫히는 시간(가정) | 2시간 | 4시간 | 8시간 | 24시간 |
|---|---:|---:|---:|---:|
| 조기경보 블록이 남는 비율 | 69% | 52% | 37% | 18% |

치명 20건은 같은 가정에서 모두 남는다. 코드로 발생한 치명은 같은 코드가 24시간 안에 다시 뜨면 반복이라 치명이 아니고, 모델 단독 1건은 `rule_code`가 없어 코드 케이스에 합쳐지지 않는다.

api·RAG 줄 번호는 origin/main 기준이다.

## 13. 남는 위험

- **상위 3개가 설정값과 에러 상태에 치우친다.** 8개 파일의 윈도우 벡터를 오프라인으로 다시 만들어 실제 모델로 계산했다. 저장된 점수와의 차이는 1e-16이고, test_5의 치명 2건은 서버 경로로 재생한 상위 3개와 같다. 경보 244건은 모두 3개가 나왔고, 732칸 중 341칸이 설정값(c13~c16)이며 `c16_mean`이 159건에 든다. 34건에는 c 채널이 아닌 피처가 있다. 치명 20건은 `replay.py` 구간 기준 17건에 `error_share_10min`이 든다. 코드로 발생한 19건은 코드가 뜬 분이면 19건 모두, 직전 1분이면 2건이다. 구현 후 서버를 `RSW_EARLY_WARNING_HANDOFFS=1`과 임시 `RSW_STATE_DIR`로 띄우고 test_3를 `--out-dir`로 재생하고, 모델 단독 치명이 있는 test_5를 `--hours 53`으로 재생해 서버 경로로도 확인한다. `--chunk-s` 기본값 60이면 요청당 약 0.2초라 test_3는 10,091요청에 약 35분, test_5 앞 53시간은 약 11분 걸린다. test_1 앞 5,900요청을 재생해 잰 값으로 만든 추정이다. 7분은 C-8의 1200초 청크 값이고, 1200초 청크는 요청당 윈도우가 20개라 1분 전송을 확인하지 못한다. 메모리를 확인한 뒤 돌린다.
- **코드가 뜬 초가 블록 밖일 수 있다.** 코드로 발생한 치명 19건 기준으로, 전송 시점을 고르게 놓으면 평균 74%가 이 경우다. 코드가 분의 앞쪽 초에 몰려 있어서다. 시계 분에 맞춰 보내면 1건만 남는다(test_2 E016: 05:46:59 행이 없어 그 분이 다음 요청에서 채점된다. 전체 윈도우의 0.93%가 :59 행이 없다). `replay.py`처럼 파일 첫 행부터 60초씩 보내면 4건이지만 대표값이 아니다. 파일이 끝나기 10분 전의 코드 8건은 모두 파일 첫 행과 같은 초에 떠서 구간 첫 행에 놓이고 항상 블록 안에 든다. 나머지 11건만 보면 4건이다. 코드가 뜬 분의 직전 1분은 19건 중 16건이 자기 임계값 이하라, 이 경우의 상위 3개는 모델이 정상으로 본 분의 피처다. 9절 2번 규칙이 경고하고, 코드와 시각은 `trigger`와 `error_timeline`에 있다.
- **RAG 모델이 원시값을 제대로 못 읽는다.** gpt-5.4-mini에게 RAG와 같은 호출 설정으로 실제 경보 1분을 주고 질문 20개를 물으니 `[초, 값]` 형식에서 12개를 맞혔다. 단순 조회도 틀렸다. 60개 행 객체도 11개로 같은 수준이었고, Claude Haiku는 같은 문제를 58~60/60 맞혔으므로 문제 자체는 정상이다. 출력이 200토큰 남짓이라 추론 없이 답한 것으로 보인다(추정). 1분, 질문 20개의 작은 표본이다. 값은 무손실이고 RAG가 입력 형태를 바꿔도 계약은 바뀌지 않는다. 다만 시험한 두 형식은 12개와 11개로 차이가 없었고, 다른 입력 방식이 나아지는지는 아직 시험하지 않았다(14절).
- **블록과 기존 문구가 어긋난다.** `sensor_findings`의 c10 문구는 test_5 02:00에서 실제로 켜진 값과 반대로 읽힌다. 9절이 경고하지만 근본 수정은 14절이다.
- **상위 피처의 값과 블록이 다르다.** 블록의 평균이나 std가 모델 입력과 0.5σ 이상 벌어지는 분은 244건 중 143건이다. `c16 <= 0`인 초가 있는 185건 중 141건, 없는 59건 중 2건이다. 나머지는 빈 초 채움 때문이다. 주로 c16(142건), c14(112건), c3(64건), c15(56건)이고, c16을 빼면 113건이다. 9절 3번 규칙만이 이를 막는다.

## 14. v1.4 뒤에 따로 결정할 것

- **v4 삭제.** `/predict`의 `sensor_features`, `GET /guns/{id}/sensor-features`, `.py/sensor_features.py`, 관련 스키마·예시·테스트를 지운다. 계약 삭제라 CLAUDE.md의 예외를 한 번 더 쓰는 결정이다. ml 밖 코드는 읽지 않고, `ui/docs` 04, 07, 08, 09가 낡게 되므로 담당자에게 알린다. `sensor_features_for_rag.md`의 읽기 규칙은 9절의 금지 목록에 걸리지 않는 것만 옮긴다.
- **c10 문구.** "꺼짐(평소 켜짐)"은 `test_c10_reads_as_dropout`으로 고정돼 있어 add-only인 v1.4에서 바꾸지 않는다. 다음 작은 수정으로 중립 표현을 쓰는 안을 권한다.
- **조기경보의 비용접 조건.** 경보 244건 중 185건이 `c16 <= 0`인 초가 섞인 분에서 나왔고, 그런 분은 임계값 초과율이 0.38%에서 10~17%로 오른다. 일반 알람은 비용접 비율 50% 초과에서 보류되지만 조기경보 규칙에는 조건이 없다(`gate: null`). 이유를 모르므로 규칙을 바꾸기 전에 원인 분석이 먼저다.
- **RAG 모델의 읽기 능력.** 추론을 켜거나 사실 요약을 넘기면 나아지는지는 시험하지 않았다. 유료 호출이 필요하므로 RAG 담당자와 입력 방식을 정할 때 함께 판단한다.
