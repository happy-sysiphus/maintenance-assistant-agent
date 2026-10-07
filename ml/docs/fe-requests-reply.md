# FE 요청에 대한 ML 답변 (`ui/docs/05-fe-api-requests.md` "ML 담당에게")

2026-10-08 · ML

요약: 요청 3·4·5는 코드에 반영했다(핸드오프 v1.1, 필드 추가만이라 1.0을 받던 쪽은 그대로 동작). 요청 1·2·6은 코드 변경 없이 아래처럼 쓰면 된다. `event`(api가 FE에 주는 묶음)에 필요한 ML 필드는 이제 전부 있다.

| 요청 | 상태 | 어디 |
|---|---|---|
| 1 서버를 띄울 수 있게 | 방법 안내 (아래) | README 0절 |
| 2 핸드오프를 api 서버로 | 지금 코드로 가능 (설정만) | `RSW_RAG_URL` |
| 3 코드 이력·센서 추세 | **반영** — `error_timeline`, `sensor_trend` | `main.py`, README 5.10 |
| 4 시각 형식 | **반영** — 핸드오프 시각에 `Z`. 나머지 API 시각은 전부 UTC(오프셋 없음) | `rag_mapping.utc_iso` |
| 5 한국어 센서 이름 | **반영** — `contributing_features[]`·trace `features[]`에 `sensor_name_ko` | `main.py` |
| 6 "재발 없음" 확인 | 답변 (아래) | — |
| caveats 수치 | **반영** — 현재 테스트 지표로 (AUROC 0.71, 종료 코드 이전 0.67, 지속 알람 2/8, 오트리거 0.18회/일/건) | `rag_mapping.CAVEATS` |

## 요청 1 · 서버 띄우기

- 모델 파일: README 0절의 구글 드라이브에서 `baseline_ensemble.joblib`을 받아 `ml/models/`에, 원본 CSV는 `ml/train/`·`ml/test/`에 둔다. 그다음 `uvicorn main:app`.
- mock부터 개발하려면: 서버를 띄운 PC에서 `python .py/replay.py test/test_0.csv --out-dir mock/` → 응답이 한 줄씩 `mock/test_0.jsonl`에 쌓인다(핸드오프가 붙은 줄이 critical 이벤트). trace는 재생 직후 `GET /guns/test_0/trace?minutes=360`을 저장. 8건 다 만들어 공유할 수 있다 — 필요하면 말해 달라.

## 요청 2 · 핸드오프를 api 서버로

코드 변경 없이 된다. `RSW_RAG_URL=http://api:9000/diagnose`로 띄우면 critical 이벤트마다 그 주소로 POST한다(헤더 `Idempotency-Key: <event_id>`, 실패 시 백오프 재시도, 4xx는 재시도 안 함 — README 5.10.3).

- 응답 본문은 아무 형식이나 된다. **2xx면 전달 완료**로 보고, JSON이면 `rag_response`로 보관만 한다. 그러니 `{"event_id": "...", "status": "ok"}`로 접수만 바로 돌려주는 안이 맞다. 7칸 리포트를 이 응답에 담을 필요는 없다.
- 같은 `event_id`가 다시 오면(재시작 뒤 재전송) 새 케이스를 만들지 말고 무시하면 된다.

## 요청 3 · 코드 이력·센서 추세 (핸드오프 v1.1)

```json
"error_timeline": [
  { "code": "E012", "start": "2021-09-11T02:46:08Z", "end": "2021-09-11T02:46:23Z",
    "duration_s": 16, "class_hint": "E01" }
],
"sensor_trend": [
  { "feature": "c5_mean", "sensor": "c5", "sensor_name_ko": "보정(밸런스) 압력", "statistic": "mean",
    "points": [ { "t": "2021-09-11T02:17:59Z", "z": -0.2 }, { "t": "2021-09-11T02:18:59Z", "z": -0.4 } ] }
]
```

- `error_timeline`: 이벤트 시점까지 이 건 버퍼(약 30분)의 0이 아닌 코드 구간, 오래된 것부터. 코드가 바뀌거나 60초 넘게 끊기면 구간이 나뉜다. `end`는 마지막으로 보인 초라 `duration_s = end - start + 1`.
- `sensor_trend`: 이번 이벤트의 기여 상위 피처(최대 4개, 시각 피처 제외) × 최근 30분의 1분 창. `z`는 trace의 `deviation`과 같은 값(평소 대비 편차, ±1·±2 띠는 `axes.deviation_bands`). 기여 피처가 없는 규칙 이벤트면 빈 배열이다.
- 한계: 코드 이력은 30분까지다(techspec D-3의 1시간 이력은 아직). "16초" 같은 지속 시간은 1초 샘플 기준이다.

## 요청 4 · 시각

핸드오프 안의 시각은 끝에 `Z`가 붙는다. `/predict`·`/guns`·trace 응답의 시각은 `Z` 없이 나오지만 **전부 UTC**다 — FE에서 UTC로 읽고 한국 시간으로 바꾸면 된다.

## 요청 5 · 한국어 센서 이름

`contributing_features[]`(응답·trace의 `focus`)와 trace `features[]`에 `sensor_name_ko`를 넣었다. 이름표는 `rag_mapping.SENSOR_KO` 하나다.

## 요청 6 · 조치 뒤 "재발 없음"

계획대로 하면 된다: 조치 저장 시각 이후 `GET /handoffs?gun_id=`에 새 이벤트가 없고 trace `events`에 `rule`·`rule_repeat`가 없으면 재발 없음.

- **얼마나 볼지**: 최소 30분(규칙 쿨다운 `model.rule_cooldown_s` = 1,800초). 같은 코드는 24시간 안에 다시 뜨면 `rule_repeat`(warning, 핸드오프 없음)이니, trace `events`의 `rule_repeat`도 재발로 세야 한다. 데모에서는 30분 시뮬레이션이 현실적이다.
- **`DELETE /guns/{id}`**: 부품 교체·캡 교체·축 보정처럼 **센서 기준값이 바뀌는 조치**에만 부른다. 부르면 6시간 워밍업을 다시 하고(그동안 모델 알람은 글로벌 기준) 규칙 쿨다운·24h 반복 이력·trace도 지워진다(건 프로파일은 유지) — 그래서 단순 점검 뒤에는 부르지 않는다.
- 공개 데이터에는 고장 뒤 10분까지만 있어서 "시뮬레이션" 표시가 맞다.

## ML과 RAG가 같이 정할 것 (ML 입장)

| 항목 | ML 입장 |
|---|---|
| 원인 후보의 단위 | FE 제안에 동의(카드 1개 = 상황 ID 1개). 단 증상 기반 상황(S05~S10)은 데이터 근거가 없어 판단을 RAG 쪽으로 옮기는 안을 제안 중(검증 결과: 고장 직전 S01~S04 일치 96%, 증상 규칙은 데이터로 확인 안 됨) |
| 핸드오프를 받는 곳 | api 서버 — 요청 2대로 설정만 하면 된다 |
| 상황 ID 이름·정의 | RAG 것만 쓰는 데 동의. ML은 ID만 보낸다 |
| 해석 한계 문구 | `caveats`는 ML 성능·규칙 한계만 담는다. 매뉴얼 적용 한계는 RAG `coverage_note` |
| 사건 키 | `event_id` — 같은 이벤트는 재전송돼도 같은 값 |
