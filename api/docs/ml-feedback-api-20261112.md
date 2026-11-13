# ML 모델 루프용 판정 조회: `GET /api/ml/feedback` (2026-11-12)

닫힌 조기 경보(WARNING) 케이스를 ML 모델 루프에 내주는 읽기 전용 주소다. ML은 이 응답을 가져가 다음 재학습과 A/B 판정에 쓴다(D-12, 설계: `ml/docs/superpowers/specs/2026-11-12-feedback-labels-design.md`).

- **사실만 준다.** 어떻게 닫혔는지, 어느 경보들이었는지, 점검에서 이상이 나왔는지를 준다. 그 케이스가 ML의 "정답"인지는 ML이 정한다.
- **기존 주소는 바뀌지 않는다.** 새 파일 `api/ml_feedback.py` 하나와 `api/main.py`의 두 줄(가져오기, 등록), `api/store.py`의 읽기 함수 하나(`warning_deliveries`: 전송 기록 전부를 한 번에 읽는다)를 더했다. 저장하는 것도 없다.
- 인증은 다른 조회 주소와 같다(없음).
- **이상한 케이스 하나가 나머지를 막지 않는다.** `POST /api/ml/handoffs`는 어떤 모양이든 받아 저장하므로, 객체여야 할 필드 하나가 다른 것이면 그 값만 `null`로 주고, 케이스를 아예 읽을 수 없으면 그 케이스만 건너뛰고 api 로그에 남긴다.
- 고른 경보와 전송 상태(`warning_outbox`)는 요청 한 번에 한 번만 읽는다(`store.warning_deliveries()`. 케이스마다 DB를 열지 않는다).

## 대상

`event.event_type == "early_warning"`이고 `records.closure.outcome`이 `resolved` 또는 `false_alarm`인 케이스. 열려 있는 케이스, 인계(`handed_over`)로 닫힌 케이스, E 코드(CRITICAL) 케이스는 나오지 않는다. 닫힌 순서로 준다.

## 응답

```json
{
  "schema_version": "1.0",
  "items": [
    {
      "case_id": "CASE-0007",
      "outcome": "resolved",
      "closed_at": "2026-11-12T08:00:00Z",
      "resolution_status": "saved",
      "selected_event_id": "11a9ca71eaa03037b93f5787ea797152",
      "checks": {"abnormal": 1, "normal": 2, "unchecked": 0},
      "events": [
        {
          "event_id": "11a9ca71eaa03037b93f5787ea797152",
          "representative": true,
          "gun_id": "E01_3",
          "window_start": "2021-09-01T10:00:00Z",
          "onset_at": "2021-09-01T10:00:59Z",
          "detected_at": "2021-09-01T10:00:59Z",
          "detector_created": "2026-11-01T18:46:47"
        }
      ]
    }
  ]
}
```

| 필드 | 출처 | 비고 |
|---|---|---|
| `outcome`, `closed_at` | `records.closure`의 `outcome`, `at` | `closed_at`은 실제 시각이다. 경보의 시각(`window_start`, `onset_at`)은 센서 데이터의 시각이다 |
| `resolution_status` | `warning_outbox`의 상태, 없으면 케이스의 `warning_resolution_status` | `saved` / `pending` / `not_linked` 등, 없으면 `null`. ML은 기록만 한다 |
| `selected_event_id` | `/warning/resolve`로 닫을 때 고른 경보(`warning_outbox`) | 예전 `/close` 경로로 닫았으면 `null` |
| `checks` | `records.checks`와, 고른 경보(없으면 대표 경보)의 점검 초안을 점검 ID로 합쳐 값별로 센 수 | 값의 두 형식(`{"result": "abnormal"}`, `"abnormal"`)을 모두 읽는다. 같은 ID는 초안이 이긴다. 세 값이 아닌 것은 세지 않는다 |
| `events` | 대표 경보(`event`)가 먼저, 그 뒤 `repeats` | 원본(`handoff`)이 없는 옛 반복 경보는 `event_id`와 `detected_at`만 채우고 나머지는 `null`. `onset_at`은 핸드오프의 `early_warning.onset_at`(경보가 시작된 창의 마지막 초)이고 없으면 `null`이다. 한 요청에 창이 여럿이면 `window_start`는 그 요청의 마지막 창이라 `onset_at`보다 뒤일 수 있어, ML은 `onset_at`을 먼저 쓴다 |

## ML이 읽는 방법

```bash
python ml/.py/model_loop.py feedback pull --api-url http://localhost:9000
```

ML은 응답을 모델 등록부 폴더의 `feedback.json`에 쌓는다(대표 경보의 `event_id`가 열쇠). api가 꺼져 있어도 ML 루프는 그대로 돈다.

## 테스트

`api/tests/test_ml_feedback.py` (14개). api 전체: 56개 통과, 1개 건너뜀.
