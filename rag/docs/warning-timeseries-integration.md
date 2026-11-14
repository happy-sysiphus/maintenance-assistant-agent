# 조기 경보 시계열 RAG 연동

RAG는 장비가 보고한 E코드, 과거 해결 사례, 매뉴얼 검색 순서로 안내 경로를 정한다. ML이 경보를 발행하면 API는 원본을 RAG에 등록한다. 등록한 경보의 `event_id`로 판단을 요청하고, 화면을 다시 열 때는 저장된 판단을 읽는다. 공용 Aura에 원본 센서와 해결 이력이 연결된다.

## ML에서 전달할 데이터

이상 에피소드의 첫 이벤트를 RAG에 보낸다. ML의 10분 지속 조건이나 기존 `event_type`은 RAG의 안내 경로에 사용되지 않는다. 같은 에피소드에서 장비가 E코드를 뒤늦게 보고하면, 같은 `episode_id`에 **새 `event_id`**를 발급해 보낸다. API는 열려 있는 같은 에피소드 케이스에 이벤트를 추가하고 새 E코드 안내를 요청한다. 정비 완료 때 실제 판단에 사용한 이벤트 ID를 보낸다.

아래는 RAG CLI의 `register --input`에 사용하는 시연용 축약 입력이다. ML이 `POST /api/ml/handoffs`로 전송할 때는 바깥의 `handoff` 키 없이 그 객체를 요청 본문으로 보낸다. 실운영에서는 ML이 확보한 전체 채널의 원본 블록을 전달한다.

```json
{
  "handoff": {
    "schema_version": "1.4",
    "event_type": "early_warning",
    "event_id": "alert-20261113-001",
    "episode_id": "episode-20261113-001",
    "gun_id": "gun-1",
    "detected_at": "2026-11-13T00:00:59Z",
    "early_warning": {"onset_at": "2026-11-13T00:00:59Z"},
    "contributing_features": [
      {"feature": "c10_std", "sensor": "c10"},
      {"feature": "error_share_10min_std", "sensor": "error_share_10min"}
    ],
    "sensor_features": {
      "window_start": "2026-11-13T00:00:00Z",
      "window_end": "2026-11-13T00:00:59Z",
      "missing_s": [41],
      "channels": {
        "c10": [[0, 0], [40, 1]],
        "c5": [[0, 1.0], [40, 1.3]]
      }
    }
  }
}
```

- ML은 현재 v1.4의 원본 센서 블록을 보낸다. RAG는 블록에 들어 있는 `c1`~`c19`를 읽으며, 상위 피처가 비어 있어도 비교할 수 있다. 응답의 `comparison_channels`는 RAG가 센서 블록에서 읽은 채널 목록이다. 과거 시연용 요청에 포함된 이 필드의 값은 비교 대상을 제한하지 않는다.
- `sensor_features`는 이벤트와 관련된 1분의 변화점 원본이다. `missing_s`는 시계 초를 사용한다. RAG는 센서 창 끝을 0초로 놓고 최대 60개 초를 복원한다. 결측 초는 모델에 `null`로 전달한다. 단위가 확인되지 않은 값은 매뉴얼의 기준 수치와 비교하지 않는다.
- 장비가 현재 이벤트에 귀속되는 코드를 보고하면 `"observed_error": {"code":"E016","observed_at":"2026-11-13T00:01:09Z"}`를 넣는다. 기존 v1.4의 `trigger.rule_code`와 `trigger.rule_trigger_time`도 읽는다. 최신 상태나 모델이 붙인 분류, 원시 블록에 남은 다른 사건의 코드는 이 필드를 채우는 근거로 쓰지 않는다. 동일 이벤트 재전송 시 ID와 데이터는 동일하게 보낸다.

## API 연결과 UI 표시

API는 `WarningAssessment(WarningMemory(driver, database=..., namespace=...), equipment_scope="cpx-weld-2g-sa")`를 사용한다. Aura의 매뉴얼 KG와 인덱스가 준비된 환경에서 `WarningMemory.initialize()`를 한 번 실행한다.

| 작업 | Python 호출 | API·UI에서 할 일 |
|---|---|---|
| 경보 수신 | `register(handoff)` | 원본과 ID를 보존한다. Aura 일시 장애는 재시도한다. |
| 판단 요청 | `assess(event_id)` | 케이스 또는 이벤트별 작업으로 한 번 요청한다. 같은 이벤트의 동시 판단 요청을 API에서 합친다. |
| 판단 조회 | `read_decision(event_id)` | 재접속 및 새로고침에 사용한다. `null`이면 결과가 준비되지 않은 상태다. |
| 후보 요청 | `manual_candidates(event_id)` | 원인 후보 보기에서 호출한다. 최대 세 개 카드의 관측, 선정 이유, `step_kind`(점검 또는 조치), PDF 원문과 페이지를 표시한다. |
| 수동 검색 | `manual_search(event_id, question, explain=True)` | 현재 이벤트와 이번 질문으로 검색한다. 답변과 선택 가능한 PDF 근거를 표시한다. |
| 해결 종료 | `close(event_id, ...)` | 실제 조치·작업자 확인·선택 근거가 있는 해결을 동일한 요청으로 재시도한다. |
| 출처 검토 | `review_resolution(event_id, ...)` | 검토 대기 건에 기존 사례를 연결하거나 새 검토 사례를 등록한다. |

`assess` 결과의 주요 필드는 아래와 같다. 카드 예시는 핵심 필드만 발췌했다.

```json
{
  "event_id": "alert-20261113-002",
  "route": "history",
  "status": "seen",
  "reason": "model_comparison",
  "past_event_id": "alert-20261113-001",
  "candidate_ids": ["alert-20261113-001"],
  "intervals": [{"channel": "c10", "start_offset_s": -20, "end_offset_s": -18}],
  "card": {
    "manual_case_id": "manual:E016-C1",
    "origin": "history",
    "repair": {
      "event_id": "alert-20261113-001",
      "case_id": "repair-001",
      "actual_action": "작업자가 저장한 실제 조치",
      "worker_confirmed": true,
      "outcome": "resolved"
    }
  }
}
```

| 응답 | 화면 |
|---|---|
| `route=error_code, status=ok` | 장비가 보고한 코드, 오류 설명, `cards[]`와 PDF 근거를 표시한다. |
| `route=history, status=seen` | 한 건부터 이전 해결 사례를 표시한다. 과거 실제 조치, 모델 비교 이유, PDF 출처를 보여준다. 작업자가 거절하면 원인 후보 보기 또는 수동 검색으로 이동한다. |
| `route=manual, status=unseen` | 센서 관측, 원인 후보 보기, 수동 검색을 표시한다. |
| `route=manual, status=uncertain` | 채널 부족·이력 충돌·서비스 장애 등 `reason`을 안내한다. 매뉴얼 검색으로 이동하거나 판단을 재시도할 수 있다. |
| `route=manual, status=unsupported_code` | 관측 코드와 KG 미등록 상태를 표시하고 매뉴얼 검색으로 연결한다. |

선택 가능한 카드와 PDF 출처는 API가 해당 이벤트 응답에서 보존한다. UI는 `POST /api/cases/{case_id}/warning/select`에 `event_id`와 함께, 검토된 카드면 `manual_case_id`, 새 후보나 수동 검색 근거면 `evidence_id`를 보낸다. API는 이 ID를 이전에 반환한 자료와 대조하고 PDF 출처는 서버에 보존한 `source`를 사용한다. 아래 예시는 선택·종료를 처리한 API가 RAG의 `close`를 호출할 때 전달하는 값이다.

```json
{
  "event_id": "alert-20261113-001",
  "case_id": "repair-001",
  "actual_action": "작업자가 수행한 정비를 기록",
  "worker_confirmed": true,
  "closed_at": "2026-11-13T00:06:00Z",
  "manual_case_id": "manual:E016-C1"
}
```

새 출처에 대한 응답은 `pending_review`다. 운영자는 `review_resolution(event_id, existing_case_id=...)` 또는 `review_resolution(event_id, title=..., equipment_scope=..., checks=[...], actions=[...])`로 검토한다. 점검·조치는 PDF 원문의 발췌를 사용한다. 오알람은 기존 케이스 종료로 처리하고 `close`를 호출하지 않는다.

## 로컬 실행과 평가 기록

저장소 루트에서 팀 Aura 환경을 읽는 RAG 가상환경으로 실행한다. `register.json`과 `close.json`은 위 요청 예시이며 `event.json`은 `{"event_id":"alert-20261113-001"}`이다. 시연은 전용 이력 이름으로 실행한다.

```powershell
python -m rag.warning_assessment init --namespace warning-demo
python -m rag.warning_assessment register --namespace warning-demo --equipment-scope cpx-weld-2g-sa --input register.json
python -m rag.warning_assessment assess --namespace warning-demo --input event.json
python -m rag.warning_assessment candidates --namespace warning-demo --input event.json
python -m rag.warning_assessment close --namespace warning-demo --input close.json
python -m rag.warning_assessment read --namespace warning-demo --input event.json
```

같은 채널에서 비슷한 변화가 있는 다음 이벤트를 새 ID와 해결 시각 이후의 시간으로 등록하고 판단한다. 코드 없이 입력하면 첫 해결 한 건부터 검색한다. 파일 간 동일 채널의 관측이 겹치는 초를 비교하고, 채널별 거리와 관측 범위를 합쳐 최대 다섯 건의 해결 사례를 고른다. Claude는 그 사례의 비교 가능한 전체 파형과 해결 내용을 받아 `seen`, `unseen`, `uncertain` 중 하나를 반환한다. `c19`는 의미가 확인되지 않아 숫자 비교에만 쓰고 매뉴얼 검색어에는 넣지 않는다. E코드 경로에는 Claude나 임베딩 호출이 없다. `--keyword-only`는 매뉴얼 검색의 임베딩을 끈다.

기존 해결 기록은 대응하는 원본 핸드오프를 확인해 같은 Aura 이력 이름으로 `register`하면 기존 `event_id`에 연결할 수 있다. 정비 이력 조회는 입력 시점에 해결된 이전 이벤트만 대상으로 하며, 동일 `episode_id`는 제외한다. 점검·조치 출처의 문서 버전과 장비 적용 범위도 확인한다.

현재 검색은 적용 범위의 검증된 이력을 읽어 시계열 점수를 계산한다. 운영 이력이 많아질 때는 Aura 조회량과 지연 시간을 측정해 후보 조회 방식을 조정한다. 통합 브랜치에서는 `RAG_WARNING_TIMESERIES=1`을 켠 뒤 수신한 v1.4 케이스에 이 경로를 적용한다. [통합 변경 보고서](../../docs/timeseries-rag-integration-report.md)

판정 결과에는 검색 후보, 비교 채널과 관측 범위, 선택 사례, 모델 이유가 남는다. 평가에서는 사건을 시간순으로 재생하고 같은 에피소드를 분리한다. 검색 상위 5건의 정답 사례 포함률, 잘못된 `seen`, 판단 보류율을 각각 계산한다. 매뉴얼 후보는 인용과 점검의 일치 여부, 과거 사례 카드는 저장된 조치·결과·출처가 정확히 표시되는지 확인한다.
