# mock 고정 JSON

MSW가 api 서버 대신 돌려주는 예시 데이터. **값은 모두 예시**이며, 어디서 왔는지 아래에 적는다.

## cases.json (`GET /api/cases`)

| 케이스 | 출처 | 지어내지 않고 비운 값 |
|---|---|---|
| CASE-0001 · test_0 · E012 | 공개 데이터 test_0의 종료 코드와 시각(2021-09-11 02:46:08 UTC), 클래스 E01, 이름은 `ml/.py/rag_mapping.py` | `event_id`는 예시 문자열. `detected_at`은 실제 ML 판정 시각을 몰라 종료 코드 시각과 같게 둠 (가정). `summary_ko` 없음 |
| CASE-0002 · test_3 · E016 | `ml/docs/rag_handoff.example.json` 값 그대로 | — |

- `status`, `severity`, `assignee`는 api가 관리할 값이라 와이어프레임 예시를 따랐다. 담당자는 모두 null.
- 와이어프레임의 `[설비]` E028(인계됨) 행과 E029 반복 묶음 행은 근거 값이 없어 넣지 않았다.
- 와이어프레임의 test_2 "모델 이상 신호 · 코드 없음" 행은 **가정 사례**라 넣지 않았다 (실제 test_2의 클래스는 E02). 수동 모드 화면을 만들 때 다시 다룬다.

## 화면 상태 확인용

브라우저 주소 뒤에 붙이면 MSW가 다르게 응답한다 (개발용).

- `?mock=empty` 빈 목록
- `?mock=error` 500 오류
- `?mock=slow` 3초 지연 (로딩 상태)
