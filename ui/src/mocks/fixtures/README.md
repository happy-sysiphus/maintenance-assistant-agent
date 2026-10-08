# mock 고정 JSON

MSW가 api 서버 대신 돌려주는 데이터. 손으로 지어낸 값이 아니라 **실제 ML · RAG 출력**으로 만들었다 (2026-10-08).

## 어떻게 만들었나

1. ML 서버(`ml/main.py`, 모델 `baseline_ensemble.joblib`)를 띄우고 공개 데이터 `test_0.csv`, `test_3.csv`의 마지막 약 12시간을 재생했다
   (`python .py/replay.py <csv> --start-hours 156 --chunk-s 600`, ML의 `docs/make_examples.py`와 같은 방식).
2. 서버에서 `GET /handoffs?gun_id=`(고장 알림, 핸드오프 v1.1)와 `GET /guns/{id}/trace?minutes=360`(이상 점수)을 받았다.
3. 원인 후보의 이름 · 정의는 `rag/pipeline/build.py`, 점검 항목 · 주의 문구 · 근거 쪽은 `rag/data/situation_mapping.json`에서 그대로 가져왔다.

모델 파일 · 원본 CSV · 변환 스크립트는 레포 밖(`Desktop/자동화ui/ml-data/`)에 있다. 다시 만들 때는 같은 방식으로 재생하면 된다.

## 파일

| 파일 | 내용 |
|---|---|
| `cases.json` | 작업함 목록 (`GET /api/cases`). 재생에서 나온 critical 고장 5건 |
| `case-details.json` | 케이스 상세 (`GET /api/cases/:id`). `event` = 핸드오프 원문, `score_trace` = 고장 전 30분 이상 점수, `guidance` = RAG 매핑, `records` = 사람 입력(처음엔 비어 있음) |
| `health.json` | 연결 확인용 |
| `guns.json` | 설비 화면 (`GET /api/guns`). 재생 뒤 ML `GET /guns` 응답 그대로 (test_0, test_3). 열린 고장은 MSW가 붙인다 |
| `manual-index.json` | 수동 모드 "매뉴얼에서 직접 찾기". `rag/data/situation_mapping.json`의 근거 쪽 · 부품과 `rag/pipeline/build.py`의 원인 이름을 그대로 모은 것 (56개). 레포 밖 `ml-data/tools/build_manual_index.py`로 만들었다 |

| 케이스 | 설비 · 코드 | 원인 후보 (ML 순서) | 화면에서 보여주는 것 |
|---|---|---|---|
| CASE-0005 | test_0 · E012 | S01 | 와이어프레임 예시 고장. 원인 1개뿐이라 "아니에요"면 바로 "남은 원인 없음" |
| CASE-0004 | test_0 · 코드 없음 (모델) | S07, S02 | "다음 원인" 흐름 |
| CASE-0003 | test_0 · 코드 없음 (모델) | 없음 | 매뉴얼 안내 없음 → 수동 모드. 알림 전 25분간 E029가 켜져 있었다 |
| CASE-0002 | test_0 · E029 | S04, S10 | 고장 전 30분 이상 점수가 없다 (trace가 6시간치만 남아서) |
| CASE-0001 | test_3 · E016 | S02, S10 | 다른 설비 |

## 지어내지 않은 부분 (화면에 비어 보이는 이유)

- **점검 항목 id**(`S01-K1` 등): RAG에 아직 없어 FE가 붙인 임시 값이다 (`docs/06` 요청 3).
- **이상일 때 할 일**: 원문을 확인한 Festo 오류 12 문구만 넣었다. 다른 오류 번호는 번호 · 근거 쪽만 있다.
- **상태 · 담당**: 모두 "새 고장", 담당 없음. 사람이 저장하면 MSW가 메모리에서 바꾼다 (새로고침하면 처음으로).
- 점검 항목 이름의 맞춤법(예: "레귈레이터")은 RAG 매핑 원문 그대로다.

## 화면 상태 확인용

브라우저 주소 뒤에 붙이면 MSW가 다르게 응답한다 (개발용).

- `?mock=empty` 빈 목록
- `?mock=error` 500 오류
- `?mock=slow` 3초 지연 (로딩 상태)
