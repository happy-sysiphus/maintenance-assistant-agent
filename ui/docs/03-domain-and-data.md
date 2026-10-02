# 03. 도메인과 데이터 (FE가 헷갈리기 쉬운 것)

출처를 항목마다 적었다. 값·이름은 레포와 이전 대화에서 확인한 것만 넣었다.

## 장비

- **점용접(RSW)**: 철판 두 장을 전극 두 개로 누른 채 전류를 흘려 한 점씩 붙이는 방식. 로봇이 차체에 수천 점을 찍는다.
- 데이터의 용접건은 **서보 공압식**이다 (공기압 실린더를 제어기가 정밀하게 조절). 제조사·모델은 논문에 공개되지 않았다.
- 주요 부품: 주 실린더(전극을 눌러 힘을 만듦), 보정(밸런스) 실린더(건이 판 양쪽에 고르게 닿게 함), 제어기, 전극 캡(닳으면 깎거나 교체 = 드레싱).

## 데이터셋

- Wang, Zhang & Wang, "Benchmark for welding gun fault prediction with multivariate time series data", Scientific Data 11:83 (2024). Zenodo 7655193.
- 학습 72파일(고장 유형 E01~E04 각 18개), 테스트 8파일(`test_0`~`test_7`). 파일 하나 = 설비 하나의 고장 7일 전 ~ 고장 10분 후, 1초 간격.
- **있는 것**: 센서 19종(c1~c19), 1초마다의 컨트롤러 코드(`error` 컬럼), 고장 순간의 종료 코드와 시각, 결측·비용접 구간.
- **없는 것 (화면에서 추정 금지)**: 실제 정비 조치 기록, 조치 후 신호, 센서 단위, 장비 모델, 코드 의미 사전, 설비별 과거 이력.
- 데이터 시각은 2019~2021년이다. "방금 전" 같은 현재 기준 표시는 쓸 수 없고, 화면에는 "데이터 시각"으로 표시한다.

### 예시 케이스 (와이어프레임 전체가 이 한 건 기준)

- `test_0`: 종료 코드 **E012**, 2021-09-11 02:46:08 UTC, 16초 지속. 데이터셋 클래스 E01.
- 같은 파일에 E029가 09-06 · 07 · 08 · 10일 19:30 UTC에 반복(각 약 2시간 15~19분), 09-09 19:30에는 E003, 09-04 16:45~23:47에도 E029. → 작업함의 "E029 반복 묶음" 줄의 근거.
- `test_3`: E016이 E029보다 먼저 떠서 규칙은 E02로 힌트를 주지만, 파일의 실제 클래스는 E04다.
- `test_2`: 실제 클래스는 E02다. 작업함의 "모델 이상 신호 · 코드 없음" 줄은 수동 모드를 보여주려는 **가정 사례**다.

## 코드 체계 네 가지 — 섞지 말 것

| 체계 | 예 | 뜻 | 화면 규칙 |
|---|---|---|---|
| 컨트롤러 코드 | E012 | 제어기가 1초마다 남기는 상태 코드 11종. 고장 순간의 4개(E012·E016·E028·E029)가 "종료 코드" | 의미 사전이 없으면 코드 원문만 표시 |
| 데이터셋 클래스 | E01~E04 | 논문이 파일에 붙인 고장 유형 | "추정" 표시와 함께. E012와 같은 칸에 넣지 않기 |
| 상황 ID | S01~S10 | 팀이 정한 상황 정의. ML과 RAG를 잇는 키 | 후보 카드의 단위 (FE 제안) |
| 매뉴얼 오류 번호 | Festo 12 | 다른 회사 제어기의 번호 체계 | "점검 후보"일 뿐, 실제로 그 오류가 떴다는 뜻이 아님 |

종료 코드 → 클래스 → 상황 ID (레포 `ml/.py/rag_mapping.py`):

| 종료 코드 | 클래스 | 한국어 이름 | 상황 ID | 정의 (RSW 상황 정의서) |
|---|---|---|---|---|
| E012 | E01 | 보정 압력 도달 지연 | S01 | 설정한 보정 압력에 1800 ms 안에 도달하지 못함 |
| E016 | E02 | 전극 파손 | S02 | 전극 위치가 기준 이동의 영점보다 작음 |
| E028 | E03 | 원치 않는 이동 | S03 | 실제 위치가 스트로크의 6.5% 넘게 이탈 |
| E029 | E04 | 드리프트 | S04 | 잠긴 실린더가 분당 5 mm 넘게 움직임 |

나머지 상황 ID (레포 `rag/pipeline/build.py`, `rag/data/situation_mapping.json`):

| ID | 이름 |
|---|---|
| S05 | 공기 공급 부족·누설로 전극 힘 저하 |
| S06 | 기계 마찰·걸림·윤활 부족 |
| S07 | 전극 캡 마모·드레싱 불량 |
| S08 | 축 보정 틀어짐·간섭에 의한 위치 이탈 |
| S09 | 냉각수·변압기·용접 제어기 등 주변 설비 이상 |
| S10 | 설정값 변경·잔고장 반복 (오경보 구분) |

## 센서 이름 (레포 `ml/main.py` SENSOR_NAME, `rag_mapping.py` SENSOR_KO)

| 키 | 영어 | 한국어 |
|---|---|---|
| c1 | Electrode cap offset | 전극 캡 오프셋 |
| c2 | Electrode force | 전극 힘 |
| c3 | Electrode position | 전극 위치 |
| c4 | Force build-up | 힘 형성 시간 |
| c5 | Balance pressure | 보정(밸런스) 압력 |
| c6 | Friction | 마찰 |
| c7 | Maximum aperture | 최대 열림 폭 |
| c8 | Maximum electrode force | 최대 전극 힘 |
| c9 | Start friction | 시작 마찰 |
| c10 | US2 | US2 작동 허용 신호 |
| c11 | Welding point count | (한국어 없음) |
| c12 | Position count | (한국어 없음) |
| c13 | Setpoint counterbalance pressure | 보정 압력 설정값 |
| c14 | Setpoint electrode force | 전극 힘 설정값 |
| c15 | Setpoint electrode position | 전극 위치 설정값 |
| c16 | Setpoint sheet thickness | 판 두께 설정값 |
| c17 | Setpoint velocity | 속도 설정값 |
| c18 | Setpoint force build-up | 힘 형성 설정값 |
| c19 | Offset value in robot | (한국어 없음, 모델에서 제외) |

- **단위는 어디에도 없다.** 화면에 단위를 지어 붙이지 않는다.
- ML이 주는 값(`value`, `reference`, `deviation`)은 실제 측정값이 아니라 **평소 대비 편차(z-score)**다.
- 한국어 이름은 핸드오프의 `sensor_findings`에만 들어 있다. trace와 `contributing_features`에는 영어 이름만 있다.

## 용어

| 용어 | 뜻 | FE에서 |
|---|---|---|
| 설정값 vs 실제값 | 설정값(c13~c18)은 제어기가 목표로 준 값, 실제값(c2~c6)은 센서가 잰 값 | 설정값은 계단선으로 |
| 비용접 구간 | 용접을 안 하는 시간(캡 드레싱 등). `c16 ≤ 0`. 이 구간의 이상 점수는 믿지 않음 | 빗금 띠 + "판단 보류" |
| 워밍업 (기준 수집) | 설비마다 처음 6시간(21600초) 동안 "평소 값"을 모아 기준을 만듦. 그동안 모델 알람은 보류되고 종료 코드 규칙만 동작 | 진행률 + "기준 수집 중" |
| 이상 점수 · 임계값 | 평소와 얼마나 다른지 나타내는 점수(대략 0~1)와 기준선(약 0.95). 고장 확률이 아님 | %로 바꾸지 않기 |
| 결측 | 통신이 끊겨 그 초의 기록이 없는 것. 값 0과 다름 | 선을 끊기. 보간·0 채우기 금지 |
| 오탐 | 시스템이 이상이라고 알렸는데 실제로는 문제가 없는 경우 | — |
| severity | normal / warning / critical. critical = 알람이 180초 이상 지속되거나 종료 코드 규칙 발화 | 작업함 경보 칸 |
| 핸드오프 | ML이 critical 때 만드는 전달 문서 (요약, 센서 증상, 고장 유형, 상황 ID) | 케이스의 재료 |
| 케이스 | 고장 사건 하나. 우리 서비스(api)가 만들고 저장해야 함 | 작업함의 한 줄 |
| 인계 | 권한·판단이 부족할 때 전문가에게 넘기는 것 | 어느 단계에서든 버튼 |
| MTTR | 평균 복구 시간. 팀 RAG 평가 문서의 KPI | 케이스 타임라인에서 계산 가능 |
| RAGAS | RAG 답변 품질 평가 지표 묶음 (충실성 등) | 후보 판단(맞음/아님)을 평가 데이터로 쓸 수 있음 |

## 매뉴얼 인용 (이전 대화에서 원문 확인한 것)

- Festo Servopneumatic drive system V1.5, Table 10.1
  - 오류 12 Timeout equalizer, p.90: "Equalizer pressure was not reached within 2 seconds." / "Check cable to the MPYD. Check supply pressure at least 5 bar."
  - 오류 3, p.89: "Check compressed air supply. Check tubing connection."
  - 오류 7: "… replace main cylinder"
- MILCO Modular Weld Guns, §4 p.11: "Air pressure should be verified at the weld cylinder with an air pressure gauge, not the gauge on the tooling." / "Air leaks from hoses, fittings, valves or the weld cylinder can affect tip pressure."

주의:
- Festo 오류 12의 "2초"와 RSW 정의서의 "1800 ms"는 서로 다른 기준이다.
- Festo 오류 12 칸에는 점검 문구만 있고 수리 단계가 없다. 화면에서 수리 방법을 지어내지 않는다.
- 현재 RAG에 들어 있는 문서는 **Festo 하나뿐**이다 (101쪽 중 32쪽). MILCO·ABB·FANUC·Yaskawa는 들어 있지 않다.
- 캔버스에서 쓰던 "적용 등급"(직접 적용 / 구조 참고 / 비교만)은 FE가 만든 개념이고 RAG에는 없다. Festo는 구동 방식이 같지만 장비 모델이 확인되지 않아 "구조 참고"로 표시해 왔다.
