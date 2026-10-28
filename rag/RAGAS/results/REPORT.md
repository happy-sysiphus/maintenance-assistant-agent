# RAGAS 평가 결과

평가일: 2026-10-28 (Asia/Seoul)

- 실행 상태: **complete**.
- 데이터셋: 50/50문항, 실제 답변: 50건, 채점 기록: 250/250건.
- 답변 상태: {'ok': 50}
- 이번 평가 기록 비용: **$1.19326** (표준 단가 추정, 입력 캐시 할인 미적용).
- 실제 토큰 사용량이 확인된 호출 비용: **$1.15839**. 나머지는 실패 요청의 보수적 예약 금액입니다.
- 토큰 사용량 미확인 요청: 3건. 해당 요청은 예약 상한으로 보수적으로 합산.
- 합성 기준 답변 추가 자동 검토: 50건 중 49건 지지 판정. 사람이 검증한 GT는 아닙니다.
- 채점 시도 총 255건, 실패/미정의 시도 5건. 일시 오류 재시도 시 이전 기록을 보존하고 최신 결과를 집계합니다.
- 재시도 입력의 수학 기호 표기 정규화(+/-·>=·<=) 적용: 5건. 평가자 유니코드 반복·출력 잘림에 대응한 동등 표기이며 원본 데이터는 보존했습니다. 재시도 출력 한도는 8192, 최초 한도는 4096입니다.

## 지표별 결과

평균은 유효 점수만 사용합니다. 실패·미정의 점수는 0으로 바꾸지 않으며 분모를 함께 보고합니다.

coded는 E 코드 질문 32건, general은 코드 없는 일반 질문 18건입니다. 아래 표의 n은 계획된 문항 수이며 실제 채점 분모는 상세 표를 확인하세요.

| 그룹 | n | Faithfulness | Relevancy | Context Precision | Context Recall | Factual F1 |
|---|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.9795 | 0.5280 | 0.7023 | 0.8750 | 0.4404 |
| coded | 32 | 0.9984 | 0.5540 | 0.8480 | 0.9714 | 0.4100 |
| E012 | 8 | 1.0000 | 0.6455 | 0.7876 | 0.8854 | 0.3600 |
| E016 | 8 | 1.0000 | 0.5764 | 0.8140 | 1.0000 | 0.4525 |
| E028 | 8 | 0.9938 | 0.5303 | 0.8634 | 1.0000 | 0.4163 |
| E029 | 8 | 1.0000 | 0.4637 | 0.9272 | 1.0000 | 0.4113 |
| general | 18 | 0.9457 | 0.4817 | 0.4433 | 0.7037 | 0.4944 |

### 유효 건수와 표준편차

| 그룹 | 지표 | 평균 | 표준편차 | 유효 | 실패/미정의 | 미실행 |
|---|---|---:|---:|---:|---:|---:|
| all | faithfulness | 0.9795 | 0.0774 | 50 | 0 | 0 |
| all | answer_relevancy | 0.5280 | 0.1762 | 50 | 0 | 0 |
| all | context_precision | 0.7023 | 0.2938 | 50 | 0 | 0 |
| all | context_recall | 0.8750 | 0.2898 | 50 | 0 | 0 |
| all | factual_correctness | 0.4404 | 0.2214 | 50 | 0 | 0 |
| coded | faithfulness | 0.9984 | 0.0088 | 32 | 0 | 0 |
| coded | answer_relevancy | 0.5540 | 0.0973 | 32 | 0 | 0 |
| coded | context_precision | 0.8480 | 0.1335 | 32 | 0 | 0 |
| coded | context_recall | 0.9714 | 0.0913 | 32 | 0 | 0 |
| coded | factual_correctness | 0.4100 | 0.1580 | 32 | 0 | 0 |
| E012 | faithfulness | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E012 | answer_relevancy | 0.6455 | 0.0758 | 8 | 0 | 0 |
| E012 | context_precision | 0.7876 | 0.1107 | 8 | 0 | 0 |
| E012 | context_recall | 0.8854 | 0.1602 | 8 | 0 | 0 |
| E012 | factual_correctness | 0.3600 | 0.1143 | 8 | 0 | 0 |
| E016 | faithfulness | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E016 | answer_relevancy | 0.5764 | 0.0720 | 8 | 0 | 0 |
| E016 | context_precision | 0.8140 | 0.1483 | 8 | 0 | 0 |
| E016 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E016 | factual_correctness | 0.4525 | 0.0671 | 8 | 0 | 0 |
| E028 | faithfulness | 0.9938 | 0.0177 | 8 | 0 | 0 |
| E028 | answer_relevancy | 0.5303 | 0.0868 | 8 | 0 | 0 |
| E028 | context_precision | 0.8634 | 0.1708 | 8 | 0 | 0 |
| E028 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E028 | factual_correctness | 0.4163 | 0.2610 | 8 | 0 | 0 |
| E029 | faithfulness | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E029 | answer_relevancy | 0.4637 | 0.0577 | 8 | 0 | 0 |
| E029 | context_precision | 0.9272 | 0.0512 | 8 | 0 | 0 |
| E029 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E029 | factual_correctness | 0.4113 | 0.1409 | 8 | 0 | 0 |
| general | faithfulness | 0.9457 | 0.1235 | 18 | 0 | 0 |
| general | answer_relevancy | 0.4817 | 0.2621 | 18 | 0 | 0 |
| general | context_precision | 0.4433 | 0.3245 | 18 | 0 | 0 |
| general | context_recall | 0.7037 | 0.4222 | 18 | 0 | 0 |
| general | factual_correctness | 0.4944 | 0.3015 | 18 | 0 | 0 |

## 시간 및 비용

- RAG 응답 평균: 4.30954초.
- RAG 응답 p95: 5.895초. 예비 2건·확대 3건 병렬 환경에서 측정했으며 단일 사용자 지연과 다를 수 있습니다.

- 최종 데이터셋 답변 생성 비용: $0.205047, 답변 1건 평균: 0.004100937 USD. 평가자 비용·제외한 초안 비용은 이 평균에서 제외했습니다.

| 단계 | 비용(USD) |
|---|---:|
| answer_relevancy | $0.070568 |
| context_precision | $0.278649 |
| context_recall | $0.088259 |
| dataset_generation | $0.100050 |
| factual_correctness | $0.231045 |
| faithfulness | $0.187028 |
| rag | $0.225584 |
| reference_audit | $0.012075 |

## Factual Correctness F1 해석 보조

아래 값은 저장된 평가자 NLI 판단에서 파생한 설명용 비율이며 추가 공식 RAGAS 지표가 아닙니다.
공식 F1 점수 재계산이 일치하는 문항만 집계하고 새 LLM 호출은 하지 않았습니다.

- 분석 가능한 문항: 48개.
- 기준 답변 주장 중 실제 RAG 답변이 지지한다고 평가된 비율의 평균: 0.6399305555555556.
- RAG 답변 주장 중 짧은 기준 답변이 지지한다고 평가된 비율의 평균: 0.33693380649334514.
- 앞 비율보다 뒤 비율이 낮다면 정답의 핵심은 포함하지만 기준에 없는 보충 설명이 많아 F1이 낮아지는 양상이 있을 수 있습니다. 보충 설명이 옳다는 보장은 아니며 Faithfulness·원문 대조와 함께 봐야 합니다.
- 주장별 수와 분모는 factual_breakdown.json에 있습니다.

## 낮은 점수 사례

정답 여부의 확정 판정이 아니라 우선 검토 대상입니다.

### faithfulness

- general-manual82-59-01: 0.5000 — WinSPAS에서 용접건을 제어할 때, ‘Controller’ 패널이나 필드버스로 마스터 제어를 전환하면 어떤 위험이 있으며 어떤 주의가 필요한가?
- general-festo-41-01: 0.8333 — 압력 상승 설정은 어떤 목적이며, 설정할 때 어떤 값을 조정해야 합니까?
- general-manual82-21-01: 0.8571 — 보정 실린더를 연결할 때 차압 설정은 용접물 쪽 이동과 스톱 쪽 이동에 각각 어떻게 해야 합니까?

### answer_relevancy

- general-manual82-66-01: 0.0000 — WinSPAS를 사용해 컨트롤러의 펌웨어를 업데이트할 때 시작 전에 어떤 주의사항과 절차를 따라야 합니까?
- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?
- general-whitepaper11-6-01: 0.0000 — 예지보전을 위한 단계별 계획은 어떤 단계들로 구성되는가?

### context_precision

- general-manual82-59-01: 0.0000 — WinSPAS에서 용접건을 제어할 때, ‘Controller’ 패널이나 필드버스로 마스터 제어를 전환하면 어떤 위험이 있으며 어떤 주의가 필요한가?
- general-manual82-66-01: 0.0000 — WinSPAS를 사용해 컨트롤러의 펌웨어를 업데이트할 때 시작 전에 어떤 주의사항과 절차를 따라야 합니까?
- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?

### context_recall

- general-manual82-59-01: 0.0000 — WinSPAS에서 용접건을 제어할 때, ‘Controller’ 패널이나 필드버스로 마스터 제어를 전환하면 어떤 위험이 있으며 어떤 주의가 필요한가?
- general-manual82-66-01: 0.0000 — WinSPAS를 사용해 컨트롤러의 펌웨어를 업데이트할 때 시작 전에 어떤 주의사항과 절차를 따라야 합니까?
- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?

### factual_correctness

- E028-04: 0.0000 — E028에서 우선 확인해야 할 조건은 무엇인가?
- general-manual82-59-01: 0.0000 — WinSPAS에서 용접건을 제어할 때, ‘Controller’ 패널이나 필드버스로 마스터 제어를 전환하면 어떤 위험이 있으며 어떤 주의가 필요한가?
- general-whitepaper11-5-01: 0.0000 — 서보-공압 용접건의 시스템 구성 요소는 무엇인가?

## 기준 답변 자동 검토에서 표시된 문항

- E016-07: 원문 인용에는 E016 오류의 정의와 감시 조건에 대한 설명만 있으며, 매뉴얼 기준 조치 범위에 대한 구체적인 내용이나 수리 절차가 포함되어 있지 않음.; 질문은 E016의 조치 범위를 묻고 있으나, 인용된 원문은 조치 범위가 아닌 오류 감시 조건과 설정값 확인 수준만 언급함.; 따라서 질문에 대한 답변으로서 조치 범위를 명확히 제시하지 못하고 있어 불충분함.

표시된 문항도 삭제·수정하지 않고 전체 점수에 포함했습니다. 해당 reference를 먼저 검토해야 합니다.

## 평가 해석과 한계

- 기준 답변은 PDF 원문에서 별도로 생성한 합성 reference입니다. 사람이 검증한 gold GT가 아닙니다.
- 생성기는 층화 비율을 맞추는 자체 PDF 기반 생성기이며 RAGAS TestsetGenerator를 실행한 결과는 아닙니다. 채점은 설치한 공식 ragas 라이브러리의 지표를 그대로 사용합니다.
- 최종 질문·기준 답변 생성기는 gpt-5.4-mini, 평가자는 gpt-4.1-mini입니다. 생성기와 RAG가 같은 모델 계열이며 자동 평가의 편향이 가능하고 독립 전문가 평가가 아닙니다.
- 사전 품질검사에서 다른 코드 질문·불완전한 기준 답변·시간 임계값 오독이 발견된 초기 초안은 최종 평가 전에 제외했습니다. 생성기를 강화하고 오류별 표 행만 추출했으며, 제외된 초안의 API 비용도 모두 포함했습니다.
- E 코드 32건, 일반 질문 18건이며 history·사진·센서·미지원 질문·답변 보류 전용 세트는 평가하지 않습니다.
- 같은 오류 코드를 반복하는 질문들은 상관되어 있습니다. 50문항을 완전히 독립인 50개 문제로 해석하지 않습니다.
- 기준 답변 원문 인용이 해당 PDF 페이지에 존재하는지는 자동 검사했습니다. 이것만으로 기준 답변 전체의 정확성을 보증하지 않습니다.
- Faithfulness에는 실제 생성 입력의 검색 원문과 검토 매핑·적용 범위를 포함합니다. Context Precision/Recall에는 순서가 보존된 검색 원문만 사용합니다.
- Context Precision은 reference 기준, Context Recall은 reference의 사실 회수율입니다. Factual Correctness는 짧은 reference와의 F1으로, 정당한 보충 설명도 감점될 수 있습니다.
- 공식 기본 영문 평가 프롬프트를 유지했습니다. Answer Relevancy의 역생성 질문은 영어로 출력되는 경우가 있어 한국어 질문과의 다국어 임베딩 유사도 영향을 받습니다. 이 점수를 단독으로 한국어 답변 품질의 절대 척도로 해석하지 않습니다.
- 사용자에게 전달되는 전체 answer(점검·기준·조치 포함)를 평가합니다. 일부만 골라 점수를 높이지 않습니다.
- 캐시는 사용하지 않았고, 기존 서비스의 실패는 그대로 보존합니다. 점수에 맞춰 프롬프트나 KG를 수정하지 않았습니다.
- 숫자는 LLM 기반 자동 평가의 추정치이며 0.9 같은 보편적 합격선은 적용하지 않았습니다.
- 지표별 점수는 metrics.jsonl, 평가자 판단 원문은 judge_traces.jsonl, 질문·합성 기준은 dataset.jsonl, RAG 원본 응답은 responses.jsonl, 비용은 costs.jsonl에 있습니다.

## 공식 근거

- [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)
- [Response Relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/)
- [Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/)
- [Context Recall](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/)
- [Factual Correctness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/factual_correctness/)
- [Testset Generation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/)
- [평가자 단가](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
