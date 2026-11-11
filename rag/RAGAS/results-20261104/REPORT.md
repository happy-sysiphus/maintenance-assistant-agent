# RAGAS 평가 결과

평가일: 2026-11-04 (Asia/Seoul)

- 실행 상태: **complete**.
- 데이터셋: 50/50문항, 실제 답변: 50건, 채점 기록: 250/250건.
- 답변 상태: {'ok': 50}
- 이번 평가 기록 비용: **$2.47302** (표준 단가 추정, 입력 캐시 할인 미적용).
- 실제 토큰 사용량이 확인된 호출 비용: **$2.47302**. 나머지는 실패 요청의 보수적 예약 금액입니다.
- 토큰 사용량 미확인 요청: 0건. 해당 요청은 예약 상한으로 보수적으로 합산.
- 합성 기준 답변 추가 자동 검토: 50건 중 49건 지지 판정. 사람이 검증한 GT는 아닙니다.
- 채점 시도 총 255건, 실패/미정의 시도 5건. 일시 오류 재시도 시 이전 기록을 보존하고 최신 결과를 집계합니다.
- 재시도 입력의 수학 기호 표기 정규화(+/-·>=·<=) 적용: 5건. 평가자 유니코드 반복·출력 잘림에 대응한 동등 표기이며 원본 데이터는 보존했습니다. 재시도 출력 한도는 8192, 최초 한도는 4096입니다.

## 지표별 결과

평균은 유효 점수만 사용합니다. 실패·미정의 점수는 0으로 바꾸지 않으며 분모를 함께 보고합니다.

coded는 E 코드 질문 32건, general은 코드 없는 일반 질문 18건입니다. 아래 표의 n은 계획된 문항 수이며 실제 채점 분모는 상세 표를 확인하세요.

| 그룹 | n | Faithfulness | Relevancy | Context Precision | Context Recall | Factual F1 |
|---|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.9794 | 0.5337 | 0.6713 | 0.9133 | 0.5382 |
| coded | 32 | 0.9961 | 0.5453 | 0.7926 | 0.9427 | 0.5737 |
| E012 | 8 | 0.9931 | 0.6510 | 0.8651 | 0.7708 | 0.4413 |
| E016 | 8 | 0.9955 | 0.5778 | 0.6554 | 1.0000 | 0.6438 |
| E028 | 8 | 0.9957 | 0.5180 | 0.9000 | 1.0000 | 0.6038 |
| E029 | 8 | 1.0000 | 0.4345 | 0.7500 | 1.0000 | 0.6062 |
| general | 18 | 0.9497 | 0.5131 | 0.4556 | 0.8611 | 0.4750 |

### 유효 건수와 표준편차

| 그룹 | 지표 | 평균 | 표준편차 | 유효 | 실패/미정의 | 미실행 |
|---|---|---:|---:|---:|---:|---:|
| all | faithfulness | 0.9794 | 0.0443 | 50 | 0 | 0 |
| all | answer_relevancy | 0.5337 | 0.1462 | 50 | 0 | 0 |
| all | context_precision | 0.6713 | 0.2901 | 50 | 0 | 0 |
| all | context_recall | 0.9133 | 0.1992 | 50 | 0 | 0 |
| all | factual_correctness | 0.5382 | 0.2948 | 50 | 0 | 0 |
| coded | faithfulness | 0.9961 | 0.0106 | 32 | 0 | 0 |
| coded | answer_relevancy | 0.5453 | 0.1065 | 32 | 0 | 0 |
| coded | context_precision | 0.7926 | 0.1679 | 32 | 0 | 0 |
| coded | context_recall | 0.9427 | 0.1224 | 32 | 0 | 0 |
| coded | factual_correctness | 0.5737 | 0.3250 | 32 | 0 | 0 |
| E012 | faithfulness | 0.9931 | 0.0127 | 8 | 0 | 0 |
| E012 | answer_relevancy | 0.6510 | 0.0459 | 8 | 0 | 0 |
| E012 | context_precision | 0.8651 | 0.0858 | 8 | 0 | 0 |
| E012 | context_recall | 0.7708 | 0.1460 | 8 | 0 | 0 |
| E012 | factual_correctness | 0.4413 | 0.3408 | 8 | 0 | 0 |
| E016 | faithfulness | 0.9955 | 0.0126 | 8 | 0 | 0 |
| E016 | answer_relevancy | 0.5778 | 0.0810 | 8 | 0 | 0 |
| E016 | context_precision | 0.6554 | 0.2037 | 8 | 0 | 0 |
| E016 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E016 | factual_correctness | 0.6438 | 0.2981 | 8 | 0 | 0 |
| E028 | faithfulness | 0.9957 | 0.0122 | 8 | 0 | 0 |
| E028 | answer_relevancy | 0.5180 | 0.0641 | 8 | 0 | 0 |
| E028 | context_precision | 0.9000 | 0.1458 | 8 | 0 | 0 |
| E028 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E028 | factual_correctness | 0.6038 | 0.3371 | 8 | 0 | 0 |
| E029 | faithfulness | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E029 | answer_relevancy | 0.4345 | 0.0927 | 8 | 0 | 0 |
| E029 | context_precision | 0.7500 | 0.1091 | 8 | 0 | 0 |
| E029 | context_recall | 1.0000 | 0.0000 | 8 | 0 | 0 |
| E029 | factual_correctness | 0.6062 | 0.3480 | 8 | 0 | 0 |
| general | faithfulness | 0.9497 | 0.0633 | 18 | 0 | 0 |
| general | answer_relevancy | 0.5131 | 0.2005 | 18 | 0 | 0 |
| general | context_precision | 0.4556 | 0.3379 | 18 | 0 | 0 |
| general | context_recall | 0.8611 | 0.2873 | 18 | 0 | 0 |
| general | factual_correctness | 0.4750 | 0.2266 | 18 | 0 | 0 |

## 시간 및 비용

- RAG 응답 평균: 10.55544초.
- RAG 응답 p95: 14.763초. 병렬 평가 환경의 측정이며 단일 사용자 지연과 다를 수 있습니다.

- 최종 데이터셋 답변 생성 비용: $1.135037, 답변 1건 평균: 0.0227007356 USD. 평가자 비용·제외한 초안 비용은 이 평균에서 제외했습니다.

| 단계 | 비용(USD) |
|---|---:|
| answer_relevancy | $0.087928 |
| chunk_comparison | $0.002057 |
| connection_check | $0.000502 |
| context_precision | $0.241056 |
| context_recall | $0.069397 |
| factual_correctness | $0.304170 |
| faithfulness | $0.292718 |
| index | $0.000001 |
| rag | $1.475191 |

## Factual Correctness F1 해석 보조

아래 값은 저장된 평가자 NLI 판단에서 파생한 설명용 비율이며 추가 공식 RAGAS 지표가 아닙니다.
공식 F1 점수 재계산이 일치하는 문항만 집계하고 새 LLM 호출은 하지 않았습니다.

- 분석 가능한 문항: 49개.
- 기준 답변 주장 중 실제 RAG 답변이 지지한다고 평가된 비율의 평균: 0.8336734693877551.
- RAG 답변 주장 중 짧은 기준 답변이 지지한다고 평가된 비율의 평균: 0.4438613201040966.
- 앞 비율보다 뒤 비율이 낮다면 정답의 핵심은 포함하지만 기준에 없는 보충 설명이 많아 F1이 낮아지는 양상이 있을 수 있습니다. 보충 설명이 옳다는 보장은 아니며 Faithfulness·원문 대조와 함께 봐야 합니다.
- 주장별 수와 분모는 factual_breakdown.json에 있습니다.

## 낮은 점수 사례

정답 여부의 확정 판정이 아니라 우선 검토 대상입니다.

### faithfulness

- general-manual82-64-01: 0.8000 — WinSPAS의 Simulation 탭은 어떤 용도로 사용할 수 있으며, 시뮬레이션은 어떤 조건이 충족되면 다음 단계로 전환되는가?
- general-whitepaper11-9-01: 0.8462 — 상태 모니터링과 AI 모델을 활용한 결과, 용접건 데이터 분석을 통해 어떤 유지보수 관련 개선 효과가 나타났는가?
- general-manual82-53-01: 0.8667 — 전송 보정(travel calibration)은 무엇을 결정하는 데 사용되며, 어떤 조건의 전극 캡에서 수행해야 합니까?

### answer_relevancy

- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?
- general-festo-40-01: 0.3137 — 이퀄라이저 설정 전에 준비 단계에서 PC 제어, 건의 위치 모드 전환, 전극 개방 폭, 용접력 지정은 어떻게 해야 합니까?
- E029-08: 0.3280 — E029 오류의 감지조건은 상대 드리프트와 절대 드리프트에서 어떻게 다르게 적용되는가?

### context_precision

- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?
- general-festo-43-01: 0.0000 — 프로젝트 데이터를 파일로 저장할 때 수행해야 하는 기본 절차는 무엇인가요?
- general-manual82-66-01: 0.1250 — WinSPAS를 사용해 컨트롤러의 펌웨어를 업데이트할 때 시작 전에 어떤 주의사항과 절차를 따라야 합니까?

### context_recall

- general-manual82-77-01: 0.0000 — CPX-WELD-2G-SA의 전원 공급 전압과 정격 동작 전류는 얼마인가?
- general-manual82-53-01: 0.5000 — 전송 보정(travel calibration)은 무엇을 결정하는 데 사용되며, 어떤 조건의 전극 캡에서 수행해야 합니까?
- general-manual82-76-01: 0.5000 — WinSPAS에서 진단 기능으로 지원하는 항목은 무엇인가?

### factual_correctness

- E012-04: 0.0000 — E012 오류에서 우선 확인해야 할 조건은 무엇인가요?
- general-whitepaper11-9-01: 0.0000 — 상태 모니터링과 AI 모델을 활용한 결과, 용접건 데이터 분석을 통해 어떤 유지보수 관련 개선 효과가 나타났는가?
- E029-03: 0.0800 — E029 오류의 원인 관계는 상대 드리프트와 절대 드리프트 중 무엇과 더 직접적으로 연결되는가?

## 기준 답변 자동 검토에서 표시된 문항

- E016-07: 원문 인용에는 E016 오류의 정의와 감시 조건에 대한 설명만 있으며, 매뉴얼 기준 조치 범위에 대한 구체적인 내용이나 수리 절차가 포함되어 있지 않음.; 질문은 E016의 조치 범위를 묻고 있으나, 인용된 원문은 조치 범위가 아닌 오류 감시 조건과 설정값 확인 수준만 언급함.; 따라서 질문에 대한 답변으로서 조치 범위를 명확히 제시하지 못하고 있어 불충분함.

표시된 문항도 삭제·수정하지 않고 전체 점수에 포함했습니다. 해당 reference를 먼저 검토해야 합니다.

## 평가 해석과 한계

- 기준 답변은 PDF 원문에서 별도로 생성한 합성 reference입니다. 사람이 검증한 gold GT가 아닙니다.
- 생성기는 층화 비율을 맞추는 자체 PDF 기반 생성기이며 RAGAS TestsetGenerator를 실행한 결과는 아닙니다. 채점은 설치한 공식 ragas 라이브러리의 지표를 그대로 사용합니다.
- 합성 질문·기준 답변 생성기는 gpt-5.4-mini, 평가자는 gpt-4.1-mini, 이번 RAG 답변 모델은 claude-sonnet-5-5입니다. 자동 평가의 편향이 가능하고 독립 전문가 평가가 아닙니다.
- 초기 데이터셋 생성에서 제외한 초안과 그 생성 비용은 기존 평가 결과에 보존되어 있습니다. 비교 평가에서는 같은 최종 50문항을 재사용하며 비용 집계는 이번 결과 디렉터리의 호출만 포함합니다.
- E 코드 32건, 일반 질문 18건이며 history·사진·센서·미지원 질문·답변 보류 전용 세트는 평가하지 않습니다.
- 같은 오류 코드를 반복하는 질문들은 상관되어 있습니다. 50문항을 완전히 독립인 50개 문제로 해석하지 않습니다.
- 기준 답변 원문 인용이 해당 PDF 페이지에 존재하는지는 자동 검사했습니다. 이것만으로 기준 답변 전체의 정확성을 보증하지 않습니다.
- Faithfulness에는 실제 생성 입력의 검색 원문과 검토 매핑·적용 범위를 포함합니다. Context Precision/Recall에는 순서가 보존된 검색 원문만 사용합니다.
- Context Precision은 reference 기준, Context Recall은 reference의 사실 회수율입니다. Factual Correctness는 짧은 reference와의 F1으로, 정당한 보충 설명도 감점될 수 있습니다.
- 공식 기본 영문 평가 프롬프트를 유지했습니다. Answer Relevancy의 역생성 질문은 영어로 출력되는 경우가 있어 한국어 질문과의 다국어 임베딩 유사도 영향을 받습니다. 이 점수를 단독으로 한국어 답변 품질의 절대 척도로 해석하지 않습니다.
- 사용자에게 전달되는 전체 answer(점검·기준·조치 포함)를 평가합니다. 일부만 골라 점수를 높이지 않습니다.
- 캐시는 사용하지 않았고, 서비스 실패도 보존합니다. 이 실행의 채점 도중 점수에 맞춰 답변을 수정하지 않습니다. 개선 전후의 파싱·검색·모델 차이는 별도 비교 보고서에서 설명합니다.
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
