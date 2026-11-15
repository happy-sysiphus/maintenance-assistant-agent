# 11/15 main 호환 시계열 RAG 통합

기준 main: `236fb09f26855508e9c7463cb5ad5c88c6149a80` (`maintenance-assistant-agent-main (15).zip`).
입력 브랜치: `maintenance-assistant-agent-feat-timeseries-rag-demo (1).zip`.
이 문서는 제공된 두 ZIP을 로컬에서 통합·검증한 결과다. GitHub에 업로드하거나 PR을 생성·머지한 상태는 아니다.

## 무엇을 합쳤는가

- PR #7의 19채널 센서 시계열 복원·과거 해결 이력 비교, Seen/Unseen/판단 보류, 제어기 E코드 우선 경로, Aura 기록을 가져왔다.
- ML의 `episode_id`를 가져와 같은 에피소드의 WARNING과 후속 제어기 CRITICAL을 연결했다. 반복 중 CRITICAL이 있으면 작업함도 긴급으로 표시한다.
- main의 EarlyPanel/EarlyManual/EarlySteps 화면을 유지했다. 삭제된 WarningPage와 그 옛 테스트를 복원하지 않는다.
- EarlyPanel에서 판정을 한 번 요청하고, 비교 사례·매뉴얼 코드 카드·원문 기반 후보를 표시한다. 이전 해결 카드에는 실제 과거 조치를 보여 준다.
- 등록 사례는 `manual_case_id`, 신규 PDF 후보는 `evidence_id`로 선택한다. 수동 검색에서도 등록 사례와 원문 근거를 각각 선택할 수 있다.
- 선택한 `event_id`를 점검→조치→정비일지와 뒤로 가기 URL에 유지한다. 이후 경보를 선택한 기록이 있어도 명시적으로 고른 이전 경보의 기록을 덮어쓰지 않는다.
- SQLite의 원본 이벤트와 반복 handoff를 유지하고, 화면용 최신 코드만 `display_event`로 따로 제공한다.
- 설명 생성만 실패한 수동 검색은 `retrieval_only` 상태와 찾은 PDF 근거를 계속 표시한다.

## main에서 보존한 것

| 항목 | 처리 |
|---|---|
| 원인 입력 `cause` | 초안·해결 기록·정비일지에 보존. 수동 PDF 근거로 해결할 때 필수 |
| 대화 이력 `history` | 기존 경로와 시계열 수동 검색 모두 전달·검증 |
| 반복 경보 마지막 시각 `last_detected_at` | 목록 API와 UI 표시 유지 |
| WARNING/CRITICAL JSON 공통 필드 | main의 rag_mapping 기본 필드·null 전송 유지 |
| 기존 SQLite 저장·전송 대기열 | 기존 DB를 삭제·초기화하지 않음. 기존 테이블 사용 |
| ML 피드백 | 기존 `/api/ml/feedback`, model_feedback/model_loop 유지 |
| 현장 기록과 R2 화면 | main 코드 보존 |
| 기존 케이스 | `rag_mode`가 없는 케이스는 원래 WARNING/CRITICAL 경로 사용 |

시계열 해결 이력 재사용은 기존 방향 패턴의 '2건부터 자동 카드' 규칙과 다르다. 검토된 해결 이력과 현재 파형을 비교해 판단하므로 화면에서 해결 수 2건 규칙을 잘못 안내하지 않는다.

## 적용 및 설정

Revert된 옛 브랜치를 다시 머지하는 대신, 최신 main에서 새 브랜치를 만든 뒤 이 변경을 적용한다.
제공한 패치는 위 main 스냅샷 기준이다. 이후 같은 파일이 바뀌었다면 자동 적용 검사가 멈추므로 새 main에 맞춰 다시 조정한다.

실제 `rag/.env`는 패치에 포함하지 않았다. 환경 담당자가 기존 연결 정보를 유지하고 아래 값을 확인한다.

```dotenv
RAG_WARNING_TIMESERIES=1
RAG_WARNING_NAMESPACE=team-rag
RAG_WARNING_EQUIPMENT_SCOPE=cpx-weld-2g-sa
RAG_WARNING_FEEDBACK=1
```

namespace와 equipment scope는 팀이 사용 중인 구역·설비에 맞춘다. 위 이름으로 임의 변경하지 않는다.
설정 예제 파일의 `RAG_WARNING_TIMESERIES=0`은 기본값이며, 실제 환경에서 1로 켜야 새 v1.4 이벤트가 시계열 경로로 들어간다.
이미 저장된 케이스는 자동 전환되지 않는다. 적용 후에는 새 event_id로 새 경보를 보내 확인한다.

RAG 의존성이 설치된 Python으로 저장소 루트에서 한 번 실행한다.

```bash
python -m rag.warning_assessment init
```

기존 Aura에 필요한 제약과 인덱스를 준비하는 명령이다. 이후 API·ML을 재시작하고 UI를 다시 빌드한다.

## 검증

테스트는 외부 검색·LLM·Aura를 대역으로 바꾸고 실제 API 트랜잭션과 임시 SQLite를 사용했다.

- API·RAG: 112 통과, 실제 DB 설정이 필요한 테스트 11개 건너뜀.
- UI: 실제 API 테스트 서버를 붙인 8개 흐름 통과. 기존 조기경보 정비, 후속 CRITICAL 해결 종료, Unseen PDF 선택·원인 저장, Seen 비교 사례 표시, 설명 실패 후 근거 선택, 수동 검색 등록 사례 선택, 과거 이벤트 명시 선택을 확인했다.
- UI 타입 검사·프로덕션 빌드·린트 통과. 빌드의 기존 큰 번들 안내는 남아 있다.
- ML 핸드오프 계약: 26 통과.
- ML 파이프라인: 88 통과. 전처리·학습·온라인 추론·episode_id·핸드오프·피드백 루프 회귀 포함.
- Python 구문 컴파일 통과.

ML 검증 환경에는 pandas 2.3.3을 사용했다. pandas 3.x에서는 기존 변경하지 않은 조기경보 테스트가 시간 인덱스 정밀도 차이로 실패해, 저장소의 개발 계열인 pandas 2.x 환경에서 확인했다. 의존성 범위 파일은 이번 통합에서 변경하지 않았다.

이번 환경에서 실제 Aura·Claude·현장 센서 연동과 실제 브라우저 클릭 검증은 수행하지 않았다. UI 흐름 테스트는 jsdom에서 실행했다.
`docs/timeseries-rag-browser-demo.html`은 원래 PR #7의 과거 화면 캡처다. 현재 화면의 검증 자료로 해석하지 않는다.
브라우저 시연 스크립트는 Early* 화면으로 수정했으며 실제 제공자 연결 재실행은 별도다.

## 검토 후 정리

패치 바깥의 적용 도구·기준 해시·변경 파일 목록은 PR에 넣지 않아도 된다. PR 완료 후 내려받은 적용 패키지는 삭제해도 된다.
실제 코드와 `docs/timeseries-main-integration-20261115.md`는 저장소에 남긴다.
