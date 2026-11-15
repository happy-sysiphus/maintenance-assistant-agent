# 11. 조기 경보 화면 — 11/12 FE

2026-11-12 · FE 담당 (한혜원)

기준:
- 11/11 회의 흐름도: 사례 카드 / 첫 화면 → 후보 목록(LLM) → 수동 모드 → 정비 후 해결 = Aura에 연결 저장. 오알람은 닫고 끝.
- api: PR #5(구동열, 11/12 머지) `/api/cases/:id/warning/*`, 계약은 [ui/docs/10-early-warning-rag-ui.md](10-early-warning-rag-ui.md) 3절.
- PR #5의 `WarningPage`(한 화면에 전부)는 쓰지 않고 지웠다. api 연결 방식은 그대로 가져왔다.

[10-card-order-early-warning.md](10-card-order-early-warning.md)의 "조기 경보" 부분(11/08)은 이 문서로 바뀌었다. R2 부분은 그대로 유효하다.

9절은 원본 main의 반영 이력이다. 이 통합본은 PR #7의 API·RAG를 새 Early* 화면에 연결했다. 현재 동작은 10절과 [통합 안내](../../docs/timeseries-main-integration-20261115.md)를 참고한다.

## 1. 화면 흐름

```
작업함 "주의" 줄
  └ 상황 화면 (긴급과 같은 틀: 왼쪽 그래프 + 오른쪽 할 일 칸)
      ├ A. 사례 카드     같은 신호 패턴을 2번 이상 고친 방법 → [이 방법으로 점검] / 다른 원인 보기
      ├ B. 첫 화면       → [원인 후보 보기] (Claude 호출은 이 버튼에서만)
      └ C. 후보 목록     이전에 고친 방법이 맨 위 + 매뉴얼 후보 → [N번 원인으로 점검] / 맞는 원인이 없어요 · 직접 찾기
            └ D. 수동 모드  케이스 안 챗봇 모양 검색 → [이 근거로 정비]
  점검 (이상 있음 → 조치 / 모두 정상 → "이상이 없어요" + [다음 원인 보기]) → 조치 → 정비일지 ("조치 후 정상으로 돌아온 것을 확인했어요" + [일지 승인하고 해결 종료])
  └ 상황 화면: "✓ 해결로 닫았어요 · 이 신호 패턴 해결 기록 n/2"
오알람으로 닫기: 어느 화면에서든, 기존 팝업 · 사유 칩. 해결 기록에는 안 들어간다.
```

원칙 (기존 화면과 같게): 화면마다 파란 큰 버튼 하나, 설명은 1~2줄, 근거는 쪽 칩 → 오른쪽 PDF.

## 2. 정비 단계에서 뺀 것

조기 경보는 **점검 · 조치 · 정비일지** 3단계다. 긴급의 판단 · 결과 단계가 없다.

| 긴급 | 조기 경보에서 대신하는 것 | 이유 |
|---|---|---|
| 판단 ("이 원인이 맞나요?") | 후보 목록에서 이미 고름. 점검이 모두 정상이면 "점검한 항목에 이상이 없어요" + [다음 원인 보기] → 후보 목록(방금 원인은 "점검함 · 이상 없음", 다음 원인이 골라짐). 사례 카드로 들어왔으면 그 카드를 거절하고 후보를 받는다. 후보를 다 점검했으면 목록을 거치지 않고 버튼이 [매뉴얼에서 직접 찾기]로 바뀌어 바로 수동 모드. 사례 카드에서 받은 후보가 0개여도 바로 수동 모드 | 같은 질문을 두 번 하지 않는다. api에 저장할 곳도 없다 |
| 결과 ("고쳐졌나요?") | 정비일지의 해결 확인 체크 하나 | api `/resolve`가 `worker_confirmed`를 요구한다 |

조치는 api에 글 한 칸(`actual_action`)뿐이라 "종류 — 한 일"로 합쳐 저장한다. 부품 · 작업 시각 칸은 없다.

## 3. 피드백 루프가 보이는 곳

| 화면 | 문구 |
|---|---|
| 해결로 닫은 뒤 (상황 화면) | 이 신호 패턴 해결 기록 **1/2** · 2건이 되면 다음 경보부터 바로 보여 줘요 |
| 다음 경보 · 후보 목록 | 이전에 **1번** 고친 방법 |
| 그다음 경보 · 사례 카드 | 이 신호 패턴에서 **2번** 해결했어요 · 지난 조치 |

수동 모드에서 고른 근거로 해결하면 검토 대기(`pending_review`)라 검토 전에는 사례 카드에 쓰이지 않는다(화면 문구로 안내).

## 4. 반복 경보

같은 설비에서 여러 번 울려 한 케이스로 묶였으면 상황 화면에 "이 설비에서 N회" 고르기가 나온다(원본이 남은 경보만). 원인을 고른 경보가 이후 점검 · 조치 · 정비일지 · 종료의 경보가 된다(`lib/warning.ts` `activeEventId`).

## 5. api 변경 (FE, 11/12)

- `PUT /warning/draft` · `POST /warning/resolve`에 `cause`(작업자가 적은 원인) 추가. 직접 찾은 근거로 해결할 때는 필수(없으면 422), 정비일지 · 이력의 원인이 된다. 사례 카드면 카드 제목.
- `POST /warning/manual`에 `history`(대화 이력) 추가 — 수동 모드가 앞 질문을 이어서 찾는다. RAG `search_manuals(history=)`는 원래 받았다. 테스트 `test_manual_search_passes_chat_history`.
- 테스트 환경: `RAG_WARNING_EQUIPMENT_SCOPE`를 지우던 것을 빈 값으로. 지우면 `rag/.env`의 개인 설정이 다시 읽혀 PC마다 테스트 결과가 달랐다(api 14개 · UI 1개 실패).

## 6. 설정 · 시연 데이터

- `rag/.env`: `RAG_WARNING_NAMESPACE=fe-test`(FE 테스트 이력 구역, 시연 때 `team-rag`), `RAG_WARNING_EQUIPMENT_SCOPE=cpx-weld-2g-sa`(동열 11/12 "일단 이대로"), `RAG_WARNING_FEEDBACK=1`.
- 적용 범위가 비어 있으면 사례 카드가 절대 안 뜬다(해결 수가 0으로 세짐).
- 실제 재생(test_0·1·3·4·6 마지막 12시간)의 조기 경보 9건 중 채널 3개는 2건, 패턴이 서로 달라 사례 카드 시연이 안 된다. 시연용: test_0 경보를 번호 · 설비(`demo-ew`)만 바꿔 3번 보냄 → CASE-0018 · 0019 해결 → CASE-0020 사례 카드. 요약 · 조치에 `[시연용]`.

## 7. 확인

- `npm run test:warning` (`WARNING_TEST_PYTHON=../.venv/bin/python`, 맥은 `python`이 없다): 첫 화면 → 후보 → 점검 → 조치 → 정비일지 → 해결 / 후보 → 수동 모드 → 근거 선택. 실제 api(합성 서버), 검색 · Claude · Aura만 대역.
- api 109 통과 · 2 건너뜀, 타입 체크 · 린트 · 빌드 통과.
- 브라우저: 화면은 확인, 해결까지 끝까지 누른 확인은 아직.

## 8. 남은 것

- 작업함 조기 경보의 상태 글자("새 고장")를 바꿀지.
- 시연 때 namespace `team-rag`로 바꿀지, 시연용 경보를 써도 되는지 팀 확인.

## 9. main 반영 범위 — 11/15

커밋 `4814397` "feat(ui): 조기 경보 화면을 상황 화면 흐름으로 다시 구성"을 main에 바로 올렸다(PR 없음). 이 절은 **무엇을 기준으로 올렸고, 무엇이 안 들어갔고, 어디를 바꿨는지**만 적는다.

### 9.1 기준

- **main `25e4377` 위에 올렸다.** 그 시점 main에 있던 것만 기준으로 했다.
  - 들어 있던 것: PR #5(WARNING api · 공용 Aura 해결 이력), ML 루프 D-11 · D-12(`GET /api/ml/feedback`), C-9(코드 없는 CRITICAL은 케이스로 만들지 않음)
- 확인: api pytest 58 통과 · 1 건너뜀, `tsc -b` · `oxlint` · `npm run test:warning`(EarlyFlow 2개) 통과.

### 9.2 안 들어간 것

| 무엇 | 상태 | 이 화면과의 관계 |
|---|---|---|
| **PR #7 시계열 RAG** (구동열, `feat/timeseries-rag-demo`) | 11/14 머지 → 9분 뒤 PR #8로 되돌림. 지금 main에 없다 | 반영하지 않았다. 9.3 |
| 11/12 멘토링 중간 결과 표시 (Seen/Unseen, 비교한 과거 사례, 근거 충분 여부) | 안 함 | 필요한 데이터가 PR #7 api에 있다 |
| 코드 있는 CRITICAL 화면 정리 | 안 함 | 11/12 합의 "ML은 조기 경보만"과 함께 남길지 팀 확인 필요 |

### 9.3 PR #7과 겹치는 곳

PR #7은 예전 `WarningPage`(한 화면) 위에 기능을 더했다. 이번 커밋은 그 `WarningPage`를 **지웠다**. PR #7을 다시 넣으면 아래 파일이 충돌한다.

- `ui/src/pages/case/WarningPage.tsx` (이쪽은 삭제, PR #7은 수정)
- `ui/src/pages/InboxPage.tsx`, `ui/src/pages/case/OverviewPage.tsx`, `ui/src/types/case.ts`
- `api/main.py` (둘 다 `/warning/*` 근처를 고침)

다시 넣는다면 **api · RAG는 PR #7 것, 화면은 이번 `Early*` 구조**로 맞추고, PR #7이 화면에 쓰던 값을 `Early*`로 옮긴다.

| PR #7 값 | 옮길 곳 (안) |
|---|---|
| `status` `seen` · `unseen` · `uncertain` · `awaiting_assessment`, `POST /warning/assess` | 상황 화면 오른쪽 칸 맨 위 상태 한 줄 |
| `candidate_comparisons` | 후보 목록 아래 "비교한 사례 보기" 접기 |
| `route` (`error_code` · `history` · `manual`), `card.origin === 'error_code'` | 후보 카드 위 작은 라벨 |
| `card.repair` (경보 · 해결 시각) | 사례 카드의 지난 조치 줄 |
| `preferred_event_id`, `rag_mode: 'timeseries'` | 반복 경보 고르기 옆 안내 한 줄, 판단 중 자동 새로고침 |

### 9.4 바꾼 곳

#### ui

| 파일 | 바꾼 것 |
|---|---|
| `pages/case/WarningPage.tsx`, `tests/WarningPage.test.tsx` | **삭제** (PR #5의 한 화면) |
| `pages/case/EarlyPanel.tsx` (새) | 상황 화면 오른쪽 칸: 사례 카드 / 첫 화면 / 후보 목록 중 하나, 해결 뒤 "해결 기록 n/2" |
| `pages/case/EarlySteps.tsx` (새) | 조기 경보의 점검 · 조치 · 정비일지. 점검이 모두 정상이면 [다음 원인 보기] |
| `pages/case/EarlyManual.tsx` (새) | 수동 모드: 케이스 안 챗봇 모양 매뉴얼 검색, 앞 대화 이어서 |
| `lib/warning.ts` (새) | `/warning/*` 요청 · 타입, 반복 경보 중 고른 경보(`activeEventId`) |
| `pages/InboxPage.tsx` | 긴급 / 조기 경보 탭(`?tab=early`), 탭별 새 건수와 새 줄 점, 조기 경보는 "N회 · 처음~마지막" 한 줄, 버튼 "확인하기" |
| `pages/case/OverviewPage.tsx` | 조기 경보도 긴급과 같은 틀(왼쪽 그래프 + 오른쪽 `EarlyPanel`), 경보 1분 센서 ↑↓ 한 줄 |
| `pages/case/CheckPage.tsx` · `ActionPage.tsx` · `LogPage.tsx` · `ManualModePage.tsx` | 조기 경보면 `Early*`로 넘김. 긴급은 R2로 순서를 바꾼 이유 · 안내 생성 실패 문구 |
| `pages/case/CaseLayout.tsx`, `pages/case/parts.tsx`, `lib/caseFlow.ts` | 조기 경보 단계(3단계), "코드 없음" 대신 "조기 경보" |
| `types/case.ts` | 조기 경보 · `last_detected_at` 타입 |
| `package.json` | `test:warning` → `tests/EarlyFlow.test.tsx` |
| `tests/EarlyFlow.test.tsx` (새) | 첫 화면 → 후보 → 점검 → 조치 → 정비일지 → 해결 / 후보 → 수동 모드 → 근거 선택 |
| `docs/10-card-order-early-warning.md`, `docs/11-early-warning-screens.md` (새) | 화면 흐름 · 결정 |

#### api

| 파일 | 바꾼 것 |
|---|---|
| `main.py` | `/warning/resolve` · `/warning/draft`에 `cause`(직접 찾은 근거로 해결하면 필수, 없으면 422), `/warning/manual`에 `history`, 케이스 요약에 `last_detected_at` |
| `field_memory.py` | 조기 경보 `alarm_key`를 `설비|early_warning`으로 따로 묶음 |
| `tests/test_warning_bridge.py`, `tests/warning_demo_server.py` | 위 변경 시험, `RAG_WARNING_EQUIPMENT_SCOPE`를 빈 값으로 고정(PC마다 `rag/.env` 때문에 결과가 달랐다) |

### 9.5 남은 것

- PR #7을 되돌린 이유와 다시 넣을 계획 확인 (종혁 · 동열).
- 수동 검색에서 PDF 근거는 찾았는데 Claude 설명만 실패하면 RAG가 `search_error`를 준다. 지금 `EarlyManual`은 이때 "찾지 못했어요"만 보여 준다. 찾은 근거는 보여 주도록 고칠 것(PR #7 화면의 `retrieval_only` 처리 참고).
- 브라우저에서 해결까지 끝까지 누른 확인.


## 10. 시계열 RAG 통합본 — 11/15

9절은 통합 전 main의 기록이며, 이 통합본에서는 PR #7의 API·RAG와 ML episode_id를 반영했다.
WarningPage는 복원하지 않는다. EarlyPanel에서 /warning/assess를 호출하고 Seen/Unseen/판단 보류,
제어기 코드 카드와 비교 이력을 보여 준다. EarlyManual은 PDF 근거와 등록 사례를 구분해 선택한다.
EarlySteps는 기존 점검·조치·정비일지 화면을 유지하며 선택한 event_id를 URL로 다음 단계까지 전달한다.
rag_mode=timeseries인 CRITICAL도 같은 흐름으로 처리하고, 작업함의 긴급 분류는 severity를 따른다.

원인(cause), 수동 검색 대화 이력(history), 반복 시각(last_detected_at), R2 현장 기록,
ML 공통 JSON 필드와 SQLite 저장은 main의 구현을 보존했다.
새 시계열 케이스는 RAG_WARNING_TIMESERIES=1일 때만 생성되며 기존 케이스는 원래 처리 방식을 유지한다.
