# WARNING API → RAG 연결 — UI 변경 제외 수정본

## 범위와 기준

사용자가 첨부한 `WARNING_API_RAG_UI_연동_패치.zip`의 API 코드를 기준으로 UI 변경을 제외한 수정본입니다. `ui/` 파일, UI 의존성, 기존 README, 기존 RAG 문서, RAG Python 코드의 변경은 포함하지 않습니다.

이 패치는 기존 저장소에 이미 `rag/early_warning.py`, `rag/warning_input.py`, `rag/warning_catalog.py`, `rag/warning_memory.py`가 있는 환경에 적용합니다. 실제 GitHub의 `main`은 `0db9387`(PR #3 되돌림), `develop-loop`는 `3b9e43d`로 확인했으며, 별도 `fix/warning-shared-aura`(`951ca08`)에는 공유 Neo4j 이력·outbox 변경이 있습니다. 이 첨부 패치 기반의 SQLite 어댑터로 해당 Aura 수정본을 덮어쓰면 안 됩니다. 기존 UI는 해당 적용 브랜치의 파일을 유지합니다.

## 정비이력 항목 클릭 시 센서 전달

기존 UI의 실제 호출 경로:

1. 정비이력 메뉴: `GET /api/history` → 종료 이력 목록만 조회합니다.
2. 개별 이력 클릭: 해결 항목은 `/cases/:id/log`, 나머지는 `/cases/:id`로 이동합니다.
3. 두 경로 모두 `CaseLayout` → `useCaseQuery` → `GET /api/cases/:id`를 사용합니다.
4. API는 WARNING의 전체 핸드오프를 보존한 그대로 `EarlyWarningRag.inspect(handoff)`에 전달합니다.
5. `contributing_features`(상위 기여 피처), `sensor_features`(1분 센서 블록), 기타 원본 필드를 제거·재작성하지 않습니다.
6. `warning_rag` 필드로 RAG 결과를 추가 반환합니다. 기존 `event`, `guidance`, `records` 등의 응답 필드는 유지합니다.

여기서 “전달”은 API 프로세스가 RAG Python 함수를 호출하는 방식입니다. 별도 RAG HTTP 서버에 POST하는 구조가 아닙니다. 종료된 케이스도 센서 원본을 전달하지만, 당시 저장한 안내·조치·일지를 최신 응답으로 덮어쓰지 않습니다. 목록 화면을 여는 것만으로 전체 이력의 센서를 일괄 전송하지 않습니다.

상세 열기의 `inspect`는 센서 관측 및 이력 조회입니다. Claude 후보 생성을 뜻하지 않습니다. 후보 생성은 아래 POST가 별도로 담당하며, 기존 UI가 해당 기능을 사용하려면 이 주소를 호출해야 합니다. 이번 패치는 UI 버튼이나 호출 코드를 변경하지 않습니다. 원래 UI가 `warning_rag`를 읽지 않는다면 센서 소견이 새로 화면에 표시되지는 않지만, API → RAG 전달은 상세 조회에서 이루어집니다.

## 유지되는 API

| 메서드 | 주소 | 기능 |
|---|---|---|
| GET | `/api/cases/:id` | 기존 UI 상세 조회 + WARNING 원본 센서 전달 |
| GET | `/api/cases/:id/warning?event_id=...` | 대표/반복 WARNING의 소견·선택·초안 조회 |
| POST | `/api/cases/:id/warning/candidates` | 매뉴얼 후보 생성, 표시 근거 보존 |
| POST | `/api/cases/:id/warning/manual` | 센서 문맥을 포함한 매뉴얼 검색 |
| POST | `/api/cases/:id/warning/select` | 실제 표시한 후보/근거 선택 |
| POST | `/api/cases/:id/warning/reject` | 자동 카드 거절 |
| PUT | `/api/cases/:id/warning/draft` | 점검·작업자·조치 임시 저장 |
| POST | `/api/cases/:id/warning/resolve` | 검증된 이벤트별 근거와 해결 이력 저장 |
| POST | `/api/cases/:id/close` | 기존 UI의 승인된 정비일지 종료 경로 유지 |
| POST | `/api/warning/resolutions/:event_id/review` | 운영자 수동 근거 검토 |
| POST | `/api/warning/resolutions/:event_id/acknowledge` | 운영자 반례 검토 완료 |

후보 요청 예: `{"event_id":"EW-0042"}`.

검색 요청 예: `{"event_id":"EW-0042","question":"공급 압력 점검 방법","explain":true}`.

선택은 `manual_case_id` 또는 검색 응답의 `evidence_id` 중 하나만 받습니다. 클라이언트가 만든 인용문을 받지 않습니다. `/warning/resolve`는 `event_id`, `actual_action`, `worker`, `worker_confirmed:true`, `log_approved:true`, 선택 카드의 점검 ID별 `checks`를 받습니다.

## 기존 UI의 종료와 호환

기존 UI는 일지 승인 후 `/close`로 해결을 저장합니다. 이 경로를 409로 막던 이전 패치 동작을 해제했습니다. 일지 승인 확인은 유지합니다. 실제 선택한 경보별 근거가 없는 기존 종료 기록은 `not_linked`로 보존하며, 그 기록을 검증된 RAG 해결 사례로 임의 승격하지 않습니다. 근거 선택 계약을 사용하는 UI는 `/warning/resolve`로 해결 사례 연결까지 저장할 수 있습니다.

CRITICAL은 기존 E 코드 기반 안내를 유지합니다. 센서에서 E 코드로 확정하는 신규 규칙이나 ML 재학습은 추가하지 않습니다.

## 저장과 피드백

- API 정비 원본과 RAG용 해결 레코드: SQLite (`api/state/api.db`, `API_DB`로 변경).
- 이 수정본은 첨부 패치의 `SqliteWarningMemory`를 유지하며, Neo4j 정비이력 전송 기능을 추가하지 않습니다.
- `RAG_WARNING_FEEDBACK=0`: 기본값. 해결 레코드는 저장하되 과거 해결 추천은 사용하지 않습니다.
- `RAG_WARNING_FEEDBACK=1`: 조건을 충족하는 과거 해결을 재사용합니다. 실제 설비의 매뉴얼 범위 `RAG_WARNING_EQUIPMENT_SCOPE`를 설정해야 자동 표시가 가능합니다.
- 수동 검색 근거는 검토 전 자동 추천에서 제외합니다.
- 같은 센서 패턴·사례에 대한 서로 다른 확인 케이스 2건 이상 등 기존 RAG 조건을 유지합니다.
- 상세 반복 조회 시 inspection 안에 inspection이 중첩되며 DB가 커지는 문제를 막았습니다.

## 적용

`api/` 변경 파일만 기존 프로젝트의 같은 경로에 반영합니다. `README.md`나 `ui/`는 업로드하지 않습니다. 이 ZIP은 변경 파일만 담은 패치이며 단독 실행 프로젝트가 아닙니다.

이전 UI 패치가 이미 커밋되어 있다면, 이번 패치에서 UI 파일을 뺀 것만으로 이전 UI 변경이 자동으로 되돌아가지는 않습니다. PR에서 기존 UI 기준 커밋의 `ui/`를 복원하거나 UI 변경 커밋을 제외해야 합니다. 팀의 더 최신 UI 구현이 있는 브랜치를 기준으로 복원해야 하며, 임의로 `main` UI를 덮어쓰지 않습니다.

## 실행과 검증

기존 의존성을 설치한 뒤:

```bash
python -m pip install -r api/requirements.txt
python -m pytest api/tests rag/tests -q -p no:cacheprovider
```

실제 ML WARNING 생성은 `RSW_EARLY_WARNING_HANDOFFS=1`이 필요합니다. 기존 ML push 또는 API pull을 유지합니다. 문서 검색·Claude 등 외부 서비스는 기존 `rag/.env` 설정을 사용합니다.

이 수정본의 검증: API·RAG 153개 통과, 외부 서비스 환경이 필요한 4개 건너뜀. 상세 GET에서 RAG에 전달한 핸드오프가 저장된 센서 블록·기여 피처와 완전히 같은지 확인했습니다. 종료 이력 재조회, 기존 `/close` 호환, 반복 조회 snapshot 비중첩, 기존 CRITICAL 회귀를 검증했습니다. 실제 운영 ML·Neo4j·Claude와의 네트워크 연결 및 브라우저 클릭은 검증하지 않았습니다.

## 기존 WARNING UI 확인 결과

확인 기준: `main`과 `develop-loop`의 UI 소스. 이들은 WARNING 전용 `WarningPage` 없이 공통 `OverviewPage`를 사용합니다.

- 작업함: `SeverityTag`가 `warning`을 주황색 “주의”로 표시합니다.
- 코드 없는 WARNING의 상단은 “코드 없음”, 왼쪽 제목은 “모델 이상 신호 전후”, “최근 30분”입니다.
- 왼쪽: 센서/이상 점수 그래프, 신호 선택 버튼, “그래프 읽는 법”, 사건 정보.
- 오른쪽: “확인할 원인”. 후보가 없으면 근거를 찾지 못했다는 문장과 “직접 점검해서 기록”, “오알람 닫기” 등이 표시됩니다.
- `warning-api-rag-ui` 및 `fix/warning-shared-aura`에는 이전에 추가한 `WarningPage`가 있으므로 화면이 다릅니다. 브랜치를 구분해야 합니다.

제공하는 HTML은 원래 UI 컴포넌트에 합성 WARNING 데이터를 넣어 생성한 정적 미리보기입니다. 운영 화면 캡처나 UI 업데이트가 아닙니다.
