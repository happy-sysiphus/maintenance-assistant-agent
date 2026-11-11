WARNING API·RAG·UI 패치 변경사항 (2026-11-11)

이 파일은 패치 적용 방법보다 '무엇을 변경했는지'를 설명하기 위한 PR 참고 문서입니다.
이 패치는 전체 프로젝트가 아니라 기존 main에 적용할 변경 파일만 포함합니다.

1. 이번 패치의 핵심
- 기존 E 코드 기반 CRITICAL 안내는 유지하면서, 오류 코드가 아직 없는 WARNING(early_warning)을 별도 화면과 RAG 흐름으로 처리하도록 연결했습니다.
- ML이 API에 전달해 저장한 상위 기여 피처 3개(contributing_features)와 1분 센서 데이터(sensor_features)를 WARNING 이벤트 원본에서 읽어 기존 RAG 분석 함수로 전달하도록 했습니다.
- WARNING 상세를 열면 센서 소견·과거 이력을 조회하고, 작업자가 '원인 후보 보기'를 눌렀을 때만 매뉴얼 검색과 후보 생성(Claude 호출)을 수행하도록 분리했습니다.
- 작업자가 검토한 카드·PDF 근거·실제 점검 및 조치·해결 여부를 케이스에 연결해 저장하고, 과거 해결 이력 재사용을 위한 기반을 추가했습니다.

2. API 변경사항
- api/warning_bridge.py [추가]: API SQLite의 WARNING 원본 이벤트를 기존 RAG EarlyWarningRag에 전달하는 브리지와, SQLite 해결 이력 조회용 메모리 어댑터를 구현했습니다. 해결 기록을 Neo4j에 이중 저장하지 않습니다.
- api/main.py [수정]: WARNING 상세 조회 및 원인 후보 생성, 수동 매뉴얼 검색, 사례/근거 선택, 자동 카드 거절, 임시 저장, 해결 종료, 운영자 검토용 엔드포인트를 연결했습니다.
  · GET /api/cases/{id}/warning : 센서 소견·선택 이벤트 상태 조회
  · POST /api/cases/{id}/warning/candidates : 원인 후보 생성
  · POST /api/cases/{id}/warning/manual : 센서 문맥을 사용한 매뉴얼 검색
  · POST /api/cases/{id}/warning/select, /reject : 후보 선택·거절
  · PUT /api/cases/{id}/warning/draft : 점검·조치 임시 저장
  · POST /api/cases/{id}/warning/resolve : 확인된 해결 결과 저장 및 종료
- api/store.py [수정]: 대표·반복 WARNING 원본, UI에서 표시·선택한 근거, 점검 초안과 실제 해결 이력을 SQLite에 보존하도록 확장했습니다. 케이스 종료와 warning_resolutions 기록을 한 트랜잭션으로 처리하고 중복·충돌 요청을 검증합니다.
- 피드백 재사용은 RAG_WARNING_FEEDBACK=0이 기본값입니다. OFF에서도 해결 기록은 저장하지만 과거 해결 사례의 자동 추천에는 사용하지 않습니다.

3. RAG와 에러코드 관련 동작
- 이번 패치의 주된 변경은 RAG 추론 모델 자체를 새로 학습하거나 E 코드 분류기를 만드는 것이 아니라, 기존 RAG의 WARNING 분석·매뉴얼 후보 생성 기능을 API에서 실제 호출하도록 연결한 것입니다.
- RAG는 상위 피처에 해당하는 서로 다른 물리 센서 채널을 골라 1분 데이터의 상승·하락 패턴을 분석하고, 과거 사례 및 매뉴얼 후보를 검색합니다.
- 이 분석은 특정 E012/E016/E028/E029 코드를 자동 확정하거나 고장 유형별 확률을 출력하는 기능이 아닙니다. 점검이 필요한 '후보'를 근거와 함께 보여줍니다.
- 피드백 ON에서 동일 패턴·동일 사례의 서로 다른 해결 케이스 2건 이상 등 조건을 충족한 경우에만 자동 이전 해결 카드를 표시하도록 설계했습니다.

4. UI 변경사항
- ui/src/pages/case/OverviewPage.tsx, WarningPage.tsx [수정]: event_type=early_warning인 케이스를 WARNING 화면으로 분기해 센서별 변화 방향, 과거 해결 사례, 매뉴얼 근거 및 점검 후보를 표시합니다.
- '원인 후보 보기' 클릭 시 후보 생성 API를 호출합니다. 즉, WARNING 화면을 열었다는 이유만으로 매번 Claude 기반 후보를 새로 생성하지 않습니다.
- 대표·반복 경보 선택, 후보·PDF 근거 선택, 오알람 닫기, 실제 점검 결과와 조치 내용 임시 저장, 정비일지 확인 후 해결 종료 및 종료 후 읽기 전용 상태를 지원합니다.
- ui/src/types/case.ts [수정]: WARNING 응답과 상태에 필요한 타입을 추가했습니다.
- ui/package.json, ui/package-lock.json [수정]: UI 테스트 등 관련 의존성을 반영했습니다.

5. 테스트 및 문서
- api/tests/test_warning_bridge.py, api/tests/warning_demo_server.py, ui/tests/WarningPage.test.tsx: WARNING 브리지/API·UI 흐름 검증을 위한 코드입니다.
- api/docs/warning-api-20261111.md: 실제 API 주소, 저장 규칙, 옵션 및 한계를 정리했습니다.
- rag/docs/early-warning-integration.md: RAG 호출 계약과 센서 패턴·이력 규칙을 설명합니다.
- ui/docs/10-early-warning-rag-ui.md: WARNING 화면 흐름과 데이터 표현을 설명합니다.
- README.md, api/README.md, api/.env.warning.example: 진입 문서와 설정 예시를 추가·정리했습니다.

6. 적용 및 확인 시 주의
- 변경 파일을 기존 프로젝트 루트(README.md, api/, rag/, ui/가 있는 위치)에 적용합니다. 압축 파일 자체를 저장소에 올리는 방식이 아닙니다. 기존 .git은 유지합니다.
- UI 의존성 변경이 있으므로 ui/에서 npm ci가 필요합니다.
- 실제 매뉴얼 검색/후보 생성을 위해 기존 RAG 문서 KG 및 Claude 연결 등의 환경설정이 필요합니다. 실서비스의 ML→API→RAG→UI 전체 통합·운영 배포 검증이 끝났다는 의미는 아닙니다.
- 자세한 환경변수와 테스트 명령은 api/docs/warning-api-20261111.md를 참고합니다.

7. PR 완료 후 이 파일 삭제 가능
- 이 PATCH_README_변경사항.txt는 변경 내용 공유·리뷰를 위한 임시 안내 파일입니다.
- PR이 리뷰되고 main에 정상 병합(merge)된 후에는 삭제해도 됩니다. 프로젝트 실행에 필요한 파일이 아닙니다.
- 단, 실제 변경된 소스코드·테스트·정식 문서(api/docs/, rag/docs/, ui/docs/)는 삭제하지 마세요.


포함 파일:
README.md
api/.env.warning.example
api/README.md
api/docs/warning-api-20261111.md
api/main.py
api/store.py
api/tests/test_warning_bridge.py
api/tests/warning_demo_server.py
api/warning_bridge.py
rag/docs/early-warning-integration.md
ui/docs/10-early-warning-rag-ui.md
ui/package-lock.json
ui/package.json
ui/src/pages/case/OverviewPage.tsx
ui/src/pages/case/WarningPage.tsx
ui/src/types/case.ts
ui/tests/WarningPage.test.tsx
