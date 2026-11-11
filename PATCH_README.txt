WARNING API/UI 변경 파일 패치

기존 프로젝트 루트(README.md, api, rag, ui가 있는 폴더)에 이 ZIP의 내용물을 덮어쓰세요.
전체 프로젝트가 아닌 변경 파일만 포함되어 있으므로 새 프로젝트로 단독 실행할 수 없습니다.
기존 .git 폴더는 그대로 유지하세요.

GitHub 웹 업로드: ZIP 자체를 업로드하지 말고 압축을 푼 변경 파일/폴더를 저장소 루트에 올리세요.
UI 의존성 추가가 있으므로 적용 후 ui 폴더에서 npm ci를 실행하세요.
상세 실행 설정: api/docs/warning-api-20261111.md

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