# ui

사용자 화면(프론트엔드) 코드.

## 실행 방법

필요한 것: Node.js 22.22 이상 (React Router가 요구하는 최소 버전)

```bash
cd ui
npm install
cp .env.example .env.local   # 처음 한 번. 기본값은 mock(MSW) 사용
npm run dev                  # http://localhost:5173
```

- api 서버 없이 `src/mocks/fixtures/`의 고정 JSON으로 화면이 뜬다 (`VITE_USE_MOCK=true`).
- api 서버가 생기면 `.env.local`에서 `VITE_USE_MOCK=false`로 바꾸고 `VITE_API_PROXY_TARGET`에 서버 주소를 넣는다.
- `.env.local`은 커밋하지 않는다.

## 폴더 안내

| 경로 | 내용 |
|---|---|
| `docs/` | 프로젝트 배경, 도메인·데이터 설명, 팀 레포 상태, ML·RAG에 요청할 API 형식 |
| `wireframe/` | 화면 와이어프레임. 먼저 `wireframe/README.md`를 읽고 그림은 `wireframe/screenshots/` |
| `src/` | 화면 코드. mock 데이터는 `src/mocks/fixtures/` |
