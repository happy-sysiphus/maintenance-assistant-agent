# maintenance-assistant-agent

## 구조

```
maintenance-assistant-agent/
├── api/   # API 서버 — UI와 ml·rag를 연결
├── ml/    # 모델 학습·추론
├── rag/   # 문서 인덱싱·검색·LLM 호출
└── ui/    # 프론트엔드 (api만 호출)
```

각 폴더의 의존성은 그 폴더 안에서 관리한다(`requirements.txt`, `package.json` 등).
모델 가중치, 데이터셋, 벡터 DB, `.env`는 커밋하지 않는다(`.gitignore` 참고).


## WARNING API·UI 연결 (2026-11-11)

[구현 기능·API 계약·실행 방법](api/docs/warning-api-20261111.md). RAG 피드백은 기본 OFF이며 해결 이력은 SQLite에 저장합니다. CRITICAL은 기존 E 코드 경로를 유지합니다.
