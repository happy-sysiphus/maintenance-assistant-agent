# maintenance-assistant-agent

용접건 센서의 이상을 감지하고, 매뉴얼 근거가 달린 정비 안내를 현장 담당자에게 주는 정비 어시스턴트.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Neo4j](https://img.shields.io/badge/Neo4j-Aura-4581C3?logo=neo4j&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![Claude](https://img.shields.io/badge/LLM-Claude-D97757)

저항 점용접(RSW) 용접건에 이상이 생기면 담당자는 매뉴얼과 과거 이력을 직접 찾아야 합니다. 문서를 그냥 검색하면 다른 모델이나 구버전의 절차가 섞여 나옵니다. 이 프로젝트는 그 일을 세 역할로 나눕니다.

- **ML**이 센서 값에서 이상을 감지합니다.
- **지식 그래프**가 그 이상에 맞는 원인, 점검, 매뉴얼 근거를 찾습니다.
- **LLM**이 찾은 근거를 한국어 안내로 풀어 씁니다. 근거가 없으면 지어내지 않습니다.

판단과 조치는 사람이 합니다. 시스템은 케이스 한 건을 점검부터 정비일지까지 처리하도록 돕고, 장비 제어와 고장 원인 확정은 하지 않습니다.

## 작동 방식

![아키텍처: Phase 1 이상 탐지, Phase 2 근거 검색, Phase 3 현장 안내, 그 아래 api 서버와 서빙 구성](docs/assets/architecture.png)

주황 점선은 아직 구현하지 않은 부분입니다.

**Phase 1 · 이상 탐지 (`ml/`)**

1. 용접건 센서 19종의 값을 1초마다 받습니다.
2. 60초 창마다 피처 46개를 만들고, 설비별 첫 6시간을 기준으로 삼아 IsolationForest + LightGBM으로 이상 점수(0~1)를 냅니다.
3. 종료 코드(E012·E016·E028·E029)가 새로 뜨거나 점수가 임계값을 10분 이상 넘으면 CRITICAL로 판정합니다. 고장 10~60분 전을 노리는 조기 경보는 WARNING으로 판정합니다.
4. 판정 근거(E 코드, 상위 피처 3개, 1분 원시값, 30분 추이)를 담은 이상 신호를 만들어 보관하고 api로 보냅니다.

**Phase 2 · 근거 검색 (`rag/`)**

- **CRITICAL:** E 코드로 지식 그래프에서 원인과 점검을 찾고, 연결된 매뉴얼 쪽을 근거로 Claude가 안내를 씁니다. 인용 쪽과 점검 ID를 검증해 통과한 답만 내보내고, 실패하면 사람이 검토해 둔 기본 카드를 냅니다.
- **WARNING:** 코드가 없으므로 센서 채널 3개의 방향으로 과거 해결 기록을 찾습니다. 작업자가 "원인 후보 보기"를 누르면 그때 매뉴얼을 검색해 Claude가 등록된 사례 중 최대 3개를 고릅니다.
- **챗봇:** 코드가 없어도 매뉴얼에 질문할 수 있습니다. 키워드 검색과 벡터 검색을 함께 씁니다.

**Phase 3 · 현장 안내 (`ui/`)**

1. 작업함에 알림이 도착하고, 케이스를 열면 30분 그래프와 원인 카드가 보입니다.
2. CRITICAL 케이스는 점검, 판단, 조치, 정비일지 순으로 처리합니다. 원인이 아니면 다음 원인으로 넘어가고, 모르겠으면 인계하고, 근거가 없으면 수동 모드로 갑니다.
3. WARNING 케이스는 센서 소견을 먼저 보여 줍니다. 자동 카드, 원인 후보, 수동 검색 중 하나로 근거를 정해 점검과 조치를 기록합니다.
4. 오알람으로 닫거나 해결로 종료합니다.

`api/`는 세 Phase 사이에서 이상 신호를 케이스로 만들고, RAG를 호출하고, 기록을 SQLite에 저장합니다.

## 피드백 루프

이 프로젝트의 목표는 ML과 RAG 양쪽에서 피드백 루프를 돌리는 것입니다.

**RAG 피드백 루프 (구현)**

1. 작업자가 조기 경보를 정비한 뒤 해결로 종료합니다.
2. 그 경보의 센서 패턴, 적용한 매뉴얼 사례, 실제 조치가 Neo4j에 해결 기록으로 저장됩니다.
3. 같은 패턴의 경보가 다시 오면 그 기록을 후보로 보여 줍니다.
4. 같은 패턴과 같은 사례로 해결한 케이스가 2건이 되면 다음부터 자동 카드로 바로 띄웁니다. 자동 카드가 틀렸다는 기록이 생기면 자동 표시를 멈춥니다.

오알람으로 닫은 경보는 해결 기록에 넣지 않습니다. 수동 검색으로 찾은 근거는 운영자가 검토한 뒤에 다시 쓰입니다.

**ML 피드백 루프 (일부 구현)**

새로 쌓인 데이터로 모델을 다시 학습하고, 지금 쓰는 모델과 비교해 더 나으면 서빙 모델을 바꿉니다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 센서 값 저장 | ML 서버가 받은 센서 값을 계속 쌓음(SQLite) | 미구현. 지금은 새 데이터를 파일로 추가함 |
| 정답·오답 재정의 | 해결로 닫힌 경보를 정답, 오알람으로 닫힌 경보를 오답으로 삼음 | 미구현. 지금은 종료 코드로 끝난 기록을 고장으로 봄 |
| 재학습 | 그때까지 끝난 데이터 전체로 새 모델을 학습 | 구현 |
| A/B 테스트 | 두 모델이 모두 처음 보는 데이터로 비교(섀도 비교). 다섯 조건을 모두 넘어야 교체 | 구현 |
| 배포 | 더 나으면 서버를 다시 켜지 않고 모델을 교체. 되돌리기 가능 | 구현 |

점검을 한 번 부르면(`model_loop.py tick`) 학습, 비교, 교체를 사람 손 없이 진행합니다. 주기적으로 부르는 스케줄러는 아직 등록하지 않았습니다. 실제 데이터 64건을 시간순으로 재생한 결과는 판정 세 번 모두 기존 모델 유지였습니다([보고서](ml/results/D-11/runs/replay-1/report.md)).

## 구현 상태

기준은 main `8997493` (2026-11-12)입니다.

| 항목 | 상태 | 위치 |
|---|---|---|
| 실시간 이상 탐지와 종료 코드 규칙 | 구현 | `ml/main.py`, `ml/.py/train.py` |
| 조기 경보 (고장 10~60분 전) | 구현 | `ml/main.py` |
| 이상 신호 작성, 보관, 전달 | 구현 | `ml/.py/rag_mapping.py`, `ml/main.py` |
| E 코드 그래프 RAG (코드 4개) | 구현 | `rag/service.py`, `rag/retrieval.py`, `rag/response.py` |
| 조기 경보 RAG와 해결 이력 재사용 (RAG 피드백 루프) | 구현 | `rag/early_warning.py`, `rag/warning_memory.py`, `api/warning_bridge.py` |
| 챗봇과 매뉴얼 검색 | 구현 | `api/assistant.py`, `rag/retrieval.py` |
| 케이스 저장과 화면 연결 | 구현 | `api/main.py`, `api/store.py` |
| 작업함, 케이스 처리, 조기 경보 화면, 챗봇 | 구현 | `ui/src/pages/` |
| ML 피드백 루프: 재학습, A/B 테스트(섀도 비교), 모델 교체 | 구현 | `ml/.py/model_loop.py`, `ml/.py/model_registry.py`, `ml/main.py` |
| ML 피드백 루프: 받은 센서 값 저장 (SQLite) | 미구현 | |
| ML 피드백 루프: 현장의 정답·오답(해결·오알람) 반영 | 미구현 | |
| AWS 서빙 | 미구현 (계획, 서비스 종류 미정) | |

ML 모델 파일과 Neo4j 데이터는 구현되어 있지만 저장소 밖에 있습니다. 모델은 [ml/README.md](ml/README.md)의 드라이브 링크에서 받고, 그래프는 팀 공용 Neo4j Aura를 씁니다.

## 구성 요소

| 구성 요소 | 기술 | 역할 | 기본 포트 |
|---|---|---|---|
| ML 서버 (`ml/`) | Python, FastAPI, scikit-learn, LightGBM | 센서 값을 받아 이상을 판정하고 이상 신호를 보냄 | 8000 |
| api 서버 (`api/`) | FastAPI, SQLite | 이상 신호를 케이스로 만들고 RAG를 호출해 화면에 응답 | 9000 |
| RAG (`rag/`) | Python 라이브러리, Claude, OpenAI 임베딩 | 매뉴얼 근거 검색, 정비 안내 생성, 조기 경보 해결 이력. api 프로세스 안에서 호출됨 | |
| 화면 (`ui/`) | React 19, Vite, TypeScript, Tailwind CSS | 작업함, 케이스 처리, 조기 경보 화면, 챗봇 | 5173 |
| Neo4j Aura | 그래프 DB, 벡터·키워드 색인 | 지식 그래프(E 코드 4, 원인 7, 점검 9), 매뉴얼 청크 965개, 조기 경보 해결 이력 | 외부 서비스 |
| SQLite | 파일 DB (`api/state/api.db`) | 케이스, 작업 기록, Neo4j 전송 대기 | |

## 빠른 시작

### 화면만 먼저 보기

서버 없이 고정 데이터로 화면이 뜹니다. Node.js 22.22 이상이 필요합니다.

```bash
cd ui
npm install
cp .env.example .env.local
npm run dev        # http://localhost:5173
```

### 전체를 연결해서 실행

준비물은 세 가지입니다.

- **Python 3.11 이상.** api와 rag는 3.13에서 검증했습니다. ML과 api·rag는 의존성이 달라서 가상환경을 따로 두면 편합니다.
- **모델 파일과 테스트 데이터.** `ml/models/baseline_ensemble.joblib`과 `ml/test/`를 [ml/README.md](ml/README.md)의 "0. 데이터 파일" 링크에서 받습니다. 조기 경보는 2026-11-01 번들에만 들어 있습니다.
- **`rag/.env`.** `rag/.env.example`을 복사해 Neo4j Aura, OpenAI, Anthropic 값을 채웁니다. 값은 팀에서 따로 전달합니다.

```bash
# 1) ML 서버
cd ml
pip install -r requirements.txt
uvicorn main:app --port 8000

# 2) api 서버 (새 터미널, 저장소 루트에서)
pip install -r api/requirements.txt
cd api
python -m uvicorn main:app --port 9000

# 3) 화면 (새 터미널). ui/.env.local 에서 VITE_USE_MOCK=false 로 바꾼 뒤
cd ui
npm run dev

# 4) 테스트 데이터를 실시간처럼 재생 (새 터미널)
cd ml
python .py/replay.py test/test_1.csv --start-hours 156 --chunk-s 600
```

재생이 고장 직전에 닿으면 작업함에 케이스가 생깁니다. api는 케이스 목록을 조회할 때 ML 서버에서 이상 신호를 가져옵니다.

### 설정

아래 기능은 서버를 켤 때 설정을 줘야 동작합니다.

| 설정 | 주는 곳 | 주지 않으면 | 주면 |
|---|---|---|---|
| `RSW_EARLY_WARNING_HANDOFFS=1` | ML 서버 환경 변수 | 조기 경보를 계산만 하고 보내지 않음 | 조기 경보가 api로 가서 WARNING 케이스가 생김 |
| `RAG_WARNING_NAMESPACE` | `rag/.env` | 조기 경보 해결 기록을 저장하지 못함 | 팀이 같은 값을 쓰면 해결 이력을 함께 씀 |
| `RAG_WARNING_EQUIPMENT_SCOPE` | `rag/.env` | 해결 기록은 쌓이지만 자동 카드가 뜨지 않음 | 현장 설비 범위(예: `cpx-weld-2g-sa`)에 맞는 사례가 2건부터 자동 카드로 뜸 |
| `RSW_MODEL_REGISTRY=models/registry` | ML 서버 환경 변수 | 모델 파일 하나를 고정해서 씀 | 모델 등록부의 현재 모델을 쓰고, 루프가 교체하면 다시 켜지 않아도 바뀜 |
| `FIELD_MEMORY=1` | api 서버 환경 변수 | 닫힌 케이스 기록을 보여 주지 않음 | 같은 설비·코드로 닫힌 케이스를 "이전 기록"으로 보여 줌 |

```bash
RSW_EARLY_WARNING_HANDOFFS=1 uvicorn main:app --port 8000               # bash
$env:RSW_EARLY_WARNING_HANDOFFS='1'; uvicorn main:app --port 8000       # PowerShell
```

### 테스트

```bash
cd ml && pytest                              # 합성 데이터로 전처리부터 서빙까지
python -m pytest rag/tests api/tests -q      # 저장소 루트에서. 외부 모델과 DB는 모의 처리
cd ui && npm run test:warning && npm run build
```

## 저장소 구조

```
maintenance-assistant-agent/
├── api/    # API 서버 — UI와 ml·rag를 연결
├── ml/     # 모델 학습·추론
├── rag/    # 문서 인덱싱·검색·LLM 호출
├── ui/     # 프론트엔드 (api만 호출)
└── docs/   # README에 쓰는 그림
```

- 각 폴더의 의존성은 그 폴더 안에서 관리합니다(`requirements.txt`, `package.json` 등).
- 모델 가중치, 데이터셋, 벡터 DB, `.env`는 커밋하지 않습니다(`.gitignore` 참고).

## 문서

| 파트 | 문서 | 내용 |
|---|---|---|
| ML | [ml/README.md](ml/README.md) | 데이터, 접근 방식, 파이프라인 실행 순서 |
| | [ml/test.md](ml/test.md) | 현재 모델의 테스트셋 결과 |
| | [ml/techspec.md](ml/techspec.md) | 남은 개발 명세 |
| | [ml/docs/early-warning-handoff.md](ml/docs/early-warning-handoff.md) | 조기 경보 신호 안내 |
| | [ml/docs/superpowers/specs/2026-11-11-model-loop-design.md](ml/docs/superpowers/specs/2026-11-11-model-loop-design.md) | 모델 루프(재학습, 섀도 비교, 교체) 설계 |
| RAG | [rag/README.md](rag/README.md) | 개발 현황과 실행 안내 |
| | [rag/docs/rag-release-20261111.md](rag/docs/rag-release-20261111.md) | 조기 경보 RAG 변경 보고 |
| | [rag/docs/early-warning-integration.md](rag/docs/early-warning-integration.md) | 조기 경보 연동 계약 |
| api | [api/README.md](api/README.md) | 실행, 연결, 저장 |
| | [api/docs/warning-api-20261111.md](api/docs/warning-api-20261111.md) | WARNING API 계약 |
| UI | [ui/README.md](ui/README.md) | 실행 방법과 폴더 안내 |
| | [ui/docs/01-project-background.md](ui/docs/01-project-background.md) | 프로젝트 배경 |
| | [ui/docs/10-early-warning-rag-ui.md](ui/docs/10-early-warning-rag-ui.md) | 조기 경보 화면 안내 |
| | [ui/wireframe/README.md](ui/wireframe/README.md) | 와이어프레임 |

## 한계

이 저장소는 8주 프로젝트의 MVP입니다. 현업 워크플로우를 재현하는 것이 목적이고, 현업에서 바로 쓸 성능이 목표는 아닙니다.

- **수 시간 전 예지는 되지 않습니다.** 이 데이터로는 불가능하다고 판정했습니다. 보장되는 감지 시점은 종료 코드가 뜨는 고장 약 10분 전입니다.
- **조기 경보는 정밀도가 낮습니다.** 놓치지 않는 쪽을 우선해서, 고장 둘 중 하나에서 10~60분 전에 켜지고 오경보는 설비당 하루 약 16회입니다.
- **모델을 교체하면 설비마다 6시간 동안 모델 알림과 조기 경보가 나가지 않습니다.** 설비별 기준을 새 모델로 다시 잡기 때문입니다. 종료 코드 규칙 알림은 계속 나갑니다.
- **안내 카드가 있는 오류 코드는 네 개입니다.** E012, E016, E028, E029입니다.
- **센서 값을 매뉴얼의 수치 기준과 비교하지 않습니다.** 공개 데이터의 센서 이름과 단위가 가려져 있습니다.
- **조기 경보의 원인 후보와 자동 카드는 현장에서 검증하지 않았습니다.** 테스트는 합성 경보와 모의 응답으로 했습니다.
- **실제 설비에 연결한 적이 없습니다.** 센서 입력은 공개 데이터를 재생한 것입니다.
- **사진은 저장만 하고 분석하지 않습니다.**
- **첫 안내 생성에 평균 약 10초가 걸립니다.** 같은 답은 캐시에서 다시 씁니다.

## 데이터와 문서 출처

**센서 데이터.** Xiaoye Wang, Changsheng Zhang, Tao Wang. "Benchmark for welding gun fault prediction with multivariate time series data." *Scientific Data* 11, 83 (2024). https://doi.org/10.1038/s41597-024-02914-z

용접건 80대의 고장 전 약 7일을 1초 간격으로 담은 공개 데이터입니다. 학습용 72개와 테스트용 8개로 나뉩니다.

**매뉴얼 4종 (`rag/data/`).**

| 인용 이름 | 문서 | 쪽수 |
|---|---|---:|
| festo | Festo 서보 공압 용접건 매뉴얼 (구판) | 101 |
| manual82 | CPX-WELD-2G-SA 시스템 매뉴얼 | 82 |
| diagnostic4 | CPX-WELD-2G-SA 오류 진단 문서 | 4 |
| whitepaper11 | 공압 구동 장치 예지보전 문서 | 11 |
