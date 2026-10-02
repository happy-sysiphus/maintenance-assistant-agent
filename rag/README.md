# RSW 용접건 GraphRAG

ML이 전달한 **S01~S10 상황 ID**와 작업자의 질문을 받아 Neo4j에서 Festo 매뉴얼 근거를 조회하고 한국어 답변을 생성하는 모듈입니다. 답변은 **오류 상황 해석 → 원인 후보 → 점검 체크리스트 → 답변 요약** 순서로 구성하고 문서·페이지를 인용합니다.

팀 저장소 루트에 이 `rag/` 폴더를 배치합니다. 아래 명령은 모두 **`rag/`의 상위 디렉터리**, 즉 팀 저장소 루트에서 실행합니다.

## 구성

```text
rag/
├── __init__.py                 # 백엔드 호출 함수 answer_question
├── __main__.py                 # python -m rag 명령
├── service.py                  # 근거 조회, 답변 생성, 인용 검증·후처리
├── config.py                   # 환경 변수와 Neo4j DB 설정
├── paths.py                    # 원본 데이터·런타임 파일 경로
├── pipeline/
│   ├── __init__.py
│   ├── parse.py                # PDF 본문·표·그림 영역 처리
│   ├── chunk.py                # 표 행·원인/조치·출처를 보존한 청킹
│   ├── ingest.py               # Neo4j 문서·페이지·청크 동기화
│   └── build.py                # S01~S10 매핑과 지식그래프 구축
├── data/
│   ├── festo_manual.pdf        # Festo 매뉴얼 원본
│   ├── rsw_situations.pdf      # 상황 정의·점검 순서의 원본
│   └── situation_mapping.json  # 상황별 오류 후보·청크·부품 연결 규칙
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`data/`는 그래프 재구축과 답변 출처 확인에 필요한 입력 데이터입니다. 파싱 결과, 청크, API 캐시는 실행 중 생성되는 `rag/var/`에 저장되며 Git에서 제외합니다. 질문 이력과 설명용 Markdown은 자동 생성하지 않습니다.

## 설치와 설정

Python 3.10~3.14와 Neo4j가 필요합니다. 의존성 버전은 현재 개발 환경(Python 3.13)에서 사용한 버전으로 고정했습니다.

```bash
python -m venv .venv
```

가상환경을 활성화합니다.

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install -r rag/requirements.txt
```

`rag/.env.example`을 `rag/.env`로 복사하고 연결 정보를 입력합니다.

| 환경 변수 | 용도 | 기본값 / 필수 여부 |
| --- | --- | --- |
| `NEO4J_URI` | Neo4j 연결 주소 | 필수 |
| `NEO4J_USERNAME` | Neo4j 계정 | 필수 |
| `NEO4J_PASSWORD` | Neo4j 비밀번호 | 필수 |
| `NEO4J_DATABASE` | 조회·적재할 DB | `festomanualv2` |
| `OPENAI_API_KEY` | 답변 생성·선택적 KG 추출 | LLM 사용 시 필수 |
| `OPENAI_MODEL` | 답변·KG 추출 모델 | `gpt-5.4-mini` |
| `RAG_RUNTIME_DIR` | 중간 파일·캐시 저장 위치 | `var` |

애플리케이션이나 배포 환경이 제공한 환경 변수는 `rag/.env`보다 우선합니다. DB는 함수의 `database` 인자 또는 CLI의 `--database`로 지정할 수 있습니다. `RAG_RUNTIME_DIR`의 상대 경로는 `rag/` 기준이며, 배포 시 절대 경로로 지정할 수도 있습니다. 실제 `.env`는 커밋하지 않습니다.

## 지식그래프 준비

기존에 구축한 그래프를 사용하는 경우 바로 질의응답을 호출할 수 있습니다. 새 환경에서는 대상 Neo4j DB를 먼저 생성·시작하고 아래 순서로 구축합니다. 기본 DB 이름은 현재 그래프의 `festomanualv2`입니다. 해당 이름의 DB를 제공할 수 없는 환경에서는 `NEO4J_DATABASE`를 사용 가능한 DB 이름으로 바꿉니다.

```bash
python -m rag parse --pages 16-19,28-34,40-42,62-64,72,74,76,81-86,89-91,94,98-99
python -m rag chunk
python -m rag ingest
python -m rag build
```

페이지 번호는 **1부터 시작하는 PDF 페이지 번호**입니다. 위 선택 범위는 현재 S01~S10 매핑에 필요한 근거를 포함합니다. 파싱 페이지나 청킹 조건을 바꾸면 `ingest`와 `build`도 순서대로 다시 실행합니다. `build`는 매핑의 페이지·문구가 청크에 정확히 연결되는지 확인한 뒤 그래프를 구축합니다.

`ingest`는 이 Festo 문서의 기존 페이지·청크를 동기화하고, `build`는 Festo에서 파생한 오류·부품·점검 관계와 S01~S10 연결을 다시 만듭니다. 서비스에서 사용하는 대상 DB를 명확히 지정하고 구축 작업을 실행해야 합니다.

API 호출 없이 규칙·매핑 기반 그래프만 구축하려면 마지막 명령을 다음으로 대체합니다.

```bash
python -m rag build --skip-llm
```

선택적으로 그림 설명 초안을 생성할 수 있습니다. 설명 초안은 검토 상태를 유지하며, 확인되지 않은 그림 의미를 확정된 근거로 사용하지 않습니다.

```bash
python -m rag parse --pages 16-19,28-34,40-42,62-64,72,74,76,81-86,89-91,94,98-99 --vision --vision-pages 28 --max-vision-calls 1
```

## 백엔드 연동

```python
from rag import answer_question

result = answer_question(
    situation_ids=["S02"],
    question="S02 오류가 떴는데 어떤 장비를 먼저 확인해야 하나요?",
)

if result["status"] == "ok":
    answer = result["answer"]
    evidence = result["evidence"]
else:
    # 상태별로 오류, 근거 부족, 인용 검토 등을 처리합니다.
    status = result["status"]
```

`situation_ids`는 `"S02"`, `"S02,S07"` 또는 `["S02", "S07"]` 형식으로 전달합니다. 최대 10개 ID와 1,000자 이하의 질문을 받습니다. 잘못된 입력은 `ValueError`를 발생시키고, DB·LLM 실행 실패는 `status="error"`와 `error` 필드로 반환합니다. 이 함수는 동기 함수이므로 비동기 서버에서는 서버 프레임워크의 스레드풀에서 호출합니다.

`use_llm=False`를 주면 API 호출 없이 근거만 조회합니다. `use_cache=True`를 명시하면 답변 파일 캐시를 사용합니다. 기본 호출은 결과 파일이나 질문 이력을 저장하지 않습니다.

| 주요 반환 필드 | 의미 |
| --- | --- |
| `status` | 처리 상태 |
| `answer` | 작업자에게 표시할 답변 |
| `evidence` | 상황 정의·점검 순서·오류 후보·매뉴얼 청크·부품 근거 |
| `missing_ids` | 그래프에 등록되지 않은 ID |
| `citation_check`, `final_citation_check` | 생성 답변과 최종 답변의 인용 검사 |
| `review_notes` | 시각 정보나 근거 범위에 대한 검토 안내 |
| `api_calls_this_run`, `cache_hit`, `usage` | LLM 호출·캐시·토큰 정보 |
| `error` | `status="error"`일 때 실행 오류 |

| 상태 | 처리 의미 |
| --- | --- |
| `ok` | 답변 생성과 인용 검사 완료 |
| `retrieval_only` | LLM 없이 근거 조회 완료 |
| `unknown_situation_id` | 하나 이상의 ID가 그래프에 없음 |
| `no_evidence` | 상황은 있으나 연결된 매뉴얼 근거가 없음 |
| `needs_citation_review` | 답변 인용 검토 필요 |
| `error` | DB·LLM 등의 실행 실패 |

`needs_citation_review`인 답변은 정상 답변과 구분하여 처리합니다. 현재 검색은 상황 ID를 기준으로 고정된 Cypher를 실행합니다. 자연어 질문은 LLM의 설명에 사용하며, 질문으로 벡터 검색을 수행하지 않습니다. 상황 ID와 Festo 컨트롤러 오류 번호는 다른 체계이고, 연결된 오류 번호는 실제 발생 경보가 아니라 점검 후보입니다.

## CLI 질의응답

```bash
python -m rag ask --situation-ids S02 --question "S02 오류가 떴는데 어떤 장비를 먼저 확인해야 하나요?"
python -m rag ask --situation-ids S02,S07 --question "캡과 드레싱 상태를 어떻게 확인하나요?" --json
python -m rag ask --situation-ids S02 --question "점검 근거를 보여주세요." --no-llm --json
```

파일 저장이 필요한 호출에만 `--output 경로.json`, 답변 캐시를 사용할 호출에만 `--cache`를 붙입니다. `python -m rag <명령> --help`로 각 단계의 옵션을 확인할 수 있습니다.
