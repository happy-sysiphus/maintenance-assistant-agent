# RAG 개발 보고 및 실행 안내

19채널 시계열 RAG를 사용하려면 [시계열 연동과 실행 설정](docs/warning-timeseries-integration.md)을 확인하세요. 장비 E코드 안내, 과거 해결 카드, 매뉴얼 후보의 화면 흐름은 [통합 보고서](../docs/timeseries-rag-integration-report.md)에 있습니다. [50건 평가](evaluation/timeseries_report.md)에는 자동 후보와 해결 카드의 Precision·Recall을 기록했습니다.

2026-11-11 버전의 변경과 검증 결과는 [RAG 변경 보고서](docs/rag-release-20261111.md)에 정리했습니다. 팀원의 조기 경보 실행과 검증은 [공용 Aura 실행 안내](docs/early-warning-integration.md#공용-aura에서-실행하기), 화면 작업은 [UI 개발 안내](../ui/docs/10-early-warning-rag-ui.md)를 참고하세요.

2026-11-04 · E 코드 기반 정비 안내 RAG

**이번 개선:** 레이아웃·표 구조 파싱, Claude 답변 생성, 정비 이력 기반 점검 제안을 추가했습니다.
변경 이유·실측 결과·남은 검토 항목은 [개발 일지](docs/development-log.md)를 확인하세요.
기존 OpenAI 평가 결과는 `RAGAS/results/`에 보존합니다.

이전 버전과의 차이, 현재 개발 상태, UI 요청 대응 결과, 팀원의 실제 실행 절차를 정리합니다.
팀원은 전달받은 환경 설정과 공용 Neo4j Aura로 질문할 수 있습니다.

## 목차

1. [이전 버전과 달라진 점](#1-이전-버전과-달라진-점)
2. [현재 개발 현황](#2-현재-개발-현황)
3. [UI 문서 요청 대응 현황](#3-ui-문서-요청-대응-현황)
4. [실제 실행 및 API 테스트 예시](#4-실제-실행-및-api-테스트-예시)
5. [rag 폴더 외 변경 사항](#5-rag-폴더-외-변경-사항)
6. [2026-11-04 개선 사항](#6-2026-11-04-개선-사항)

## 1. 이전 버전과 달라진 점

| 구분 | 이전 팀 레포의 rag | 이번 버전 |
|---|---|---|
| 오류 입력 | S01~S10 상황 ID 필수 | 실제 오류 코드 E012·E016·E028·E029 사용 |
| 일반 질문 | 상황 ID를 먼저 지정해야 조회 가능 | 코드 없이 매뉴얼 질문 가능 |
| ML 정보 | ML이 상황 ID를 전달한다는 전제 | 실제 E 코드만 사용. 센서 데이터·E01~E04는 RAG 입력에서 제외 |
| 문서 | 기존 Festo와 RSW 상황 정의서 기반 | 기존 Festo + 새 문서 3종. RSW 정의서와 S 매핑 제거 |
| 그래프 | S 상황 노드에서 근거 연결 | Situation 중심 구조 유지, E 코드 네 개로 변경 |
| 검색 | 상황 ID 기반 그래프 조회 | E 코드 근거 우선 확보 + 키워드·벡터 하이브리드 검색 |
| 답변 | 마크다운 설명문 중심 | JSON으로 해석·요약·원인·점검·정상 기준·조치 분리 |
| 점검 ID | 문자열 점검 순서 | E012-K1 같은 고정 ID. 코드별 카드는 검토 매핑 순서로 연결 |
| 대화 문맥 | history 입력 없음 | 선택적 history로 이어 묻기 지원 |
| DB | 로컬 Neo4j 연결 전제 | 공용 Aura의 KG·임베딩·검색 인덱스 사용 |
| API | 매핑 파일을 읽어 임시 안내 조립 | 실제 RAG 함수 호출과 한국어 답변 생성 연결 |

기존 파싱·청킹 파이프라인과 “상황 → 관련 근거” 그래프 구성을 유지했습니다.
rag_dev2에서는 코드 근거 보존과 검색 순위 결합 아이디어를 참고했고, 센서 모니터링 등은 가져오지 않았습니다.

### 사용하는 문서

| 인용 식별자 | 문서 | PDF 페이지 |
|---|---|---:|
| festo | 기존 Festo 서보 공압 용접건 매뉴얼 | 101 |
| manual82 | CPX-WELD-2G-SA 시스템 매뉴얼 | 82 |
| diagnostic4 | CPX-WELD-2G-SA 오류 진단 문서 | 4 |
| whitepaper11 | 공압 구동 장치 예지보전 문서 | 11 |

새 공식 매뉴얼의 오류 명칭을 기준으로 하고, 상세 진단과 구판의 보충 근거는 출처를 구분합니다.
구판의 오류 16·28·29는 신형 E016·E028·E029와 의미가 달라 번호만으로 연결하지 않았습니다.
인용 페이지는 **PDF 파일에서 1부터 세는 실제 페이지**입니다.

## 2. 현재 개발 현황

### 완료한 기능

- Aura에 문서 4종·198페이지·Situation 4개 적재 및 연결 확인.
- 원본 942개와 인용 근거 23개, 총 965개 청크 저장. 본문 검색 대상은 732개이며 이미지 자리표시자는 제외.
- E 코드 정의와 연결 근거를 우선 확보하고 키워드·벡터 검색을 보충하는 하이브리드 검색.
- 코드가 있는 질문과 없는 질문을 같은 JSON 형식으로 처리.
- 해석·요약·원인·점검·정상 기준·조치와 문서 인용 분리.
- history를 검색·생성 문맥에 반영. 최근 8개 메시지, 각 1,500자까지 사용.
- 코드 질문의 점검 본문·기준·인용은 검토된 고정 매핑에서 연결.
- 출력 형식·인용 문서·페이지·점검 ID 검사와 검증 실패 이유 반환.
- API의 케이스 안내·챗봇·PDF 제공·수동 문서 검색 연결.
- 선택적 답변 캐시와 응답 버전 관리.

### 검증 결과와 응답 확인

오프라인 회귀 테스트 **98개를 통과**했습니다. 실제 PDF 머리글·행 경계 검사도 포함합니다.
대표 20쪽의 원문 대조 검사와 전체 198쪽 처리 검사를 통과했습니다. 원본 확인이 필요한 표 블록 18개는 별도로 표시합니다.
실제 Aura 조회, E 코드 네 개의 답변 생성, 일반 질문·이어 묻기, 캐시 및 API 연결도 확인했습니다.
사용 중 발견된 새 점검 ID 생성 문제와 Swagger 입력칸 누락은 수정하고 회귀 테스트에 포함했습니다.

이번 50문항의 캐시 없는 응답은 평균 **10.555초**, p95 **14.763초**였습니다. 캐시 적중은 외부 모델을 호출하지 않습니다.
네 코드의 실제 응답 JSON은 [docs/](docs/)에 있습니다.

| 필드 | 확인할 내용 |
|---|---|
| status | ok: 생성·검증 통과 / retrieval_only: 근거만 조회 |
| error_code | E 코드 또는 null |
| sections | interpretation·summary |
| causes | situation_id·name·why·checks·actions |
| checks | 고정 id·text·cite·criterion |
| actions | when·text·cite |
| evidence | doc·page·quote·text 등 조회 근거 |
| review_notes | 적용 범위·시각 정보 확인 안내 |
| seconds / usage | 전체 시간과 토큰 사용량 |
| cache_hit / api_calls_this_run | 캐시 재사용 여부와 이번 외부 API 호출 수 |
| maintenance_advice | 관련 정비 이력 기반 제안. 없으면 null. PDF 근거와 구분한 사례 ID 포함 |
| validation_errors | 검증 실패의 code·message·path |

근거가 없으면 actions는 빈 배열, criterion은 null로 반환합니다.
일반 질문은 error_code와 situation_id가 null일 수 있으며 원인 목록도 비어 있을 수 있습니다.
인용 위치·형식 검증이 모든 문장의 의미나 실제 장비 고장 원인을 보장하는 것은 아닙니다.
진단 첨부 문서의 적용 펌웨어와 현장 설비 적합성은 별도 확인이 필요합니다.
사진·센서 분석은 포함하지 않습니다.

### UI에 남은 작업

현재 main의 UI는 E 코드·문서 네 종류·원인 카드 표시를 지원합니다. 이번 작업에서는 ui/src를 수정하지 않았습니다.
새 maintenance_advice는 챗봇 본문과 API에서 확인하며, 케이스 화면에 별도 정비 제안 영역을 둘지는 UI 후속 작업입니다.
첫 생성이 10초를 넘을 수 있으므로 실제 화면의 로딩 상태도 확인해야 합니다.

Swagger에서 답변이 나오는 것은 테스트 페이지 → API → RAG 연결 확인입니다.
ML 감지부터 사용자용 UI 화면까지의 전체 흐름을 모두 검증했다는 의미는 아닙니다.

## 3. UI 문서 요청 대응 현황

이번 개발은 UI 담당자의 **[07-rag-requests-now.md](../ui/docs/07-rag-requests-now.md)**를 기준으로 진행했습니다.
각 요청의 충족 여부, 합의로 바뀐 방식, 호출 예시, 기준표 검토, 측정 결과와 남은 UI 작업은
**[09-rag-development-reply.md](../ui/docs/09-rag-development-reply.md)**에 답변 형식으로 남겼습니다.

| 07번 요청 | 개발 결과 |
|---|---|
| 1. E 코드 입력·원인 후보 | E 코드 직접 조회. S 변환 없이 코드당 안내 카드 하나 |
| 2. 네 코드 예시 JSON | 실제 실행 결과 네 개 제공 |
| 3. 구조화 답변·조치·정상 기준 | 필드 분리, 근거 없는 값은 비움 |
| 4. 질문만 호출·history | 일반 질문과 코드 질문, 선택적 대화 이력 지원 |
| 5. 응답 시간·PDF | 실제 시간 측정 및 PDF 네 종류 제공 |
| 6. 실행 환경·API 연결 | Aura 연결과 실제 RAG 함수 호출 완료. 접속정보는 별도 전달 |

실제 출력 예시:

- [E012.example.json](docs/E012.example.json)
- [E016.example.json](docs/E016.example.json)
- [E028.example.json](docs/E028.example.json)
- [E029.example.json](docs/E029.example.json)

기존 기록 저장 주소와 기본 후보 구조는 API에서 유지합니다.
신규 케이스는 E 코드당 후보 하나이며, 구조화 결과도 guidance에 함께 제공합니다.
기존 S 기반 저장 케이스는 작업 기록을 보존하기 위해 자동으로 E 코드로 재해석하지 않습니다.

## 4. 실제 실행 및 API 테스트 예시

아래는 **Windows PowerShell** 기준입니다.
공용 Aura가 실행 중이고 인터넷에 연결돼 있어야 합니다.
Neo4j Desktop이나 ML 서버는 RAG 질문·챗봇 테스트만 할 때 필요하지 않습니다.

### 4.1 처음 한 번: 실행 환경 준비

Python 3.13 환경에서 검증했습니다.
저장소를 내려받고 아래 경로를 본인의 저장소 위치로 바꿉니다.
저장소 루트는 바로 아래에 rag·api·ui·ml 폴더가 있는 위치입니다.

```powershell
# 본인 저장소 위치로 변경
Set-Location "C:\작업폴더\maintenance-assistant-agent"
$repoRoot = (Get-Location).Path

# 처음 한 번만 생성·설치
py -3.13 -m venv .venv
$ragPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
& $ragPython -m pip install -r .\api\requirements.txt
```

api/requirements.txt에 RAG 의존성도 포함됩니다.
가상환경 활성화 없이 $ragPython으로 실행하므로 Activate.ps1 실행 정책을 바꿀 필요가 없습니다.

### 4.2 전달받은 .env.txt 배치

RAG 담당자에게 받은 **.env.txt를 rag/.env라는 이름으로 복사**합니다.
예를 들어 받은 파일이 Downloads에 있다면 다음과 같습니다.
이미 올바른 rag/.env가 있으면 복사 단계는 생략합니다.

```powershell
$receivedEnv = Join-Path $env:USERPROFILE "Downloads\.env.txt"
Copy-Item -LiteralPath $receivedEnv -Destination (Join-Path $repoRoot "rag\.env")
Test-Path -LiteralPath (Join-Path $repoRoot "rag\.env")
```

마지막 결과가 True면 파일 위치가 맞습니다.
서버가 읽는 위치는 api/나 저장소 루트가 아닌 **rag/.env**입니다.

```text
maintenance-assistant-agent/
├─ .venv/
├─ rag/
│  ├─ .env          ← 전달받은 텍스트 파일을 이 이름으로 배치
│  ├─ .env.example  ← 비밀값 없는 참고 양식
│  └─ ...
├─ api/
├─ ui/
└─ ml/
```

| 전달받는 설정 | 내용 |
|---|---|
| OPENAI_API_KEY | 질문·문서 임베딩용 키 |
| ANTHROPIC_API_KEY | Claude 한국어 답변 생성용 키 |
| NEO4J_URI | Aura 접속 주소 |
| NEO4J_USERNAME / NEO4J_PASSWORD | DB 접속 사용자명·비밀번호 |
| NEO4J_DATABASE | 실제 DB 이름 |
| RAG_WARNING_NAMESPACE | 팀 공용 조기 경보 이력 이름 |
| ANTHROPIC_MODEL | claude-sonnet-5-5 |
| OPENAI_EMBEDDING_MODEL | text-embedding-3-small |
| RAG_SYSTEM_TRUST | OS 인증서 사용 여부. 전달받은 설정을 기준으로 사용 |

Claude 키·OpenAI 키·Aura 설정이 모두 필요합니다. 서비스 답변 생성과 임베딩은 서로 다른 모델을 사용해도 됩니다.
.env와 .env.txt는 비밀값이 들어 있는 로컬 파일이므로 커밋하지 않습니다.
이 이름들은 Git 제외 규칙에 포함돼 있습니다.

Windows에서 .env를 더블클릭했을 때 AnySign 암호창이 뜨면 연결 프로그램 문제입니다.
확인할 때는 메모장을 직접 사용합니다. 파일 자체에 별도의 열기 암호는 없습니다.

```powershell
notepad (Join-Path $repoRoot "rag\.env")
```

### 4.3 터미널에서 실제 질문하기

새 PowerShell 창에서는 저장소 루트로 이동하고 변수를 다시 지정합니다.

```powershell
Set-Location "C:\작업폴더\maintenance-assistant-agent"
$repoRoot = (Get-Location).Path
$ragPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
```

**외부 모델 호출 없이 근거만 조회:**

```powershell
& $ragPython -B -m rag ask --error-code E029 --question "점검 근거를 보여주세요." --no-llm --json
```

status가 retrieval_only이고 evidence에 문서·페이지가 있으면 조회가 정상입니다.

**오류 코드를 지정한 질문:**

```powershell
& $ragPython -B -m rag ask --error-code E029 --question "이 오류는 어떤 상황이고 무엇부터 확인해야 하나요?"
```

**오류 코드 없는 일반 질문:**

```powershell
& $ragPython -B -m rag ask --question "이동 보정은 어떤 전극 캡을 사용해서 해야 하나요?"
```

**질문 문장 안에 코드 입력:**

```powershell
& $ragPython -B -m rag ask --question "지금 E016이 발생했는데 어떤 점검을 해야 하나요?"
```

**UI에 전달할 JSON 확인:**

```powershell
& $ragPython -B -m rag ask --error-code E012 --question "원인 후보와 점검 기준, 조치를 알려주세요." --json
```

정상 생성은 status=ok입니다. answer·sections·causes·evidence를 확인합니다.
needs_citation_review라면 validation_errors, error라면 error 필드와 환경 설정·연결을 확인합니다.

실제 생성에는 Claude 답변과 OpenAI 질문 임베딩 비용이 발생합니다.
--no-llm은 두 API를 모두 호출하지 않습니다. --cache는 문서·모델·질문·대화·정비 이력이 같은 유효한 답변을 재사용합니다.
cache_hit=true, api_calls_this_run=0이면 이번에는 외부 모델을 호출하지 않은 것입니다.
기본 실행은 답변 파일을 만들지 않습니다.

**API 잔액 및 평균 호출 비용**

- 실제 잔액: [OpenAI Billing](https://platform.openai.com/settings/organization/billing/overview), [Claude Billing](https://platform.claude.com/settings/billing). README에는 변하는 잔액 숫자를 저장하지 않습니다.
- 사용 내역: [OpenAI Usage에서 기간별 사용 비용 확인](https://platform.openai.com/usage).
- **현재 Claude 버전: 질문 1건 평균 약 $0.02270**, 2026-11-04 캐시 없는 50문항 기준입니다. 질문 임베딩과 답변 생성 비용이며, 문서 임베딩·RAGAS 평가자 비용은 제외합니다.
- 같은 50문항의 이전 OpenAI 버전은 약 $0.00410/건이었습니다. 이번 전체 개발·연결 검증·평가 기록 추정 비용은 **$2.47302 / 상한 $3**입니다.
- 계산 단가: 100만 토큰당 [Claude Sonnet 5.5](https://platform.claude.com/docs/en/about-claude/pricing) 입력 $2·출력 $10, [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small) $0.02. 토큰 사용량 기반 추정치로 실제 청구액·계정 잔액과 다를 수 있습니다. 상세 내역은 [비교 보고서](RAGAS/results-20261104/COMPARISON.md)를 확인하세요.

### 4.4 API 서버 시작

같은 PowerShell에서 실행합니다.

```powershell
Set-Location (Join-Path $repoRoot "api")
& $ragPython -B -m uvicorn main:app --host 127.0.0.1 --port 9000 --reload
```

이 터미널을 켜 둔 채 브라우저에서 테스트합니다. 중지는 이 터미널에서 Ctrl+C입니다.
환경 설정을 바꾼 경우 서버를 재시작합니다.

WinError 10048이 나오면 9000번 포트에 서버가 이미 실행 중인 것입니다.
기존 테스트 페이지를 사용하거나 기존 서버를 종료한 뒤 하나만 실행합니다.
다른 포트로 실행하려면 --port 9001로 바꾸고 아래 테스트 URL도 9001로 바꿉니다.

### 4.5 브라우저에서 UI가 보내는 API 요청 테스트

[http://127.0.0.1:9000/docs](http://127.0.0.1:9000/docs)를 엽니다.
이 페이지는 FastAPI가 자동 제공하는 **Swagger UI**입니다.
실제 사용자용 ui/ 화면과는 별개로 UI가 보낼 JSON 요청을 직접 시험할 수 있습니다.

1. POST /api/assistant/ask를 펼칩니다.
2. Try it out을 누릅니다.
3. Request body의 흰색 입력칸에 JSON을 붙여넣습니다.
4. Execute를 누릅니다.
5. 아래 Server response의 Response body를 확인합니다.

**오류 코드가 있는 질문:**

```json
{
  "question": "E029 오류의 의미와 점검 방법을 알려주세요.",
  "photos": [],
  "history": []
}
```

**오류 코드가 없는 일반 질문:**

```json
{
  "question": "이동 보정에는 어떤 전극 캡을 사용하나요?",
  "photos": [],
  "history": []
}
```

**이어 묻기:**

```json
{
  "question": "그럼 가장 먼저 무엇을 확인해야 하나요?",
  "photos": [],
  "history": [
    {
      "role": "user",
      "text": "E029 오류의 의미와 점검 방법을 알려주세요."
    }
  ]
}
```

history에는 실제 이전 사용자 질문을 넣습니다. assistant 메시지도 실제 답변 내용으로 추가할 수 있습니다.
사진은 현재 분석하지 않으므로 photos는 빈 배열로 테스트합니다.

HTTP 200만으로 성공을 판단하지 말고 JSON의 status도 확인합니다.
정상이면 status=ok이며 구조화 RAG 필드와 기존 UI용 blocks·caution·situation_ids를 함께 반환합니다.
일반 질문의 error_code는 null, situation_ids는 빈 배열일 수 있습니다.

Request body가 없으면 수정 전 서버가 실행 중일 수 있습니다.
API를 중지·재실행하고 브라우저 페이지를 새로고침합니다.

```text
Swagger 또는 UI의 JSON 요청
  → POST /api/assistant/ask
  → RAG answer_question()
  → OpenAI 질문 임베딩 + Aura 근거 검색 + Claude 생성
  → 형식·인용·점검 ID 검증
  → API JSON 응답
```

유효한 캐시가 있으면 생성 결과를 재사용합니다.
직접 입력한 E029는 사용자가 준 코드이며 ML의 실제 감지 결과를 뜻하지 않습니다.

### 4.6 PowerShell에서 API에 직접 요청

서버 터미널은 그대로 두고 새 PowerShell 창에서 실행합니다.
JSON을 UTF-8 바이트로 보내 Windows PowerShell에서도 한글을 유지합니다.

```powershell
$ragRequest = @{
    question = "E029 오류의 의미와 점검 방법을 알려주세요."
    photos = @()
    history = @()
} | ConvertTo-Json -Depth 8

$ragApiParams = @{
    Uri = "http://127.0.0.1:9000/api/assistant/ask"
    Method = "Post"
    ContentType = "application/json; charset=utf-8"
    Body = [System.Text.Encoding]::UTF8.GetBytes($ragRequest)
}
$ragResult = Invoke-RestMethod @ragApiParams

$ragResult.status
$ragResult.answer
$ragResult | ConvertTo-Json -Depth 20
```

일반 질문도 question 값만 바꾸면 같은 주소와 요청 형식을 사용합니다.

### 4.7 PDF와 실제 사용자용 UI 연결 확인

- [기존 Festo](http://127.0.0.1:9000/api/manuals/festo.pdf)
- [새 시스템 매뉴얼](http://127.0.0.1:9000/api/manuals/manual82.pdf)
- [오류 진단 문서](http://127.0.0.1:9000/api/manuals/diagnostic4.pdf)
- [공압 구동 문서](http://127.0.0.1:9000/api/manuals/whitepaper11.pdf)

cite가 doc=manual82, page=44이면
[manual82.pdf#page=44](http://127.0.0.1:9000/api/manuals/manual82.pdf#page=44)를 엽니다.
GET /api/situations에서도 E 코드 네 개의 목록을 확인할 수 있습니다.

실제 사용자용 UI 담당자는 ui/.env.local에 다음을 설정해 API로 요청을 보냅니다.

```dotenv
VITE_USE_MOCK=false
VITE_API_BASE_URL=/api
VITE_API_PROXY_TARGET=http://localhost:9000
```

설정 후 별도 PowerShell에서 UI 개발 서버를 실행할 수 있습니다.
Node.js 실행 환경은 [UI README](../ui/README.md)의 요구사항을 따릅니다.

```powershell
Set-Location "C:\작업폴더\maintenance-assistant-agent\ui"
npm install
npm run dev
```

브라우저에서 http://localhost:5173 을 엽니다. API 서버도 계속 실행 중이어야 합니다.
기존 UI의 E 코드·문서 네 종류·원인·조치 표시를 유지합니다.
이번 개선은 카드 순서를 바꾸지 않으며, 정비 이력 제안은 챗봇 답변 본문과 API의 maintenance_advice에서 확인합니다.

### 4.8 오프라인 회귀 테스트

4.1의 환경 설치 후 저장소 루트에서 실행합니다.

```powershell
Set-Location $repoRoot
& $ragPython -m pip install pytest
& $ragPython -B -m pytest rag/tests api/tests -q -p no:cacheprovider
```

이 테스트는 Claude·OpenAI·Neo4j를 모의 처리하므로 외부 모델 호출 비용이 없습니다.
.env와 파싱 결과·백업·캐시 등 로컬 산출물은 Git 제외 경로에서 관리합니다.

## 5. rag 폴더 외 변경 사항

RAG를 실제로 호출하기 위한 API 연동 코드와 인수인계 문서를 함께 수정했습니다.

| 파일 | 변경 내용 |
|---|---|
| [api/main.py](../api/main.py) | E 코드 안내·문서 4종 PDF·검색 경로 연결. 챗봇 요청 스키마를 선언해 Swagger 입력칸 제공, RAG 호출을 별도 스레드에서 실행 |
| [api/guidance.py](../api/guidance.py) | 기존 임시 안내를 실제 RAG 호출로 연결. E 코드당 후보 하나, 고정 점검 ID·정상 기준·구조화 답변과 캐시 반영 |
| [api/assistant.py](../api/assistant.py) | 질문·history를 RAG에 전달하고, 구조화 결과와 기존 UI용 답변 블록·인용을 함께 반환 |
| [api/store.py](../api/store.py) | SQLite 연결을 작업 후 확실히 닫도록 수정 |
| [api/requirements.txt](../api/requirements.txt) | API 설치 시 RAG 의존성도 함께 설치하도록 연결 |
| [api/README.md](../api/README.md) | 실행 방법, E 코드 연동, 챗봇·PDF 경로 및 Swagger 테스트 절차 정리 |
| api/check_criteria.json (삭제) | 기존 S 코드 기반 임시 기준표 제거. 검토한 E 코드 기준은 RAG 매핑으로 통합 |
| [ui/docs/07-rag-requests-now.md](../ui/docs/07-rag-requests-now.md) | 요청 원문을 유지하고 상단에 09번 개발 답변 링크 추가 |
| [ui/docs/09-rag-development-reply.md](../ui/docs/09-rag-development-reply.md) (신규) | 07번 요청별 충족 여부·호출 예시·응답 계약·검증 결과·남은 UI 작업 기록 |

위 표는 10/28 최초 연동까지의 변경 이력입니다. 11/04 개발 일지에 나온 API의 종료 이력 전달과 기록 보존은 당시 로컬 검증에 사용한 변경입니다. 11/11 게시 범위는 RAG 코드와 새 UI 개발 안내이며, 조기 경보의 API·UI 작업은 [연동 안내](docs/early-warning-integration.md)에 있습니다.

## 6. 2026-11-04 개선 사항

- 파싱은 Docling의 레이아웃·표 구조와 PDF 원본 셀 경계를 대조합니다. 머리글·행·표 ID·페이지·좌표를 보존합니다.
- 페이지를 넘는 설명은 하나의 검색 단위로 연결하되, 인용 근거는 원래 페이지별로 제공합니다. 그림은 위치만 기록합니다.
- 회전된 비트 행렬 등 검증되지 않은 시각 관계는 원본 확인 대상으로 남깁니다.
- 청크 크기는 600·900·1200자의 검색 비교로 정합니다. 비교 방법과 선택 근거는 개발 일지에 기록합니다.
- 답변은 Claude, 임베딩은 OpenAI입니다. 코드 질문의 원인·점검 카드는 검토된 매핑 순서를 유지합니다.
- RAG는 호출자가 `maintenance_context`로 넘긴 같은 E 코드의 최근 종료 기록을 최대 5건 사용합니다. 당시 로컬 API 검증에서는 현재 케이스를 제외해 전달했습니다.
- 반복 실패·점검 결과를 고려한 제안은 maintenance_advice에 분리합니다. 이력으로 매뉴얼 기준·수리 조치를 새로 만들지 않습니다.

CLI에서는 `--maintenance-context 기록.json`으로 정규화된 기록을 선택적으로 줄 수 있습니다.
형식은 case_id·error_code·closed_at(시간대 포함)·outcome·checks·actions·note이며,
checks는 점검 ID → normal/abnormal/skipped, actions는 수행 내용 문자열 목록입니다.
API에서 이 기능을 쓰려면 종료 이력을 정규화해 RAG 호출에 전달해야 합니다.

파싱용 의존성은 `requirements-parse.txt`로 분리되어 있어 서비스 실행 시 Docling을 설치할 필요가 없습니다.
전체 변경·검증 보고는 [개발 일지](docs/development-log.md)에 있습니다. 공유 Aura를 사용하는 팀원은 KG를 재구축하지 않습니다.

같은 50문항의 [RAGAS 비교 보고서](RAGAS/results-20261104/COMPARISON.md)에는 개선된 지표와 하락한 지표, 응답 시간·비용 증가를 함께 기록했습니다.
