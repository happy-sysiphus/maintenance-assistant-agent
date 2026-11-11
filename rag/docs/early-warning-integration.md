# 조기 경보 RAG 연동

## 로컬에서 호출하기

저장소 루트에서 기존 RAG 가상환경을 사용합니다.

```powershell
$ragPython = '.\rag\.venv\Scripts\python.exe'
& $ragPython -B -m rag.early_warning analyze --input rag/tests/fixtures/early_warning_v14.json
```

위 예시는 GitHub `develop-loop`의 [v1.4 조기 경보 문서](https://github.com/happy-sysiphus/maintenance-assistant-agent/blob/develop-loop/ml/docs/rag_handoff_early_warning.example.json)입니다. 기여 피처는 3개지만 실제 채널은 `c10` 하나입니다. c10은 40초에서 0에서 1로 바뀝니다. `pattern_key=null`이라 자동 조회를 보류하고, 센서 관측은 후보 검색에서 쓸 수 있습니다.

Neo4j를 연결할 때는 기존 환경 설정의 Neo4j 주소와 인증을 사용합니다. 호출자가 대상 DB와 이력 구역을 명시합니다. 초기화는 별도 명령으로 새 제약과 인덱스를 추가합니다. 조회나 실행 시마다 초기화하지 않습니다.

```powershell
& $ragPython -B -m rag.early_warning init --database neo4j --namespace local-demo
```

`init`은 현장 이력 스키마만 만듭니다. 후보·수동 검색에는 해당 DB에 기존 문서 KG와 검색 인덱스가 준비돼 있어야 합니다. 기본 검색은 기존 키워드와 OpenAI 임베딩을 함께 쓰고, 후보 생성과 명시적인 수동 질문 답변은 Claude를 씁니다. 로컬 문서 키워드 검색만 확인하려면 `manual --keyword-only`를 사용하고 `explain`을 생략합니다. 격리 테스트와 실제 서비스의 DB 및 namespace는 따로 지정합니다.

## RAG 호출 계약

공통 Python 객체는 `EarlyWarningRag(WarningMemory(driver, database=..., namespace=...))`입니다. API 프로세스에서 Neo4j 드라이버를 재사용합니다. CLI에서는 아래 호출에 대응하는 명령과 `--input`의 JSON 객체를 사용합니다. `analyze`만 핸드오프 자체를 받습니다.

| 호출 | 주요 입력 | 결과 |
|---|---|---|
| `inspect` | `handoff`, 선택적 `equipment_scope`, `pending` | 관측, `pattern_key`, `history_candidates`, `automatic_card`, `status`. 외부 모델 호출 없음 |
| `generate_candidates` / `candidates` | `handoff`, 선택적 `equipment_scope`, `pending` | `event_id`, 관측, 이력 후보, 새 생성 후보 최대 3개. 카드에는 `manual_case_id`, `origin`, `title`, `source`, `checks`, `actions` 포함 |
| `search_manuals` / `manual` | `handoff`, `question`, 선택적 `history`, `explain` | 선택 가능한 `hits[].source`와 등록 사례 ID. `explain=true`일 때만 근거를 붙인 대화 답변 `notes`를 생성 |
| `record_resolution` / `record` | `handoff`, 아래 종료 JSON의 필드 | `saved` 또는 `pending_review`. 동일 내용 재시도는 같은 기록으로 귀결 |
| `review_resolution` / `review` | `event_id`와 `existing_case_id`, 또는 신규 사례의 제목·설비 범위·점검과 조치 발췌 | 검토된 사례가 연결된 해결 기록 |
| `acknowledge` | `event_id` | 검토 완료한 반례 한 건의 보류를 해제. 다른 반례나 복수 승격 후보가 있으면 자동 표시는 계속 보류 |

`equipment_scope`는 API가 확인한 현장의 매뉴얼 적용 범위입니다. 모든 용접건이 같은 모델이면 `EarlyWarningRag(memory, equipment_scope="cpx-weld-2g-sa")`처럼 서비스에 공통값을 한 번 지정합니다. 값은 해당 현장 모델에 맞춰 선택합니다. 적용 범위 안에서 같은 센서 패턴과 매뉴얼 사례로 해결된 서로 다른 케이스 두 건부터 자동 표시합니다. 같은 용접건의 케이스 두 건도 해당됩니다.

`gun_id`는 선택적 이력 정보입니다. 집계는 `case_id`로 하며, 같은 케이스의 반복 경보와 저장 재시도는 한 건으로 표시합니다. 후보의 `resolved_cases`는 확인된 해결 케이스 수이고 `applicable_cases`는 현재 매뉴얼 적용 범위에서 자동 표시 조건에 포함되는 해결 케이스 수입니다. 범위 불명 사례는 후보 조회에 사용할 수 있습니다.

비교 키는 `direction-v1:c10:up|c3:up|c5:down`처럼 정렬된 서로 다른 세 채널과 방향으로 만듭니다. 파생 피처와 중복 채널은 별도 물리 채널로 세지 않습니다. 비교 가능한 세 채널을 확보하지 못해도 해결은 저장할 수 있습니다. 이 경우 패턴 연결은 없으며 다른 경보에 자동 재사용하지 않습니다.

### 조회와 후보 요청

API의 케이스 상세 조회에서는 `inspect`만 호출합니다. UI는 `automatic_card`가 있을 때 **이전 해결 사례**로 표시합니다. `insufficient_channels`, `new_pattern`, `candidates`, `pending`, `memory_unavailable`, `conflict`, `counterexample`는 자동 카드 없이 표시하는 상태입니다. 센서 소견, 저장 후보, 원인 후보 보기 버튼을 사용합니다.

작업자가 **원인 후보 보기**를 누르거나 자동 카드를 거절한 뒤 후보를 요청하면 별도의 POST에서 `generate_candidates`를 호출합니다. `history_candidates`와 `candidates`를 구분해 표시하고, `no_evidence`나 `generation_error`일 때는 수동 모드로 안내합니다. 기존 E 코드의 순번 키와 별도로, 조기 경보 카드의 선택·판단·조치는 `event_id`와 `manual_case_id`로 연결합니다. 작업자가 본 카드 응답을 케이스에 보존하면 재조회 때에도 판단의 대상이 유지됩니다.

### 해결 종료 입력

아래는 RAG 저장 호출에 전달하는 종료 필드입니다. API는 표시한 후보 또는 자동 카드와 실제 판단에 사용한 이벤트를 확인하고, 해당 `handoff`를 별도 인자로 전달합니다. 날짜와 조치는 합성 예시입니다.

```json
{
  "case_id": "CASE-0042",
  "manual_case_id": "manual:E016-C1",
  "equipment_scope": "cpx-weld-2g-sa",
  "actual_action": "매뉴얼에 따라 전극과 캡을 점검하고 기록한 조치를 수행함",
  "worker_confirmed": true,
  "closed_at": "2026-11-10T03:00:00Z"
}
```

수동 모드에서 목록에 없는 근거를 선택하면 `manual_case_id` 대신 검색 결과의 `manual_source`를 넣습니다. UI에서 임의 인용문을 작성하게 하는 대신 API가 보존한 `hits[].source`를 전달합니다. RAG는 신규 저장에서 현재 검색 근거의 문서·쪽·본문과 대조합니다. 검토 요청의 예시는 아래와 같습니다.

```json
{
  "event_id": "event-manual-1",
  "title": "공급 압력 점검",
  "equipment_scope": "festo-legacy",
  "checks": ["Check compressed air supply"]
}
```

`checks`와 `actions`는 선택한 PDF 근거 안의 발췌입니다. 이미 등록된 사례와 같은 근거라면 `existing_case_id`로 연결할 수 있습니다. 자동 카드를 거절한 케이스에서 다른 사례로 실제 해결했다면, API가 저장한 거절 기록에서 `rejected_auto_case_id`를 전달합니다. 거절만 있거나 검토 전 수동 기록이면 확정 반례 집계에 포함하지 않습니다.

### 실시간 반영을 위해 API에서 붙일 부분

1. 반복 경보를 묶어도 실제로 판단한 `event_id`의 전체 센서 블록과 선택한 근거를 보존합니다. 목록의 첫 경보를 대신 사용하면 다른 패턴에 정비 결과가 쌓일 수 있습니다.
2. 작업자의 확인이 포함된 해결 종료와 Neo4j 전송 대기 레코드를 같은 SQLite 트랜잭션에 기록합니다. 오알람·미해결은 저장 함수에 보내지 않습니다.
3. 별도 작업이 바로 전송하고 연결 실패는 지수 백오프로 재시도합니다. 재시도는 종료 시각과 본문까지 같은 데이터를 전달합니다. 같은 경보에 내용이 다른 요청이 오면 기존 기록이 변경되지 않으므로 운영자가 충돌을 검토합니다.
4. 해당 패턴의 전송 대기 기록이 남아 있으면 `inspect(pending=True)`로 자동 카드를 잠시 보류합니다. Neo4j 반영을 확인한 뒤 다음 조회에서 최신 결과를 표시합니다. 문서 그래프 재구축과 해결 이력 반영은 독립적입니다.

## 검증

```powershell
& $ragPython -B -m pytest rag/tests api/tests -q -p no:cacheprovider
```

빈 데이터 디렉터리로 격리된 Neo4j를 띄운 뒤 아래 테스트를 실행합니다. 매뉴얼 검색 테스트는 전용 샘플 청크로 검색과 후보 선택을 검증합니다. 통합 테스트는 실제 매뉴얼 그래프 재구축 경로도 실행하므로 전용 로컬 DB를 사용합니다.

```powershell
$env:RAG_WARNING_TEST_BOLT = 'bolt://127.0.0.1:17687'
& $ragPython -B -m pytest rag/tests/test_early_warning.py -q -p no:cacheprovider
```
