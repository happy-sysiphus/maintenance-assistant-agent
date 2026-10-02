// api 서버의 케이스 응답 타입.
// 모양은 docs/05-fe-api-requests.md 맨 아래 GET /cases/{id} 예시와 같은 계열이고,
// event 안의 필드 이름은 ML 핸드오프(ml/docs/rag_handoff.example.json)를 그대로 따른다.
// api 서버가 아직 없으므로 FE 제안이다. 확정되면 여기를 고친다.

/** 경보 수준. ML의 severity 값 중 케이스가 되는 것 */
export type Severity = 'critical' | 'warning'

/** 케이스 진행 상태 (api가 관리). 작업함 와이어프레임의 신규 / 원인 찾는 중 / 인계됨 */
export type CaseStatus = 'new' | 'in_progress' | 'handed_over'

/** ML 핸드오프 trigger 중 작업함이 쓰는 부분 */
export interface CaseTrigger {
  /** 'rule' | 'model' | 'model+rule' 등 ML이 준 값 그대로 */
  source: string
  /** 컨트롤러 종료 코드 (예: E012). 모델 신호만 있으면 null */
  rule_code: string | null
  /** 종료 코드가 뜬 데이터 시각 (UTC, Z 없을 수 있음) */
  rule_trigger_time: string | null
}

/** 데이터셋 고장 유형(E01~E04) 추정. 컨트롤러 코드와 다른 체계 */
export interface FaultClass {
  code: string
  name_ko: string
}

/** ML 핸드오프 중 작업함이 쓰는 부분 */
export interface CaseEvent {
  event_id: string
  gun_id: string
  /** ML이 판정한 데이터 시각 (UTC, Z 없을 수 있음) */
  detected_at: string
  trigger: CaseTrigger
  fault_class: FaultClass | null
  summary_ko: string | null
}

/** GET /api/cases 의 목록 한 줄 */
export interface CaseSummary {
  case_id: string
  status: CaseStatus
  severity: Severity
  /** 담당자. 아직 배정 기능이 없으므로 mock에서는 모두 null */
  assignee: string | null
  event: CaseEvent
}

/** GET /api/cases 응답 */
export interface CaseListResponse {
  cases: CaseSummary[]
}
