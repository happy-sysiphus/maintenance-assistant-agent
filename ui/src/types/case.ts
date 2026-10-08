// api 서버의 케이스 응답 타입.
// 모양은 docs/05-fe-api-requests.md 맨 아래 GET /cases/{id} 예시와 같은 계열이고,
// event 안의 필드 이름은 ML 핸드오프 v1.1(ml/docs/rag_handoff.example.json)을 그대로 따른다.
// guidance는 RAG 매핑(rag/data/situation_mapping.json)에서 온다.
// api 서버가 아직 없으므로 FE 제안이다. 확정되면 여기를 고친다.

/** 경보 수준. ML의 severity 값 중 케이스가 되는 것 */
export type Severity = 'critical' | 'warning'

/**
 * 케이스 진행 상태 (api가 관리).
 * 작업함: 새 고장 / 점검 중 / 일지 작성 / 일지 승인됨 / 미해결 / 도움 요청.
 * resolved(해결 종료)는 작업함에서 빠지고 정비 이력에만 보인다.
 */
export type CaseStatus = 'new' | 'in_progress' | 'logging' | 'log_approved' | 'unresolved' | 'handed_over' | 'resolved'

/** ML 핸드오프 trigger 중 작업함이 쓰는 부분 */
export interface CaseTrigger {
  /** 'rule' | 'model' | 'model+rule' 등 ML이 준 값 그대로 */
  source: string
  /** 컨트롤러 종료 코드 (예: E012). 모델 신호만 있으면 null */
  rule_code: string | null
  /** 종료 코드가 뜬 데이터 시각 (UTC) */
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
  /** ML이 판정한 데이터 시각 (UTC) */
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
  /** 담당자. 로그인이 없어 null */
  assignee: string | null
  event: CaseEvent
}

/** GET /api/cases 응답 */
export interface CaseListResponse {
  cases: CaseSummary[]
}

// ---- 케이스 상세 (GET /api/cases/:id) ----

/** 핸드오프 v1.1 error_timeline: 컨트롤러 코드가 켜진 구간 */
export interface ErrorSpan {
  code: string
  start: string
  end: string
  duration_s: number
  class_hint: string | null
}

/** 핸드오프 v1.1 sensor_trend: 최근 30분, 1분 단위, 평소 대비 편차(z) */
export interface SensorTrend {
  feature: string
  sensor: string
  sensor_name_ko: string
  /** mean = 평균, std = 1분 안의 흔들림 */
  statistic: string
  points: { t: string; z: number }[]
}

export interface SensorFinding {
  sensor: string
  sensor_name_ko: string
  direction: string
  deviation_z: number
  text_ko: string
}

/** ML 핸드오프 전체 (작업함용 CaseEvent의 확장) */
export interface HandoffEvent extends CaseEvent {
  trigger: CaseTrigger & { anomaly_score: number; threshold: number }
  sensor_findings: SensorFinding[]
  fault_class: (FaultClass & { terminal_code: string | null; situation_id: string | null }) | null
  situation_ids: string[]
  context: { non_welding_share: number; alarm_held: boolean; gun_norm: string }
  caveats: string[]
  error_timeline: ErrorSpan[]
  sensor_trend: SensorTrend[]
}

/** api가 핸드오프를 받을 때 복사해 둔 ML trace의 이상 점수 (1분 창 시작 시각) */
export interface ScorePoint {
  t: string
  score: number
  threshold: number
  non_welding_share: number
  error_codes: string[]
}

/** RAG status 6종 (docs/06) */
export type GuidanceStatus =
  | 'ok'
  | 'retrieval_only'
  | 'unknown_situation_id'
  | 'no_evidence'
  | 'needs_citation_review'
  | 'error'

/** 원인 후보 1개 = 상황 ID 1개 */
export interface Candidate {
  situation_id: string
  rank: number
  name: string
  definition: string
  coverage_note: string | null
  /** 점검 항목. id는 RAG에 아직 없어 FE가 붙인 임시 값 (docs/06 요청 3) */
  checks: { id: string; title: string }[]
  /** Festo 오류 번호 후보. manual은 원문을 확인한 조치 문구가 있을 때만 */
  diagnostics: {
    number: number
    priority: string | null
    rationale: string | null
    manual?: { name: string; page: number; text: string }
  }[]
  evidence: { page: number; quote: string | null; rationale: string | null }[]
}

export interface Guidance {
  status: GuidanceStatus
  candidates: Candidate[]
}

// ---- 사람이 입력하는 기록 (api가 저장, ML · RAG에는 보내지 않음) ----

export type CheckResult = 'normal' | 'abnormal' | 'skipped'

export interface CheckRecord {
  result: CheckResult
  memo?: string
}

export type Verdict = 'yes' | 'no' | 'unsure'

export interface Judgment {
  situation_id: string
  verdict: Verdict
  /** 기록 시각 (작업 시각, 서버가 넣음) */
  at: string
}

export const ACTION_KINDS = ['조정', '교체', '청소', '재체결', '기타'] as const
export type ActionKind = (typeof ACTION_KINDS)[number]

export interface ActionInput {
  situation_id: string
  kind: ActionKind
  did: string
  parts: { name: string; qty: number }[]
  worker: string
  /** 작업 시각 (ISO) */
  started_at: string
  ended_at: string
}

export interface ActionRecord extends ActionInput {
  at: string
}

export type Outcome = 'resolved' | 'retry' | 'next'

export interface ResultRecord {
  situation_id: string
  outcome: Outcome
  at: string
}

export const RECURRENCE = ['처음', '반복', '모름'] as const
export type Recurrence = (typeof RECURRENCE)[number]

/** 정비일지 칸. 앞 단계 기록으로 미리 채우고 사람이 고친다 (AI 생성 아님) */
export interface LogInput {
  /** 작업한 날짜 (작업 시각 기준) */
  date: string
  work_time: string
  worker_gun: string
  problem: string
  cause: string
  action: string
  missed_checks: string
  recurrence: Recurrence
}

export interface MaintenanceLog extends LogInput {
  /** 마지막 저장 시각 (작업 시각, 서버가 넣음) */
  saved_at: string
  /** 일지 승인 시각. 승인 전이면 null */
  approved_at: string | null
}

/** 케이스를 닫은 방식. resolved = 해결 종료, unresolved = 미해결로 저장, handed_over = 도움 요청 (뒤의 둘은 작업함에 남음) */
export type Closure = 'resolved' | 'unresolved' | 'handed_over'

/** 수동 모드에서 원인을 직접 찾았을 때 조치 · 결과에 쓰는 situation_id */
export const MANUAL_CAUSE = 'MANUAL'

/** 수동 모드 점검: 매뉴얼 안내 없이 사람이 확인한 것 */
export interface ManualCheck {
  title: string
  result: 'normal' | 'abnormal'
  memo?: string
}

export interface ManualRecord {
  checks: ManualCheck[]
  /** 직접 찾은 원인. 못 찾았으면 빈 문자열 */
  cause: string
  saved_at: string
}

/** 현장 확인 입력: 누가 언제 직접 확인했나 (결과는 records.checks에 같이 저장) */
export interface FieldRecord {
  by: string
  check_ids: string[]
  at: string
}

export const URGENCY = ['보통', '급함', '매우 급함'] as const
export const HANDOVER_REASONS = ['원인을 못 찾음', '권한이 필요함', '부품이 필요함', '기타'] as const

export interface HandoverInput {
  to: string
  urgency: (typeof URGENCY)[number]
  reasons: (typeof HANDOVER_REASONS)[number][]
  note: string
}

export interface HandoverRecord extends HandoverInput {
  at: string
}

export interface CaseRecords {
  /** 점검 항목 id → 결과 */
  checks: Record<string, CheckRecord>
  /** 점검을 처음 시작한 시각 (작업 시각) */
  started_at?: string
  judgments: Judgment[]
  actions: ActionRecord[]
  results: ResultRecord[]
  log?: MaintenanceLog
  manual?: ManualRecord
  field: FieldRecord[]
  handovers: HandoverRecord[]
  /** 해결 종료 또는 미해결로 저장한 기록 (작업 시각) */
  closure?: { outcome: Closure; at: string }
}

/** GET /api/cases/:id 응답 */
export interface CaseDetail extends Omit<CaseSummary, 'event'> {
  event: HandoffEvent
  score_trace: ScorePoint[]
  guidance: Guidance
  records: CaseRecords
}

/** GET /api/history 의 한 줄: 해결 종료했거나 미해결로 저장한 케이스 */
export interface HistoryItem extends CaseSummary {
  closure: { outcome: Closure; at: string }
  /** 점검을 처음 시작한 시각 (걸린 시간 계산용, 작업 시각) */
  started_at: string | null
  /** 맞다고 판단한 원인 이름. 없으면 null */
  cause: string | null
  /** 마지막 조치 종류 */
  action_kind: string | null
  worker: string | null
}

export interface HistoryResponse {
  items: HistoryItem[]
}

// ---- 설비 (GET /api/guns) ----

/** ML GET /guns 한 줄 (그대로) + api가 붙인 열린 고장 */
export interface GunStatus {
  gun_id: string
  /** warming_up = 이 설비 기준 수집 중, gun = 이 설비 기준, global = 공통 기준 */
  gun_norm: 'warming_up' | 'gun' | 'global'
  gun_threshold: number | null
  warmup_rows: number
  last_score: number | null
  consecutive_alarms: number
  drift_warning: string[]
  open_cases: CaseSummary[]
}

export interface GunsResponse {
  guns: GunStatus[]
}

/** 매뉴얼 근거 쪽 검색 결과 (RAG 매핑의 근거 쪽에서 찾음, 매뉴얼 전문 검색 아님) */
export interface ManualHit {
  page: number
  title: string
  quote: string | null
  situation_id: string
  situation_name: string | null
}
