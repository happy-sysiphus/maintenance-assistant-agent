// 조기 경보(WARNING) api: /api/cases/:id/warning/* (PR #5, api/docs/warning-api-20261111.md 3절).
// 상세 조회는 Claude를 부르지 않는다. 원인 후보(candidates)만 Claude를 부른다.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { CaseDetail } from '../types/case'
import { ApiError, apiGet, apiSend, type ManualDoc } from './api'

export type WarningSource = { evidence_id?: string; doc: ManualDoc; page: number; quote: string }
type Cite = { doc: ManualDoc; page: number; quote: string }
export type WarningStep = { id?: string; text: string; when?: string; cite?: Cite; criterion?: { kind?: string; text: string; cite?: Cite } }
/** 점검 결과 (api 값). 화면의 "건너뜀"은 unchecked로 보낸다 */
export type WarningCheck = 'normal' | 'abnormal' | 'unchecked'
/** cause: 작업자가 적은 원인 (직접 찾은 근거로 정비할 때, api 11/12 FE 추가) */
export type WarningDraft = { worker: string; actual_action: string; cause?: string; checks: Record<string, WarningCheck> }
export type WarningCard = {
  manual_case_id: string
  title: string
  /** Claude가 이 사례를 고른 이유 (새 후보에만) */
  reason?: string
  /** history = 이전 해결 사례, generated = 매뉴얼 후보 */
  origin: 'history' | 'generated' | string
  source: WarningSource
  checks: WarningStep[]
  actions: WarningStep[]
  coverage_note?: string
  /** 확인된 해결 케이스 수 */
  resolved_cases?: number
  /** 그중 지금 설비 적용 범위에 맞는 수 (2 이상이면 자동 카드) */
  applicable_cases?: number
  repair?: { event_id: string; actual_action: string; closed_at: string }
  repairs?: { case_id: string; actual_action: string }[]
}
export type Observation = { channel: string; name: string; direction?: 'up' | 'down' | null; observed_seconds: number; first?: number; last?: number }
export type PdfCandidate = { evidence_id: string; title: string; reason: string; excerpt: string; source: WarningSource; step_kind: 'check' | 'action' }
export type WarningResult = { status: string; candidates?: (WarningCard | PdfCandidate)[]; history_candidates?: WarningCard[]; cards?: WarningCard[] }
export type WarningState = {
  route?: 'error_code' | 'history' | 'manual'
  reason?: string
  model_reason?: string
  definition?: string
  observed_error?: { code: string; observed_at: string } | null
  card?: WarningCard | null
  cards?: WarningCard[]
  candidate_comparisons?: { event_id: string; score: number; channels?: { channel: string; overlap_seconds: number }[] }[]
  event_id: string
  /** automatic · candidates · new_pattern · insufficient_channels · pending · conflict · counterexample · memory_unavailable … */
  status: string
  recording_status?: string
  observations?: Observation[]
  automatic_card?: WarningCard | null
  history_candidates?: WarningCard[]
  rejected_auto_case_id?: string
  selection?: { manual_case_id?: string; manual_source?: WarningSource }
  /** 원인 후보 보기 결과 (api가 보존) */
  generate_candidates?: WarningResult
  /** 이 경보에서 화면에 보여 준 카드 (고를 수 있는 것은 이것뿐) */
  shown_cards?: Record<string, WarningCard>
  /** 점검 · 조치 임시 저장 */
  draft?: WarningDraft
}

/** 이 케이스에서 판단할 수 있는 경보: 대표 경보 + 원본이 남은 반복 경보 (api는 반복의 handoff를 보존한다) */
export function warningEvents(d: CaseDetail) {
  return [
    { event_id: d.event.event_id, detected_at: d.event.detected_at },
    ...(d.repeats ?? []).filter((r) => r.handoff).map((r) => ({ event_id: r.event_id, detected_at: r.detected_at })),
  ]
}

/**
 * 지금 다루는 경보. 고른 값(want)이 있으면 그것, 없으면 원인을 이미 고른 경보(서버 warning_events에 selection이 있는 것),
 * 그것도 없으면 대표 경보. 점검 · 조치 · 정비일지 · 수동 모드가 같은 경보로 저장 · 종료하도록 (ui/docs/10 3.1 "같은 event_id 유지")
 */
export function activeEventId(d: CaseDetail, want?: string | null) {
  const ids = warningEvents(d).map((e) => e.event_id)
  if (want && ids.includes(want)) return want
  const states = (d as CaseDetail & { warning_events?: Record<string, { selection?: unknown }> }).warning_events ?? {}
  return [...ids].reverse().find((id) => states[id]?.selection) ?? d.preferred_event_id ?? d.event.event_id
}

/**
 * 점검했더니 이상이 없던 원인 (경보마다). 점검 화면이 "다음 원인"을 정하고 후보 목록이 회색으로 표시하는 데 같이 쓴다.
 * 이 탭(세션)에만 둔다: api에는 판단을 저장할 곳이 없다. 저장소가 막혀 있으면 빈 목록으로 동작
 */
const clearedKey = (caseId: string, eventId: string) => `early.cleared.${caseId}.${eventId}`
export function readCleared(caseId: string, eventId: string): string[] {
  try {
    return JSON.parse(sessionStorage.getItem(clearedKey(caseId, eventId)) ?? '[]') as string[]
  } catch {
    return []
  }
}
export function addCleared(caseId: string, eventId: string, manualCaseId: string) {
  try {
    const all = new Set(readCleared(caseId, eventId)).add(manualCaseId)
    sessionStorage.setItem(clearedKey(caseId, eventId), JSON.stringify([...all]))
  } catch {
    /* 저장 못 해도 흐름은 그대로 */
  }
}
/** 이 경보에서 화면에 나온 후보 (사례 카드 · 이전 해결 사례 + 매뉴얼 후보, 거절한 카드 제외) */
export function candidateIds(w: WarningState) {
  const ids = warningCards(w).map((c) => c.manual_case_id)
  return [...new Set(ids)].filter((id) => id !== w.rejected_auto_case_id)
}


/** API가 실제로 등록한 카드와 PDF 후보를 구분한다. PDF에 임의의 manual_case_id를 만들지 않는다. */
export function warningCards(w: WarningState): WarningCard[] {
  const gen = w.generate_candidates
  const all = [w.automatic_card, w.card, ...(w.cards ?? []), ...(gen?.cards ?? []), ...(gen?.history_candidates ?? []), ...(w.history_candidates ?? []), ...(gen?.candidates ?? [])]
  return [...new Map(all.filter((c): c is WarningCard => !!c && 'manual_case_id' in c).map((c) => [c.manual_case_id, c])).values()]
    .filter((c) => c.manual_case_id !== w.rejected_auto_case_id)
}
export function pdfCandidates(w: WarningState): PdfCandidate[] {
  return (w.generate_candidates?.candidates ?? []).filter((c): c is PdfCandidate => 'evidence_id' in c && !('manual_case_id' in c))
}
export const automaticCard = (w: WarningState) => {
  const c = w.automatic_card ?? w.card
  return c?.manual_case_id !== w.rejected_auto_case_id ? c : null
}

/** 고른 원인: 사례 카드면 카드, 직접 찾은 근거면 근거 (수동 모드) */
export function selectedOf(w: WarningState | undefined) {
  const id = w?.selection?.manual_case_id
  return { card: id ? w?.shown_cards?.[id] : undefined, source: w?.selection?.manual_source }
}

/** 센서 이름 (ml/.py/rag_mapping.py SENSOR_KO와 같은 글자) */
const SENSOR_KO: Record<string, string> = {
  c1: '전극 캡 오프셋', c2: '전극 힘', c3: '전극 위치', c4: '힘 도달 신호', c5: '보정 압력', c6: '마찰', c7: '최대 열림 폭',
  c8: '최대 전극 힘', c9: '시작 마찰', c10: 'US2 작동 허용 신호', c13: '보정 압력 설정값', c14: '전극 힘 설정값',
  c15: '전극 위치 설정값', c16: '판 두께 설정값', c17: '속도 설정값', c18: '건 닫기·힘 형성 명령',
}
export const sensorKo = (o: Observation) => SENSOR_KO[o.channel] ?? o.name

/** 조기 경보 정비 단계. 판단(후보 목록에서 이미 고름) · 결과(정비일지의 "해결됐어요")가 없다 (11/12) */
export const EARLY_STEPS = ['점검', '조치', '정비일지'] as const

/** 같은 신호 패턴의 해결이 몇 건 쌓이면 다음부터 바로 카드로 뜨나 (rag/early_warning.py: 서로 다른 케이스 2건) */
export const AUTO_CARD_AT = 2

export function useWarning(caseId: string, eventId: string) {
  return useQuery({
    queryKey: ['warning', caseId, eventId],
    queryFn: () => apiGet<WarningState>(`/cases/${caseId}/warning?event_id=${encodeURIComponent(eventId)}`),
    // 해결 기록을 공용 이력에 반영하는 중이면 다시 읽는다
    refetchInterval: (q) => (q.state.data?.recording_status === 'pending' || q.state.data?.recording_status === 'pending_review' || q.state.data?.status === 'pending' ? 2000 : false),
    refetchOnWindowFocus: false,
    // 422 = 센서 원본이 없는 옛 경보. 다시 불러도 같으니 재시도하지 않는다 (ui/docs/10 2.1)
    retry: (n, e) => n < 1 && !(e instanceof ApiError && e.status === 422),
  })
}

export function useWarningAction(caseId: string, eventId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ path, body }: { path: 'assess' | 'candidates' | 'select' | 'reject' | 'draft' | 'resolve'; body?: object }) =>
      // draft만 PUT (점검 · 조치 임시 저장), 나머지는 POST
      apiSend<unknown>(path === 'draft' ? 'PUT' : 'POST', `/cases/${caseId}/warning/${path}`, { event_id: eventId, ...body }),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['warning', caseId] })
      await qc.invalidateQueries({ queryKey: ['case', caseId] })
      qc.invalidateQueries({ queryKey: ['cases'] })
      qc.invalidateQueries({ queryKey: ['history'] })
    },
  })
}
