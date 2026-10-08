// 케이스 기록으로 "지금 어느 원인을 보고 있나"를 계산한다.
// 원인 후보는 ML이 준 순서(rank)대로 하나씩 본다. 마지막 판단이 "아니에요"이거나,
// 마지막 판단 뒤에 결과에서 "다음 원인"을 고른 후보는 제외된다. 다시 판단하면 되살아난다.
import { MANUAL_CAUSE, type Candidate, type CaseDetail, type CheckRecord, type Judgment } from '../types/case'

export type CandidateState = 'current' | 'excluded' | 'waiting'

function isExcluded(d: CaseDetail, sid: string) {
  const j = latestJudgment(d, sid)
  if (j?.verdict === 'no') return true
  const r = d.records.results.filter((x) => x.situation_id === sid).at(-1)
  return r?.outcome === 'next' && (!j || r.at > j.at)
}

/** 주소의 ?cause=S01 로 원인을 지정하면 그 원인, 없으면 지금 원인 */
export function pickCandidate(d: CaseDetail, cause: string | null): Candidate | null {
  return (cause && d.guidance.candidates.find((c) => c.situation_id === cause)) || currentCandidate(d)
}

/** 아직 제외되지 않은 첫 원인. 모두 제외됐으면 null */
export function currentCandidate(d: CaseDetail): Candidate | null {
  return d.guidance.candidates.find((c) => !isExcluded(d, c.situation_id)) ?? null
}

/** 지금 원인 다음에 볼 원인. 없으면 null */
export function nextCandidate(d: CaseDetail, sid: string): Candidate | null {
  const list = d.guidance.candidates
  const i = list.findIndex((c) => c.situation_id === sid)
  return list.slice(i + 1).find((c) => !isExcluded(d, c.situation_id)) ?? null
}

export function candidateState(d: CaseDetail, sid: string): CandidateState {
  if (isExcluded(d, sid)) return 'excluded'
  return currentCandidate(d)?.situation_id === sid ? 'current' : 'waiting'
}

export function latestJudgment(d: CaseDetail, sid: string): Judgment | undefined {
  return d.records.judgments.filter((j) => j.situation_id === sid).at(-1)
}

export function checkCounts(c: Candidate, checks: Record<string, CheckRecord>) {
  const counts = { normal: 0, abnormal: 0, skipped: 0, done: 0, total: c.checks.length }
  for (const { id } of c.checks) {
    const r = checks[id]?.result
    if (r) {
      counts[r] += 1
      counts.done += 1
    }
  }
  return counts
}

/** 매뉴얼 원문을 확인한 조치 문구 (없으면 null) */
export function manualAction(c: Candidate) {
  return c.diagnostics.find((x) => x.manual)?.manual ?? null
}

/** 화면에 보이는 신호 이름: 센서 이름 + 통계 (std = 1분 안의 흔들림, mean = 평균) */
export function signalLabel(t: { sensor_name_ko: string; statistic: string; sensor: string }) {
  if (t.statistic === 'std') return `${t.sensor_name_ko} 흔들림`
  // 용접 가동률처럼 센서가 아닌 파생 값의 평균은 이름만 쓴다
  return t.sensor.startsWith('c') ? `${t.sensor_name_ko} 평균` : t.sensor_name_ko
}

/** 해결에 이른 원인: 결과에서 "해결됐어요"를 고른 원인, 없으면 마지막으로 "맞아요"라고 한 원인 */
export function resolvedCandidate(d: CaseDetail): Candidate | null {
  const sid =
    d.records.results.filter((r) => r.outcome === 'resolved').at(-1)?.situation_id ??
    d.records.judgments.filter((j) => j.verdict === 'yes').at(-1)?.situation_id
  return d.guidance.candidates.find((c) => c.situation_id === sid) ?? null
}

/** 조치 · 결과가 다루는 원인: 지금 원인 후보, 없으면 수동 모드에서 직접 찾은 원인 */
export function activeCause(d: CaseDetail): { situation_id: string; name: string; candidate: Candidate | null } | null {
  const c = currentCandidate(d)
  if (c) return { situation_id: c.situation_id, name: c.name, candidate: c }
  const manual = d.records.manual?.cause.trim()
  return manual ? { situation_id: MANUAL_CAUSE, name: manual, candidate: null } : null
}

/** 해결에 이른 원인 이름 (매뉴얼 후보 또는 직접 찾은 원인). 없으면 null */
export function resolvedCauseName(d: CaseDetail): string | null {
  const last = d.records.results.filter((r) => r.outcome === 'resolved').at(-1)
  if (last?.situation_id === MANUAL_CAUSE) return d.records.manual?.cause || null
  return resolvedCandidate(d)?.name ?? (d.records.manual?.cause || null)
}
