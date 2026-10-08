import { delay, http, HttpResponse } from 'msw'
import { resolvedCauseName } from '../lib/caseFlow'
import type {
  ActionInput,
  CaseDetail,
  CaseListResponse,
  CaseSummary,
  CheckRecord,
  Closure,
  GunsResponse,
  GunStatus,
  HandoverInput,
  HistoryResponse,
  LogInput,
  ManualCheck,
  ManualHit,
  Outcome,
  Verdict,
} from '../types/case'
import caseDetails from './fixtures/case-details.json'
import guns from './fixtures/guns.json'
import health from './fixtures/health.json'
import manualIndex from './fixtures/manual-index.json'

// 고정 JSON은 src/mocks/fixtures/에 둔다.
// ui/mock/은 깃에 안 올라가는 개인 폴더이고, data/라는 폴더 이름은 레포 루트 .gitignore에 걸린다.
// 경로는 '*/...'로 써서 VITE_API_BASE_URL 앞부분과 상관없이 잡히게 한다.
//
// api 서버 대신 브라우저 메모리에 케이스를 저장한다. 새로고침하면 처음 상태로 돌아간다.

// 화면 상태 확인용: 브라우저 주소에 ?mock=empty | error | slow 를 붙인다 (핸들러는 페이지에서 돌아서 location을 읽을 수 있다)
function mockMode() {
  return new URLSearchParams(window.location.search).get('mock')
}

const store: Record<string, CaseDetail> = structuredClone(caseDetails as unknown as Record<string, CaseDetail>)

function summary(d: CaseDetail): CaseSummary {
  const { event: e } = d
  return {
    case_id: d.case_id,
    status: d.status,
    severity: d.severity,
    assignee: d.assignee,
    event: {
      event_id: e.event_id,
      gun_id: e.gun_id,
      detected_at: e.detected_at,
      trigger: { source: e.trigger.source, rule_code: e.trigger.rule_code, rule_trigger_time: e.trigger.rule_trigger_time },
      fault_class: e.fault_class ? { code: e.fault_class.code, name_ko: e.fault_class.name_ko } : null,
      summary_ko: e.summary_ko,
    },
  }
}

/** 사람이 처음 무언가를 저장하면 "점검 중"으로 바꾼다 */
function touch(d: CaseDetail) {
  if (d.status === 'new') d.status = 'in_progress'
  d.records.started_at ??= new Date().toISOString()
}

function withCase(id: string, update?: (d: CaseDetail) => void) {
  const d = store[id]
  if (!d) return HttpResponse.json({ error: 'not found' }, { status: 404 })
  if (update) update(d)
  return HttpResponse.json(d)
}

export const handlers = [
  http.get('*/api/health', () => HttpResponse.json(health)),

  http.get('*/api/cases', async () => {
    const mode = mockMode()
    if (mode === 'slow') await delay(3000)
    if (mode === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    if (mode === 'empty') return HttpResponse.json<CaseListResponse>({ cases: [] })
    // 해결 종료한 케이스는 작업함에서 빠지고 정비 이력에만 보인다
    const cases = Object.values(store)
      .filter((d) => d.status !== 'resolved')
      .map(summary)
      .sort((a, b) => b.event.detected_at.localeCompare(a.event.detected_at))
    return HttpResponse.json<CaseListResponse>({ cases })
  }),

  http.get('*/api/cases/:id', async ({ params }) => {
    if (mockMode() === 'slow') await delay(3000)
    if (mockMode() === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    return withCase(params.id as string)
  }),

  // 점검 결과: 항목 id → 결과를 통째로 덮어쓴다
  http.put('*/api/cases/:id/checks', async ({ params, request }) => {
    const body = (await request.json()) as { checks: Record<string, CheckRecord> }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.checks = { ...d.records.checks, ...body.checks }
    })
  }),

  http.post('*/api/cases/:id/judgments', async ({ params, request }) => {
    const body = (await request.json()) as { situation_id: string; verdict: Verdict }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.judgments.push({ ...body, at: new Date().toISOString() })
    })
  }),

  http.post('*/api/cases/:id/actions', async ({ params, request }) => {
    const body = (await request.json()) as ActionInput
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.actions.push({ ...body, at: new Date().toISOString() })
    })
  }),

  http.post('*/api/cases/:id/results', async ({ params, request }) => {
    const body = (await request.json()) as { situation_id: string; outcome: Outcome }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.results.push({ ...body, at: new Date().toISOString() })
      if (body.outcome === 'resolved') d.status = d.records.log?.approved_at ? 'log_approved' : 'logging'
    })
  }),

  // 정비일지 저장. approved: true면 승인까지 (승인 뒤에는 칸을 잠근다), false면 승인 취소
  http.put('*/api/cases/:id/log', async ({ params, request }) => {
    const { approved, ...fields } = (await request.json()) as LogInput & { approved: boolean }
    const d = store[params.id as string]
    if (d?.status === 'resolved') return HttpResponse.json({ error: 'closed' }, { status: 409 })
    return withCase(params.id as string, (d) => {
      const now = new Date().toISOString()
      d.records.log = { ...fields, saved_at: now, approved_at: approved ? (d.records.log?.approved_at ?? now) : null }
      d.status = approved ? 'log_approved' : 'logging'
    })
  }),

  // 해결 종료(일지 승인 뒤에만) 또는 미해결로 저장
  http.post('*/api/cases/:id/close', async ({ params, request }) => {
    const { outcome } = (await request.json()) as { outcome: Closure }
    const d = store[params.id as string]
    if (outcome === 'resolved' && !d?.records.log?.approved_at) {
      return HttpResponse.json({ error: 'log not approved' }, { status: 409 })
    }
    return withCase(params.id as string, (d) => {
      d.records.closure = { outcome, at: new Date().toISOString() }
      d.status = outcome
    })
  }),

  // 정비 이력: 해결 종료했거나 미해결로 저장한 케이스, 최근 것부터
  http.get('*/api/history', async () => {
    if (mockMode() === 'slow') await delay(3000)
    if (mockMode() === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    const items = Object.values(store)
      .filter((d) => d.records.closure)
      .map((d) => {
        const action = d.records.actions.at(-1)
        return {
          ...summary(d),
          closure: d.records.closure!,
          started_at: d.records.started_at ?? null,
          cause: resolvedCauseName(d),
          action_kind: action?.kind ?? null,
          worker: action?.worker ?? null,
        }
      })
      .sort((a, b) => b.closure.at.localeCompare(a.closure.at))
    return HttpResponse.json<HistoryResponse>({ items })
  }),

  // 수동 모드: 사람이 확인한 것과 직접 찾은 원인 (통째로 덮어씀)
  http.put('*/api/cases/:id/manual', async ({ params, request }) => {
    const body = (await request.json()) as { checks: ManualCheck[]; cause: string }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.manual = { ...body, saved_at: new Date().toISOString() }
    })
  }),

  // 현장 확인 입력: 결과는 점검 결과에 합치고, 누가 확인했는지 남긴다
  http.post('*/api/cases/:id/field', async ({ params, request }) => {
    const body = (await request.json()) as { by: string; checks: Record<string, CheckRecord> }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.checks = { ...d.records.checks, ...body.checks }
      d.records.field.push({ by: body.by, check_ids: Object.keys(body.checks), at: new Date().toISOString() })
    })
  }),

  // 도움 요청: 지금까지 기록은 케이스에 이미 있으므로 요청 내용만 받는다
  http.post('*/api/cases/:id/handover', async ({ params, request }) => {
    const body = (await request.json()) as HandoverInput
    // 점검을 시작하지 않고 바로 넘길 수도 있어서 touch()로 "점검 시작"을 남기지 않는다
    return withCase(params.id as string, (d) => {
      const at = new Date().toISOString()
      d.records.handovers.push({ ...body, at })
      d.records.closure = { outcome: 'handed_over', at }
      d.status = 'handed_over'
    })
  }),

  // 설비: ML GET /guns를 그대로 쓰고, 설비별 열린 고장(해결 종료 전)을 붙인다
  http.get('*/api/guns', async () => {
    if (mockMode() === 'slow') await delay(3000)
    if (mockMode() === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    const list = (guns.guns as Omit<GunStatus, 'open_cases'>[]).map((g) => ({
      ...g,
      open_cases: Object.values(store)
        .filter((d) => d.event.gun_id === g.gun_id && d.status !== 'resolved')
        .map(summary)
        .sort((a, b) => b.event.detected_at.localeCompare(a.event.detected_at)),
    }))
    return HttpResponse.json<GunsResponse>({ guns: list })
  }),

  // 매뉴얼 근거 쪽 검색: RAG 매핑의 근거 설명 · 인용 · 상황 이름에서 찾는다 (매뉴얼 전문 검색은 RAG에 요청 중)
  http.get('*/api/manuals/festo/search', ({ request }) => {
    const q = (new URL(request.url).searchParams.get('q') ?? '').trim().toLowerCase()
    const items = (manualIndex.items as ManualHit[]).filter(
      (i) => q && [i.title, i.quote, i.situation_name].some((v) => v?.toLowerCase().includes(q)),
    )
    return HttpResponse.json({ items })
  }),
]
