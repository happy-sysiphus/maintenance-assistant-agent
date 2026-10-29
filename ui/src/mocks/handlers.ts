import { delay, http, HttpResponse } from 'msw'
import { resolvedCauseName } from '../lib/caseFlow'
import type {
  ActionInput,
  CaseDetail,
  CaseListResponse,
  CaseSummary,
  CheckCriterion,
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
  Photo,
  Verdict,
} from '../types/case'
import caseDetails from './fixtures/case-details.json'
import checkCriteria from './fixtures/check-criteria.json'
import guns from './fixtures/guns.json'
import health from './fixtures/health.json'
import manualIndex from './fixtures/manual-index.json'
import situationsFixture from './fixtures/situations.json'
import type { AnswerBlock, AskInput, AssistantAnswer, Situation } from '../types/assistant'

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
// 점검 항목에 정상 기준을 붙인다 (api/guidance.py와 같은 표, FE 초안)
for (const d of Object.values(store))
  for (const c of d.guidance.candidates)
    for (const k of c.checks) {
      const crit = (checkCriteria.criteria as Record<string, CheckCriterion>)[k.id]
      if (crit) k.criterion = crit
    }

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

// ---- AI 챗봇 (mock) ----
// 진짜 RAG(LLM) 대신, RAG 매핑 원본(situations.json)에서 근거만 골라 답을 조립한다.
// RAG의 use_llm=False와 같은 "근거만 조회(retrieval_only)" 응답이라 AI 요약 문장은 만들지 않는다.
// 질문에 맞는 상황은 고장 코드 · Festo 오류 번호 · 글자 겹침(두 글자 단위)으로 고른다. 진짜 api에서는
// 벡터 검색이나 LLM 분류로 바꿀 자리다 (docs/05 "AI 챗봇 API").
const situations = (situationsFixture as { situations: Situation[] }).situations
const OTHER_MANUALS = /(milco|abb|fanuc|yaskawa|화낙|야스카와)/i

// 종료 코드 → 상황 ID (ml/.py/rag_mapping.py, docs/03)
const CODE_TO_SITUATION: Record<string, string> = { E012: 'S01', E016: 'S02', E028: 'S03', E029: 'S04' }

function bigrams(text: string) {
  const t = text.replace(/[^가-힣a-z0-9]/gi, '').toLowerCase()
  const out = new Set<string>()
  for (let i = 0; i < t.length - 1; i++) out.add(t.slice(i, i + 2))
  return out
}

// 앞 질문을 이어 묻는 말 ("그럼 다음은?", "그거 왜 그래?")일 때만 앞 대화 주제를 이어 쓴다
const FOLLOW_UP = /(그럼|그거|그건|이거|이건|거기|다음|더 자세히|왜 그래|그 다음)/

/** 질문에 가장 맞는 상황. 이어 묻는 말이면 앞 질문까지 보고, 못 찾으면 null */
function pickSituation(input: AskInput): Situation | null {
  const now = matchSituation(input.question)
  if (now || !FOLLOW_UP.test(input.question)) return now
  // 가장 최근 질문부터 거꾸로 보며 주제가 잡히는 첫 질문을 따른다
  for (const h of [...input.history].reverse()) {
    const hit = h.role === 'user' ? matchSituation(h.text) : null
    if (hit) return hit
  }
  return null
}

function matchSituation(text: string): Situation | null {
  if (!text.trim()) return null
  const code = text.toUpperCase().match(/E0?(12|16|28|29)\b/)
  if (code) return situations.find((s) => s.situation_id === CODE_TO_SITUATION[`E0${code[1]}`]) ?? null
  const err = text.match(/오류\s*(\d+)/)
  if (err) {
    const hit = situations.find((s) => s.diagnostics.some((d) => d.number === Number(err[1])))
    if (hit) return hit
  }
  const q = bigrams(text)
  let best: Situation | null = null
  let bestScore = 0
  for (const s of situations) {
    const name = bigrams(s.name)
    const body = bigrams([s.definition, ...s.check_sequence].join(' '))
    let score = 0
    for (const g of q) score += name.has(g) ? 3 : body.has(g) ? 1 : 0
    if (score > bestScore) [best, bestScore] = [s, score]
  }
  return bestScore >= 4 ? best : null
}

const NOT_FOUND: AssistantAnswer = {
  status: 'no_evidence',
  situation_ids: [],
  blocks: [
    {
      type: 'p',
      parts: [
        { text: '질문과 맞는 매뉴얼 근거를 찾지 못했습니다. 근거 없이 답하지 않아요. ' },
        { text: '고장 코드(E012 등)나 증상, 부품 이름', bold: true },
        { text: '을 넣어 다시 물어봐 주세요.' },
      ],
    },
  ],
  caution: null,
  evidence: [],
}

function mockAnswer(s: Situation, input: AskInput): AssistantAnswer {
  const other = input.question.match(OTHER_MANUALS)
  if (other) {
    return {
      status: 'no_evidence',
      situation_ids: [],
      blocks: [
        {
          type: 'p',
          parts: [
            { text: '지금 연결된 매뉴얼은 ' },
            { text: 'Festo 하나', bold: true },
            { text: `뿐이라 ${other[1].toUpperCase()} 내용은 찾지 못했습니다. 근거 없이 답하지 않습니다.` },
          ],
        },
      ],
      caution: null,
      evidence: [],
    }
  }
  const blocks: AnswerBlock[] = []
  if (input.photos.length) {
    blocks.push({ type: 'p', parts: [{ text: '사진 속 부품 상태는 ' }, { text: '판단하지 않습니다', bold: true }, { text: '. 사진은 이 대화에 저장해 두었어요.' }] })
  }
  blocks.push({
    type: 'p',
    parts: [
      { text: '상황 정의서의 ' },
      { text: s.name, bold: true },
      // RSW 상황 정의서 PDF는 10/28 RAG에서 빠져 쪽 칩을 달지 않는다 (mock을 새 RAG 모양으로 바꾸기 전 임시)
      { text: ' 점검 순서는 다음과 같습니다.' },
    ],
  })
  blocks.push({ type: 'ol', items: s.check_sequence })
  for (const d of s.diagnostics.filter((x) => x.manual)) {
    blocks.push({
      type: 'p',
      parts: [
        { text: `Festo 매뉴얼 오류 ${d.number}(${d.manual!.name}) 원문: “${d.manual!.text}” ` },
        { cite: { doc: 'festo', page: d.manual!.page } },
      ],
    })
  }
  if (!s.diagnostics.some((x) => x.manual) && s.diagnostics.length) {
    blocks.push({
      type: 'p',
      parts: [{ text: `관련 Festo 오류 번호 후보: ${s.diagnostics.map((d) => d.number).join(', ')}. 점검 후보일 뿐, 그 오류가 실제로 떴다는 뜻은 아닙니다.` }],
    })
  }
  return {
    status: 'retrieval_only',
    situation_ids: [s.situation_id],
    blocks,
    caution: s.coverage_note,
    evidence: [
      ...s.evidence.map((e) => ({ doc: 'festo' as const, page: e.page, title: e.rationale ?? 'Festo 매뉴얼' })),
    ],
  }
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
      .filter((d) => d.status !== 'resolved' && d.status !== 'false_alarm')
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
    const body = (await request.json()) as { by: string; checks: Record<string, CheckRecord>; photos?: Photo[] }
    return withCase(params.id as string, (d) => {
      touch(d)
      d.records.checks = { ...d.records.checks, ...body.checks }
      d.records.field.push({ by: body.by, check_ids: Object.keys(body.checks), photos: body.photos ?? [], at: new Date().toISOString() })
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
        .filter((d) => d.event.gun_id === g.gun_id && d.status !== 'resolved' && d.status !== 'false_alarm')
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

  // AI 챗봇: 상황 목록 · 질문 · 답변 피드백
  http.get('*/api/situations', () => HttpResponse.json({ situations })),

  http.post('*/api/assistant/ask', async ({ request }) => {
    const input = (await request.json()) as AskInput
    await delay(mockMode() === 'slow' ? 3000 : 700)
    if (mockMode() === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    if (OTHER_MANUALS.test(input.question)) return HttpResponse.json(mockAnswer(situations[0], input))
    const s = pickSituation(input)
    return HttpResponse.json(s ? mockAnswer(s, input) : NOT_FOUND)
  }),

  http.post('*/api/assistant/feedback', () => HttpResponse.json({ ok: true })),
]
