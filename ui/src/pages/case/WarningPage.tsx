import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiSend, manualUrl, type ManualDoc } from '../../lib/api'
import { btn, btnPrimary, card, input } from '../../components/ui'
import { isClosed } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, FalseAlarmButton } from './parts'

type Source = { evidence_id?: string; doc: ManualDoc; page: number; quote: string }
type Step = { id?: string; text: string; when?: string; criterion?: { text: string } }
type Card = { manual_case_id: string; title: string; reason?: string; origin: string; source: Source;
  checks: Step[]; actions: Step[]; coverage_note?: string; applicable_cases?: number;
  repairs?: { case_id: string; actual_action: string }[];
  repair?: { event_id: string; actual_action: string; closed_at: string } }
type ProposedCheck = { title: string; reason: string; step_kind: 'check' | 'action'; excerpt: string;
  source: Source; evidence_id: string }
type Selection = { manual_case_id?: string; manual_source?: Source }
type Draft = { actual_action: string; worker: string; checks: Record<string, 'normal' | 'abnormal' | 'unchecked'>; selection?: Selection }
type Response = { event_id: string; status: string; recording_status?: string; feedback_enabled?: boolean; automatic_card?: Card | null;
  route?: 'error_code' | 'history' | 'manual'; reason?: string; model_reason?: string; definition?: string
  observed_error?: { code: string }; card?: Card | null; cards?: Card[]
  candidate_comparisons?: { event_id: string; score: number }[]
  rejected_auto_case_id?: string;
  observations?: { channel: string; name: string; direction?: string | null; observed_seconds: number;
    first?: number; last?: number; change_count?: number;
    transitions?: { last_before_offset_s: number; first_after_offset_s: number; from: number; to: number }[] }[];
  history_candidates?: Card[]; candidates?: (Card | ProposedCheck)[]; hits?: { source: Source }[];
  notes?: { text: string; source: Source }[]; selection?: Selection; draft?: Draft;
  generate_candidates?: Response; search_manuals?: Response; shown_cards?: Record<string, Card> }
const states: Record<string, string> = {
  awaiting_assessment: '이전 해결 이력을 확인하는 중입니다.',
  seen: '현재 경보와 비교할 수 있는 이전 해결 사례를 찾았습니다.',
  unseen: '현재 경보에 적용할 이전 해결 사례를 찾지 못했습니다.',
  uncertain: '이력 재사용 판단을 보류했습니다. 매뉴얼에서 점검 근거를 찾을 수 있습니다.',
  unsupported_code: '장비가 보고한 코드의 등록 안내가 없습니다. 매뉴얼을 검색해 주세요.',
  rejected: '이전 해결 사례를 이번 경보에 적용하지 않습니다.',
  feedback_disabled: '이력 재사용 꺼짐 · 해결 기록은 저장됩니다.',
  insufficient_channels: '비교 가능한 센서가 3개 미만이어서 이전 사례 자동 표시는 보류합니다.',
  new_pattern: '같은 센서 패턴의 해결 이력이 없습니다.', candidates: '이전 해결 사례를 확인해 주세요.',
  automatic: '같은 패턴에서 확인된 이전 해결 사례입니다. 현재 설비에도 적용되는지 확인해 주세요.',
  pending: '같은 센서 패턴의 해결 기록을 반영 중이어서 이전 사례 자동 표시를 보류합니다.',
  counterexample: '다른 사례로 해결된 기록이 있어 자동 표시를 보류합니다.',
  conflict: '여러 해결 사례가 있어 자동 표시를 보류합니다.', memory_unavailable: '이전 이력을 조회하지 못했습니다.',
  invalid_input: '이 경보의 센서 데이터 형식을 확인해 주세요. v1.4 원본이 필요합니다.',
  no_evidence: '맞는 매뉴얼 근거가 부족합니다. 직접 검색해 주세요.',
  generation_error: '후보를 생성하지 못했습니다. 다시 시도하거나 직접 검색해 주세요.',
  retrieval_only: 'AI 설명을 완성하지 못했습니다. 아래에서 찾은 PDF 근거와 점검 카드를 확인할 수 있습니다.',
  search_error: '매뉴얼을 검색하지 못했습니다. 연결 상태를 확인하고 다시 시도해 주세요.',
}
const recordingStates: Record<string, string> = {
  pending: '해결 기록을 공용 이력에 반영 중입니다.',
  pending_review: '해결 기록 반영 완료 · 수동 근거 검토 대기',
  saved: '공용 해결 이력 반영 완료',
  failed: '공용 해결 이력을 반영하지 못했습니다. 관리자 확인이 필요합니다.',
}

export default function WarningPage() {
  const d = useCaseDetail()
  const timeseries = d.rag_mode === 'timeseries'
  const qc = useQueryClient()
  const closed = isClosed(d.status)
  const repeats = (d as typeof d & { repeats?: { event_id: string; handoff?: unknown }[] }).repeats ?? []
  const events = [d.event.event_id, ...repeats.filter(r => r.handoff).map(r => r.event_id)]
  const [eventId, setEventId] = useState(d.preferred_event_id || d.event.event_id)
  const [question, setQuestion] = useState('')
  const [action, setAction] = useState('')
  const [worker, setWorker] = useState('')
  const [checks, setChecks] = useState<Draft['checks']>({})
  const [confirmed, setConfirmed] = useState(false)
  const [approved, setApproved] = useState(false)
  const key = ['warning', d.case_id, eventId]
  const query = useQuery({ queryKey: key,
    queryFn: () => apiGet<Response>(`/cases/${d.case_id}/warning?event_id=${encodeURIComponent(eventId)}`),
    refetchInterval: query => {
      const result = query.state.data
      if (result?.recording_status === 'pending' || result?.status === 'pending') return 2000
      return result?.recording_status === 'pending_review' ? 5000 : false
    },
    refetchOnWindowFocus: false })
  const w = query.data
  const assessment = useMutation({
    mutationFn: (id: string) => apiSend<Response>('POST', `/cases/${d.case_id}/warning/assess`, { event_id: id }),
    onSuccess: (_, id) => qc.invalidateQueries({ queryKey: ['warning', d.case_id, id] }),
  })
  const [attempted, setAttempted] = useState<string[]>([])
  const assess = assessment.mutate
  useEffect(() => {
    if (timeseries && !closed && w?.status === 'awaiting_assessment' && !attempted.includes(eventId)) {
      setAttempted(ids => [...ids, eventId])
      assess(eventId)
    }
  }, [timeseries, closed, w?.status, attempted, eventId, assess])
  // Restore only on event switch / initial arrival, not on background refetch.
  const [restored, setRestored] = useState('')
  if (w && restored !== eventId) {
    setAction(w.draft?.actual_action ?? '')
    setWorker(w.draft?.worker ?? '')
    setChecks(w.draft?.checks ?? {})
    setConfirmed(false); setApproved(false); setRestored(eventId)
  }
  const mutation = useMutation({
    mutationFn: ({ path, body, method = 'POST' }: { path: string; body?: object; method?: 'POST' | 'PUT' }) =>
      apiSend<Response>(method, `/cases/${d.case_id}/warning/${path}`, { event_id: eventId, ...body }),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['warning', d.case_id] })
      await qc.invalidateQueries({ queryKey: ['case', d.case_id] })
      qc.invalidateQueries({ queryKey: ['cases'] })
      qc.invalidateQueries({ queryKey: ['history'] })
    },
  })
  const run = (path: string, body?: object, method?: 'POST' | 'PUT') => mutation.mutate({ path, body, method })
  const busy = mutation.isPending
  const generated = w?.generate_candidates
  const search = w?.search_manuals
  // Older saved results used search_error even when PDF retrieval had succeeded.
  const searchStatus = search?.status === 'search_error' && search.hits?.length ? 'retrieval_only' : search?.status
  const allCards = [...(w?.cards ?? []), ...(w?.history_candidates ?? []), ...(generated?.history_candidates ?? []),
    ...(generated?.candidates ?? []).filter((c): c is Card => 'manual_case_id' in c), ...(search?.cards ?? [])]
  const cards = [...new Map(allCards.map(c => [c.manual_case_id, c])).values()]
    .filter(c => c.manual_case_id !== w?.rejected_auto_case_id)
  const selected = w?.selection
  const selectedCard = selected?.manual_case_id ? w?.shown_cards?.[selected.manual_case_id] : undefined
  const selectedName = selectedCard?.title ?? (selected?.manual_source ? `${selected.manual_source.doc} p.${selected.manual_source.page}` : '')
  const historyCard = w?.card ?? w?.automatic_card
  const proposed = (generated?.candidates ?? []).filter((c): c is ProposedCheck => 'excerpt' in c)
  const feedback = w?.feedback_enabled
  const [localDirty, setLocalDirty] = useState(false)
  const select = (body: object) => { setChecks({}); setConfirmed(false); setApproved(false); run('select', body) }
  const cite = (source: Source) => <a className="text-primary-ink underline" target="_blank" rel="noreferrer"
    href={manualUrl(source.page, source.doc)}>{source.doc} PDF p.{source.page}</a>
  const renderCard = (c: Card, automatic = false) => (
    <article key={c.manual_case_id} className={`rounded-xl border p-4 ${selected?.manual_case_id === c.manual_case_id ? 'border-primary bg-primary/5' : 'border-line-soft'}`}>
      <p className="text-xs text-sub">{c.origin === 'history' ? '이전 해결 사례' : c.origin === 'error_code' ? 'E코드 정비 안내' : '매뉴얼 후보'}{c.applicable_cases !== undefined && ` · 해결 ${c.applicable_cases}건`}</p>
      <h3 className="mt-1 font-semibold">{c.title}</h3>
      {c.reason && <p className="mt-2 text-sm">검토 이유 · {c.reason}</p>}
      {c.coverage_note && <p className="mt-2 text-sm text-sub">{c.coverage_note}</p>}
      <div className="mt-2 text-sm">{cite(c.source)}<blockquote className="mt-1 text-sub">{c.source.quote}</blockquote></div>
      {c.checks.map((s, i) => <p key={s.id ?? i} className="mt-2 text-sm">점검 · {s.text}{s.criterion && ` (${s.criterion.text})`}</p>)}
      {c.actions.map((s, i) => <p key={i} className="mt-2 text-sm">매뉴얼 조치 · {s.when && `${s.when}: `}{s.text}</p>)}
      {c.repairs?.map(r => <p key={r.case_id} className="mt-2 text-sm">과거 실제 조치 · {r.actual_action}</p>)}
      {c.repair && <div className="mt-3 rounded-lg bg-canvas p-3 text-sm">
        <p className="font-semibold">과거 실제 조치</p><p>{c.repair.actual_action}</p>
        <p className="mt-1 text-xs text-sub">경보 {c.repair.event_id} · 해결 {c.repair.closed_at}</p>
      </div>}
      {!closed && <div className="mt-3 flex gap-2">
        <button className={btnPrimary} disabled={busy} onClick={() => select({ manual_case_id: c.manual_case_id })}>
          {selected?.manual_case_id === c.manual_case_id ? '선택됨' : automatic ? '맞아요 · 이 사례 선택' : '이 사례 선택'}
        </button>
        {automatic && <button className={btn} disabled={busy} onClick={() => run('reject')}>아니에요</button>}
      </div>}
    </article>
  )
  return <>
    <BackLink to="/" />
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
      <section className={`${card} flex flex-col gap-4 p-6`}>
        <div><p className="text-sm text-warn">{timeseries ? '정비 안내' : 'WARNING'}</p>
          <h2 className="text-xl font-semibold">{timeseries ? d.event.summary_ko ?? '센서 경보' : '조기 경보 · 원인 미확정'}</h2></div>
        <p className="text-sm text-sub">센서 변화는 관측 사실이며 고장 원인의 확정이 아닙니다.</p>
        <label className="flex flex-col gap-1 text-sm">판단할 경보
          <select className={input} value={eventId} disabled={busy || localDirty} onChange={e => { setEventId(e.target.value); setRestored(''); mutation.reset() }}>
            {events.map(id => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        {timeseries && d.preferred_event_id && d.preferred_event_id !== d.event.event_id && d.preferred_event_id !== eventId && <p className="text-sm text-sub">
          이 케이스에 새 장비 코드가 도착했습니다. 위 목록에서 경보를 선택해 안내를 확인할 수 있습니다.
        </p>}
        {localDirty && <p className="text-sm text-sub">입력한 조치를 저장한 뒤 다른 경보를 선택할 수 있습니다.</p>}
        {query.isPending && <p role="status">센서 소견을 불러오는 중…</p>}
        {query.isError && <div role="alert">경보를 불러오지 못했습니다. <button className={btn} onClick={() => query.refetch()}>다시 시도</button></div>}
        {w && <>
          <div role="status" className="rounded-lg bg-canvas p-3 text-sm">
            {w.route === 'error_code' ? `장비 보고 코드 ${w.observed_error?.code} · ${w.definition}` : states[w.status] ?? w.status}
          </div>
          {timeseries && w.model_reason && w.status !== 'seen' && <p className="text-sm text-ink-2">{w.model_reason}</p>}
          {timeseries && !!w.candidate_comparisons?.length && <details className="text-sm text-sub">
            <summary className="cursor-pointer">비교한 과거 사례와 검색 점수</summary>
            {w.candidate_comparisons.map(c => <p key={c.event_id} className="mt-1">{c.event_id} · {c.score.toFixed(3)}</p>)}
          </details>}
          {timeseries && !closed && (assessment.isError || w.status === 'uncertain') && <div>
            {assessment.isError && <p role="alert" className="text-sm text-alarm">이력 판단 요청을 완료하지 못했습니다.</p>}
            <button className={btn} disabled={assessment.isPending} onClick={() => assess(eventId)}>이력 판단 다시 시도</button>
          </div>}
          {timeseries && <details className="rounded-lg border border-line-soft p-3">
            <summary className="cursor-pointer text-sm font-medium">1분 센서 관측 · {w.observations?.length ?? 0}채널</summary>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">{w.observations?.map(o => <div key={o.channel} className="rounded-lg bg-canvas p-2 text-xs">
              <b>{o.channel}</b> · {o.name}
              <p>{o.first} → {o.last} · 관측 {o.observed_seconds}초 · 변화 {o.change_count}회</p>
            </div>)}</div>
          </details>}
          {!timeseries && <div className="grid gap-2 sm:grid-cols-3">{w.observations?.map(o => <div key={o.channel} className="rounded-lg border border-line-soft p-3">
            <p className="text-xs text-sub">{o.channel} · {o.name}</p>
            <p className="mt-2 text-lg font-semibold">{o.direction === 'up' ? '↑ 상승' : o.direction === 'down' ? '↓ 하락' : '방향 미확정'}</p>
            <p className="text-xs text-sub">관측 {o.observed_seconds}초</p>
            {o.transitions?.map((t, i) => <p key={i} className="text-xs text-sub">{t.from} → {t.to} · 시작 후 {t.last_before_offset_s}~{t.first_after_offset_s}초 사이</p>)}
          </div>)}</div>}
          {historyCard && renderCard(historyCard, true)}
          {!closed && <button className={btnPrimary} disabled={busy || assessment.isPending || w.status === 'invalid_input'} onClick={() => run('candidates')}>원인 후보 보기</button>}
          {busy && <p role="status">요청을 처리하는 중…</p>}
          {mutation.isError && <p role="alert" className="text-alarm">요청을 완료하지 못했습니다. 입력은 유지됩니다. 연결 상태와 선택한 근거를 확인하고 다시 시도해 주세요.</p>}
          {generated && <p role="status" className="text-sm text-sub">{generated.status === 'ok' ? '매뉴얼 후보를 가져왔습니다.' : states[generated.status] ?? generated.status}</p>}
          {proposed.map(c => <article key={c.evidence_id} className="rounded-xl border border-line-soft p-4">
            <p className="text-xs text-sub">매뉴얼 후보 · 적용 여부 확인 필요</p>
            <h3 className="mt-1 font-semibold">{c.title}</h3><p className="mt-2 text-sm">{c.reason}</p>
            <p className="mt-2 text-sm">{c.step_kind === 'check' ? '점검' : '조치'} · {c.excerpt}</p>
            <div className="mt-2 text-sm">{cite(c.source)}</div>
            {!closed && <button className={`${btn} mt-2`} disabled={busy} onClick={() => select({ evidence_id: c.evidence_id })}>이 후보의 PDF 근거 선택</button>}
          </article>)}
          {!closed && <form className="flex flex-col gap-2 border-t border-line-soft pt-4" onSubmit={e => { e.preventDefault(); run('manual', { question, explain: true }) }}>
            <label htmlFor="warning-question" className="font-semibold">직접 매뉴얼 검색</label>
            <textarea id="warning-question" className={input} value={question} maxLength={2000} onChange={e => setQuestion(e.target.value)} placeholder="현장에서 확인한 증상을 입력하세요" />
            <button className={btn} disabled={busy || !question.trim()}>검색</button>
          </form>}
          {searchStatus && <p role="status" className="text-sm text-sub">{searchStatus === 'ok' ? '검색된 PDF 근거' : states[searchStatus] ?? searchStatus}</p>}
          {search?.notes?.map((n, i) => <p key={i} className="text-sm">{n.text} · {cite(n.source)}</p>)}
          {cards.filter(c => c.manual_case_id !== historyCard?.manual_case_id).map(c => renderCard(c))}
          {search?.hits?.map((h, i) => <article key={h.source.evidence_id ?? i} className="rounded-lg border border-line-soft p-3 text-sm">
            {cite(h.source)}<p className="my-2">{h.source.quote}</p>
            {!closed && <button className={btn} disabled={busy} onClick={() => select({ evidence_id: h.source.evidence_id })}>이 근거 선택</button>}
          </article>)}
        </>}
      </section>
      <aside className={`${card} flex flex-col gap-4 self-start p-5`}>
        <h2 className="text-lg font-semibold">점검·조치 기록</h2>
        <p className="text-sm">적용 근거 · {selectedName || '후보 또는 검색 근거를 선택하세요'}</p>
        {selected?.manual_source && cite(selected.manual_source)}
        {selectedCard?.checks.map((step, i) => <label key={step.id ?? i} className="flex flex-col gap-1 text-sm">{step.text}
          <select className={input} disabled={closed || busy} value={checks[step.id ?? String(i)] ?? 'unchecked'} onChange={e => {
            setChecks(v => ({ ...v, [step.id ?? String(i)]: e.target.value as 'normal' | 'abnormal' | 'unchecked' })); setLocalDirty(true)
          }}><option value="unchecked">미점검</option><option value="normal">정상</option><option value="abnormal">이상</option></select>
        </label>)}
        <label className="flex flex-col gap-1 text-sm">작업자<input className={input} disabled={closed || busy} value={worker} maxLength={200}
          onChange={e => { setWorker(e.target.value); setLocalDirty(true); setApproved(false) }} /></label>
        <label className="flex flex-col gap-1 text-sm">실제 수행한 조치<textarea className={input} rows={5} disabled={closed || busy} value={action} maxLength={3000}
          onChange={e => { setAction(e.target.value); setLocalDirty(true); setApproved(false); setConfirmed(false) }} /></label>
        {!closed && <>
          <button className={btn} disabled={busy} onClick={() => mutation.mutate({ path: 'draft', method: 'PUT', body: { worker, actual_action: action, checks } }, { onSuccess: () => setLocalDirty(false) })}>점검·조치 임시 저장</button>
          <label className="flex gap-2 text-sm"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />실제 조치 후 해결됐음을 확인했습니다.</label>
          <div className="rounded-lg bg-canvas p-3 text-sm"><p className="font-semibold">종료할 정비일지 미리보기</p>
            <p>문제 · 조기 경보 · 원인 미확정</p><p>적용 근거 · {selectedName || '미선택'}</p><p>작업자 · {worker || '미입력'}</p><p>조치 · {action || '미입력'}</p>
          </div>
          <label className="flex gap-2 text-sm"><input type="checkbox" checked={approved} onChange={e => setApproved(e.target.checked)} />위 정비일지를 확인하고 승인합니다.</label>
          <button className={btnPrimary} disabled={busy || !selected || !worker.trim() || !action.trim() || !confirmed || !approved}
            onClick={() => mutation.mutate({ path: 'resolve', body: { actual_action: action, worker, worker_confirmed: confirmed, log_approved: approved, checks } }, { onSuccess: () => setLocalDirty(false) })}>해결 결과 저장·종료</button>
          <FalseAlarmButton />
        </>}
        <p className="text-xs text-sub">{feedback ? '확인된 해결 이력을 다음 경보에 재사용합니다.' : '이력 재사용은 꺼져 있습니다. 해결 기록은 저장합니다.'}</p>
        {closed && <><p role="status" className="text-sm">종료된 경보입니다. 기록은 읽기 전용입니다.</p>
          {w?.recording_status && recordingStates[w.recording_status] &&
            <p className="text-xs text-sub">{recordingStates[w.recording_status]}</p>}</>}
      </aside>
    </div>
  </>
}
