// 조기 경보 수동 모드 (11/12, 방식 c): 후보 목록에 맞는 원인이 없을 때, 케이스 안에서 챗봇처럼 매뉴얼을 찾는다.
// 속은 POST /warning/manual (질문 + 이 경보의 센서 소견으로 검색, explain이면 Claude 설명). 앞 대화는 history로 넘긴다 (api 11/12 FE 추가).
// 고를 수 있는 근거는 서버가 이 경보에서 돌려준 것뿐 → "이 근거로 정비"는 evidence_id로 고르고 점검으로 간다.
// 직접 고른 근거로 해결하면 검토 대기(pending_review)라 검토 전에는 다음 경보의 사례 카드에 쓰이지 않는다.
import { useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { ChatIcon, SendIcon } from '../../components/Icons'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { card as cardBox } from '../../components/ui'
import { apiSend, docFromParam, MANUAL_SHORT, type ManualDoc } from '../../lib/api'
import { activeEventId, sensorKo, useWarning, useWarningAction, type WarningSource } from '../../lib/warning'
import { useCaseDetail } from './context'
import { BackLink, SideDivider, SideLabel, sideCard, TwoColumns } from './parts'

type SearchResult = {
  status: 'ok' | 'no_evidence' | 'search_error' | string
  hits: { source: WarningSource & { evidence_id: string }; review_needed?: boolean }[]
  notes: { text: string; source: WarningSource }[]
}
type Msg = { q: string; state: 'pending' | 'done' | 'error'; res?: SearchResult }

export function EarlyManual() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  // 상황 화면에서 고른 경보 (?event=), 없으면 원인을 고른 경보 · 대표 경보
  const eventId = activeEventId(d, params.get('event'))
  const w = useWarning(d.case_id, eventId).data
  const act = useWarningAction(d.case_id, eventId)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [text, setText] = useState('')
  const openPage = Number(params.get('manual')) || null
  const openDoc = docFromParam(params.get('doc'))
  const open = (p: number | null, doc: ManualDoc = openDoc) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (p) next.set('manual', String(p))
        else next.delete('manual')
        if (p) next.set('doc', doc)
        else next.delete('doc')
        return next
      },
      { replace: true },
    )

  const ask = async (q: string) => {
    const question = q.trim()
    if (!question) return
    setText('')
    const i = msgs.length
    // 앞 대화를 이어서 찾는다 ("그럼 캡은요?"). 답은 설명 문장, 없으면 찾은 근거 쪽으로 (RAG는 최근 8개만 씀)
    const history = msgs
      .filter((m) => m.state === 'done')
      .flatMap((m) => [
        { role: 'user', text: m.q },
        {
          role: 'assistant',
          text: m.res?.notes.map((n) => n.text).join(' ') || m.res?.hits.map((h) => `${MANUAL_SHORT[h.source.doc]} ${h.source.page}쪽`).join(', ') || '근거 없음',
        },
      ])
      .slice(-8)
    setMsgs((m) => [...m, { q: question, state: 'pending' }])
    try {
      const res = await apiSend<SearchResult>('POST', `/cases/${d.case_id}/warning/manual`, {
        event_id: eventId,
        question,
        explain: true,
        ...(history.length ? { history } : {}),
      })
      setMsgs((m) => m.map((x, j) => (j === i ? { ...x, state: 'done', res } : x)))
    } catch {
      setMsgs((m) => m.map((x, j) => (j === i ? { ...x, state: 'error' } : x)))
    }
  }
  // 직접 찾은 근거로 정비: 사례 카드가 있었으면 그때 거절로 남긴다 (반례, Picked와 같은 규칙)
  const choose = (evidence_id: string) => {
    const select = () => act.mutate({ path: 'select', body: { evidence_id } }, { onSuccess: () => navigate('../check') })
    if (w?.automatic_card && w.automatic_card.manual_case_id !== w.rejected_auto_case_id) act.mutate({ path: 'reject' }, { onSuccess: select })
    else select()
  }
  // 처음 화면의 질문 예: 이 경보에서 방향이 뚜렷한 센서로
  const examples = (w?.observations ?? [])
    .filter((o) => o.direction)
    .slice(0, 3)
    .map((o) => `${sensorKo(o)}이(가) ${o.direction === 'up' ? '올라갈' : '내려갈'} 때 확인할 것`)
  const busy = msgs.some((m) => m.state === 'pending')

  return (
    <>
      <BackLink to=".." />
      <TwoColumns
        wide={openPage !== null}
        side={
          openPage ? (
            <ManualPanel page={openPage} doc={openDoc} onPage={(p) => open(p)} onClose={() => open(null)} />
          ) : (
            <aside className={sideCard} aria-label="찾는 방법">
              <SideLabel>찾는 방법</SideLabel>
              <p className="text-sm">현장에서 본 증상이나 궁금한 점을 적으면, 이 경보의 센서 변화와 함께 매뉴얼에서 근거를 찾아요.</p>
              <p className="text-sm">맞는 근거에서 <b className="font-semibold">이 근거로 정비</b>를 누르면 점검으로 넘어가요.</p>
              <SideDivider />
              <p className="text-sm text-sub">직접 찾은 근거로 해결한 기록은 검토를 거친 뒤에 다음 경보에 쓰여요.</p>
            </aside>
          )
        }
      >
        <section className={`${cardBox} flex min-h-[520px] flex-col`}>
          <div className="border-b border-line-soft px-[22px] py-4">
            <h2 className="text-xl font-semibold tracking-[-0.015em]">매뉴얼에서 직접 찾기</h2>
          </div>
          <div className="flex grow flex-col gap-6 px-[22px] py-5">
            {msgs.length === 0 && (
              <div className="flex flex-col gap-3">
                <p className="text-sm text-sub">무엇을 찾을까요?</p>
                <div className="flex flex-wrap gap-2">
                  {examples.map((q) => (
                    <button
                      key={q}
                      type="button"
                      onClick={() => ask(q)}
                      className="min-h-9 rounded-full border border-field bg-white px-3.5 text-sm font-medium text-ink-2 hover:border-primary"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className="flex flex-col gap-3">
                <div className="self-end rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-base whitespace-pre-wrap text-white">{m.q}</div>
                <div className="flex items-start gap-3">
                  <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary-soft text-primary-ink">
                    <ChatIcon size={18} />
                  </span>
                  <Answer m={m} onRetry={() => ask(m.q)} onOpen={open} onChoose={choose} choosing={act.isPending} />
                </div>
              </div>
            ))}
            {act.isError && <p className="text-sm text-alarm">근거를 고르지 못했어요. 다시 눌러 주세요.</p>}
          </div>
          <form
            className="border-t border-line-soft p-3"
            onSubmit={(e) => {
              e.preventDefault()
              void ask(text)
            }}
          >
            <div className="flex items-end gap-2 rounded-2xl border border-field bg-white py-2 pr-2 pl-4 focus-within:border-primary">
              <textarea
                rows={1}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault()
                    void ask(text)
                  }
                }}
                placeholder="현장에서 본 증상을 적어 주세요"
                aria-label="질문"
                className="max-h-40 min-h-10 grow resize-none bg-transparent py-2 text-base outline-none"
              />
              <button
                type="submit"
                aria-label="찾기"
                disabled={busy || !text.trim()}
                className="flex size-10 items-center justify-center rounded-xl bg-primary text-white hover:bg-primary-ink disabled:opacity-40"
              >
                <SendIcon size={18} />
              </button>
            </div>
          </form>
        </section>
      </TwoColumns>
    </>
  )
}

function Answer({
  m,
  onRetry,
  onOpen,
  onChoose,
  choosing,
}: {
  m: Msg
  onRetry: () => void
  onOpen: (p: number, doc: ManualDoc) => void
  onChoose: (evidenceId: string) => void
  choosing: boolean
}) {
  if (m.state === 'pending')
    return (
      <p className="pt-1 text-sm text-sub" role="status">
        매뉴얼에서 찾는 중…
      </p>
    )
  if (m.state === 'error' || m.res?.status === 'search_error')
    return (
      <div className="flex items-center gap-3 pt-1 text-sm" role="alert">
        <span className="text-alarm">찾지 못했어요.</span>
        <button type="button" className="font-medium text-primary-ink hover:underline" onClick={onRetry}>
          다시 시도
        </button>
      </div>
    )
  const res = m.res!
  if (res.hits.length === 0) return <p className="pt-1 text-base text-ink-2">맞는 매뉴얼 근거를 찾지 못했어요. 다르게 물어봐 주세요.</p>
  return (
    <div className="flex min-w-0 grow flex-col gap-3">
      {res.notes.map((n, i) => (
        <p key={i} className="text-base leading-relaxed">
          {n.text}
        </p>
      ))}
      {/* 근거 카드: 쪽 열기 · 발췌 · 이 근거로 정비. 원문 확인이 필요한 근거는 표시 (ui/docs/10 2.3) */}
      {res.hits.slice(0, 4).map((h) => (
        <div key={h.source.evidence_id} className="flex flex-col gap-2 rounded-lg border border-line px-3.5 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <DocButton page={h.source.page} label={`${MANUAL_SHORT[h.source.doc]} ${h.source.page}쪽`} on={false} onOpen={(p) => onOpen(p, h.source.doc)} />
            {h.review_needed && (
              <span className="rounded border-l-[3px] border-hint-line bg-hint px-2 py-0.5 text-xs font-semibold text-warn">PDF 원문 확인 필요</span>
            )}
            <div className="grow" />
            <button
              type="button"
              disabled={choosing}
              onClick={() => onChoose(h.source.evidence_id)}
              className="inline-flex min-h-9 items-center rounded-lg border border-primary px-3 text-sm font-semibold text-primary-ink hover:bg-primary-soft"
            >
              이 근거로 정비
            </button>
          </div>
          <p className="line-clamp-3 text-sm text-ink-2">{h.source.quote}</p>
        </div>
      ))}
    </div>
  )
}
