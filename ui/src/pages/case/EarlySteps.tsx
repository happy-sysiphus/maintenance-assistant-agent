// 조기 경보 정비 단계 (11/12): 점검 → 조치 → 정비일지. 긴급 케이스의 같은 화면 모양을 쓰고 저장만 조기 경보 api로.
// 판단 · 결과 단계는 없다: 원인은 후보 목록에서 이미 골랐고(점검이 모두 정상이면 "다음 원인 보기"로 후보 목록), 해결 여부는 정비일지에서 체크한다.
// 저장: PUT /warning/draft (점검 · 조치 · 작업자), 종료: POST /warning/resolve (해결 확인 + 일지 승인).
import { useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { DotTag } from '../../components/Tag'
import { btn, btnLarge, btnPrimary, btnQuiet, card as cardBox, input, label } from '../../components/ui'
import { docFromParam, MANUAL_SHORT, type ManualDoc } from '../../lib/api'
import { formatKst } from '../../lib/time'
import { activeEventId, addCleared, candidateIds, EARLY_STEPS, readCleared, selectedOf, useWarning, useWarningAction, type WarningCheck, type WarningDraft } from '../../lib/warning'
import { ACTION_KINDS, type CheckResult } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, SideDivider, SideLabel, sideCard, Steps, TwoColumns } from './parts'

const EMPTY: WarningDraft = { worker: '', actual_action: '', cause: '', checks: {} }
// 화면의 건너뜀 ↔ api의 unchecked
const toApi = (r: CheckResult): WarningCheck => (r === 'skipped' ? 'unchecked' : r)
const fromApi = (r: WarningCheck | undefined): CheckResult | undefined => (r === 'unchecked' ? 'skipped' : r)

/** 세 화면이 같이 쓰는 것: 고른 원인 · 임시 저장 · 저장 함수 */
function useEarlyWork() {
  const d = useCaseDetail()
  // 원인을 고른 경보 (반복 경보를 보고 정비했으면 그 경보로 저장 · 종료)
  const eventId = activeEventId(d)
  const q = useWarning(d.case_id, eventId)
  const act = useWarningAction(d.case_id, eventId)
  const { card, source } = selectedOf(q.data)
  // api 응답의 draft에는 selection · saved_at도 붙어 온다. 보낼 때는 이 세 칸만 (api가 다른 칸은 422로 거부)
  const raw = q.data?.draft ?? EMPTY
  const draft: WarningDraft = { worker: raw.worker ?? '', actual_action: raw.actual_action ?? '', cause: raw.cause ?? '', checks: raw.checks ?? {} }
  const save = (patch: Partial<WarningDraft>, onSuccess?: () => void) =>
    act.mutate({ path: 'draft', body: { ...draft, ...patch } }, { onSuccess })
  return { d, q, act, card, source, draft, save, eventId }
}

/** ?manual=쪽&doc= 로 오른쪽에 PDF를 연다 (긴급 점검 · 조치 화면과 같은 방식) */
function useManualParam() {
  const [params, setParams] = useSearchParams()
  const page = Number(params.get('manual')) || null
  const doc = docFromParam(params.get('doc'))
  const open = (p: number | null, dc: ManualDoc = doc) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (p) next.set('manual', String(p))
        else next.delete('manual')
        if (p) next.set('doc', dc)
        else next.delete('doc')
        return next
      },
      { replace: true },
    )
  return { page, doc, open }
}

function Loading({ q }: { q: ReturnType<typeof useWarning> }) {
  return q.isError ? <p className="text-sm text-alarm">경보 정보를 불러오지 못했어요.</p> : <p className="text-sm text-sub">불러오는 중…</p>
}

// ---------------- 점검 ----------------

export function EarlyCheck() {
  const { d, q, act, card, source, draft, save, eventId } = useEarlyWork()
  const navigate = useNavigate()
  const manual = useManualParam()
  const [checks, setChecks] = useState<Record<string, CheckResult> | null>(null)
  const [picked, setPicked] = useState<string | null>(null)
  const [noIssue, setNoIssue] = useState(false)
  // 직접 찾은 근거로 정비할 때는 원인 이름을 작업자가 적는다 (정비일지 · 이력의 "원인", 11/12)
  const [causeText, setCauseText] = useState<string | null>(null)
  if (!q.data) return <Loading q={q} />
  // 원인을 안 골랐으면 상황 화면(후보 목록)으로
  if (!card && !source) return <Navigate to=".." replace />

  const items = (card?.checks ?? []).map((c, i) => ({ ...c, key: c.id ?? String(i) }))
  const value = checks ?? Object.fromEntries(Object.entries(draft.checks).map(([k, v]) => [k, fromApi(v)!]))
  const done = items.filter((it) => value[it.key]).length
  const active = items.find((it) => it.key === picked) ?? items.find((it) => !value[it.key]) ?? items[0]
  const ref = card?.source ?? source!
  const anyAbnormal = Object.values(value).includes('abnormal')
  const asApi = () => Object.fromEntries(Object.entries(value).map(([k, v]) => [k, toApi(v)]))
  // 판단 단계 대신 (11/12): 이상이 하나라도 있으면 조치로, 하나도 없으면 "이 원인이 아닌 것 같아요"를 묻는다
  const cause = causeText ?? draft.cause ?? ''
  const needCause = !card && !cause.trim()
  const submit = (next: boolean) =>
    save({ checks: asApi(), ...(card ? {} : { cause: cause.trim() }) }, () => {
      if (!next) return
      if (anyAbnormal || items.length === 0) navigate('../action')
      else setNoIssue(true)
    })
  // 이상 없음 다음 (흐름도 "맞는 후보가 있나?"):
  //  · 사례 카드로 들어왔으면(후보 목록 전) Claude 후보를 받는다. 새 후보가 하나도 없으면 바로 수동 모드
  //  · 아직 안 본 후보가 있으면 후보 목록으로 (방금 원인은 "점검함 · 이상 없음", 다음 원인이 골라져 있다)
  //  · 후보를 다 봤으면 목록을 거치지 않고 바로 수동 모드
  const w = q.data
  const fromAuto = !w.generate_candidates && card?.manual_case_id === w.automatic_card?.manual_case_id
  const checked = new Set([...readCleared(d.case_id, eventId), ...(card ? [card.manual_case_id] : [])])
  const remaining = candidateIds(w).filter((id) => !checked.has(id))
  const toManual = !fromAuto && remaining.length === 0
  const manualPath = `../manual?event=${encodeURIComponent(eventId)}`
  const nextCause = () => {
    if (card) addCleared(d.case_id, eventId, card.manual_case_id)
    if (toManual) return navigate(manualPath)
    if (!fromAuto) return navigate('..')
    // 사례 카드로 들어왔으면 후보를 받는다 (사례 카드는 거절하지 않고 목록 맨 위에 남는다. 거절은 다른 원인을 고를 때)
    act.mutate(
      { path: 'candidates' },
      {
        onSuccess: (res) => {
          const r = res as { candidates?: unknown[] }
          navigate((r.candidates?.length ?? 0) > 0 ? '..' : manualPath)
        },
      },
    )
  }

  return (
    <>
      <BackLink to=".." />
      <Steps current="점검" steps={EARLY_STEPS} />
      <TwoColumns
        wide={manual.page !== null}
        side={
          manual.page ? (
            <ManualPanel page={manual.page} doc={manual.doc} onPage={(p) => manual.open(p)} onClose={() => manual.open(null)} />
          ) : (
            <aside className={sideCard} aria-label="점검 기준">
              {active ? (
                <>
                  <SideLabel>
                    {items.indexOf(active) + 1}. {active.criterion?.kind === 'value' ? '정상 기준' : '확인할 것'}
                  </SideLabel>
                  <p className="text-base font-semibold">{active.criterion?.text ?? active.text}</p>
                  {(active.criterion?.cite ?? active.cite) && (
                    <CiteBox cite={(active.criterion?.cite ?? active.cite)!} onOpen={manual.open} />
                  )}
                </>
              ) : (
                <>
                  <SideLabel>매뉴얼 근거</SideLabel>
                  <CiteBox cite={ref} onOpen={manual.open} />
                </>
              )}
              {card?.coverage_note && (
                <>
                  <SideDivider />
                  <SideLabel>적용 조건</SideLabel>
                  <p className="text-sm">{card.coverage_note}</p>
                </>
              )}
            </aside>
          )
        }
      >
        <section className={`${cardBox} overflow-hidden`}>
          <div className="flex items-center gap-3 border-b border-line-soft px-[22px] py-4">
            <span className="text-xs font-medium text-sub">원인</span>
            {card ? (
              <b className="min-w-0 grow text-xl font-semibold tracking-[-0.015em]">{card.title}</b>
            ) : (
              <input
                className={`${input} grow text-lg font-semibold`}
                value={cause}
                onChange={(e) => setCauseText(e.target.value)}
                maxLength={200}
                placeholder="매뉴얼을 보고 확인한 원인을 적어 주세요 (예: 전극 캡 마모)"
                aria-label="원인"
              />
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 border-b border-line-soft px-[22px] py-3">
            <span className="mr-1 text-xs font-medium text-sub">매뉴얼 근거</span>
            <DocButton
              page={ref.page}
              label={`${MANUAL_SHORT[ref.doc]} ${ref.page}쪽`}
              on={manual.page === ref.page && manual.doc === ref.doc}
              onOpen={(p) => manual.open(p, ref.doc)}
            />
          </div>
          {items.length === 0 ? (
            <p className="px-[22px] py-5 text-sm text-sub">
              {card ? '매뉴얼에 이 원인의 점검 항목이 없어요. 근거 쪽을 확인하고 다음을 누르세요.' : '근거 쪽을 확인하고, 확인한 원인을 위에 적은 뒤 다음을 누르세요.'}
            </p>
          ) : (
            <ol>
              {items.map((it, i) => (
                <li
                  key={it.key}
                  onClickCapture={() => setPicked(it.key)}
                  onFocusCapture={() => setPicked(it.key)}
                  className={`flex items-center gap-3 border-t border-l-[3px] border-t-line-soft px-5 py-3.5 transition-colors first:border-t-0 ${
                    it === active ? 'border-l-primary bg-primary-soft/40' : 'border-l-transparent'
                  }`}
                >
                  <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-fill text-xs font-semibold text-sub">{i + 1}</span>
                  <span className="min-w-0 grow text-base">{it.text}</span>
                  <ResultSeg label={it.text} value={value[it.key]} onChange={(v) => setChecks({ ...value, [it.key]: v })} />
                </li>
              ))}
            </ol>
          )}
        </section>
        {noIssue && !anyAbnormal ? (
          // 점검이 모두 정상(또는 건너뜀): 긴급의 판단 "아니에요"처럼 다음 원인으로
          <div className="flex items-center gap-3 rounded-xl border border-line bg-white px-5 py-4">
            <div className="flex min-w-0 grow flex-col gap-0.5">
              <b className="font-semibold">점검한 항목에 이상이 없어요</b>
              <span className="text-sm text-sub">
                {toManual ? '후보를 모두 확인했어요. 매뉴얼에서 직접 찾아보세요.' : '이 원인은 아닌 것 같아요. 다음 원인을 확인해 보세요.'}
              </span>
            </div>
            {act.isError && <span className="text-sm text-alarm">다시 눌러 주세요.</span>}
            <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={act.isPending} onClick={nextCause}>
              {act.isPending ? '다음 원인 찾는 중…' : toManual ? '매뉴얼에서 직접 찾기' : '다음 원인 보기'}
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            {items.length > 0 && (
              <span className="text-sm">
                <b className="font-semibold">
                  {done} / {items.length} 확인
                </b>
              </span>
            )}
            {act.isError && <span className="text-sm text-alarm">저장하지 못했어요. 입력 내용은 남아 있어요.</span>}
            <div className="grow" />
            <button type="button" className={btn} disabled={act.isPending} onClick={() => submit(false)}>
              임시 저장
            </button>
            <button
              type="button"
              className={`${btnPrimary} ${btnLarge}`}
              disabled={act.isPending || needCause || (items.length > 0 && done === 0)}
              onClick={() => submit(true)}
            >
              다음
            </button>
          </div>
        )}
      </TwoColumns>
    </>
  )
}

function CiteBox({ cite, onOpen }: { cite: { doc: ManualDoc; page: number; quote: string }; onOpen: (p: number, doc: ManualDoc) => void }) {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg bg-canvas px-3.5 py-3">
      <p className="text-sm leading-relaxed text-ink-2">“{cite.quote}”</p>
      <button
        type="button"
        onClick={() => onOpen(cite.page, cite.doc)}
        className="inline-flex min-h-10 w-full items-center justify-center rounded-lg border border-field bg-white text-sm font-medium text-primary-ink hover:bg-primary-soft"
      >
        {MANUAL_SHORT[cite.doc]} {cite.page}쪽 열기
      </button>
    </div>
  )
}

// ---------------- 조치 ----------------

/** api에는 조치가 글 한 칸(actual_action)뿐이라 "종류 — 한 일"로 합쳐 저장하고, 다시 열면 나눈다 */
function splitAction(text: string) {
  const m = text.match(new RegExp(`^(${ACTION_KINDS.join('|')}) — ([\\s\\S]*)$`))
  return m ? { kind: m[1], did: m[2] } : { kind: '', did: text }
}

export function EarlyAction() {
  const { q, act, card, source, draft, save } = useEarlyWork()
  const navigate = useNavigate()
  const manual = useManualParam()
  const [form, setForm] = useState<{ kind: string; did: string; worker: string } | null>(null)
  const [tried, setTried] = useState(false)
  if (!q.data) return <Loading q={q} />
  if (!card && !source) return <Navigate to=".." replace />

  const v = form ?? { ...splitAction(draft.actual_action), worker: draft.worker }
  const set = (patch: Partial<typeof v>) => setForm({ ...v, ...patch })
  const ok = v.kind && v.did.trim() && v.worker.trim()
  const manualAction = card?.actions[0]
  const submit = () => {
    setTried(true)
    if (!ok) return
    save({ worker: v.worker.trim(), actual_action: `${v.kind} — ${v.did.trim()}` }, () => navigate('../log'))
  }

  return (
    <>
      <BackLink to="../check" />
      <Steps current="조치" steps={EARLY_STEPS} />
      <TwoColumns
        wide={manual.page !== null}
        side={
          manual.page ? (
            <ManualPanel page={manual.page} doc={manual.doc} onPage={(p) => manual.open(p)} onClose={() => manual.open(null)} />
          ) : (
            <aside className={sideCard} aria-label="매뉴얼 조치">
              <SideLabel>매뉴얼 조치 (참고)</SideLabel>
              {card && card.actions.length > 0 ? (
                card.actions.map((a, i) => (
                  <div key={i} className="flex flex-col gap-1">
                    {a.when && <span className="text-xs text-sub">{a.when}</span>}
                    <p className="text-sm">{a.text}</p>
                    {a.cite && (
                      <button type="button" className={`${btnQuiet} self-start`} onClick={() => manual.open(a.cite!.page, a.cite!.doc)}>
                        {MANUAL_SHORT[a.cite.doc]} {a.cite.page}쪽 열기
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <p className="text-sm text-sub">매뉴얼에 적힌 조치가 없어요. 실제로 한 일을 적어 주세요.</p>
              )}
              {card?.repairs && card.repairs.length > 0 && (
                <>
                  <SideDivider />
                  <SideLabel>지난번 실제 조치</SideLabel>
                  {card.repairs.slice(0, 2).map((r) => (
                    <p key={r.case_id} className="line-clamp-2 text-sm text-ink-2">
                      {r.actual_action}
                    </p>
                  ))}
                </>
              )}
            </aside>
          )
        }
      >
        <section className={`${cardBox} flex flex-col gap-5 p-[26px]`}>
          <div className="flex items-center gap-3">
            <h2 className="grow text-[22px] leading-snug font-medium tracking-[-0.015em]">어떤 조치를 했나요?</h2>
            {manualAction && (
              <button type="button" className={btn} onClick={() => set({ did: `${manualAction.text} (매뉴얼 조치, 실제로 한 일로 고치세요)` })}>
                매뉴얼 조치 불러오기
              </button>
            )}
          </div>
          <div className="flex items-center gap-3 rounded-lg bg-canvas px-3.5 py-2.5 text-sm">
            <span className="text-xs font-medium text-sub">원인</span>
            <b className="font-semibold">{card?.title ?? (draft.cause || '직접 찾은 근거')}</b>
            <div className="grow" />
            <Link to=".." className={btnQuiet}>
              바꾸기
            </Link>
          </div>
          <fieldset className="flex flex-col gap-1.5">
            <legend className={label}>조치 종류 *</legend>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {ACTION_KINDS.map((k) => (
                <button
                  key={k}
                  type="button"
                  aria-pressed={v.kind === k}
                  onClick={() => set({ kind: k })}
                  className={`inline-flex min-h-10 items-center rounded-lg border px-3.5 text-sm ${
                    v.kind === k ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-ink-2'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
            {tried && !v.kind && <span className="text-xs text-alarm">조치 종류를 골라 주세요</span>}
          </fieldset>
          <label className="flex flex-col gap-1.5">
            <span className={label}>한 일 *</span>
            <textarea rows={3} className={`${input} resize-y`} placeholder="실제로 한 작업을 적어 주세요" value={v.did} onChange={(e) => set({ did: e.target.value })} />
            {tried && !v.did.trim() && <span className="text-xs text-alarm">한 일을 적어 주세요</span>}
          </label>
          <label className="flex max-w-[280px] flex-col gap-1.5">
            <span className={label}>작업자 *</span>
            <input className={input} placeholder="이름" value={v.worker} onChange={(e) => set({ worker: e.target.value })} />
            {tried && !v.worker.trim() && <span className="text-xs text-alarm">작업자를 적어 주세요</span>}
          </label>
          <div className="flex items-center gap-3 border-t border-line-soft pt-4">
            {act.isError && <span className="text-sm text-alarm">저장하지 못했어요. 입력 내용은 남아 있어요.</span>}
            <div className="grow" />
            <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={act.isPending} onClick={submit}>
              다음
            </button>
          </div>
        </section>
      </TwoColumns>
    </>
  )
}

// ---------------- 정비일지 ----------------

export function EarlyLog() {
  const { d, q, act, card, source, draft } = useEarlyWork()
  const navigate = useNavigate()
  const [fixed, setFixed] = useState(false)
  if (!q.data) return <Loading q={q} />
  const closed = d.status === 'resolved'
  if (!closed && !card && !source) return <Navigate to=".." replace />

  const abnormal = (card?.checks ?? []).filter((c, i) => draft.checks[c.id ?? String(i)] === 'abnormal').map((c) => c.text)
  const missed = (card?.checks ?? []).filter((c, i) => (draft.checks[c.id ?? String(i)] ?? 'unchecked') === 'unchecked').map((c) => c.text)
  const rows: [string, string][] = [
    ['문제', `조기 경보 발생 (데이터 시각 ${formatKst(d.event.detected_at) ?? '—'})`],
    ['원인', [card?.title ?? (source ? `${draft.cause || '원인 미입력'} (직접 찾은 근거 · ${MANUAL_SHORT[source.doc]} ${source.page}쪽)` : '—'), abnormal.length ? `${abnormal.join(', ')} 이상` : ''].filter(Boolean).join(' — ')],
    ['조치', draft.actual_action || '—'],
    ['작업자', draft.worker || '—'],
    ['못 한 점검', missed.join(' · ') || '없음'],
  ]
  const ready = Boolean(draft.actual_action.trim() && draft.worker.trim() && (card || draft.cause?.trim()))
  const resolve = () =>
    act.mutate(
      { path: 'resolve', body: { ...draft, worker_confirmed: true, log_approved: true } },
      // 닫으면 상황 화면: 이 해결이 다음 경보에 어떻게 쓰이는지(피드백 루프)를 보여 준다
      { onSuccess: () => navigate('..') },
    )

  return (
    <>
      {!closed && <BackLink to="../action" />}
      {!closed && <Steps current="정비일지" steps={EARLY_STEPS} />}
      <TwoColumns>
        <section className={`${cardBox} flex flex-col gap-3.5 p-[26px]`}>
          <div className="flex items-center gap-3">
            <h2 className="grow text-xl font-semibold tracking-[-0.015em]">정비일지</h2>
            {closed && <DotTag color="gray">해결 종료 · 읽기 전용</DotTag>}
          </div>
          <dl className="flex flex-col">
            {rows.map(([k, val]) => (
              <div key={k} className="flex gap-4 border-t border-line-soft py-3 first:border-t-0">
                <dt className="w-24 shrink-0 text-sm text-sub">{k}</dt>
                <dd className="min-w-0 grow text-sm whitespace-pre-wrap">{val}</dd>
              </div>
            ))}
          </dl>
          {!closed && (
            <div className="mt-2 flex flex-col gap-3 border-t border-line-soft pt-4">
              {!ready && (
                <p className="text-sm text-sub">
                  {!card && !draft.cause?.trim() ? '원인이 비어 있어요.' : '조치와 작업자가 비어 있어요.'}{' '}
                  <Link to={!card && !draft.cause?.trim() ? '../check' : '../action'} className="font-medium text-primary-ink hover:underline">
                    {!card && !draft.cause?.trim() ? '원인 적기' : '조치 적기'}
                  </Link>
                </p>
              )}
              {/* 결과 단계 대신 한 줄 체크: api는 해결 확인 · 일지 승인이 있어야 닫는다 */}
              <label className="flex items-center gap-2.5 text-base">
                <input type="checkbox" className="size-5 accent-primary" checked={fixed} onChange={(e) => setFixed(e.target.checked)} />
                조치 후 정상으로 돌아온 것을 확인했어요
              </label>
              <div className="flex items-center gap-3">
                {act.isError && <span className="text-sm text-alarm">저장하지 못했어요. 다시 눌러 주세요.</span>}
                <div className="grow" />
                <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={!ready || !fixed || act.isPending} onClick={resolve}>
                  일지 승인하고 해결 종료
                </button>
              </div>
            </div>
          )}
        </section>
      </TwoColumns>
    </>
  )
}
