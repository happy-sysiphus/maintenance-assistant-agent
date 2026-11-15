// 조기 경보 상황 화면의 오른쪽 칸 (11/11 회의 흐름도, 11/12 시안).
// 화면마다 할 일 하나: A 사례 카드(맞아요/아니에요) · B 첫 화면(원인 후보 보기) · C 후보 목록(원인 하나 고르기).
// 닫힌 경보는 해결 기록이 몇 건 쌓였는지(피드백 루프)를 보여 준다.
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { Select } from '../../components/Select'
import { btn, btnLarge, btnPrimary, btnQuiet } from '../../components/ui'
import { ApiError } from '../../lib/api'
import { formatKst } from '../../lib/time'
import { activeEventId, AUTO_CARD_AT, readCleared, sensorKo, warningEvents, useWarning, useWarningAction, type WarningCard, type WarningState } from '../../lib/warning'
import { isClosed, type CaseDetail } from '../../types/case'
import { FalseAlarmButton, SideLabel } from './parts'

const ARROW = { up: '↑', down: '↓' } as const

/** 상황 화면에서 볼 경보 (?event=). 반복 경보가 있으면 고를 수 있다. 정비가 시작되면 그 경보가 기본 */
function useChosenEvent(d: CaseDetail) {
  const [params, setParams] = useSearchParams()
  const id = activeEventId(d, params.get('event'))
  const choose = (v: string) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.set('event', v)
        return next
      },
      { replace: true },
    )
  return [id, choose] as const
}

/** 왼쪽 그래프 카드 아래: 경보가 울린 1분 동안 변한 센서 (RAG observations). 방향이 없으면 회색 – */
export function ObservationRow({ d }: { d: CaseDetail }) {
  const [eventId] = useChosenEvent(d)
  const q = useWarning(d.case_id, eventId)
  const obs = q.data?.observations ?? []
  if (obs.length === 0) return null
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="mr-1 text-xs font-medium text-sub">경보 때 변한 센서</span>
      {obs.map((o) => (
        <span key={o.channel} className="inline-flex min-h-8 items-center gap-1.5 rounded-lg bg-fill px-3 text-sm font-medium text-ink-2">
          {sensorKo(o)}
          <b className={o.direction ? 'text-ink' : 'font-normal text-sub'}>{o.direction ? ARROW[o.direction] : '–'}</b>
        </span>
      ))}
    </div>
  )
}

export function EarlyPanel({ d }: { d: CaseDetail }) {
  const [eventId, chooseEvent] = useChosenEvent(d)
  const q = useWarning(d.case_id, eventId)
  const act = useWarningAction(d.case_id, eventId)

  if (q.isPending) return <p className="text-sm text-sub">불러오는 중…</p>
  if (q.isError)
    return q.error instanceof ApiError && q.error.status === 422 ? (
      <p className="text-sm text-sub">이 경보에는 분석할 센서 원본이 없어요.</p>
    ) : (
      <div className="flex flex-col gap-2">
        <p className="text-sm text-alarm">경보 정보를 불러오지 못했어요.</p>
        <button type="button" className={btn} onClick={() => q.refetch()}>
          다시 불러오기
        </button>
      </div>
    )
  const w = q.data
  if (isClosed(d.status)) return <Closed d={d} w={w} />

  const gen = w.generate_candidates
  const auto = w.automatic_card && w.automatic_card.manual_case_id !== w.rejected_auto_case_id ? w.automatic_card : null
  const asking = act.isPending && act.variables?.path === 'candidates'

  const events = warningEvents(d)
  return (
    <>
      <Header />
      {/* 같은 설비에서 여러 번 울렸으면 어느 경보를 보고 정비할지 (정비 중에는 바꾸지 않는다) */}
      {events.length > 1 && (
        <label className="flex items-center gap-3">
          <span className="shrink-0 text-xs font-medium text-sub">이 설비에서 {events.length}회</span>
          <Select
            value={eventId}
            onChange={chooseEvent}
            ariaLabel="판단할 경보"
            options={[...events].reverse().map((e, i) => ({
              value: e.event_id,
              label: `${formatKst(e.detected_at)?.slice(5, 16).replace('-', '/') ?? e.event_id}${i === 0 ? ' · 최근' : ''}`,
            }))}
          />
        </label>
      )}
      {asking || gen ? (
        <CandidateList w={w} act={act} asking={asking} eventId={eventId} caseId={d.case_id} />
      ) : auto ? (
        <AutoCard card={auto} act={act} selected={w.selection?.manual_case_id} />
      ) : (
        <First w={w} onAsk={() => act.mutate({ path: 'candidates' })} />
      )}
      {act.isError && <p className="text-sm text-alarm">요청을 처리하지 못했어요. 다시 눌러 주세요.</p>}
      <FalseAlarmButton kind="early" />
    </>
  )
}

function Header() {
  return (
    <div className="flex flex-col gap-1">
      <span className="inline-flex items-center gap-[7px] text-sm font-semibold text-warn">
        <span className="size-[7px] rounded-full bg-warn-dot" />
        주의 · 확정 아님
      </span>
      <p className="text-sm text-ink-2">10~60분 안에 고장이 날 수 있다는 신호예요.</p>
    </div>
  )
}

type Act = ReturnType<typeof useWarningAction>

/** B. 첫 화면: 할 일은 "원인 후보 보기" 하나. 이전 해결이 있으면 한 줄로 알려 준다 */
function First({ w, onAsk }: { w: WarningState; onAsk: () => void }) {
  const past = w.history_candidates?.length ?? 0
  return (
    <>
      {past > 0 && <p className="rounded-lg bg-canvas px-3.5 py-3 text-sm text-ink-2">같은 신호 패턴을 고친 기록이 있어요.</p>}
      <button type="button" className={`${btnPrimary} ${btnLarge}`} onClick={onAsk}>
        원인 후보 보기
      </button>
    </>
  )
}

/** A. 사례 카드: 같은 신호 패턴을 2번 이상 고친 방법. 할 일은 맞아요 / 아니에요 */
function AutoCard({ card, act, selected }: { card: WarningCard; act: Act; selected?: string }) {
  const last = card.repairs?.[0]?.actual_action
  return (
    <>
      <div className="flex flex-col gap-2 rounded-lg border border-primary px-4 py-3.5 ring-1 ring-primary">
        <SideLabel>이전에 고친 방법</SideLabel>
        <b className="text-lg font-semibold tracking-[-0.015em]">{card.title}</b>
        <p className="text-sm font-medium text-primary-ink">이 신호 패턴에서 {card.applicable_cases ?? card.resolved_cases}번 해결했어요</p>
        {last && <p className="line-clamp-2 text-sm text-sub">지난 조치 · {last}</p>}
      </div>
      <Picked card={card} act={act} selected={selected} label="이 방법으로 점검" />
      <button
        type="button"
        className={btn}
        disabled={act.isPending}
        // 아니에요 → 후보 목록 (흐름도). 이전 방법은 지우지 않고 목록 맨 위에 추천으로 남긴다.
        // 거절 기록(반례)은 다른 원인으로 정비를 시작할 때 남긴다 (Picked, 11/12)
        onClick={() => act.mutate({ path: 'candidates' })}
      >
        다른 원인 보기
      </button>
    </>
  )
}

/** C. 후보 목록: 이전 해결 사례를 맨 위에, 그다음 매뉴얼 후보. 할 일은 원인 하나 고르기 */
function CandidateList({ w, act, asking, eventId, caseId }: { w: WarningState; act: Act; asking: boolean; eventId: string; caseId: string }) {
  const gen = w.generate_candidates
  // 이전에 고친 방법(사례 카드 · 이력)이 맨 위 추천, 그 아래 매뉴얼 후보 (11/12)
  const auto = w.automatic_card && w.automatic_card.manual_case_id !== w.rejected_auto_case_id ? w.automatic_card : null
  const all = [...(auto ? [auto] : []), ...(gen?.history_candidates ?? []), ...(w.history_candidates ?? []), ...(gen?.candidates ?? [])]
  const cards = [...new Map(all.map((c) => [c.manual_case_id, c])).values()].filter((c) => c.manual_case_id !== w.rejected_auto_case_id)
  // 점검했더니 이상이 없던 원인 (점검 화면 "다음 원인 보기"가 기록). 지금 고른 원인의 임시 점검이 모두 정상이어도 같다
  const draftChecks = Object.values(w.draft?.checks ?? {})
  const cleared = new Set([
    ...readCleared(caseId, eventId),
    ...(draftChecks.length > 0 && !draftChecks.includes('abnormal') && w.selection?.manual_case_id ? [w.selection.manual_case_id] : []),
  ])
  const [picked, setPicked] = useState<string | null>(null)
  // 기본으로 고르는 원인: 직접 누른 것 → 지금 고른 원인(이상 없음이 아니면) → 아직 안 본 첫 원인
  const keep = w.selection?.manual_case_id && !cleared.has(w.selection.manual_case_id) ? w.selection.manual_case_id : null
  const current = cards.find((c) => c.manual_case_id === (picked ?? keep)) ?? cards.find((c) => !cleared.has(c.manual_case_id))
  const allCleared = cards.length > 0 && cards.every((c) => cleared.has(c.manual_case_id))

  if (asking) return <p className="rounded-lg bg-canvas px-3.5 py-3 text-sm text-ink-2">매뉴얼에서 원인을 찾는 중이에요…</p>
  return (
    <>
      <SideLabel>확인할 원인</SideLabel>
      {cards.length === 0 && (
        <p className="text-sm text-sub">
          {gen?.status === 'generation_error' ? '원인 후보를 만들지 못했어요.' : '이 신호에 맞는 매뉴얼 근거를 찾지 못했어요.'}
        </p>
      )}
      {cards.map((c, i) => {
        const on = c === current
        const past = c.origin === 'history' ? c.resolved_cases : 0
        const done = cleared.has(c.manual_case_id) && !on
        return (
          <button
            key={c.manual_case_id}
            type="button"
            aria-pressed={on}
            onClick={() => setPicked(c.manual_case_id)}
            className={`flex items-start gap-3 rounded-lg border px-3.5 py-3 text-left ${on ? 'border-primary ring-1 ring-primary' : 'border-line hover:bg-canvas'}`}
          >
            <span
              className={`mt-px flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${on ? 'bg-primary text-white' : 'bg-fill text-sub'}`}
            >
              {i + 1}
            </span>
            <span className="flex min-w-0 grow flex-col gap-0.5">
              <b className={`font-semibold ${done ? 'text-sub' : ''}`}>{c.title}</b>
              {done && <span className="text-sm text-sub">점검함 · 이상 없음</span>}
              {!!past && !done && <span className="text-sm font-medium text-primary-ink">이전에 {past}번 고친 방법</span>}
              {/* 이유는 고른 원인만, 두 줄까지 (긴 설명은 점검 화면에서) */}
              {on && c.reason && <span className="line-clamp-2 text-sm text-sub">{c.reason}</span>}
            </span>
          </button>
        )
      })}
      {allCleared && <p className="text-sm text-sub">후보를 모두 점검했는데 이상이 없었어요. 매뉴얼에서 직접 찾아보세요.</p>}
      {current && (
        <Picked card={current} act={act} selected={keep ?? undefined} autoId={auto?.manual_case_id} label={`${cards.indexOf(current) + 1}번 원인으로 점검`} />
      )}
      {/* 흐름도: 맞는 후보가 없으면 수동 모드. 후보가 없거나 다 점검했으면 이게 할 일이라 파란 버튼 */}
      <Link
        to={`manual?event=${encodeURIComponent(eventId)}`}
        className={cards.length === 0 || allCleared ? `${btnPrimary} ${btnLarge}` : btnQuiet}
      >
        {cards.length === 0 || allCleared ? '매뉴얼에서 직접 찾기' : '맞는 원인이 없어요 · 직접 찾기'}
      </Link>
    </>
  )
}

/**
 * 파란 큰 버튼: 이 원인을 고르고 점검으로. 이미 고른 원인이면 고르기 없이 "이어서 점검".
 * 사례 카드(autoId)가 있는데 다른 원인을 고르면 그때 사례 카드를 거절로 남긴다: 다른 사례로 해결되면 RAG가 반례로 쓴다
 */
function Picked({ card, act, label, selected, autoId }: { card: WarningCard; act: Act; label: string; selected?: string; autoId?: string }) {
  const navigate = useNavigate()
  if (selected === card.manual_case_id)
    return (
      <button type="button" className={`${btnPrimary} ${btnLarge}`} onClick={() => navigate('check')}>
        이어서 점검
      </button>
    )
  return (
    <button
      type="button"
      className={`${btnPrimary} ${btnLarge}`}
      disabled={act.isPending}
      onClick={() => {
        const select = () => act.mutate({ path: 'select', body: { manual_case_id: card.manual_case_id } }, { onSuccess: () => navigate('check') })
        if (autoId && autoId !== card.manual_case_id) act.mutate({ path: 'reject' }, { onSuccess: select })
        else select()
      }}
    >
      {label}
    </button>
  )
}

/** 닫힌 경보: 해결이면 이 기록이 다음 경보에 어떻게 쓰이는지(피드백 루프), 오알람이면 이력에 안 들어간다는 것만 */
function Closed({ d, w }: { d: CaseDetail; w: WarningState }) {
  if (d.status !== 'resolved')
    return <p className="text-sm text-sub">오알람으로 닫았어요. 해결 기록에는 들어가지 않아요.</p>
  const id = w.selection?.manual_case_id
  const card = [w.automatic_card, ...(w.history_candidates ?? [])].find((c) => c && c.manual_case_id === id)
  // 조회 결과의 해결 수는 이 경보 자신을 빼고 센다 → 이 해결을 더한다
  const n = (card?.resolved_cases ?? 0) + 1
  const saving = w.recording_status === 'pending'
  return (
    <div className="flex flex-col gap-2">
      <b className="text-lg font-semibold">✓ 해결로 닫았어요</b>
      {saving ? (
        <p className="text-sm text-sub">해결 기록을 저장하는 중이에요…</p>
      ) : w.recording_status === 'saved' && id ? (
        <div className="flex flex-col gap-1 rounded-lg bg-canvas px-3.5 py-3">
          <span className="text-sm font-semibold">
            이 신호 패턴 해결 기록 {n >= AUTO_CARD_AT ? `${n}건` : `${n}/${AUTO_CARD_AT}`}
          </span>
          <span className="text-sm text-ink-2">
            {n >= AUTO_CARD_AT ? '다음 경보부터 이 방법을 바로 보여 줘요.' : `${AUTO_CARD_AT}건이 되면 다음 경보부터 이 방법을 바로 보여 줘요.`}
          </span>
        </div>
      ) : w.recording_status === 'pending_review' ? (
        <p className="text-sm text-sub">직접 찾은 근거라 검토 뒤에 다음 경보에 쓰여요.</p>
      ) : w.recording_status === 'failed' ? (
        <p className="text-sm text-alarm">해결 기록을 저장하지 못했어요. 관리자 확인이 필요해요.</p>
      ) : null}
    </div>
  )
}
