import { useForm } from 'react-hook-form'
import { useNavigate } from 'react-router'
import { DotTag } from '../../components/Tag'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { latestJudgment, resolvedCandidate, resolvedCauseName } from '../../lib/caseFlow'
import { formatKst } from '../../lib/time'
import { useCaseMutation } from '../../lib/useCase'
import { MANUAL_CAUSE, RECURRENCE, type CaseDetail, type Closure, type LogInput } from '../../types/case'
import { useCaseDetail } from './context'
import { SidePanel, TwoColumns } from './parts'

// 정비일지 (wireframe/boards/V2Log.dc.html 승인 전, V2LogDone.dc.html 승인 후).
// 앞 단계에서 사람이 저장한 기록으로 미리 채운다 (AI 생성 아님). 고칠 곳만 고친다.
// 순서: "일지 승인"이 먼저, "해결 종료"는 승인 전까지 잠김. "미해결로 저장"은 언제든.

type FormValues = Omit<LogInput, 'recurrence'> & {
  recurrence: LogInput['recurrence'] | ''
}

const hhmm = (v: string | undefined) => formatKst(v)?.slice(11) ?? ''

/** 저장된 기록으로 일지 초안 만들기. 기록에 없는 값은 비워 둔다 */
function draft(d: CaseDetail): FormValues {
  const e = d.event
  // 수동 모드로 해결했으면 직접 찾은 원인과 사람이 적은 점검을 쓴다
  const manual = d.records.results.filter((r) => r.outcome === 'resolved').at(-1)?.situation_id === MANUAL_CAUSE
  const cand = manual ? null : resolvedCandidate(d)
  const sid = manual ? MANUAL_CAUSE : cand?.situation_id
  const action = d.records.actions.filter((a) => !sid || a.situation_id === sid).at(-1)
  const occurred = formatKst(e.trigger.rule_trigger_time ?? e.detected_at)
  const withMemo = (title: string, memo?: string) => `${title} 이상${memo ? ` (${memo})` : ''}`
  const abnormal = manual
    ? (d.records.manual?.checks ?? []).filter((c) => c.result === 'abnormal').map((c) => withMemo(c.title, c.memo))
    : (cand?.checks ?? []).filter((c) => d.records.checks[c.id]?.result === 'abnormal').map((c) => withMemo(c.title, d.records.checks[c.id]?.memo))
  const causeName = resolvedCauseName(d)
  const missed = (cand?.checks ?? []).filter((c) => {
    const r = d.records.checks[c.id]?.result
    return !r || r === 'skipped'
  })
  const parts = action?.parts.map((p) => `${p.name} ${p.qty}개`).join(', ')

  return {
    date: formatKst(action?.started_at)?.slice(0, 10) ?? '',
    work_time: action ? `${hhmm(action.started_at)} ~ ${hhmm(action.ended_at)}` : '',
    worker_gun: [action?.worker, e.gun_id].filter(Boolean).join(' · '),
    problem: [`${e.trigger.rule_code ?? '모델 이상 신호'} 발생 (데이터 시각 ${occurred ?? '—'}).`, e.summary_ko && `ML 요약: ${e.summary_ko}`]
      .filter(Boolean)
      .join(' '),
    cause: causeName ? [causeName, abnormal.join(', ')].filter(Boolean).join(' — ') : '',
    action: action ? [`${action.kind} — ${action.did}`, parts && `교체 부품: ${parts}`].filter(Boolean).join('\n') : '',
    missed_checks: missed.map((c) => c.title).join(' · '),
    recurrence: '',
  }
}

export default function LogPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const saveLog = useCaseMutation<LogInput & { approved: boolean }>('PUT', '/log')
  const close = useCaseMutation<{ outcome: Closure }>('POST', '/close')
  const saved = d.records.log
  const { register, handleSubmit, getValues, formState } = useForm<FormValues>({
    defaultValues: saved ?? draft(d),
  })
  const { errors } = formState

  const closed = d.status === 'resolved'
  const approved = Boolean(saved?.approved_at)
  const locked = approved || closed
  const cand = resolvedCandidate(d)
  const checklist = [
    {
      // 매뉴얼 후보를 "맞아요"로 판단했거나, 수동 모드에서 원인을 직접 적었으면 확인됨
      ok: Boolean((cand && latestJudgment(d, cand.situation_id)?.verdict === 'yes') || d.records.manual?.cause.trim()),
      text: '원인 확인됨',
    },
    { ok: d.records.actions.length > 0, text: '조치 기록됨' },
    {
      ok: d.records.results.at(-1)?.outcome === 'resolved',
      text: '결과 · 해결됐어요',
    },
    { ok: approved, text: approved ? '일지 승인됨' : '일지 승인 전' },
  ]
  const canClose = checklist.every((c) => c.ok)
  const busy = saveLog.isPending || close.isPending

  const approve = handleSubmit((v) =>
    saveLog.mutate({
      ...v,
      recurrence: v.recurrence as LogInput['recurrence'],
      approved: true,
    }),
  )
  const unapprove = () => saveLog.mutate({ ...(saved as LogInput), approved: false })
  const saveUnresolved = () => {
    // 미해결은 칸이 덜 채워져도 저장할 수 있다 (부품 대기 등)
    const v = getValues()
    const body = {
      ...v,
      recurrence: (v.recurrence || '모름') as LogInput['recurrence'],
      approved,
    }
    saveLog.mutate(body, {
      onSuccess: () => close.mutate({ outcome: 'unresolved' }, { onSuccess: () => navigate('/') }),
    })
  }
  const resolve = () => close.mutate({ outcome: 'resolved' }, { onSuccess: () => navigate(`/history?case=${d.case_id}`) })

  const field = 'flex min-w-0 flex-col gap-1.5'
  const ro = locked ? 'bg-canvas text-[#4e5968]' : ''
  const err = (show: unknown, text: string) => (show ? <span className="text-[13px] text-alarm">{text}</span> : null)

  return (
    <TwoColumns
      side={
        <div className="flex flex-col gap-4">
          <section className={`${card} flex flex-col gap-3 p-5`} aria-label="종료 전 확인">
            <span className={label}>종료 전 확인</span>
            {checklist.map((c) => (
              <div key={c.text} className="flex items-center gap-3">
                <span className={`inline-flex w-[18px] justify-center font-semibold ${c.ok ? 'text-[#12805c]' : 'text-faint'}`} aria-hidden="true">
                  {c.ok ? '✓' : '–'}
                </span>
                <span className={c.ok ? '' : 'text-sub'}>
                  {c.text}
                  <span className="sr-only">{c.ok ? ' (완료)' : ' (아직)'}</span>
                </span>
              </div>
            ))}
          </section>
          <SidePanel d={d} initial="기록" />
        </div>
      }
    >
      <form onSubmit={approve} className={`${card} flex flex-col gap-3.5 p-[26px]`} noValidate>
        <div className="flex items-center gap-3">
          <h2 className="grow text-[19px] font-semibold tracking-[-0.015em]">정비일지</h2>
          {closed ? (
            <DotTag color="green">해결 종료 · 읽기 전용</DotTag>
          ) : (
            !saved && <span className="text-[13.5px] text-sub">앞 단계 기록으로 채웠습니다. 고칠 곳만 고치세요.</span>
          )}
        </div>
        <fieldset disabled={locked} className="flex flex-col gap-3.5">
          <div className="grid grid-cols-3 gap-3">
            <label className={field}>
              <span className={label}>날짜 (작업)</span>
              <input className={`${input} ${ro}`} placeholder="YYYY-MM-DD" {...register('date', { required: true })} />
              {err(errors.date, '날짜를 적어 주세요')}
            </label>
            <label className={field}>
              <span className={label}>작업 시간</span>
              <input className={`${input} ${ro}`} placeholder="00:00 ~ 00:00" {...register('work_time')} />
            </label>
            <label className={field}>
              <span className={label}>작업자 · 설비</span>
              <input className={`${input} ${ro}`} {...register('worker_gun', { required: true })} />
              {err(errors.worker_gun, '작업자를 적어 주세요')}
            </label>
          </div>
          <label className={field}>
            <span className={label}>문제</span>
            <textarea rows={2} className={`${input} ${ro} resize-y`} {...register('problem', { required: true })} />
            {err(errors.problem, '문제를 적어 주세요')}
          </label>
          <label className={field}>
            <span className={label}>원인</span>
            <input className={`${input} ${ro}`} placeholder="확인한 원인" {...register('cause', { required: true })} />
            {err(errors.cause, '원인을 적어 주세요')}
          </label>
          <label className={field}>
            <span className={label}>조치</span>
            <textarea rows={3} className={`${input} ${ro} resize-y`} {...register('action', { required: true })} />
            {err(errors.action, '조치를 적어 주세요')}
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className={field}>
              <span className={label}>못 한 점검</span>
              <input className={`${input} ${ro}`} placeholder="없음" {...register('missed_checks')} />
            </label>
            <label className={field}>
              <span className={label}>같은 고장 반복</span>
              <select className={`${input} ${ro}`} {...register('recurrence', { required: true })}>
                <option value="" disabled>
                  골라 주세요
                </option>
                {RECURRENCE.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
              {err(errors.recurrence, '반복 여부를 골라 주세요')}
            </label>
          </div>
        </fieldset>

        {!closed && (
          <div className="mt-2 flex items-center gap-3 border-t border-line-soft pt-4">
            <button type="button" className={btn} disabled={busy} onClick={saveUnresolved}>
              미해결로 저장
            </button>
            {(saveLog.isError || close.isError) && <span className="text-[13.5px] text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
            <div className="grow" />
            {approved ? (
              <>
                <DotTag color="green">일지 승인됨 {hhmm(saved?.approved_at ?? undefined)}</DotTag>
                <button type="button" className={btnQuiet} disabled={busy} onClick={unapprove}>
                  승인 취소하고 고치기
                </button>
                <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={busy || !canClose} onClick={resolve}>
                  해결 종료
                </button>
              </>
            ) : (
              <>
                <button type="submit" className={`${btnPrimary} ${btnLarge}`} disabled={busy}>
                  일지 승인
                </button>
                <button type="button" className={`${btn} ${btnLarge}`} disabled title="일지를 승인한 뒤에 종료할 수 있습니다">
                  해결 종료
                </button>
              </>
            )}
          </div>
        )}
      </form>
    </TwoColumns>
  )
}
