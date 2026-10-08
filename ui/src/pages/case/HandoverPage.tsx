import { useState } from 'react'
import { Controller, useForm } from 'react-hook-form'
import { useNavigate } from 'react-router'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { candidateState, checkCounts, latestJudgment } from '../../lib/caseFlow'
import { formatKst } from '../../lib/time'
import { useCaseMutation } from '../../lib/useCase'
import { HANDOVER_REASONS, URGENCY, type CaseDetail, type HandoverInput } from '../../types/case'
import { useCaseDetail } from './context'
import { RecordTab } from './parts'

// 도움 요청 · 인계 (wireframe/boards/V2Handover.dc.html): 받는 사람 · 급한 정도 · 이유 · 내용만 적는다.
// 지금까지의 점검 · 판단 · 조치 기록은 케이스에 이미 있어서 자동으로 함께 전달된다.

const RECEIVERS = ['설비 전문가', '공정 담당']
const OUTCOME = { resolved: '해결됨', retry: '해결 안 됨', next: '해결 안 됨' }

const chip = (on: boolean) =>
  `inline-flex min-h-10 items-center rounded-lg border px-3.5 text-sm whitespace-nowrap ${
    on ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-[#4e5968]'
  }`

/** 함께 전달되는 내용: 저장된 기록에서 계산한다 */
function summary(d: CaseDetail) {
  const e = d.event
  const yes = d.guidance.candidates.filter((c) => latestJudgment(d, c.situation_id)?.verdict === 'yes').map((c) => c.name)
  if (d.records.manual?.cause) yes.push(`${d.records.manual.cause} (직접 찾음)`)
  const counts = { abnormal: 0, normal: 0, skipped: 0 }
  for (const c of d.guidance.candidates) {
    const k = checkCounts(c, d.records.checks)
    counts.abnormal += k.abnormal
    counts.normal += k.normal
    counts.skipped += k.skipped
  }
  for (const c of d.records.manual?.checks ?? []) counts[c.result] += 1
  const action = d.records.actions.at(-1)
  const result = d.records.results.at(-1)
  const remaining = d.guidance.candidates.filter((c) => candidateState(d, c.situation_id) !== 'excluded').map((c) => c.name)
  return [
    { k: '고장', v: `${e.gun_id} · ${e.trigger.rule_code ?? '코드 없음'} · ${formatKst(e.trigger.rule_trigger_time ?? e.detected_at)}` },
    { k: '확인한 원인', v: yes.join(', ') || '—' },
    { k: '점검', v: `이상 ${counts.abnormal} · 정상 ${counts.normal} · 건너뜀 ${counts.skipped}` },
    { k: '조치', v: action ? `${action.kind}${result ? ` · ${OUTCOME[result.outcome]}` : ''}` : '—' },
    { k: '남은 원인', v: remaining.join(', ') || '—' },
  ]
}

interface FormValues extends Omit<HandoverInput, 'to'> {
  to: string
}

export default function HandoverPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const [all, setAll] = useState(false)
  const send = useCaseMutation<HandoverInput>('POST', '/handover')
  const { register, control, handleSubmit, formState } = useForm<FormValues>({
    defaultValues: { to: '', urgency: '보통', reasons: [], note: '' },
  })
  const { errors } = formState
  const onSubmit = handleSubmit((v) => send.mutate(v, { onSuccess: () => navigate('/') }))

  return (
    <>
      <h2 className="text-[19px] font-semibold tracking-[-0.015em]">도움 요청</h2>
      <div className="grid grow grid-cols-[minmax(0,1fr)_380px] items-start gap-4">
        <form onSubmit={onSubmit} className={`${card} flex flex-col gap-[18px] p-[26px]`} noValidate>
          <div className="grid grid-cols-2 gap-3.5">
            <label className="flex flex-col gap-1.5">
              <span className={label}>받는 사람 *</span>
              <select className={input} {...register('to', { required: true })}>
                <option value="" disabled>
                  선택
                </option>
                {RECEIVERS.map((r) => (
                  <option key={r}>{r}</option>
                ))}
              </select>
              {errors.to && <span className="text-[13px] text-alarm">받는 사람을 골라 주세요</span>}
            </label>
            <fieldset className="flex flex-col gap-1.5">
              <legend className={label}>급한 정도</legend>
              <Controller
                control={control}
                name="urgency"
                render={({ field }) => (
                  <div className="mt-1.5 flex gap-2">
                    {URGENCY.map((u) => (
                      <button key={u} type="button" aria-pressed={field.value === u} className={chip(field.value === u)} onClick={() => field.onChange(u)}>
                        {u}
                      </button>
                    ))}
                  </div>
                )}
              />
            </fieldset>
          </div>
          <fieldset className="flex flex-col gap-1.5">
            <legend className={label}>요청 이유 *</legend>
            <Controller
              control={control}
              name="reasons"
              rules={{ validate: (v) => v.length > 0 }}
              render={({ field }) => (
                <div className="mt-1.5 flex flex-wrap gap-2">
                  {HANDOVER_REASONS.map((r) => {
                    const on = field.value.includes(r)
                    return (
                      <button
                        key={r}
                        type="button"
                        aria-pressed={on}
                        className={chip(on)}
                        onClick={() => field.onChange(on ? field.value.filter((x) => x !== r) : [...field.value, r])}
                      >
                        {r}
                      </button>
                    )
                  })}
                </div>
              )}
            />
            {errors.reasons && <span className="text-[13px] text-alarm">이유를 하나 이상 골라 주세요</span>}
          </fieldset>
          <label className="flex flex-col gap-1.5">
            <span className={label}>요청 내용 *</span>
            <textarea rows={5} className={`${input} resize-y`} placeholder="무엇을 도와주면 되는지 적어주세요" {...register('note', { required: true })} />
            {errors.note && <span className="text-[13px] text-alarm">요청 내용을 적어 주세요</span>}
          </label>
          <div className="mt-2 flex items-center gap-3">
            <button type="button" className={btn} onClick={() => navigate(-1)}>
              취소
            </button>
            {send.isError && <span className="text-[13.5px] text-alarm">보내지 못했습니다. 입력 내용은 남아 있습니다.</span>}
            <div className="grow" />
            <button type="submit" className={`${btnPrimary} ${btnLarge}`} disabled={send.isPending}>
              보내기
            </button>
          </div>
        </form>

        <aside className={`${card} flex flex-col gap-3 p-5`} aria-label="함께 전달되는 내용">
          <span className={label}>함께 전달되는 내용</span>
          <dl className="grid grid-cols-[88px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-[9px] text-[14.5px]">
            {summary(d).map((r) => (
              <div key={r.k} className="contents">
                <dt className="text-[13.5px] text-faint">{r.k}</dt>
                <dd>{r.v}</dd>
              </div>
            ))}
          </dl>
          <div className="h-px bg-line-soft" />
          <button type="button" className={`${btnQuiet} self-start`} aria-expanded={all} onClick={() => setAll((v) => !v)}>
            {all ? '기록 접기' : '기록 전체 보기'}
          </button>
          {all && <RecordTab d={d} />}
        </aside>
      </div>
    </>
  )
}
