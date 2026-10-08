import { useFieldArray, useForm, useWatch } from 'react-hook-form'
import { Link, Navigate, useNavigate } from 'react-router'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { currentCandidate, latestJudgment, manualAction } from '../../lib/caseFlow'
import { useCaseMutation } from '../../lib/useCase'
import { ACTION_KINDS, type ActionInput, type ActionKind } from '../../types/case'
import { useCaseDetail } from './context'
import { SidePanel, Steps, TwoColumns } from './parts'

// 조치 (wireframe/boards/V2Action.dc.html): 사람이 실제로 한 일만 적는다. 정비일지에는 여기 적은 내용만 들어간다.
// 시작 · 종료는 작업 시각(지금 시계)이고, 고장 시각(데이터 시각)과 다르다.

interface FormValues {
  kind: ActionKind | ''
  reason: string
  did: string
  parts: { name: string; qty: number }[]
  worker: string
  started_at: string
  ended_at: string
}

/** datetime-local 입력용 "YYYY-MM-DDTHH:mm" (이 컴퓨터 시간대) */
function localNow() {
  const d = new Date()
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset())
  return d.toISOString().slice(0, 16)
}

export default function ActionPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const save = useCaseMutation<ActionInput>('POST', '/actions')
  const cand = currentCandidate(d)
  const { register, control, handleSubmit, setValue, formState } = useForm<FormValues>({
    defaultValues: { kind: '', reason: '', did: '', parts: [], worker: '', started_at: localNow(), ended_at: localNow() },
  })
  const parts = useFieldArray({ control, name: 'parts' })
  const kind = useWatch({ control, name: 'kind' })
  if (!cand) return <Navigate to="../no-cause" replace />

  const judgment = latestJudgment(d, cand.situation_id)
  const manual = manualAction(cand)
  const { errors } = formState

  const onSubmit = handleSubmit((v) =>
    save.mutate(
      {
        situation_id: cand.situation_id,
        kind: v.kind as ActionKind,
        reason: v.reason || undefined,
        did: v.did,
        parts: v.parts.filter((p) => p.name.trim()),
        worker: v.worker,
        started_at: new Date(v.started_at).toISOString(),
        ended_at: new Date(v.ended_at).toISOString(),
      },
      { onSuccess: () => navigate('../result') },
    ),
  )

  return (
    <>
      <Steps current="조치" />
      <TwoColumns side={<SidePanel d={d} initial="상황" />}>
        <form onSubmit={onSubmit} className={`${card} flex flex-col gap-5 p-[26px]`} noValidate>
          <div className="flex items-center gap-3">
            <h2 className="grow text-xl font-semibold tracking-[-0.015em]">어떤 조치를 했나요?</h2>
            {manual && (
              <button type="button" className={btn} onClick={() => setValue('did', `${manual.text} (Festo ${manual.page}쪽, 초안 — 실제로 한 일로 고치세요)`)}>
                매뉴얼 조치 불러오기
              </button>
            )}
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-[#f7f8fa] px-3.5 py-2.5 text-sm">
            <span className="text-[13px] font-medium text-sub">판단</span>
            <b className="font-semibold">{cand.name} · {judgment?.verdict === 'yes' ? '맞아요' : '판단 전'}</b>
            <input className={`${input} max-w-[320px] min-h-9 py-1.5 text-sm`} placeholder="이유 (선택)" {...register('reason')} />
            <div className="grow" />
            <Link to="../judge" className={btnQuiet}>
              바꾸기
            </Link>
          </div>

          <fieldset className="flex flex-col gap-1.5">
            <legend className={label}>조치 종류 *</legend>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {ACTION_KINDS.map((k) => (
                <label
                  key={k}
                  className={`inline-flex min-h-10 cursor-pointer items-center rounded-lg border px-3.5 text-sm ${
                    kind === k ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-[#4e5968]'
                  }`}
                >
                  <input type="radio" value={k} className="sr-only" {...register('kind', { required: true })} />
                  {k}
                </label>
              ))}
            </div>
            {errors.kind && <span className="text-[13px] text-alarm">조치 종류를 골라 주세요</span>}
          </fieldset>

          <label className="flex flex-col gap-1.5">
            <span className={label}>한 일 *</span>
            <textarea rows={3} className={`${input} resize-y`} placeholder="실제로 한 작업을 적어 주세요" {...register('did', { required: true })} />
            {errors.did && <span className="text-[13px] text-alarm">한 일을 적어 주세요</span>}
          </label>

          <div className="flex flex-col gap-1.5">
            <span className={label}>교체 부품 (선택)</span>
            {parts.fields.map((f, i) => (
              <div key={f.id} className="flex gap-2">
                <input className={input} placeholder="부품명" aria-label={`부품 ${i + 1} 이름`} {...register(`parts.${i}.name`)} />
                <input type="number" min={1} className={`${input} w-24`} aria-label={`부품 ${i + 1} 수량`} {...register(`parts.${i}.qty`, { valueAsNumber: true })} />
                <button type="button" className={btn} onClick={() => parts.remove(i)}>
                  빼기
                </button>
              </div>
            ))}
            <button type="button" className={`${btnQuiet} self-start`} onClick={() => parts.append({ name: '', qty: 1 })}>
              + 부품 추가
            </button>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <label className="flex flex-col gap-1.5">
              <span className={label}>작업자 *</span>
              <input className={input} placeholder="이름" {...register('worker', { required: true })} />
              {errors.worker && <span className="text-[13px] text-alarm">작업자를 적어 주세요</span>}
            </label>
            <label className="flex flex-col gap-1.5">
              <span className={label}>시작 (작업 시각)</span>
              <input type="datetime-local" className={input} {...register('started_at', { required: true })} />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className={label}>종료 (작업 시각)</span>
              <input type="datetime-local" className={input} {...register('ended_at', { required: true })} />
            </label>
          </div>

          <div className="flex items-center gap-3 border-t border-line-soft pt-4">
            <span className="rounded-full bg-[#eef3ff] px-2.5 py-1 text-[13px] font-medium text-primary-ink">정비일지에는 여기 적은 내용이 들어갑니다</span>
            {save.isError && <span className="text-[13.5px] text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
            <div className="grow" />
            <button type="submit" className={`${btnPrimary} ${btnLarge}`} disabled={save.isPending}>
              저장
            </button>
          </div>
        </form>
      </TwoColumns>
    </>
  )
}
