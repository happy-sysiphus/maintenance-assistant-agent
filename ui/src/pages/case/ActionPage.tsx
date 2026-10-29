import { useFieldArray, useForm, useWatch } from 'react-hook-form'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router'
import { ManualPanel } from '../../components/ManualPanel'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { docFromParam, MANUAL_SHORT, type ManualDoc } from '../../lib/api'
import { activeCause, latestJudgment, manualAction } from '../../lib/caseFlow'
import { useCaseMutation } from '../../lib/useCase'
import { ACTION_KINDS, type ActionInput, type ActionKind } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, ManualActionPanel, Steps, TwoColumns } from './parts'

// 조치 (wireframe/boards/V2Action.dc.html): 사람이 실제로 한 일만 적는다. 정비일지에는 여기 적은 내용만 들어간다.
// 시작 · 종료는 작업 시각(지금 시계)이고, 고장 시각(데이터 시각)과 다르다.
// 원인은 지금 원인 후보, 없으면 수동 모드에서 직접 찾은 원인이다.
// (10/23) 맨 위는 "확정 원인" 한 줄, 오른쪽은 매뉴얼 조치(참고) 패널. 원문 보기를 누르면 그 자리에 PDF가 열린다.

interface FormValues {
  kind: ActionKind | ''
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
  const cause = activeCause(d)
  const cand = cause?.candidate ?? null
  const [params, setParams] = useSearchParams()
  const openPage = Number(params.get('manual')) || null
  const openDoc = docFromParam(params.get('doc'))
  const setPage = (page: number | null, doc: ManualDoc = openDoc) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (page) next.set('manual', String(page))
        else next.delete('manual')
        if (page) next.set('doc', doc)
        else next.delete('doc')
        return next
      },
      { replace: true },
    )
  const { register, control, handleSubmit, setValue, formState } = useForm<FormValues>({
    defaultValues: { kind: '', did: '', parts: [], worker: '', started_at: localNow(), ended_at: localNow() },
  })
  const parts = useFieldArray({ control, name: 'parts' })
  const kind = useWatch({ control, name: 'kind' })
  if (!cause) return <Navigate to="../no-cause" replace />

  const judgment = cand ? latestJudgment(d, cand.situation_id) : undefined
  const phenomenon = d.event.fault_class?.name_ko
  const manual = cand ? manualAction(cand) : null
  const { errors } = formState

  const onSubmit = handleSubmit((v) =>
    save.mutate(
      {
        situation_id: cause.situation_id,
        kind: v.kind as ActionKind,
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
      <BackLink to={cand ? `../judge?cause=${cand.situation_id}` : '../manual'} />
      <Steps current="조치" />
      <TwoColumns
        wide={openPage !== null}
        side={
          openPage ? (
            <ManualPanel page={openPage} doc={openDoc} onPage={(p) => setPage(p)} onClose={() => setPage(null)} />
          ) : (
            <ManualActionPanel cand={cand} onOpen={setPage} />
          )
        }
      >
        <form onSubmit={onSubmit} className={`${card} flex flex-col gap-5 p-[26px]`} noValidate>
          <div className="flex items-center gap-3">
            <h2 className="grow text-[22px] leading-snug font-medium tracking-[-0.015em]">어떤 조치를 했나요?</h2>
            {manual && (
              <button
                type="button"
                className={btn}
                onClick={() =>
                  setValue('did', `${manual.text} (${MANUAL_SHORT[manual.doc]}${manual.page ? ` ${manual.page}쪽` : ''}, 초안 — 실제로 한 일로 고치세요)`)
                }
              >
                매뉴얼 조치 불러오기
              </button>
            )}
          </div>

          <div className="flex items-center gap-3 rounded-lg bg-canvas px-3.5 py-2.5 text-sm">
            <span className="text-xs font-medium text-sub">{cand && judgment?.verdict !== 'yes' ? '원인 (판단 전)' : '확정 원인'}</span>
            <b className="font-semibold">{cand ? cand.name : cause.name}</b>
            {!cand && <span className="text-sub">· 직접 찾음</span>}
            {phenomenon && phenomenon !== cause.name && <span className="text-sub">· 현상 {phenomenon}</span>}
            <div className="grow" />
            <Link to={cand ? '../judge' : '../manual'} className={btnQuiet}>
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
                    kind === k ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-ink-2'
                  }`}
                >
                  <input type="radio" value={k} className="sr-only" {...register('kind', { required: true })} />
                  {k}
                </label>
              ))}
            </div>
            {errors.kind && <span className="text-xs text-alarm">조치 종류를 골라 주세요</span>}
          </fieldset>

          <label className="flex flex-col gap-1.5">
            <span className={label}>한 일 *</span>
            <textarea rows={3} className={`${input} resize-y`} placeholder="실제로 한 작업을 적어 주세요" {...register('did', { required: true })} />
            {errors.did && <span className="text-xs text-alarm">한 일을 적어 주세요</span>}
          </label>

          <div className="flex flex-col gap-1.5">
            <span className={label}>교체 부품 (선택)</span>
            {parts.fields.map((f, i) => (
              <div key={f.id} className="flex gap-2">
                <input className={input} placeholder="부품명" aria-label={`부품 ${i + 1} 이름`} {...register(`parts.${i}.name`)} />
                <input
                  type="number"
                  min={1}
                  className={`${input} w-24`}
                  aria-label={`부품 ${i + 1} 수량`}
                  {...register(`parts.${i}.qty`, { valueAsNumber: true })}
                />
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
              {errors.worker && <span className="text-xs text-alarm">작업자를 적어 주세요</span>}
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
            {save.isError && <span className="text-sm text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
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
