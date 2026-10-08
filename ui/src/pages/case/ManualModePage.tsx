import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useFieldArray, useForm, useWatch } from 'react-hook-form'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { SearchIcon } from '../../components/Icons'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { apiGet } from '../../lib/api'
import { useCaseMutation } from '../../lib/useCase'
import type { ManualCheck, ManualHit } from '../../types/case'
import { useCaseDetail } from './context'
import { TwoColumns } from './parts'

// 수동 모드 (wireframe/boards/V2NoGuide.dc.html): 맞는 매뉴얼 안내가 없을 때.
// 점검 순서를 지어내지 않고 사람이 확인한 것을 적는다. 오른쪽에서 매뉴얼 근거 쪽을 찾아 원본을 연다.
// 들어오는 길: 안내 없는 고장을 열 때 / 원인이 모두 "아니에요"일 때.

type Row = { title: string; result: ManualCheck['result'] | ''; memo: string }
interface FormValues {
  checks: Row[]
  cause: string
}

const EMPTY_ROW: Row = { title: '', result: '', memo: '' }

export default function ManualModePage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const [why, setWhy] = useState(false)
  const save = useCaseMutation<{ checks: ManualCheck[]; cause: string }>('PUT', '/manual')
  const saved = d.records.manual
  const { register, control, handleSubmit, setValue, getValues, formState } = useForm<FormValues>({
    defaultValues: {
      checks: saved?.checks.length ? saved.checks.map((c) => ({ ...c, memo: c.memo ?? '' })) : [EMPTY_ROW, EMPTY_ROW],
      cause: saved?.cause ?? '',
    },
  })
  const rows = useFieldArray({ control, name: 'checks' })
  const values = useWatch({ control, name: 'checks' })

  const openPage = Number(params.get('manual')) || null
  const setPage = (page: number | null) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (page) next.set('manual', String(page))
        else next.delete('manual')
        return next
      },
      { replace: true },
    )

  const body = (v: FormValues) => ({
    // 이름을 적고 결과를 고른 줄만 저장한다
    checks: v.checks
      .filter((c) => c.title.trim() && c.result)
      .map((c) => ({ title: c.title.trim(), result: c.result as ManualCheck['result'], memo: c.memo.trim() || undefined })),
    cause: v.cause.trim(),
  })
  const saveDraft = () => save.mutate(body(getValues()))
  const toAction = handleSubmit((v) => save.mutate(body(v), { onSuccess: () => navigate('../action') }))

  // "왜 없나요?": 실제로 안내가 없는 이유 (ML이 후보를 안 줬는지, 사람이 모두 아니라고 했는지)
  const reason =
    d.guidance.candidates.length === 0
      ? 'ML이 이 고장에 맞는 원인 후보(상황 ID)를 넘기지 않아서, 연결할 매뉴얼 점검 순서가 없습니다.'
      : '매뉴얼 안내가 있는 원인 후보를 모두 "아니에요"로 판단했습니다.'

  return (
    <>
      <div className="flex items-center gap-3 rounded-lg border border-[#f3d9a8] bg-[#fff6e6] px-[18px] py-3.5 text-[#6b3a00]">
        <b className="font-semibold">이 고장에 맞는 매뉴얼 안내를 찾지 못했습니다.</b>
        <span className="grow">직접 점검한 내용을 기록하거나 도움을 요청하세요.</span>
        <button type="button" className={btnQuiet} aria-expanded={why} onClick={() => setWhy((v) => !v)}>
          왜 없나요?
        </button>
      </div>
      {why && <p className="-mt-2 px-1 text-sm text-[#6b3a00]">{reason}</p>}

      <TwoColumns
        wide={openPage !== null}
        side={openPage ? <ManualPanel page={openPage} onPage={setPage} onClose={() => setPage(null)} /> : <ManualSearch openPage={openPage} onOpen={setPage} />}
      >
        <form onSubmit={toAction} className="flex flex-col gap-4" noValidate>
          <section className={`${card} flex flex-col gap-3.5 p-6`}>
            <h2 className="text-[17px] font-semibold tracking-[-0.015em]">점검 기록</h2>
            {rows.fields.map((f, i) => (
              <div key={f.id} className="flex flex-col gap-2">
                <div className="flex items-center gap-2.5">
                  <input className={input} placeholder="확인한 것" aria-label={`확인한 것 ${i + 1}`} {...register(`checks.${i}.title`)} />
                  <ResultSeg
                    label={`확인한 것 ${i + 1} 결과`}
                    options={['normal', 'abnormal']}
                    value={values[i]?.result || undefined}
                    onChange={(v) => setValue(`checks.${i}.result`, v)}
                  />
                  <button type="button" className={btn} aria-label={`${i + 1}번 줄 빼기`} onClick={() => rows.remove(i)}>
                    빼기
                  </button>
                </div>
                {values[i]?.result === 'abnormal' && (
                  <label className="ml-1 flex items-center gap-3">
                    <span className={label}>메모</span>
                    <input className={`${input} max-w-[420px]`} placeholder="확인한 내용 (예: 측정한 값)" {...register(`checks.${i}.memo`)} />
                  </label>
                )}
              </div>
            ))}
            <button type="button" className={`${btn} self-start`} onClick={() => rows.append(EMPTY_ROW)}>
              + 항목 추가
            </button>
          </section>

          <section className={`${card} flex flex-col gap-3 p-6`}>
            <h2 className="text-[17px] font-semibold tracking-[-0.015em]">원인을 찾았나요?</h2>
            <input className={input} placeholder="찾은 원인을 적어주세요" aria-label="찾은 원인" {...register('cause', { validate: (v) => v.trim() !== '' })} />
            {formState.errors.cause && <span className="text-[13px] text-alarm">조치를 기록하려면 찾은 원인을 적어 주세요. 못 찾았으면 도움을 요청하세요.</span>}
            <div className="mt-2 flex items-center gap-3">
              <button type="button" className={btn} disabled={save.isPending} onClick={saveDraft}>
                {save.isSuccess && !save.isPending ? '저장됨' : '임시 저장'}
              </button>
              {save.isError && <span className="text-[13.5px] text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
              <div className="grow" />
              <Link to="../handover" className={btn}>
                도움 요청
              </Link>
              <button type="submit" className={`${btnPrimary} ${btnLarge}`} disabled={save.isPending}>
                조치 기록으로
              </button>
            </div>
          </section>
        </form>
      </TwoColumns>
    </>
  )
}

/** 매뉴얼에서 직접 찾기: RAG 매핑의 근거 쪽(설명 · 인용 · 원인 이름)에서 찾는다. 매뉴얼 전문 검색은 아직 없다 */
function ManualSearch({ openPage, onOpen }: { openPage: number | null; onOpen: (page: number) => void }) {
  const [text, setText] = useState('')
  const [q, setQ] = useState('')
  const { data, isFetching, isError } = useQuery({
    queryKey: ['manual-search', q],
    queryFn: () => apiGet<{ items: ManualHit[] }>(`/manuals/festo/search?q=${encodeURIComponent(q)}`),
    enabled: q !== '',
  })
  const items = data?.items ?? []

  return (
    <aside className={`${card} flex flex-col gap-3 p-5`} aria-label="매뉴얼에서 직접 찾기">
      <span className={label}>매뉴얼에서 직접 찾기</span>
      <form
        className="relative"
        onSubmit={(e) => {
          e.preventDefault()
          setQ(text.trim())
        }}
      >
        <span className="pointer-events-none absolute top-[13px] left-3.5 text-sub">
          <SearchIcon size={18} />
        </span>
        <input type="search" className={`${input} pl-[42px]`} value={text} onChange={(e) => setText(e.target.value)} placeholder="예: 압력, 캡, MPYD" aria-label="매뉴얼 검색" />
      </form>
      <p className="text-[13px] text-sub">매뉴얼 전문이 아니라 RAG가 원인별로 연결해 둔 근거 쪽에서 찾습니다.</p>
      {isError && <p className="text-sm text-alarm">찾지 못했습니다. 다시 시도해 주세요.</p>}
      {q && !isFetching && !isError && items.length === 0 && <p className="text-sm text-sub">"{q}"에 맞는 근거 쪽이 없습니다.</p>}
      <ul className="flex flex-col gap-2">
        {items.map((i) => (
          <li key={`${i.page}-${i.situation_id}-${i.title}`} className="flex items-center gap-3 rounded-lg border border-line px-3.5 py-3">
            <div className="min-w-0 grow">
              <b className="font-semibold">{i.title}</b>
              <div className="truncate text-[13px] text-sub" title={i.quote ?? undefined}>
                {i.situation_name ?? i.situation_id} · Festo 매뉴얼
              </div>
            </div>
            <DocButton page={i.page} label={`${i.page}쪽`} on={openPage === i.page} onOpen={onOpen} />
          </li>
        ))}
      </ul>
      <div className="h-px bg-line-soft" />
      <button type="button" className={`${btnQuiet} self-start`} onClick={() => onOpen(1)}>
        매뉴얼 처음부터 보기
      </button>
    </aside>
  )
}
