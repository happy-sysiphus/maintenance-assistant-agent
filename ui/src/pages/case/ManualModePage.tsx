import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useFieldArray, useForm, useWatch } from 'react-hook-form'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { SearchIcon } from '../../components/Icons'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input, label } from '../../components/ui'
import { apiGet, docFromParam, MANUAL_NAME, type ManualDoc } from '../../lib/api'
import { useCaseMutation } from '../../lib/useCase'
import type { ManualCheck, ManualHit } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, FalseAlarmButton, SideLabel, sideCard, TwoColumns } from './parts'

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

  const body = (v: FormValues) => ({
    // 이름을 적고 결과를 고른 줄만 저장한다
    checks: v.checks
      .filter((c) => c.title.trim() && c.result)
      .map((c) => ({ title: c.title.trim(), result: c.result as ManualCheck['result'], memo: c.memo.trim() || undefined })),
    cause: v.cause.trim(),
  })
  const saveDraft = () => save.mutate(body(getValues()))
  const toAction = handleSubmit((v) => save.mutate(body(v), { onSuccess: () => navigate('../action') }))

  return (
    <>
      <BackLink />
      <div className="flex items-center gap-3 rounded-lg border border-warn-dot/30 bg-warn-soft px-[18px] py-3.5 text-warn">
        {/* 안내가 아예 없을 때와, 안내의 원인을 모두 "아니에요"로 넘겼을 때를 구분한다 */}
        <b className="font-semibold">
          {d.guidance.candidates.length ? '매뉴얼 안내의 원인이 모두 맞지 않았습니다.' : '이 고장에 맞는 매뉴얼 안내를 찾지 못했습니다.'}
        </b>
        <span className="grow">직접 점검한 내용을 기록하거나 도움을 요청하세요.</span>
      </div>

      <TwoColumns
        wide={openPage !== null}
        side={
          openPage ? (
            <ManualPanel page={openPage} doc={openDoc} onPage={(p) => setPage(p)} onClose={() => setPage(null)} />
          ) : (
            <ManualSearch onOpen={setPage} />
          )
        }
      >
        <form onSubmit={toAction} className="flex flex-col gap-4" noValidate>
          <section className={`${card} flex flex-col gap-3.5 p-6`}>
            <h2 className="text-lg font-semibold tracking-[-0.015em]">점검 기록</h2>
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
            <h2 className="text-lg font-semibold tracking-[-0.015em]">원인을 찾았나요?</h2>
            <input
              className={input}
              placeholder="찾은 원인을 적어주세요"
              aria-label="찾은 원인"
              {...register('cause', { validate: (v) => v.trim() !== '' })}
            />
            {formState.errors.cause && (
              <span className="text-xs text-alarm">조치를 기록하려면 찾은 원인을 적어 주세요. 못 찾았으면 도움을 요청하세요.</span>
            )}
            <div className="mt-2 flex items-center gap-3">
              <button type="button" className={btn} disabled={save.isPending} onClick={saveDraft}>
                {save.isSuccess && !save.isPending ? '저장됨' : '임시 저장'}
              </button>
              {save.isError && <span className="text-sm text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
              <div className="grow" />
              <FalseAlarmButton />
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

/** 매뉴얼에서 직접 찾기: api가 문서 4종에서 키워드로 찾는다 (10/28). 검색 중에는 PDF가 닫혀 있다 */
function ManualSearch({ onOpen }: { onOpen: (page: number, doc: ManualDoc) => void }) {
  const [text, setText] = useState('')
  const [q, setQ] = useState('')
  const { data, isFetching, isError } = useQuery({
    queryKey: ['manual-search', q],
    queryFn: () => apiGet<{ items: ManualHit[] }>(`/manuals/festo/search?q=${encodeURIComponent(q)}`),
    enabled: q !== '',
  })
  // 같은 문서의 같은 쪽은 한 줄만 (api 검색이 한 쪽의 여러 조각을 따로 준다)
  const items = (data?.items ?? []).filter((i, n, all) => all.findIndex((x) => (x.doc ?? 'festo') === (i.doc ?? 'festo') && x.page === i.page) === n)

  return (
    <aside className={sideCard} aria-label="매뉴얼에서 직접 찾기">
      <SideLabel>매뉴얼에서 직접 찾기</SideLabel>
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
        <input
          type="search"
          className={`${input} pl-[42px]`}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="예: 압력, 캡, MPYD"
          aria-label="매뉴얼 검색"
        />
      </form>
      {isError && <p className="text-sm text-alarm">찾지 못했습니다. 다시 시도해 주세요.</p>}
      {q && !isFetching && !isError && items.length === 0 && <p className="text-sm text-sub">"{q}"에 맞는 근거 쪽이 없습니다.</p>}
      <ul className="flex flex-col gap-2">
        {items.map((i) => {
          const doc = i.doc ?? 'festo'
          return (
            <li key={`${doc}-${i.page}`} className="flex items-center gap-3 rounded-lg border border-line px-3.5 py-3">
              {i.doc ? (
                // 10/28 api: title은 문서의 영어 제목이라 문서 이름표 + 찾은 원문 한 줄을 보여준다
                <div className="min-w-0 grow">
                  <b className="font-semibold">{MANUAL_NAME[doc]}</b>
                  {i.quote && (
                    <p className="line-clamp-2 text-xs text-sub" title={i.quote}>
                      {i.quote}
                    </p>
                  )}
                </div>
              ) : (
                <div className="min-w-0 grow">
                  <b className="font-semibold">{i.title}</b>
                  <div className="truncate text-xs text-sub" title={i.quote ?? undefined}>
                    {i.situation_name ?? i.situation_id} · {MANUAL_NAME[doc]}
                  </div>
                </div>
              )}
              <DocButton page={i.page} label={`${i.page}쪽`} on={false} onOpen={(p) => onOpen(p, doc)} />
            </li>
          )
        })}
      </ul>
      <div className="h-px bg-line-soft" />
      <button type="button" className={`${btnQuiet} self-start`} onClick={() => onOpen(1, 'manual82')}>
        매뉴얼 처음부터 보기
      </button>
    </aside>
  )
}
