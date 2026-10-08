import { useState } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { btn, btnLarge, btnPrimary, btnQuiet, card, input } from '../../components/ui'
import { checkCounts, manualAction, pickCandidate } from '../../lib/caseFlow'
import { useCaseMutation } from '../../lib/useCase'
import type { CheckRecord } from '../../types/case'
import { useCaseDetail } from './context'
import { SidePanel, Steps, TwoColumns } from './parts'

// 점검 (wireframe/boards/V2Check.dc.html)
// 항목마다 정상 / 이상 / 건너뜀. "이상"이면 메모 칸이 열린다 (측정값 · 단위 · 기준값 칸은 두지 않음, 10/8 결정)
// 매뉴얼 쪽 버튼을 누르면 오른쪽에 원본 PDF가 열린다 (V2Manual, 주소 ?manual=쪽).
// RAG가 점검 항목별 쪽수를 아직 주지 않아서, 쪽 버튼은 원인 단위 근거 쪽과 오류 조치 쪽에만 단다 (docs/06 요청 3).

export default function CheckPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const cand = pickCandidate(d, params.get('cause'))
  const [checks, setChecks] = useState<Record<string, CheckRecord>>(d.records.checks)
  const [openNote, setOpenNote] = useState(false)
  const save = useCaseMutation<{ checks: Record<string, CheckRecord> }>('PUT', '/checks')

  if (!cand) return <Navigate to="../no-cause" replace />
  const index = d.guidance.candidates.indexOf(cand) + 1
  const total = d.guidance.candidates.length
  const counts = checkCounts(cand, checks)
  const manual = manualAction(cand)
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
  // 근거 쪽: 같은 쪽이 여러 번 나오면 설명을 합친다
  const pages = new Map<number, string[]>()
  for (const e of cand.evidence) pages.set(e.page, [...(pages.get(e.page) ?? []), e.rationale ?? ''].filter(Boolean))

  const set = (id: string, patch: Partial<CheckRecord>) =>
    setChecks((prev) => ({ ...prev, [id]: { ...(prev[id] ?? { result: 'skipped' }), ...patch } }))

  const submit = (next: boolean) =>
    save.mutate({ checks }, { onSuccess: () => next && navigate(`../judge?cause=${cand.situation_id}`) })

  return (
    <>
      <Steps current="점검" />
      <TwoColumns wide={openPage !== null} side={openPage ? <ManualPanel page={openPage} onPage={setPage} onClose={() => setPage(null)} /> : <SidePanel d={d} initial="상황" />}>
        <section className={`${card} overflow-hidden`}>
          <div className="flex items-center gap-3 border-b border-[#eef0f4] px-[22px] py-4">
            <span className="text-[13px] font-medium text-sub">
              원인 {index} / {total}
            </span>
            <b className="text-base font-semibold">{cand.name}</b>
          </div>
          {cand.coverage_note && (
            <div className="border-b border-[#eef0f4] bg-[#f7f8fb] px-[22px] py-3 text-sm text-[#4a5468]">
              <div className="flex items-center gap-3">
                <span className="grow">매뉴얼 기준은 이 설비의 기준과 다를 수 있습니다</span>
                <button type="button" className={btnQuiet} aria-expanded={openNote} onClick={() => setOpenNote((v) => !v)}>
                  {openNote ? '접기' : '자세히'}
                </button>
              </div>
              {openNote && <p className="mt-2 text-[13.5px] leading-relaxed">{cand.coverage_note}</p>}
            </div>
          )}
          {pages.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 border-b border-[#eef0f4] px-[22px] py-3">
              <span className="mr-1 text-[13px] font-medium text-sub">매뉴얼 근거</span>
              {[...pages].map(([page, why]) => (
                <DocButton key={page} page={page} label={`${page}쪽`} title={why.join(' · ')} on={openPage === page} onOpen={setPage} />
              ))}
            </div>
          )}
          <ol>
            {cand.checks.map((item, i) => {
              const rec = checks[item.id]
              return (
                <li key={item.id} className="flex flex-col gap-2.5 border-t border-line-soft px-5 py-3.5 first:border-t-0">
                  <div className="flex items-center gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-[#f2f4f6] text-[12.5px] font-semibold text-sub">
                      {i + 1}
                    </span>
                    <span className="min-w-0 grow text-base font-semibold tracking-[-0.015em]">{item.title}</span>
                    <ResultSeg label={item.title} value={rec?.result} onChange={(v) => set(item.id, { result: v })} />
                  </div>
                  {rec?.result === 'abnormal' && (
                    <>
                      <label className="ml-9 flex items-center gap-3">
                        <span className="text-[13px] font-medium text-sub">메모</span>
                        <input
                          className={`${input} max-w-[420px]`}
                          value={rec.memo ?? ''}
                          placeholder="확인한 내용 (예: 측정한 값)"
                          onChange={(e) => set(item.id, { memo: e.target.value })}
                        />
                      </label>
                      {manual && (
                        <div className="ml-9 flex items-center gap-3 rounded border-l-[3px] border-[#f79009] bg-[#fffaeb] px-3 py-2 text-sm text-[#4e5968]">
                          <span className="grow">
                            <b className="font-semibold text-[#93570a]">이 원인의 매뉴얼 조치</b> · Festo 오류 {cand.diagnostics.find((x) => x.manual)?.number}: {manual.text}
                          </span>
                          <DocButton page={manual.page} label={`${manual.page}쪽`} on={openPage === manual.page} onOpen={setPage} />
                        </div>
                      )}
                    </>
                  )}
                </li>
              )
            })}
          </ol>
        </section>
        <div className="flex items-center gap-3">
          <span className="text-[13.5px] text-sub">
            {counts.total}개 중 {counts.done}개 확인
          </span>
          {save.isError && <span className="text-[13.5px] text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
          <div className="grow" />
          <button type="button" className={btn} disabled={save.isPending} onClick={() => submit(false)}>
            {save.isSuccess && !save.isPending ? '저장됨' : '임시 저장'}
          </button>
          <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={save.isPending || counts.done === 0} onClick={() => submit(true)}>
            다음
          </button>
        </div>
      </TwoColumns>
    </>
  )
}
