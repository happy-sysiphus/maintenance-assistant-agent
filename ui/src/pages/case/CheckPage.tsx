import { useState } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { CameraIcon } from '../../components/Icons'
import { btn, btnLarge, btnPrimary, card, input } from '../../components/ui'
import { checkCounts, manualAction, pickCandidate } from '../../lib/caseFlow'
import { addPhotos, MAX_PHOTOS } from '../../lib/photo'
import { useCaseMutation } from '../../lib/useCase'
import type { CheckRecord } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, StepPanel, Steps, TwoColumns } from './parts'

// 점검 (wireframe/boards/V2Check.dc.html)
// 항목마다 정상 / 이상 / 건너뜀. "이상"이면 메모 칸이 열린다 (측정값 · 단위 · 기준값 칸은 두지 않음, 10/8 결정)
// 매뉴얼 쪽 버튼을 누르면 오른쪽에 원본 PDF가 열린다 (V2Manual, 주소 ?manual=쪽).
// RAG가 점검 항목별 쪽수를 아직 주지 않아서, 쪽 버튼은 원인 단위 근거 쪽과 오류 조치 쪽에만 단다 (docs/06 요청 3).
// (10/23) 근거 버튼은 쪽수만 (설명은 마우스를 올리면). "이상"을 고르면 이상일 때 줄에 원인 · 조치 쪽 버튼을 보여준다.
// 오른쪽은 탭 대신 이 단계에 필요한 것만 (정의 · 주의).
// (10/23) 현장 확인 입력을 여기로 합쳤다: 항목마다 사진(최대 4장)을 붙인다. 인쇄는 정비일지에만 둔다.

export default function CheckPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const cand = pickCandidate(d, params.get('cause'))
  const [checks, setChecks] = useState<Record<string, CheckRecord>>(d.records.checks)
  const [picked, setPicked] = useState<string | null>(null)
  const save = useCaseMutation<{ checks: Record<string, CheckRecord> }>('PUT', '/checks')

  if (!cand) return <Navigate to="../no-cause" replace />
  const index = d.guidance.candidates.indexOf(cand) + 1
  const total = d.guidance.candidates.length
  const counts = checkCounts(cand, checks)
  const manual = manualAction(cand)
  const openPage = Number(params.get('manual')) || null
  // 어느 문서를 열지 (기본 Festo 매뉴얼, 항목 기준이 상황 정의서에 있으면 rsw)
  const openDoc = params.get('doc') === 'rsw' ? 'rsw' : 'festo'
  const setPage = (page: number | null, doc: 'festo' | 'rsw' = 'festo') =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (page) next.set('manual', String(page))
        else next.delete('manual')
        if (page && doc === 'rsw') next.set('doc', 'rsw')
        else next.delete('doc')
        return next
      },
      { replace: true },
    )
  // 오른쪽 칸에 기준을 보여줄 항목: 사용자가 마지막으로 누른 항목, 없으면 아직 안 한 첫 항목
  const activeId = picked ?? cand.checks.find((c) => !checks[c.id])?.id ?? cand.checks[0]?.id
  const activeIndex = cand.checks.findIndex((c) => c.id === activeId)
  const active = activeIndex >= 0 ? cand.checks[activeIndex] : undefined
  // 근거 쪽: 같은 쪽이 여러 번 나오면 설명을 합친다
  const pages = new Map<number, string[]>()
  for (const e of cand.evidence) pages.set(e.page, [...(pages.get(e.page) ?? []), e.rationale ?? ''].filter(Boolean))

  const set = (id: string, patch: Partial<CheckRecord>) =>
    setChecks((prev) => ({ ...prev, [id]: { ...(prev[id] ?? { result: 'skipped' }), ...patch } }))

  const attach = async (id: string, files: File[]) => set(id, { photos: await addPhotos(checks[id]?.photos ?? [], files) })
  const detach = (id: string, i: number) => set(id, { photos: (checks[id]?.photos ?? []).filter((_, j) => j !== i) })

  const submit = (next: boolean) => save.mutate({ checks }, { onSuccess: () => next && navigate(`../judge?cause=${cand.situation_id}`) })

  return (
    <>
      <BackLink to=".." />
      <Steps current="점검" />
      <TwoColumns
        wide={openPage !== null}
        side={
          openPage ? (
            <ManualPanel page={openPage} doc={openDoc} onPage={(p) => setPage(p, openDoc)} onClose={() => setPage(null)} />
          ) : (
            <StepPanel
              d={d}
              cand={cand}
              item={active && { no: activeIndex + 1, title: active.title, criterion: active.criterion }}
              onOpenDoc={setPage}
            />
          )
        }
      >
        <section className={`${card} overflow-hidden`}>
          <div className="flex items-center gap-3 border-b border-line-soft px-[22px] py-4">
            <span className="text-xs font-medium text-sub">
              원인 {index} / {total}
            </span>
            <b className="text-base font-semibold">{cand.name}</b>
          </div>
          {pages.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 border-b border-line-soft px-[22px] py-3">
              <span className="mr-1 text-xs font-medium text-sub">매뉴얼 근거</span>
              {[...pages].map(([page, why]) => (
                <DocButton key={page} page={page} label={`${page}쪽`} title={why.join(' · ')} on={openPage === page} onOpen={(p) => setPage(p)} />
              ))}
            </div>
          )}
          <ol>
            {cand.checks.map((item, i) => {
              const rec = checks[item.id]
              return (
                // 항목을 누르면(결과 · 메모 · 사진 포함) 오른쪽 칸에 그 항목의 기준이 나온다
                <li
                  key={item.id}
                  onClickCapture={() => setPicked(item.id)}
                  onFocusCapture={() => setPicked(item.id)}
                  className={`flex flex-col gap-2.5 border-t border-l-[3px] border-t-line-soft px-5 py-3.5 transition-colors first:border-t-0 ${
                    item.id === activeId ? 'border-l-primary bg-primary-soft/40' : 'border-l-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-fill text-xs font-semibold text-sub">
                      {i + 1}
                    </span>
                    <span className="min-w-0 grow text-base font-semibold tracking-[-0.015em]">{item.title}</span>
                    {/* 결과를 고른 항목에만 사진을 붙인다 (사진만 붙여서 '건너뜀'으로 기록되지 않게) */}
                    {rec && (rec.photos?.length ?? 0) < MAX_PHOTOS && (
                      <label
                        className="inline-flex size-10 shrink-0 cursor-pointer items-center justify-center rounded-lg border border-field bg-white text-ink-2 transition-colors hover:bg-canvas print:hidden"
                        title="사진 붙이기"
                      >
                        <CameraIcon size={18} />
                        <span className="sr-only">{item.title} 사진 붙이기</span>
                        <input
                          type="file"
                          accept="image/*"
                          capture="environment"
                          multiple
                          className="sr-only"
                          onChange={(e) => {
                            const files = [...(e.target.files ?? [])]
                            e.target.value = ''
                            void attach(item.id, files)
                          }}
                        />
                      </label>
                    )}
                    <ResultSeg label={item.title} value={rec?.result} onChange={(v) => set(item.id, { result: v })} />
                  </div>
                  {rec?.photos && rec.photos.length > 0 && (
                    <div className="ml-9 flex flex-wrap gap-3">
                      {rec.photos.map((p, j) => (
                        <div key={j} className="relative">
                          <img src={p.url} alt={`${item.title} 사진 ${j + 1}`} className="size-16 rounded-lg border border-line object-cover" />
                          <button
                            type="button"
                            aria-label={`${item.title} 사진 ${j + 1} 빼기`}
                            onClick={() => detach(item.id, j)}
                            className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full border border-field bg-white text-xs text-ink-2 print:hidden"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                  {rec?.result === 'abnormal' && (
                    <>
                      <label className="ml-9 flex items-center gap-3">
                        <span className="text-xs font-medium text-sub">메모</span>
                        <input
                          className={`${input} max-w-[420px]`}
                          value={rec.memo ?? ''}
                          placeholder="확인한 내용 (예: 측정한 값)"
                          onChange={(e) => set(item.id, { memo: e.target.value })}
                        />
                      </label>
                      {manual && (
                        <div
                          className="ml-9 flex items-center gap-3 rounded border-l-[3px] border-hint-line bg-hint py-2 pr-2 pl-3 text-sm text-ink-2"
                          title={manual.text}
                        >
                          <b className="font-semibold text-warn">이상일 때</b>
                          <span className="grow">
                            매뉴얼 원인 · 조치 — Festo 오류 {cand.diagnostics.find((x) => x.manual)?.number} {manual.name}
                          </span>
                          {/* RAG 검토 메모(정의서 기준과 매뉴얼 기준의 차이)는 매뉴얼 숫자를 읽는 이 자리에서 */}
                          {cand.coverage_note && (
                            <span title={cand.coverage_note} className="shrink-0 cursor-help text-xs text-sub underline decoration-dotted">
                              기준 차이
                            </span>
                          )}
                          <DocButton page={manual.page} name="원인 · 조치" on={openPage === manual.page} onOpen={(p) => setPage(p)} />
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
          <span className="text-sm">
            <b className="font-semibold">
              {counts.done} / {counts.total} 확인
            </b>
            {counts.done < counts.total && <span className="text-sub"> · 남음 {counts.total - counts.done}</span>}
          </span>
          {save.isError && <span className="text-sm text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
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
