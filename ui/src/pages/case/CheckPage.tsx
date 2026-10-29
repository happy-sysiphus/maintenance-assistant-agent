import { useState } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router'
import { DocButton, ManualPanel } from '../../components/ManualPanel'
import { ResultSeg } from '../../components/ResultSeg'
import { CameraIcon } from '../../components/Icons'
import { btn, btnLarge, btnPrimary, card, input } from '../../components/ui'
import { docFromParam, MANUAL_NAME, MANUAL_SHORT, type ManualDoc } from '../../lib/api'
import { checkCounts, pickCandidate } from '../../lib/caseFlow'
import { askAboutCause } from '../../lib/guidance'
import { addPhotos, MAX_PHOTOS } from '../../lib/photo'
import { useCaseMutation } from '../../lib/useCase'
import type { CheckRecord } from '../../types/case'
import { useCaseDetail } from './context'
import { BackLink, CauseList, SideLabel, StepPanel, Steps, TwoColumns } from './parts'

// 점검 (wireframe/boards/V2Check.dc.html)
// 항목마다 정상 / 이상 / 건너뜀. "이상"이면 메모 칸이 열린다 (측정값 · 단위 · 기준값 칸은 두지 않음, 10/8 결정)
// 매뉴얼 쪽 버튼을 누르면 오른쪽에 원본 PDF가 열린다 (V2Manual, 주소 ?manual=쪽).
// RAG가 점검 항목별 쪽수를 아직 주지 않아서, 쪽 버튼은 원인 단위 근거 쪽과 오류 조치 쪽에만 단다 (docs/06 요청 3).
// (10/23) 근거 버튼은 쪽수만 (설명은 마우스를 올리면). "이상"이면 메모 칸만 열린다 (매뉴얼은 오른쪽 칸에서 연다).
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
  // 앞 원인 (원인 2부터): "이전"은 그 원인의 판단 화면으로, 머리글에서 그 원인의 점검으로 바로 갈 수 있다
  const prev = index > 1 ? d.guidance.candidates[index - 2] : null
  const counts = checkCounts(cand, checks)
  // 점검 항목이 없는 원인(RAG 근거만)은 할 일이 근거를 읽는 것뿐이라 첫 근거 PDF를 처음부터 열어 둔다 (10/29).
  // 닫으면 ?manual=0 으로 닫힌 채 둔다
  const firstEvidence = cand.checks.length === 0 ? cand.evidence[0] : undefined
  if (firstEvidence && !params.has('manual')) {
    const next = new URLSearchParams(params)
    next.set('manual', String(firstEvidence.page))
    next.set('doc', firstEvidence.doc ?? 'festo')
    return <Navigate to={`?${next}`} replace />
  }
  const openPage = Number(params.get('manual')) || null
  // 어느 문서를 열지 (?doc=, 기본 신형 시스템 매뉴얼). 근거 · 기준마다 문서가 다르다 (10/28 RAG, 4종)
  const openDoc = docFromParam(params.get('doc'))
  const setPage = (page: number | null, doc: ManualDoc = openDoc) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (page) next.set('manual', String(page))
        else if (firstEvidence) next.set('manual', '0')
        else next.delete('manual')
        if (page) next.set('doc', doc)
        else next.delete('doc')
        return next
      },
      { replace: true },
    )
  // 오른쪽 칸에 기준을 보여줄 항목: 사용자가 마지막으로 누른 항목, 없으면 아직 안 한 첫 항목
  // 원인을 바꾸면(원인 1 ↔ 2) 앞 원인에서 고른 항목은 무시한다
  const pickedHere = cand.checks.some((c) => c.id === picked) ? picked : null
  const activeId = pickedHere ?? cand.checks.find((c) => !checks[c.id])?.id ?? cand.checks[0]?.id
  const activeIndex = cand.checks.findIndex((c) => c.id === activeId)
  const active = activeIndex >= 0 ? cand.checks[activeIndex] : undefined
  // 근거 쪽: 같은 문서의 같은 쪽이 여러 번 나오면 설명을 합친다
  const pages = new Map<string, { doc: ManualDoc; page: number; why: string[] }>()
  for (const e of cand.evidence) {
    const doc = e.doc ?? 'festo'
    const key = `${doc}:${e.page}`
    const why = [...(pages.get(key)?.why ?? []), e.rationale ?? ''].filter(Boolean)
    pages.set(key, { doc, page: e.page, why })
  }

  const set = (id: string, patch: Partial<CheckRecord>) =>
    setChecks((prev) => ({ ...prev, [id]: { ...(prev[id] ?? { result: 'skipped' }), ...patch } }))

  const attach = async (id: string, files: File[]) => set(id, { photos: await addPhotos(checks[id]?.photos ?? [], files) })
  const detach = (id: string, i: number) => set(id, { photos: (checks[id]?.photos ?? []).filter((_, j) => j !== i) })

  const submit = (next: boolean) => save.mutate({ checks }, { onSuccess: () => next && navigate(`../judge?cause=${cand.situation_id}`) })

  return (
    <>
      <BackLink to={prev ? `../judge?cause=${prev.situation_id}` : '..'} />
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
              // 점검 항목이 없는 원인이면 오른쪽에 보여줄 기준이 없으므로 원인 목록을 대신 둔다.
              // 항목이 있으면 기준이 주인공이라 목록은 두지 않는다 (몇 번째 원인인지는 왼쪽 머리에)
              top={
                total > 1 &&
                cand.checks.length === 0 && (
                  <>
                    <SideLabel>원인 목록</SideLabel>
                    <CauseList d={d} viewing={cand.situation_id} />
                  </>
                )
              }
              item={active && { no: activeIndex + 1, title: active.title, criterion: active.criterion }}
              onOpenDoc={setPage}
            />
          )
        }
      >
        <section className={`${card} overflow-hidden`}>
          <div className="flex items-center gap-3 border-b border-line-soft px-[22px] py-4">
            <span className="text-xs font-medium text-sub">
              원인 {index}
            </span>
            {/* 의심하는 이유(RAG why)는 길고 근거 쪽과 겹쳐서 보여주지 않는다 (10/29) */}
            <b className="min-w-0 grow text-xl font-semibold tracking-[-0.015em]">{cand.name}</b>
            {/* 앞 원인으로는 맨 위 "‹ 이전"(앞 원인의 판단)으로 간다 — 머리의 "원인 N 점검 보기"는 겹쳐서 뺐다 (10/29) */}
          </div>
          {pages.size > 0 && (
            <div className="flex flex-wrap items-center gap-2 border-b border-line-soft px-[22px] py-3">
              <span className="mr-1 text-xs font-medium text-sub">매뉴얼 근거</span>
              {[...pages].map(([key, { doc, page, why }]) => (
                <DocButton
                  key={key}
                  page={page}
                  label={`${MANUAL_SHORT[doc]} ${page}쪽`}
                  title={[MANUAL_NAME[doc], ...why].join(' · ')}
                  on={openPage === page && openDoc === doc}
                  onOpen={(p) => setPage(p, doc)}
                />
              ))}
            </div>
          )}
          {/* RAG 원인 중에는 점검 항목 없이 근거만 있는 것도 있다 (예: E012 원인 2 · 3) */}
          {cand.checks.length === 0 && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-[22px] py-5 text-sm">
              <p className="text-sub">매뉴얼에 이 원인의 점검 항목이 없어요. 오른쪽 매뉴얼에서 확인하고 다음을 눌러 판단하세요.</p>
              {/* 더 물어볼 곳: 챗봇은 매뉴얼 4종 전체에서 찾는다 (질문만 채워 두고 보내기는 사람이) */}
              <Link to={askAboutCause(cand, '확인')} className="font-medium text-primary-ink hover:underline">
                AI 챗봇에게 물어보기
              </Link>
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
                    <span className="min-w-0 grow text-base">{item.title}</span>
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
          <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={save.isPending || (counts.total > 0 && counts.done === 0)} onClick={() => submit(true)}>
            다음
          </button>
        </div>
      </TwoColumns>
    </>
  )
}
