import { useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { ResultSeg } from '../../components/ResultSeg'
import { btn, btnLarge, btnPrimary, card, input, label } from '../../components/ui'
import { activeCause } from '../../lib/caseFlow'
import { useCaseMutation } from '../../lib/useCase'
import { addPhotos as addTo, MAX_PHOTOS } from '../../lib/photo'
import type { CheckRecord, Photo } from '../../types/case'
import { useCaseDetail } from './context'
import { SideLabel, sideCard } from './parts'

// 현장 확인 입력 (wireframe/boards/V2Field.dc.html): 센서로 알 수 없어 직접 확인해야 할 때 본인이 결과를 넣는다.
// 항목은 지금 원인의 점검 항목이다 (지어내지 않음). 결과는 점검 결과에 같이 저장되고, 저장하면 점검 화면으로 돌아간다.
// 인쇄하면 종이 체크시트로 쓸 수 있다. (10/8) 측정값 칸 대신 메모 칸 · 확인한 사람은 직접 입력.
// (10/23) 사진을 4장까지 붙일 수 있다. 업로드 api가 없어 지금은 data URL로 케이스 기록에 같이 저장한다 (lib/photo).

export default function FieldPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const cand = activeCause(d)?.candidate ?? null
  const [checks, setChecks] = useState<Record<string, CheckRecord>>({})
  const [by, setBy] = useState('')
  const [tried, setTried] = useState(false)
  const [photos, setPhotos] = useState<Photo[]>([])
  const save = useCaseMutation<{ by: string; checks: Record<string, CheckRecord>; photos: Photo[] }>('POST', '/field')

  const addPhotos = async (files: FileList | null) => setPhotos(await addTo(photos, files))

  const ctx = d.event.context
  const set = (id: string, patch: Partial<CheckRecord>) =>
    setChecks((prev) => ({ ...prev, [id]: { ...(prev[id] ?? d.records.checks[id] ?? { result: 'skipped' }), ...patch } }))
  const changed = Object.keys(checks).length

  const submit = () => {
    setTried(true)
    if (!by.trim() || changed === 0) return
    save.mutate({ by: by.trim(), checks, photos }, { onSuccess: () => navigate('../check') })
  }

  return (
    <>
      <div className="flex items-center gap-3">
        <h2 className="grow text-xl font-semibold tracking-[-0.015em]">현장 확인</h2>
        <button type="button" className={`${btn} print:hidden`} onClick={() => window.print()}>
          인쇄 · PDF
        </button>
      </div>
      <div className="grid grow grid-cols-[minmax(0,1fr)_380px] items-start gap-4 print:block">
        <section className={`${card} overflow-hidden`}>
          {cand ? (
            <>
              <div className="flex items-center gap-3 border-b border-line-soft px-[22px] py-4">
                <span className="text-xs font-medium text-sub">{d.event.gun_id}</span>
                <b className="text-base font-semibold">{cand.name}</b>
              </div>
              <ol>
                {cand.checks.map((item, i) => {
                  const rec = checks[item.id] ?? d.records.checks[item.id]
                  return (
                    <li key={item.id} className="flex items-center gap-3 border-t border-line-soft px-5 py-3.5 first:border-t-0">
                      <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-fill text-xs font-semibold text-sub">
                        {i + 1}
                      </span>
                      <span className="min-w-0 grow text-base font-semibold tracking-[-0.015em]">{item.title}</span>
                      <div className="w-[220px] shrink-0">
                        <input
                          className={input}
                          placeholder="메모"
                          aria-label={`${item.title} 메모`}
                          value={rec?.memo ?? ''}
                          onChange={(e) => set(item.id, { memo: e.target.value })}
                        />
                      </div>
                      <ResultSeg label={item.title} value={rec?.result} onChange={(v) => set(item.id, { result: v })} />
                    </li>
                  )
                })}
              </ol>
              <div className="grid grid-cols-2 gap-3 border-t border-line-soft px-5 py-4">
                <label className="flex flex-col gap-1.5">
                  <span className={label}>확인한 사람 *</span>
                  <input className={input} placeholder="이름" value={by} onChange={(e) => setBy(e.target.value)} />
                  {tried && !by.trim() && <span className="text-xs text-alarm print:hidden">확인한 사람을 적어 주세요</span>}
                </label>
                <label className="flex flex-col gap-1.5">
                  <span className={label}>확인 시각</span>
                  <input className={`${input} bg-canvas text-ink-2`} value="저장할 때 자동 기록" readOnly />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-3 border-t border-line-soft px-5 py-4 print:hidden">
                <span className={label}>사진</span>
                {photos.map((p, i) => (
                  <div key={i} className="relative">
                    <img src={p.url} alt={`현장 사진 ${i + 1}`} className="size-16 rounded-lg border border-line object-cover" />
                    <button
                      type="button"
                      aria-label={`사진 ${i + 1} 빼기`}
                      onClick={() => setPhotos((prev) => prev.filter((_, j) => j !== i))}
                      className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full border border-field bg-white text-xs text-ink-2"
                    >
                      ✕
                    </button>
                  </div>
                ))}
                {photos.length < MAX_PHOTOS && (
                  <label className={`${btn} cursor-pointer`}>
                    + 사진 추가
                    <input
                      type="file"
                      accept="image/*"
                      capture="environment"
                      multiple
                      className="sr-only"
                      onChange={(e) => {
                        void addPhotos(e.target.files)
                        e.target.value = ''
                      }}
                    />
                  </label>
                )}
                <span className="text-xs text-sub">
                  {photos.length} / {MAX_PHOTOS}장
                </span>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-start gap-3 p-6">
              <p className="text-sm text-sub">지금 확인할 매뉴얼 점검 항목이 없습니다. 수동 모드에서 확인한 것을 직접 적어 주세요.</p>
              <Link to="../manual" className={btn}>
                수동 모드로
              </Link>
            </div>
          )}
        </section>

        <div className="flex flex-col gap-4 print:hidden">
          {(ctx.alarm_held || ctx.non_welding_share > 0) && (
            <section className={sideCard} aria-label="알림 시점 상태">
              <SideLabel>알림 시점 상태</SideLabel>
              <p className="text-sm text-ink-2">
                {ctx.alarm_held && <b className="mr-1 font-semibold text-warn">판단 보류 ·</b>}
                용접 안 한 비율 {Math.round(ctx.non_welding_share * 100)}%
              </p>
            </section>
          )}
          <div className="flex items-center gap-3">
            {save.isError && <span className="text-sm text-alarm">저장하지 못했습니다. 입력 내용은 남아 있습니다.</span>}
            {tried && changed === 0 && <span className="text-sm text-alarm">확인한 항목이 없습니다</span>}
            <div className="grow" />
            <Link to="../check" className={btn}>
              취소
            </Link>
            {cand && (
              <button type="button" className={`${btnPrimary} ${btnLarge}`} disabled={save.isPending} onClick={submit}>
                저장하고 돌아가기
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
