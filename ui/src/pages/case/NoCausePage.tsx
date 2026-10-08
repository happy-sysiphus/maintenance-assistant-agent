import { Link } from 'react-router'
import { DotTag } from '../../components/Tag'
import { bigChoice, card } from '../../components/ui'
import { checkCounts } from '../../lib/caseFlow'
import { useCaseDetail } from './context'
import { RecordTab, TwoColumns } from './parts'

// 남은 원인 없음 (wireframe/boards/V2NoCause.dc.html): 모든 원인이 "아니에요"일 때 막히지 않게
export default function NoCausePage() {
  const d = useCaseDetail()
  const unchecked = (c: (typeof d.guidance.candidates)[number]) => {
    const k = checkCounts(c, d.records.checks)
    return k.total - k.normal - k.abnormal
  }
  const skipped = d.guidance.candidates.reduce((n, c) => n + unchecked(c), 0)
  // 다시 볼 원인: 건너뛰거나 확인 안 한 점검이 있는 첫 원인
  const revisit = d.guidance.candidates.find((c) => unchecked(c) > 0)

  return (
    <TwoColumns
      side={
        <aside className={`${card} flex flex-col gap-3.5 p-[18px]`} aria-label="기록">
          <span className="text-[13px] font-medium text-sub">기록</span>
          <RecordTab d={d} />
        </aside>
      }
    >
      <section className={`${card} flex flex-col gap-[22px] p-[30px]`}>
        <h2 className="text-[22px] font-semibold tracking-[-0.015em]">더 확인할 원인이 없습니다</h2>
        <div className="flex flex-col gap-2.5">
          {d.guidance.candidates.map((c, i) => {
            const k = checkCounts(c, d.records.checks)
            return (
              <div key={c.situation_id} className="flex items-center gap-3 rounded-lg border border-line px-3.5 py-3">
                <span className="flex size-6 items-center justify-center rounded-full bg-[#f2f4f6] text-[12.5px] font-semibold text-sub">{i + 1}</span>
                <div className="grow">
                  <b className="font-semibold">{c.name}</b>
                  <div className="text-[13.5px] text-sub">
                    점검 {k.total}개 중 {k.done}개 확인
                  </div>
                </div>
                <DotTag color="gray">제외</DotTag>
              </div>
            )
          })}
        </div>
        <span className="text-[13px] font-medium text-sub">어떻게 할까요?</span>
        <div className="grid grid-cols-3 gap-3.5">
          <Link to="../manual" className={bigChoice}>
            <b className="text-[17px] font-semibold">직접 점검해서 기록</b>
            <span className="text-[13.5px] text-sub">안내 없이 확인한 것을 적습니다</span>
          </Link>
          <Link to="../handover" className={bigChoice}>
            <b className="text-[17px] font-semibold">도움 요청</b>
            <span className="text-[13.5px] text-sub">지금까지 기록이 함께 전달됩니다</span>
          </Link>
          {revisit ? (
            <Link to={`../check?cause=${revisit.situation_id}`} className={bigChoice}>
              <b className="text-[17px] font-semibold">원인 다시 보기</b>
              <span className="text-[13.5px] text-sub">건너뛰거나 확인 안 한 점검 {skipped}개</span>
            </Link>
          ) : (
            <button type="button" className={bigChoice} disabled>
              <b className="text-[17px] font-semibold">원인 다시 보기</b>
              <span className="text-[13.5px] text-sub">건너뛴 점검 없음</span>
            </button>
          )}
        </div>
      </section>
    </TwoColumns>
  )
}
