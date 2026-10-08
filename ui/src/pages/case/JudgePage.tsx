import { Link, Navigate, useNavigate, useSearchParams } from 'react-router'
import { DotTag } from '../../components/Tag'
import { bigChoice, btnQuiet, card } from '../../components/ui'
import { checkCounts, nextCandidate, pickCandidate } from '../../lib/caseFlow'
import { josa } from '../../lib/josa'
import { useCaseMutation } from '../../lib/useCase'
import type { CheckResult, Verdict } from '../../types/case'
import { useCaseDetail } from './context'
import { SidePanel, Steps, TwoColumns } from './parts'

// 판단 (wireframe/boards/V2Judge.dc.html): 점검 결과를 보고 카드 하나를 누르면 바로 넘어간다

const RESULT_TAG: Record<CheckResult, { label: string; tone: string }> = {
  normal: { label: '정상', tone: 'text-[#0f7b5f]' },
  abnormal: { label: '이상', tone: 'text-[#c4362b]' },
  skipped: { label: '건너뜀', tone: 'text-sub' },
}

export default function JudgePage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const judge = useCaseMutation<{ situation_id: string; verdict: Verdict }>('POST', '/judgments')
  const [params] = useSearchParams()
  const cand = pickCandidate(d, params.get('cause'))
  if (!cand) return <Navigate to="../no-cause" replace />

  const index = d.guidance.candidates.indexOf(cand) + 1
  const next = nextCandidate(d, cand.situation_id)
  const counts = checkCounts(cand, d.records.checks)

  const choose = (verdict: Verdict) =>
    judge.mutate(
      { situation_id: cand.situation_id, verdict },
      {
        onSuccess: () => {
          if (verdict === 'yes') navigate('../action')
          else if (verdict === 'no') navigate(next ? '../check' : '../no-cause')
          else navigate('../handover')
        },
      },
    )

  return (
    <>
      <Steps current="판단" />
      <TwoColumns side={<SidePanel d={d} initial="원인" />}>
        <section className={`${card} flex flex-col gap-[22px] p-[30px]`}>
          <div>
            <span className="text-[13px] font-medium text-sub">
              원인 {index} / {d.guidance.candidates.length}
            </span>
            <h2 className="mt-3 text-[22px] font-semibold tracking-[-0.015em]">{josa(cand.name, '이', '가')} 맞나요?</h2>
          </div>
          <div className="overflow-hidden rounded-lg border border-line">
            <div className="flex items-center gap-3 bg-head py-1.5 pr-1.5 pl-3.5">
              <span className="text-[13px] font-medium text-sub">점검 결과</span>
              {counts.abnormal > 0 && <span className="text-[13.5px] font-medium text-[#c4362b]">이상 {counts.abnormal}</span>}
              {counts.normal > 0 && <span className="text-[13.5px] font-medium text-[#0f7b5f]">정상 {counts.normal}</span>}
              {counts.total - counts.normal - counts.abnormal > 0 && (
                <span className="text-[13.5px] font-medium text-sub">건너뜀 {counts.skipped} · 미확인 {counts.total - counts.normal - counts.abnormal - counts.skipped}</span>
              )}
              <div className="grow" />
              <Link to={`../check?cause=${cand.situation_id}`} className={btnQuiet}>
                점검 다시 보기
              </Link>
            </div>
            {cand.checks.map((item, i) => {
              const rec = d.records.checks[item.id]
              const tag = rec ? RESULT_TAG[rec.result] : null
              return (
                <div key={item.id} className="flex items-center gap-3 border-t border-line-soft px-3.5 py-2.5 text-[14.5px]">
                  <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-[#f2f4f6] text-[12.5px] font-semibold text-sub">{i + 1}</span>
                  <span className="grow">{item.title}</span>
                  {rec?.memo && <span className="text-[13.5px] text-sub">메모: {rec.memo}</span>}
                  <span className={`w-16 text-[13.5px] font-medium ${tag?.tone ?? 'text-faint'}`}>{tag?.label ?? '미확인'}</span>
                </div>
              )
            })}
          </div>
          <div className="grid grid-cols-3 gap-3.5">
            <button type="button" className={bigChoice} disabled={judge.isPending} onClick={() => choose('yes')}>
              <b className="text-[17px] font-semibold">맞아요</b>
              <span className="text-[13.5px] text-sub">조치를 기록합니다</span>
            </button>
            <button type="button" className={bigChoice} disabled={judge.isPending} onClick={() => choose('no')}>
              <b className="text-[17px] font-semibold">아니에요</b>
              <span className="text-[13.5px] text-sub">{next ? `다음 원인: ${next.name}` : '남은 원인이 없습니다'}</span>
            </button>
            <button type="button" className={bigChoice} disabled={judge.isPending} onClick={() => choose('unsure')}>
              <b className="text-[17px] font-semibold">잘 모르겠어요</b>
              <span className="text-[13.5px] text-sub">도움 요청</span>
            </button>
          </div>
          {judge.isError && <p className="text-[13.5px] text-alarm">저장하지 못했습니다. 다시 눌러 주세요.</p>}
          {counts.done === 0 && (
            <p className="text-[13.5px] text-sub">
              <DotTag color="gray">점검한 항목이 없습니다</DotTag>
            </p>
          )}
        </section>
      </TwoColumns>
    </>
  )
}
