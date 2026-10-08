import { Link, Navigate, useNavigate } from 'react-router'
import { bigChoice, btnQuiet, card } from '../../components/ui'
import { activeCause, checkCounts, latestJudgment, nextCandidate } from '../../lib/caseFlow'
import { josa } from '../../lib/josa'
import { formatKst } from '../../lib/time'
import { useCaseMutation } from '../../lib/useCase'
import type { Outcome } from '../../types/case'
import { useCaseDetail } from './context'
import { SidePanel, Steps, TwoColumns } from './parts'

// 결과 (wireframe/boards/V2Result.dc.html): 지금까지 한 일을 보고 해결됐는지 고른다.
// "조치 후 재발 없음" 자동 표시는 지금은 뺐다 (10/8 결정).
// 수동 모드에서 직접 찾은 원인이면 점검 줄에 수동 점검을 보여준다.
export default function ResultPage() {
  const d = useCaseDetail()
  const navigate = useNavigate()
  const save = useCaseMutation<{ situation_id: string; outcome: Outcome }>('POST', '/results')
  const cause = activeCause(d)
  if (!cause) return <Navigate to="../no-cause" replace />
  const cand = cause.candidate

  const action = d.records.actions.filter((a) => a.situation_id === cause.situation_id).at(-1)
  if (!action) return <Navigate to="../action" replace />
  const judgment = cand ? latestJudgment(d, cand.situation_id) : undefined
  // 점검 줄: 매뉴얼 후보면 그 후보의 점검, 수동 모드면 사람이 적은 점검
  const manualChecks = d.records.manual?.checks ?? []
  const counts = cand
    ? checkCounts(cand, d.records.checks)
    : { normal: manualChecks.filter((c) => c.result === 'normal').length, abnormal: 0, skipped: 0, total: manualChecks.length }
  const abnormal = cand
    ? cand.checks.filter((c) => d.records.checks[c.id]?.result === 'abnormal').map((c) => ({ key: c.id, title: c.title, memo: d.records.checks[c.id]?.memo }))
    : manualChecks.filter((c) => c.result === 'abnormal').map((c, i) => ({ key: String(i), title: c.title, memo: c.memo }))
  if (!cand) counts.abnormal = abnormal.length
  const next = cand ? nextCandidate(d, cand.situation_id) : null

  const choose = (outcome: Outcome) =>
    save.mutate(
      { situation_id: cause.situation_id, outcome },
      {
        onSuccess: () => {
          if (outcome === 'resolved') navigate('../log')
          else if (outcome === 'retry') navigate('../action')
          else navigate(next ? '../check' : '../no-cause')
        },
      },
    )

  const row = 'flex items-center gap-3 border-t border-line-soft px-3.5 py-2.5 text-[14.5px] first:border-t-0'
  return (
    <>
      <Steps current="결과" />
      <TwoColumns side={<SidePanel d={d} initial="기록" />}>
        <section className={`${card} flex flex-col gap-[22px] p-[30px]`}>
          <div>
            <span className="text-[13px] font-medium text-sub">
              {cand ? `원인 ${d.guidance.candidates.indexOf(cand) + 1} / ${d.guidance.candidates.length}` : `직접 찾은 원인 · ${cause.name}`}
            </span>
            <h2 className="mt-3 text-[22px] font-semibold tracking-[-0.015em]">해결됐나요?</h2>
          </div>
          <div className="overflow-hidden rounded-lg border border-line">
            <div className="flex items-center bg-head py-1.5 pr-1.5 pl-3.5">
              <span className="grow text-[13px] font-medium text-sub">지금까지 한 일</span>
              <Link to="../action" className={btnQuiet}>
                조치 수정
              </Link>
            </div>
            <div className={row}>
              <span className="w-14 shrink-0 text-[13.5px] text-faint">점검</span>
              <span className="grow">
                {abnormal.length ? (
                  abnormal.map((c) => (
                    <span key={c.key} className="mr-2">
                      {c.title} <b className="font-semibold text-alarm">이상</b>
                      {c.memo && ` · ${c.memo}`}
                    </span>
                  ))
                ) : (
                  <span className="text-sub">이상 항목 없음</span>
                )}
              </span>
              <span className="text-[13.5px] text-sub">
                정상 {counts.normal} · 건너뜀 {counts.skipped} · 미확인 {counts.total - counts.normal - counts.abnormal - counts.skipped}
              </span>
            </div>
            <div className={row}>
              <span className="w-14 shrink-0 text-[13.5px] text-faint">판단</span>
              <span className="grow">{cand ? `${josa(cand.name, '이', '가')} ${judgment?.verdict === 'yes' ? '맞아요' : '판단 전'}` : `직접 찾음 · ${cause.name}`}</span>
              <span className="text-[13.5px] text-sub">{judgment && formatKst(judgment.at)?.slice(11)}</span>
            </div>
            <div className={row}>
              <span className="w-14 shrink-0 text-[13.5px] text-faint">조치</span>
              <span className="grow">
                <b className="font-semibold">{action.kind}</b> — {action.did}
              </span>
              <span className="text-[13.5px] text-sub">
                {formatKst(action.started_at)?.slice(11)} ~ {formatKst(action.ended_at)?.slice(11)}
              </span>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3.5">
            <button type="button" className={bigChoice} disabled={save.isPending} onClick={() => choose('resolved')}>
              <b className="text-[17px] font-semibold">해결됐어요</b>
              <span className="text-[13.5px] text-sub">정비일지를 작성합니다</span>
            </button>
            <button type="button" className={bigChoice} disabled={save.isPending} onClick={() => choose('retry')}>
              <b className="text-[17px] font-semibold">다른 조치를 해볼게요</b>
              <span className="text-[13.5px] text-sub">같은 원인으로 다시 기록</span>
            </button>
            <button type="button" className={bigChoice} disabled={save.isPending} onClick={() => choose('next')}>
              <b className="text-[17px] font-semibold">다음 원인 확인</b>
              <span className="text-[13.5px] text-sub">{next ? next.name : '남은 원인이 없습니다'}</span>
            </button>
          </div>
          {save.isError && <p className="text-[13.5px] text-alarm">저장하지 못했습니다. 다시 눌러 주세요.</p>}
        </section>
      </TwoColumns>
    </>
  )
}
