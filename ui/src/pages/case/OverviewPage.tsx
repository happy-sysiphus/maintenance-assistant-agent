import { useState } from 'react'
import { Link } from 'react-router'
import { SignalChart } from '../../components/SignalChart'
import { btnLarge, btnPrimary, card } from '../../components/ui'
import { checkCounts, currentCandidate, signalLabel } from '../../lib/caseFlow'
import { useCaseDetail } from './context'
import { CaseFacts, CauseTab } from './parts'

// 상황 (wireframe/boards/V2Case.dc.html): 고장 전후 30분 그래프 + 확인할 원인
export default function OverviewPage() {
  const d = useCaseDetail()
  const trends = d.event.sensor_trend
  const [selected, setSelected] = useState(trends.find((t) => t.sensor.startsWith('c'))?.feature ?? trends[0]?.feature ?? '')
  const current = currentCandidate(d)
  const started = Object.keys(d.records.checks).length > 0 || d.records.judgments.length > 0
  const noGuide = d.guidance.candidates.length === 0

  return (
    <div className="grid grow grid-cols-[minmax(0,1fr)_380px] gap-4">
      <section className={`${card} flex flex-col gap-3.5 p-6`}>
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-semibold tracking-[-0.015em]">
            {d.event.trigger.rule_code ? `${d.event.trigger.rule_code} 발생 전후` : '모델 이상 신호 전후'}
          </h2>
          <div className="grow" />
          <span className="text-[13.5px] text-sub">최근 30분</span>
        </div>
        {d.event.summary_ko && <p className="-mt-1 text-sm text-sub">ML 요약 · {d.event.summary_ko}</p>}
        <SignalChart trends={trends} selected={selected} score={d.score_trace} codes={d.event.error_timeline} />
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[13px] font-medium text-sub">신호</span>
          {trends.map((t) => (
            <button
              key={t.feature}
              type="button"
              aria-pressed={t.feature === selected}
              onClick={() => setSelected(t.feature)}
              className={`min-h-10 rounded-lg border px-3.5 text-sm ${
                t.feature === selected ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-[#4e5968]'
              }`}
            >
              {signalLabel(t)}
            </button>
          ))}
          <div className="grow" />
          <span className="text-[13.5px] text-sub">값 = 이 설비 평소 대비 편차 · 1분 단위</span>
        </div>
        <div className="mt-1 border-t border-line-soft pt-4">
          <CaseFacts d={d} />
        </div>
      </section>

      <section className={`${card} flex flex-col gap-3.5 p-[22px]`}>
        <span className="text-[13px] font-medium text-sub">확인할 원인</span>
        {noGuide ? (
          <p className="text-sm text-sub">이 고장에 맞는 매뉴얼 안내를 찾지 못했습니다. 직접 점검한 내용을 기록하거나 도움을 요청하세요.</p>
        ) : (
          <CauseTab d={d} />
        )}
        <div className="grow" />
        {noGuide ? (
          <Link to="manual" className={`${btnPrimary} ${btnLarge}`}>
            직접 점검해서 기록
          </Link>
        ) : current ? (
          <Link to="check" className={`${btnPrimary} ${btnLarge}`}>
            {started ? '이어서 점검' : '점검 시작'}
            {current && started && ` · ${checkCounts(current, d.records.checks).done}/${current.checks.length}`}
          </Link>
        ) : (
          <Link to="no-cause" className={`${btnPrimary} ${btnLarge}`}>
            남은 원인 없음 · 다음 할 일
          </Link>
        )}
      </section>
    </div>
  )
}
