import { useState } from 'react'
import { Link } from 'react-router'
import { SignalChart } from '../../components/SignalChart'
import { btnLarge, btnPrimary, card } from '../../components/ui'
import { checkCounts, currentCandidate, signalLabel } from '../../lib/caseFlow'
import { firstSentence, stripCites } from '../../lib/guidance'
import { useCaseDetail } from './context'
import { BackLink, CaseFacts, CauseList, FalseAlarmButton } from './parts'

// 상황 (wireframe/boards/V2Case.dc.html): 고장 전후 30분 그래프 + 확인할 원인
export default function OverviewPage() {
  const d = useCaseDetail()
  const trends = d.event.sensor_trend
  // ML은 sensor_trend를 이번 고장에 기여가 큰 신호부터 준다 → 첫 번째를 먼저 보여준다
  const [selected, setSelected] = useState(trends[0]?.feature ?? '')
  const [howTo, setHowTo] = useState(false)
  const current = currentCandidate(d)
  const started = Object.keys(d.records.checks).length > 0 || d.records.judgments.length > 0
  const noGuide = d.guidance.candidates.length === 0
  const summary = d.guidance.sections?.summary ? firstSentence(stripCites(d.guidance.sections.summary)) : null

  return (
    <>
      <BackLink to="/" />
      <div className="grid grow grid-cols-[minmax(0,1fr)_380px] items-start gap-4">
        <section className={`${card} flex flex-col gap-3.5 p-6`}>
          <div className="flex items-baseline gap-3">
            <h2 className="text-lg font-semibold tracking-[-0.015em]">
              {d.event.trigger.rule_code ? `${d.event.trigger.rule_code} 발생 전후` : '모델 이상 신호 전후'}
            </h2>
            <div className="grow" />
            <span className="text-sm text-sub">최근 30분</span>
          </div>
          {/* 이 고장이 무엇인지 한 줄 (RAG 요약의 첫 문장) */}
          {summary && <p className="-mt-1.5 text-sm leading-relaxed text-ink-2">{summary}</p>}
          <SignalChart trends={trends} selected={selected} score={d.score_trace} codes={d.event.error_timeline} />
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-medium text-sub">신호</span>
            {trends.map((t) => (
              <button
                key={t.feature}
                type="button"
                aria-pressed={t.feature === selected}
                onClick={() => setSelected(t.feature)}
                className={`min-h-10 rounded-lg border px-3.5 text-sm ${
                  t.feature === selected ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-ink-2'
                }`}
              >
                {signalLabel(t)}
              </button>
            ))}
            <div className="grow" />
            <button
              type="button"
              aria-expanded={howTo}
              onClick={() => setHowTo((v) => !v)}
              className="inline-flex min-h-9 items-center px-2 text-sm font-medium text-primary-ink hover:underline"
            >
              {howTo ? '설명 접기' : '그래프 읽는 법'}
            </button>
          </div>
          {/* 글을 줄이려고 설명 · ML 요약은 눌렀을 때만 보여준다 (10/23) */}
          {howTo && (
            <div className="flex flex-col gap-1 rounded-lg bg-canvas px-3.5 py-3 text-sm text-ink-2">
              <p>값은 실제 측정값이 아니라 이 설비 평소 대비 편차입니다 (1분 단위, ±1이 평소 범위). 빨간 선은 고장 코드가 뜬 시각입니다.</p>
              <p>이상 점수는 0~1이며 고장 확률이 아닙니다. 점선이 기준선입니다.</p>
              {d.event.summary_ko && <p>ML 요약 · {d.event.summary_ko}</p>}
            </div>
          )}
          <div className="mt-1 border-t border-line-soft pt-4">
            <CaseFacts d={d} />
          </div>
        </section>

        <section className={`${card} flex flex-col gap-3.5 p-[22px]`}>
          <span className="text-xs font-medium text-sub">확인할 원인</span>
          {/* RAG 안내 생성 실패: api는 기본 점검 항목(매핑)만 남긴다. 다시 열면 다시 만든다 */}
          {!noGuide && (d.guidance.status === 'error' || d.guidance.status === 'needs_citation_review') && (
            <p role="status" className="rounded-lg border border-warn-dot/30 bg-warn-soft px-3.5 py-3 text-sm text-warn">
              매뉴얼 안내를 만들지 못해 기본 점검 항목만 보여줘요. 잠시 뒤 새로고침하면 다시 만들어요.
            </p>
          )}
          {noGuide ? (
            <p className="text-sm text-sub">이 고장에 맞는 매뉴얼 안내를 찾지 못했습니다. 직접 점검한 내용을 기록하거나 도움을 요청하세요.</p>
          ) : (
            <CauseList d={d} />
          )}
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
              원인 찾기
            </Link>
          )}
          <FalseAlarmButton />
        </section>
      </div>
    </>
  )
}
