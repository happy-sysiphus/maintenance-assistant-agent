import { useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import { DotTag } from '../../components/Tag'
import { card } from '../../components/ui'
import { candidateState, checkCounts, latestJudgment, signalLabel } from '../../lib/caseFlow'
import { formatKst } from '../../lib/time'
import type { Candidate, CaseDetail } from '../../types/case'

// 케이스 화면들이 같이 쓰는 조각: 진행 표시, 오른쪽 패널(상황 / 원인 / 기록)

const STEPS = ['점검', '판단', '조치', '결과'] as const
export type Step = (typeof STEPS)[number]

export function Steps({ current }: { current: Step }) {
  const at = STEPS.indexOf(current)
  return (
    <nav aria-label="진행 단계" className="flex items-center gap-3">
      {STEPS.map((s, i) => (
        <div key={s} className={`flex items-center gap-3 ${i < STEPS.length - 1 ? 'grow' : ''}`}>
          <span
            aria-current={i === at ? 'step' : undefined}
            className={`flex items-center gap-2 text-sm whitespace-nowrap ${i === at ? 'font-semibold text-ink' : i < at ? 'font-medium text-ink-2' : 'font-medium text-faint'}`}
          >
            <b
              className={`flex size-6 items-center justify-center rounded-full text-xs ${
                i === at ? 'bg-primary text-white' : i < at ? 'bg-primary-soft text-primary ring-[1.5px] ring-primary ring-inset' : 'bg-line text-sub'
              }`}
            >
              {i < at ? '✓' : i + 1}
            </b>
            {s}
          </span>
          {i < STEPS.length - 1 && <span className="h-px min-w-6 grow bg-field" />}
        </div>
      ))}
    </nav>
  )
}

/** 본문 + 오른쪽 380px 패널 (wide: 매뉴얼이 열렸을 때 540px) */
export function TwoColumns({ children, side, wide = false }: { children: ReactNode; side: ReactNode; wide?: boolean }) {
  return (
    <div className={`grid grow items-start gap-4 ${wide ? 'grid-cols-[minmax(0,1fr)_540px]' : 'grid-cols-[minmax(0,1fr)_380px]'}`}>
      <div className="flex min-w-0 flex-col gap-4">{children}</div>
      {side}
    </div>
  )
}

type Tab = '상황' | '원인' | '기록'

export function SidePanel({ d, initial }: { d: CaseDetail; initial: Tab }) {
  const [tab, setTab] = useState<Tab>(initial)
  return (
    <aside className={`${card} overflow-hidden`} aria-label="고장 정보">
      <div role="tablist" className="flex gap-1 border-b border-line px-4">
        {(['상황', '원인', '기록'] as Tab[]).map((t) => (
          <button
            key={t}
            role="tab"
            type="button"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 px-2.5 pt-[13px] pb-[11px] text-sm ${tab === t ? 'border-ink font-semibold text-ink' : 'border-transparent font-medium text-faint'}`}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="flex flex-col gap-3.5 p-[18px]">
        {tab === '상황' && <CaseFacts d={d} />}
        {tab === '원인' && <CauseTab d={d} />}
        {tab === '기록' && <RecordTab d={d} />}
      </div>
    </aside>
  )
}

/** 점검 단계의 오른쪽 패널 (와이어프레임 V2Check 10/23): 탭 대신 지금 필요한 것만. 그래프 · 기록은 눌렀을 때만 */
export function StepPanel({ d, cand, remaining }: { d: CaseDetail; cand: Candidate; remaining: number }) {
  const [note, setNote] = useState(false)
  const [records, setRecords] = useState(false)
  const quiet = 'inline-flex min-h-9 items-center px-2 text-sm font-medium text-primary-ink hover:underline'
  return (
    <aside className={`${card} flex flex-col gap-3.5 p-5`} aria-label="이 단계에서 볼 것">
      <span className="text-xs font-medium text-sub">정의</span>
      <p className="text-sm">{cand.definition}</p>
      {cand.coverage_note && (
        <div className="rounded-lg bg-canvas px-3 py-2.5 text-sm text-ink-2">
          <div className="flex items-center gap-2">
            <span className="grow">매뉴얼 기준은 이 설비와 다를 수 있음</span>
            <button type="button" className={quiet} aria-expanded={note} onClick={() => setNote((v) => !v)}>
              {note ? '접기' : '자세히'}
            </button>
          </div>
          {note && <p className="mt-1.5 leading-relaxed">{cand.coverage_note}</p>}
        </div>
      )}
      <div className="h-px bg-line-soft" />
      <span className="text-xs font-medium text-sub">다음 할 일</span>
      <div className="flex flex-col gap-0.5 rounded-lg bg-primary-soft px-3.5 py-3 text-sm text-primary-ink">
        <b className="font-semibold">{remaining > 0 ? `점검 ${remaining}개 남음` : '점검을 마쳤습니다'}</b>
        <span>{remaining > 0 ? '끝나면 판단에서 이 원인이 맞는지 고릅니다' : '"다음"을 눌러 이 원인이 맞는지 고릅니다'}</span>
      </div>
      <div className="h-px bg-line-soft" />
      <div className="flex gap-1">
        <Link to=".." className={quiet}>
          그래프 보기
        </Link>
        <button type="button" className={quiet} aria-expanded={records} onClick={() => setRecords((v) => !v)}>
          {records ? '기록 접기' : '기록 전체 보기'}
        </button>
      </div>
      {records && <RecordTab d={d} />}
    </aside>
  )
}

/** 조치 단계의 오른쪽 패널 (와이어프레임 V2Action 10/23): 매뉴얼 조치는 참고용 */
export function ManualActionPanel({ cand, onOpen }: { cand: Candidate | null; onOpen?: (page: number) => void }) {
  const diag = cand?.diagnostics.find((x) => x.manual)
  const manual = diag?.manual
  return (
    <aside className={`${card} flex flex-col gap-3 p-5`} aria-label="매뉴얼 조치">
      <span className="text-xs font-medium text-sub">매뉴얼 조치 (참고)</span>
      {manual && diag ? (
        <>
          <b className="font-semibold">
            Festo 오류 {diag.number} · {manual.name}
          </b>
          {/* 원문 그대로 (번역 데이터가 없어 지어내지 않는다) */}
          <p className="text-sm leading-relaxed text-ink-2">{manual.text}</p>
          <span className="text-xs text-sub">영어 원문 · 이 설비 기준과 다를 수 있음</span>
          {onOpen && (
            <button
              type="button"
              onClick={() => onOpen(manual.page)}
              className="inline-flex min-h-10 items-center self-start rounded-lg border border-field bg-white px-3 text-sm font-medium text-primary-ink hover:bg-canvas"
            >
              원문 보기 <span className="ml-1 font-normal text-sub">{manual.page}쪽</span>
            </button>
          )}
        </>
      ) : (
        <p className="text-sm text-sub">원문을 확인한 매뉴얼 조치가 없습니다.</p>
      )}
      <div className="h-px bg-line-soft" />
      <span className="text-sm text-sub">매뉴얼 조치는 참고용이고, 왼쪽에는 실제로 한 일만 적습니다.</span>
    </aside>
  )
}

export function CaseFacts({ d }: { d: CaseDetail }) {
  const e = d.event
  const span = e.error_timeline.at(-1)
  const signals = e.sensor_trend.filter((t) => t.sensor.startsWith('c')).map(signalLabel)
  return (
    <dl className="grid grid-cols-[88px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-[9px] text-sm">
      <dt className="text-sm text-faint">고장 코드</dt>
      <dd className="font-semibold">{e.trigger.rule_code ?? <span className="font-normal text-sub">없음 · 모델 이상 신호</span>}</dd>
      <dt className="text-sm text-faint">발생</dt>
      <dd>
        {formatKst(e.trigger.rule_trigger_time ?? e.detected_at)}
        {span && span.code === e.trigger.rule_code && ` · ${span.duration_s}초간`}
      </dd>
      <dt className="text-sm text-faint">현상 (예상)</dt>
      <dd>{e.fault_class?.name_ko ?? <span className="text-sub">—</span>}</dd>
      <dt className="text-sm text-faint">달라진 신호</dt>
      <dd>{signals.length ? signals.join(' · ') : <span className="text-sub">—</span>}</dd>
      {/* 정상(판단 가능)은 표시하지 않고, 판단 보류일 때만 주황으로 알린다 */}
      {e.context.alarm_held && (
        <>
          <dt className="text-sm text-faint">데이터</dt>
          <dd>
            <DotTag color="amber">판단 보류 · 용접 멈춤 구간</DotTag>
          </dd>
        </>
      )}
    </dl>
  )
}

const STATE_TAG = {
  current: <DotTag color="blue">확인 중</DotTag>,
  excluded: <DotTag color="gray">제외</DotTag>,
  waiting: <DotTag color="gray">대기</DotTag>,
}

export function CauseTab({ d }: { d: CaseDetail }) {
  if (d.guidance.candidates.length === 0) return <p className="text-sm text-sub">매뉴얼 안내가 있는 원인이 없습니다.</p>
  return (
    <>
      {d.guidance.candidates.map((c, i) => {
        const st = candidateState(d, c.situation_id)
        const n = checkCounts(c, d.records.checks)
        const j = latestJudgment(d, c.situation_id)
        return (
          <div
            key={c.situation_id}
            className={`flex items-center gap-3 rounded-lg border px-3.5 py-3 ${st === 'current' ? 'border-primary ring-1 ring-primary' : 'border-line'}`}
          >
            <span className={`flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${st === 'current' ? 'bg-primary text-white' : 'bg-fill text-sub'}`}>
              {i + 1}
            </span>
            <div className="min-w-0 grow">
              <b className={`font-semibold ${st === 'excluded' ? 'text-sub' : ''}`}>{c.name}</b>
              <div className="text-sm text-sub">
                {j?.verdict === 'no' ? `아니에요 · ${formatKst(j.at)?.slice(11)}` : `점검 ${n.total}개${n.done ? ` 중 ${n.done}개 확인` : ''}`}
              </div>
            </div>
            {STATE_TAG[st]}
          </div>
        )
      })}
    </>
  )
}

const VERDICT = { yes: '맞아요', no: '아니에요', unsure: '잘 모르겠어요' }
const OUTCOME = { resolved: '해결됐어요', retry: '다른 조치를 해볼게요', next: '다음 원인 확인' }
const CLOSURE = { resolved: '해결 종료', unresolved: '미해결로 저장', handed_over: '도움 요청' }

export function RecordTab({ d }: { d: CaseDetail }) {
  const name = (sid: string) => d.guidance.candidates.find((c) => c.situation_id === sid)?.name ?? sid
  const items: { at: string; text: ReactNode }[] = []
  if (d.records.started_at) items.push({ at: d.records.started_at, text: '점검 시작' })
  for (const j of d.records.judgments) items.push({ at: j.at, text: `${name(j.situation_id)} · ${VERDICT[j.verdict]}` })
  for (const a of d.records.actions) items.push({ at: a.at, text: `조치 기록 · ${a.kind}` })
  for (const r of d.records.results) items.push({ at: r.at, text: `결과 · ${OUTCOME[r.outcome]}` })
  if (d.records.manual) items.push({ at: d.records.manual.saved_at, text: `수동 점검 기록${d.records.manual.cause ? ` · 원인 ${d.records.manual.cause}` : ''}` })
  for (const f of d.records.field) items.push({ at: f.at, text: `현장 확인 · ${f.by} · ${f.check_ids.length}개${f.photos?.length ? ` · 사진 ${f.photos.length}장` : ''}` })
  for (const h of d.records.handovers) items.push({ at: h.at, text: `도움 요청 · ${h.to} · ${h.equipment_state}` })
  if (d.records.log?.approved_at) items.push({ at: d.records.log.approved_at, text: '정비일지 승인' })
  // 도움 요청은 위에서 이미 한 줄로 보여준다
  if (d.records.closure && d.records.closure.outcome !== 'handed_over') items.push({ at: d.records.closure.at, text: CLOSURE[d.records.closure.outcome] })
  items.sort((a, b) => a.at.localeCompare(b.at))
  const code = d.event.trigger.rule_code
  return (
    <ol className="flex flex-col gap-3">
      <li className="flex gap-3 text-sm">
        <span className="w-11 shrink-0" />
        <i className="mt-[7px] size-[7px] shrink-0 rounded-full bg-alarm-dot" />
        <div>
          <b className="font-semibold">{code ? `${code} 발생` : '모델 이상 신호'}</b> · 데이터 시각 {formatKst(d.event.trigger.rule_trigger_time ?? d.event.detected_at)}
        </div>
      </li>
      {items.map((it, i) => (
        <li key={i} className="flex gap-3 text-sm">
          <span className="w-11 shrink-0 text-faint">{formatKst(it.at)?.slice(11)}</span>
          <i className="mt-[7px] size-[7px] shrink-0 rounded-full bg-primary" />
          <div>{it.text}</div>
        </li>
      ))}
      {items.length === 0 && <li className="pl-14 text-sm text-sub">아직 기록이 없습니다.</li>}
    </ol>
  )
}
