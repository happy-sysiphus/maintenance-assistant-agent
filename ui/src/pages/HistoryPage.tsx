import { useQuery } from '@tanstack/react-query'
import { Fragment, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router'
import { ChevronIcon } from '../components/Icons'
import { Select } from '../components/Select'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { DotTag } from '../components/Tag'
import { PageHeader, SearchBox } from '../components/PageHeader'
import { btn, btnQuiet, card, page } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst } from '../lib/time'
import { useCaseQuery } from '../lib/useCase'
import { RecordTab } from './case/parts'
import type { Closure, HistoryItem, HistoryResponse } from '../types/case'

// 정비 이력 (wireframe/boards/V2History.dc.html): 해결 종료했거나 미해결로 저장한 고장 (읽기 전용).
// 줄을 누르면 그 줄 바로 아래로 요약이 펼쳐진다 (한 번에 한 줄). 시간순 기록은 "기록 전체 보기"를 눌러야 보인다.
// 종료 · 걸린 시간은 작업 시각 기준이다. 데이터 시각(고장 발생)과 섞지 않는다.

const EMPTY = '—'
const RESULT: Record<Closure, { label: string; color: 'gray' | 'amber' }> = {
  resolved: { label: '해결', color: 'gray' },
  unresolved: { label: '미해결', color: 'amber' },
  handed_over: { label: '도움 요청', color: 'amber' },
}
const FILTERS: { key: Closure | 'all'; label: string }[] = [
  { key: 'all', label: '전체' },
  { key: 'resolved', label: '해결' },
  { key: 'unresolved', label: '미해결' },
  { key: 'handed_over', label: '도움 요청' },
]
const PERIODS = [
  { days: 30, label: '최근 30일' },
  { days: 7, label: '최근 7일' },
]

/** 걸린 시간: 점검 시작 → 종료 (예: 44분, 1시간 12분) */
function duration(from: string | null, to: string) {
  if (!from) return null
  const min = Math.max(0, Math.round((new Date(to).getTime() - new Date(from).getTime()) / 60_000))
  return min < 60 ? `${min}분` : `${Math.floor(min / 60)}시간${min % 60 ? ` ${min % 60}분` : ''}`
}

export default function HistoryPage() {
  const [params, setParams] = useSearchParams()
  const [keyword, setKeyword] = useState('')
  const [result, setResult] = useState<Closure | 'all'>('all')
  // 설비 화면의 "이 설비 정비 이력"에서 오면 ?gun=으로 설비를 골라 둔다
  const [gun, setGun] = useState(params.get('gun') ?? '')
  const [days, setDays] = useState(30)
  // 기간 기준 시각: 화면을 연 시각으로 고정 (그릴 때마다 바뀌지 않게)
  const [openedAt] = useState(() => Date.now())
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['history'],
    queryFn: () => apiGet<HistoryResponse>('/history'),
  })

  const all = data?.items ?? []
  const guns = [...new Set(all.map((i) => i.event.gun_id))].sort()
  const since = openedAt - days * 86_400_000
  const k = keyword.trim().toLowerCase()
  const shown = all.filter(
    (i) =>
      (result === 'all' || i.closure.outcome === result) &&
      (!gun || i.event.gun_id === gun) &&
      new Date(i.closure.at).getTime() >= since &&
      (!k || [i.event.gun_id, i.event.trigger.rule_code].some((v) => v?.toLowerCase().includes(k))),
  )
  // 펼친 줄은 ?case=로 둔다 (정비일지에서 해결 종료하면 그 건이 펼쳐진 채로 온다)
  const openId = params.get('case')
  const filtered = result !== 'all' || gun || days !== 30 || k

  return (
    <>
      <div className={`${page()} flex grow flex-col gap-4 pt-7 pb-7`}>
        <PageHeader title="정비 이력" description="해결했거나 미해결로 저장한 정비 기록을 모았습니다.">
          <SearchBox value={keyword} onChange={setKeyword} placeholder="설비 · 코드 검색" />
        </PageHeader>

        <div className="flex items-center gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={result === f.key}
              onClick={() => setResult(f.key)}
              className={`min-h-10 shrink-0 rounded-lg border px-3.5 text-sm whitespace-nowrap ${
                result === f.key ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-ink-2'
              }`}
            >
              {f.label}
            </button>
          ))}
          <div className="grow" />
          <div className="w-40 shrink-0">
            <Select
              ariaLabel="설비"
              value={gun}
              onChange={setGun}
              options={[{ value: '', label: '설비 전체' }, ...guns.map((g) => ({ value: g, label: g }))]}
            />
          </div>
          <div className="w-40 shrink-0">
            <Select
              ariaLabel="기간"
              value={String(days)}
              onChange={(v) => setDays(Number(v))}
              options={PERIODS.map((p) => ({
                value: String(p.days),
                label: p.label,
              }))}
            />
          </div>
        </div>

        <section className={`${card} overflow-hidden`}>
          {isPending ? (
            <LoadingRows label="정비 이력을 불러오는 중" />
          ) : isError ? (
            <ErrorState onRetry={() => refetch()} />
          ) : shown.length === 0 ? (
            filtered && all.length > 0 ? (
              <EmptyState title="조건에 맞는 이력이 없습니다" note="결과 · 설비 · 기간 · 검색어를 확인해 주세요." />
            ) : (
              <EmptyState
                title="아직 정비 이력이 없습니다"
                note="작업함에서 고장을 해결 종료하거나 미해결로 저장하면 여기에 쌓입니다."
                action={
                  <Link to="/" className={`${btn} mt-2`}>
                    작업함으로
                  </Link>
                }
              />
            )
          ) : (
            <HistoryTable items={shown} openId={openId} onToggle={(id) => setParams(id === openId ? {} : { case: id }, { replace: true })} />
          )}
        </section>
      </div>
    </>
  )
}

const COLUMNS = [
  { label: '종료 (작업 시각)', width: 'w-[17%]' },
  { label: '설비', width: 'w-[11%]' },
  { label: '고장 코드', width: 'w-[12%]' },
  { label: '원인', width: 'w-[22%]' },
  { label: '조치', width: 'w-[11%]' },
  { label: '결과', width: 'w-[12%]' },
  { label: '걸린 시간', width: 'w-[11%]' },
  { label: '', width: 'w-16' },
]

function HistoryTable({ items, openId, onToggle }: { items: HistoryItem[]; openId: string | null; onToggle: (id: string) => void }) {
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-base'
  return (
    <table className="w-full table-fixed border-collapse text-left">
      <thead>
        <tr>
          {COLUMNS.map((c, i) => (
            <th
              key={i}
              scope="col"
              className={`border-b border-line bg-head px-5 py-[11px] text-xs font-medium whitespace-nowrap text-faint ${c.width}`}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {items.map((i) => {
          const on = i.case_id === openId
          const r = RESULT[i.closure.outcome]
          return (
            <Fragment key={i.case_id}>
              <tr className={`cursor-pointer ${on ? 'bg-primary-soft' : 'hover:bg-canvas'}`} onClick={() => onToggle(i.case_id)}>
                <td className={td}>
                  {/* 키보드로도 펼칠 수 있게 첫 칸을 버튼으로 */}
                  <button
                    type="button"
                    aria-expanded={on}
                    aria-controls={`history-${i.case_id}`}
                    className="text-left whitespace-nowrap"
                    onClick={(e) => (e.stopPropagation(), onToggle(i.case_id))}
                  >
                    {formatKst(i.closure.at)}
                  </button>
                </td>
                <td className={`${td} font-semibold`}>{i.event.gun_id}</td>
                <td className={td}>
                  {i.event.trigger.rule_code ? (
                    <span className="font-semibold">{i.event.trigger.rule_code}</span>
                  ) : (
                    <span className="text-xs font-medium text-sub">코드 없음</span>
                  )}
                </td>
                <td className={`${td} truncate`}>{i.cause ?? <span className="text-sub">찾지 못함</span>}</td>
                <td className={td}>{i.action_kind ?? <span className="text-sub">{EMPTY}</span>}</td>
                <td className={td}>
                  <DotTag color={r.color}>{r.label}</DotTag>
                </td>
                <td className={td}>{duration(i.started_at, i.closure.at) ?? <span className="text-sub">{EMPTY}</span>}</td>
                <td className={`${td} text-right text-sub`}>
                  <ChevronIcon size={18} className={`inline transition-transform ${on ? 'rotate-180' : ''}`} />
                </td>
              </tr>
              {on && (
                <tr id={`history-${i.case_id}`}>
                  <td colSpan={COLUMNS.length} className="border-t border-line-soft bg-head px-5 py-5">
                    <HistoryDetail item={i} />
                  </td>
                </tr>
              )}
            </Fragment>
          )
        })}
      </tbody>
    </table>
  )
}

/** 펼친 칸: 요약 한 줄 + 버튼. 긴 시간순 기록은 "기록 전체 보기"를 눌렀을 때만 */
function HistoryDetail({ item }: { item: HistoryItem }) {
  const { data } = useCaseQuery(item.case_id)
  const [showRecord, setShowRecord] = useState(false)
  const judged = data ? new Set(data.records.judgments.map((j) => j.situation_id)).size : null
  return (
    <div className="flex flex-col gap-4" aria-label="선택한 이력">
      <div className="flex flex-wrap items-end gap-x-10 gap-y-3">
        <dl className="flex flex-wrap gap-x-10 gap-y-3">
          <Fact label="원인">{item.cause ?? <span className="text-sub">찾지 못함</span>}</Fact>
          <Fact label="조치">{item.action_kind ?? EMPTY}</Fact>
          <Fact label="확인한 원인">{judged === null ? EMPTY : `${judged}개`}</Fact>
          <Fact label="작업자">{item.worker ?? EMPTY}</Fact>
        </dl>
        <div className="grow" />
        <div className="flex items-center gap-2">
          <button type="button" className={btnQuiet} aria-expanded={showRecord} onClick={() => setShowRecord((v) => !v)}>
            {showRecord ? '기록 접기' : '기록 전체 보기'}
          </button>
          {data?.records.log && (
            <Link to={`/cases/${item.case_id}/log`} className={btn}>
              정비일지 보기
            </Link>
          )}
        </div>
      </div>
      {showRecord && <div className={`${card} p-5`}>{data ? <RecordTab d={data} /> : <LoadingRows rows={2} label="기록을 불러오는 중" />}</div>}
    </div>
  )
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs font-medium text-faint">{label}</dt>
      <dd className="text-base">{children}</dd>
    </div>
  )
}
