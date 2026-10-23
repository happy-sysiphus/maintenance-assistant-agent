import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'
import { Select } from '../components/Select'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { DotTag } from '../components/Tag'
import { PageHeader, SearchBox } from '../components/PageHeader'
import { btn, card, page } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst } from '../lib/time'
import type { Closure, HistoryItem, HistoryResponse } from '../types/case'

// 정비 이력 (wireframe/boards/V2History.dc.html): 해결 종료했거나 미해결로 저장한 고장 (읽기 전용).
// 줄을 누르면 그 건으로 바로 간다 (해결 종료 = 정비일지, 나머지 = 상황 화면). 펼치는 칸은 두지 않는다 (10/23).
// 종료 시각은 작업 시각 기준이다. 데이터 시각(고장 발생)과 섞지 않는다.

const EMPTY = '—'
const RESULT: Record<Closure, { label: string; color: 'gray' | 'amber' }> = {
  resolved: { label: '해결', color: 'gray' },
  unresolved: { label: '미해결', color: 'amber' },
  false_alarm: { label: '오알람', color: 'gray' },
  handed_over: { label: '도움 요청', color: 'amber' },
}
const FILTERS: { key: Closure | 'all'; label: string }[] = [
  { key: 'all', label: '전체' },
  { key: 'resolved', label: '해결' },
  { key: 'unresolved', label: '미해결' },
  { key: 'false_alarm', label: '오알람' },
  { key: 'handed_over', label: '도움 요청' },
]
const PERIODS = [
  { days: 30, label: '최근 30일' },
  { days: 7, label: '최근 7일' },
]

export default function HistoryPage() {
  const [params] = useSearchParams()
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

        <section className={`${card} overflow-x-auto`}>
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
            <HistoryTable items={shown} />
          )}
        </section>
      </div>
    </>
  )
}

// 짧은 칸은 고정 비율, 원인 칸이 남는 폭을 쓴다. 맨 끝은 버튼 칸 (작업함과 같은 모양)
const COLUMNS = [
  { label: '종료 (작업 시각)', width: 'w-[16%]' },
  { label: '설비', width: 'w-[10%]' },
  { label: '고장 코드', width: 'w-[11%]' },
  { label: '원인', width: '' },
  { label: '조치', width: 'w-[9%]' },
  { label: '결과', width: 'w-[10%]' },
  { label: '', width: 'w-40' },
]

function HistoryTable({ items }: { items: HistoryItem[] }) {
  // 줄을 누르면 그 건으로 바로 간다: 해결 종료한 건은 정비일지(오른쪽에 기록도 있음), 나머지는 상황 화면
  const navigate = useNavigate()
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-base'
  const target = (i: HistoryItem) => (i.closure.outcome === 'resolved' ? `/cases/${i.case_id}/log` : `/cases/${i.case_id}`)
  return (
    <table className="w-full min-w-[960px] table-fixed border-collapse text-left">
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
          const r = RESULT[i.closure.outcome]
          return (
            <tr key={i.case_id} className="group cursor-pointer transition-colors hover:bg-canvas" onClick={() => navigate(target(i))}>
              <td className={`${td} whitespace-nowrap`}>{formatKst(i.closure.at)}</td>
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
              <td className={`${td} text-right`}>
                {/* 버튼은 흰색, 줄에 마우스를 올리면 파랗게 (작업함과 같은 규칙) */}
                <Link
                  to={target(i)}
                  onClick={(e) => e.stopPropagation()}
                  className={`${btn} min-h-9 w-[112px] px-3 group-hover:border-primary! group-hover:bg-primary! group-hover:font-semibold group-hover:text-white! focus-visible:border-primary! focus-visible:bg-primary! focus-visible:text-white!`}
                >
                  {i.closure.outcome === 'resolved' ? '정비일지' : '상황 보기'}
                </Link>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
