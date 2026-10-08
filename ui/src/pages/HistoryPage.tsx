import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { SearchIcon } from '../components/Icons'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { DotTag } from '../components/Tag'
import { TopBar } from '../components/TopBar'
import { btn, card, input } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst } from '../lib/time'
import { useCaseQuery } from '../lib/useCase'
import { RecordTab } from './case/parts'
import type { Closure, HistoryItem, HistoryResponse } from '../types/case'

// 정비 이력 (wireframe/boards/V2History.dc.html): 해결 종료했거나 미해결로 저장한 고장 (읽기 전용).
// 줄을 누르면 오른쪽에 요약과 시간순 기록, 정비일지 보기.
// 종료 · 걸린 시간은 작업 시각 기준이다. 데이터 시각(고장 발생)과 섞지 않는다.

const EMPTY = '—'
const RESULT: Record<Closure, { label: string; color: 'green' | 'amber' }> = {
  resolved: { label: '해결', color: 'green' },
  unresolved: { label: '미해결', color: 'amber' },
}
const FILTERS: { key: Closure | 'all'; label: string }[] = [
  { key: 'all', label: '전체' },
  { key: 'resolved', label: '해결' },
  { key: 'unresolved', label: '미해결' },
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
  const [gun, setGun] = useState('')
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
  const selectedId = params.get('case') ?? shown[0]?.case_id ?? null
  const selected = shown.find((i) => i.case_id === selectedId) ?? null
  const filtered = result !== 'all' || gun || days !== 30 || k

  return (
    <>
      <TopBar>
        <h1 className="text-[19px] font-bold tracking-[-0.02em]">정비 이력</h1>
        <div className="grow" />
        <label className="relative w-[300px]">
          <span className="sr-only">검색</span>
          <span className="pointer-events-none absolute top-[13px] left-3.5 text-sub">
            <SearchIcon size={18} />
          </span>
          <input type="search" value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="설비 · 코드 검색" className={`${input} pl-[42px]`} />
        </label>
      </TopBar>

      <div className="flex grow flex-col gap-4 px-8 pt-6 pb-7">
        <div className="flex items-center gap-2">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={result === f.key}
              onClick={() => setResult(f.key)}
              className={`min-h-10 shrink-0 rounded-lg border px-3.5 text-sm whitespace-nowrap ${
                result === f.key ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-[#4e5968]'
              }`}
            >
              {f.label}
            </button>
          ))}
          <div className="grow" />
          <div className="w-40 shrink-0">
            <select aria-label="설비" className={input} value={gun} onChange={(e) => setGun(e.target.value)}>
              <option value="">설비 전체</option>
              {guns.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </select>
          </div>
          <div className="w-40 shrink-0">
            <select aria-label="기간" className={input} value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {PERIODS.map((p) => (
                <option key={p.days} value={p.days}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grow grid-cols-[minmax(0,1fr)_380px] items-start gap-4">
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
              <HistoryTable items={shown} selectedId={selected?.case_id ?? null} onSelect={(id) => setParams({ case: id }, { replace: true })} />
            )}
          </section>
          {selected && <HistoryDetail item={selected} />}
        </div>
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
  { label: '걸린 시간', width: 'w-[15%]' },
]

function HistoryTable({ items, selectedId, onSelect }: { items: HistoryItem[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-[15px]'
  return (
    <table className="w-full table-fixed border-collapse text-left">
      <thead>
        <tr>
          {COLUMNS.map((c) => (
            <th
              key={c.label}
              scope="col"
              className={`border-b border-line bg-head px-5 py-[11px] text-[12.5px] font-medium whitespace-nowrap text-faint ${c.width}`}
            >
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {items.map((i) => {
          const on = i.case_id === selectedId
          const r = RESULT[i.closure.outcome]
          return (
            <tr key={i.case_id} className={`cursor-pointer ${on ? 'bg-[#f3f6ff]' : 'hover:bg-canvas'}`} onClick={() => onSelect(i.case_id)}>
              <td className={td}>
                {/* 키보드로도 고를 수 있게 첫 칸을 버튼으로 */}
                <button type="button" aria-pressed={on} className="text-left whitespace-nowrap" onClick={() => onSelect(i.case_id)}>
                  {formatKst(i.closure.at)}
                </button>
              </td>
              <td className={`${td} font-semibold`}>{i.event.gun_id}</td>
              <td className={td}>
                {i.event.trigger.rule_code ? (
                  <span className="font-semibold">{i.event.trigger.rule_code}</span>
                ) : (
                  <span className="text-[13px] font-medium text-sub">코드 없음</span>
                )}
              </td>
              <td className={`${td} truncate`}>{i.cause ?? <span className="text-sub">찾지 못함</span>}</td>
              <td className={td}>{i.action_kind ?? <span className="text-sub">{EMPTY}</span>}</td>
              <td className={td}>
                <DotTag color={r.color}>{r.label}</DotTag>
              </td>
              <td className={td}>{duration(i.started_at, i.closure.at) ?? <span className="text-sub">{EMPTY}</span>}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function HistoryDetail({ item }: { item: HistoryItem }) {
  const { data } = useCaseQuery(item.case_id)
  const r = RESULT[item.closure.outcome]
  const judged = data ? new Set(data.records.judgments.map((j) => j.situation_id)).size : null
  const code = item.event.trigger.rule_code ?? '코드 없음'
  return (
    <aside className={`${card} flex flex-col gap-3.5 p-5`} aria-label="선택한 이력">
      <div className="flex items-center gap-3">
        <b className="grow text-base font-semibold">
          {item.event.gun_id} · {code}
        </b>
        <DotTag color={r.color}>{r.label}</DotTag>
      </div>
      <dl className="grid grid-cols-[88px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-[9px] text-[14.5px]">
        <dt className="text-[13.5px] text-faint">원인</dt>
        <dd>{item.cause ?? <span className="text-sub">찾지 못함</span>}</dd>
        <dt className="text-[13.5px] text-faint">조치</dt>
        <dd>{item.action_kind ?? EMPTY}</dd>
        <dt className="text-[13.5px] text-faint">확인한 원인</dt>
        <dd>{judged === null ? EMPTY : `${judged}개`}</dd>
        <dt className="text-[13.5px] text-faint">작업자</dt>
        <dd>{item.worker ?? EMPTY}</dd>
      </dl>
      <div className="h-px bg-line-soft" />
      {data ? <RecordTab d={data} /> : <LoadingRows rows={2} label="기록을 불러오는 중" />}
      {data?.records.log && (
        <Link to={`/cases/${item.case_id}/log`} className={`${btn} self-start`}>
          정비일지 보기
        </Link>
      )}
    </aside>
  )
}
