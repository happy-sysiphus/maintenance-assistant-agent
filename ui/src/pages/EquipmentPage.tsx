import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router'
import { SearchIcon } from '../components/Icons'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { DotTag, StatusTag } from '../components/Tag'
import { TopBar } from '../components/TopBar'
import { btn, card, input } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst } from '../lib/time'
import type { GunStatus, GunsResponse } from '../types/case'

// 설비 (wireframe/boards/V2Equip.dc.html): "지금 이 설비를 판단할 수 있나"를 한 줄씩.
// 값은 ML GET /guns 그대로 + 열린 고장. 고장 통계 · 평균 고장 간격은 근거 데이터가 없어 넣지 않는다.
// 설비 단위 "판단 보류" 값은 ML /guns에 없어서 칩을 두지 않는다 (판단 보류는 고장 알림마다 온다).

const EMPTY = '—'
const NORM: Record<GunStatus['gun_norm'], string> = {
  gun: '이 설비 기준',
  warming_up: '기준 수집 중',
  global: '공통 기준',
}

type Filter = 'all' | 'open' | 'warming_up'

export default function EquipmentPage() {
  const [keyword, setKeyword] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ['guns'], queryFn: () => apiGet<GunsResponse>('/guns') })

  const all = data?.guns ?? []
  const k = keyword.trim().toLowerCase()
  const match = (g: GunStatus) => (filter === 'all' || (filter === 'open' ? g.open_cases.length > 0 : g.gun_norm === 'warming_up')) && (!k || g.gun_id.toLowerCase().includes(k))
  const shown = all.filter(match)
  const selected = shown.find((g) => g.gun_id === selectedId) ?? shown[0] ?? null
  const filters: { key: Filter; label: string; n: number }[] = [
    { key: 'all', label: '전체', n: all.length },
    { key: 'open', label: '열린 고장 있음', n: all.filter((g) => g.open_cases.length > 0).length },
    { key: 'warming_up', label: '기준 수집 중', n: all.filter((g) => g.gun_norm === 'warming_up').length },
  ]

  return (
    <>
      <TopBar>
        <h1 className="text-[19px] font-bold tracking-[-0.02em]">설비</h1>
        <div className="grow" />
        <label className="relative w-[300px]">
          <span className="sr-only">설비 검색</span>
          <span className="pointer-events-none absolute top-[13px] left-3.5 text-sub">
            <SearchIcon size={18} />
          </span>
          <input type="search" value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="설비 검색" className={`${input} pl-[42px]`} />
        </label>
      </TopBar>

      <div className="flex grow flex-col gap-4 px-8 pt-6 pb-7">
        <div className="flex items-center gap-2">
          {filters.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              onClick={() => setFilter(f.key)}
              className={`min-h-10 shrink-0 rounded-lg border px-3.5 text-sm whitespace-nowrap ${
                filter === f.key ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-[#4e5968]'
              }`}
            >
              {f.label} {data ? f.n : ''}
            </button>
          ))}
        </div>

        <div className="grid grow grid-cols-[minmax(0,1fr)_380px] items-start gap-4">
          <section className={`${card} overflow-hidden`}>
            {isPending ? (
              <LoadingRows label="설비 목록을 불러오는 중" />
            ) : isError ? (
              <ErrorState onRetry={() => refetch()} />
            ) : shown.length === 0 ? (
              <EmptyState title="조건에 맞는 설비가 없습니다" />
            ) : (
              <GunTable guns={shown} selectedId={selected?.gun_id ?? null} onSelect={setSelectedId} />
            )}
          </section>
          {selected && <GunDetail gun={selected} />}
        </div>
      </div>
    </>
  )
}

const COLUMNS = [
  { label: '설비', width: 'w-[14%]' },
  { label: '상태', width: 'w-[17%]' },
  { label: '최근 고장 (데이터 시각)', width: 'w-[27%]' },
  { label: '판단 기준', width: 'w-[18%]' },
  { label: '비고', width: 'w-[24%]' },
]

function OpenTag({ gun }: { gun: GunStatus }) {
  const n = gun.open_cases.length
  if (n === 0) return <DotTag color="gray">열린 고장 없음</DotTag>
  const critical = gun.open_cases.some((c) => c.severity === 'critical')
  return (
    <span className={`inline-flex items-center gap-[7px] text-[13.5px] font-medium whitespace-nowrap ${critical ? 'text-alarm' : 'text-warn'}`}>
      <span className={`size-[7px] rounded-full ${critical ? 'bg-alarm-dot' : 'bg-warn-dot'}`} />
      열린 고장 {n}
    </span>
  )
}

function GunTable({ guns, selectedId, onSelect }: { guns: GunStatus[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-[15px]'
  return (
    <table className="w-full table-fixed border-collapse text-left">
      <thead>
        <tr>
          {COLUMNS.map((c) => (
            <th key={c.label} scope="col" className={`border-b border-line bg-head px-5 py-[11px] text-[12.5px] font-medium whitespace-nowrap text-faint ${c.width}`}>
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {guns.map((g) => {
          const on = g.gun_id === selectedId
          const latest = g.open_cases[0]
          return (
            <tr key={g.gun_id} className={`cursor-pointer ${on ? 'bg-[#f3f6ff]' : 'hover:bg-canvas'}`} onClick={() => onSelect(g.gun_id)}>
              <td className={`${td} font-semibold`}>
                <button type="button" aria-pressed={on} onClick={() => onSelect(g.gun_id)}>
                  {g.gun_id}
                </button>
              </td>
              <td className={td}>
                <OpenTag gun={g} />
              </td>
              <td className={td}>
                {latest ? (
                  <>
                    <span className="font-semibold">{latest.event.trigger.rule_code ?? '코드 없음'}</span>{' '}
                    <span className="text-[13.5px] text-sub">{formatKst(latest.event.trigger.rule_trigger_time ?? latest.event.detected_at)}</span>
                  </>
                ) : (
                  <span className="text-sub">{EMPTY}</span>
                )}
              </td>
              <td className={td}>{NORM[g.gun_norm]}</td>
              <td className={`${td} truncate text-[13.5px] text-sub`}>
                {g.gun_norm === 'warming_up' ? '그동안은 공통 기준으로만 판단' : g.drift_warning.join(' · ')}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function GunDetail({ gun }: { gun: GunStatus }) {
  return (
    <aside className={`${card} flex flex-col gap-3.5 p-5`} aria-label={`${gun.gun_id} 상세`}>
      <div className="flex items-center gap-3">
        <b className="grow text-[17px] font-semibold">{gun.gun_id}</b>
        <OpenTag gun={gun} />
      </div>
      <dl className="grid grid-cols-[88px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-[9px] text-[14.5px]">
        <dt className="text-[13.5px] text-faint">판단 기준</dt>
        <dd>
          {NORM[gun.gun_norm]}
          {gun.gun_norm === 'warming_up' && <span className="text-[13.5px] text-sub"> · 수집 {gun.warmup_rows.toLocaleString()}행</span>}
        </dd>
        <dt className="text-[13.5px] text-faint">마지막 점수</dt>
        <dd>
          {gun.last_score === null ? EMPTY : gun.last_score.toFixed(2)}
          {gun.gun_threshold !== null && <span className="text-[13.5px] text-sub"> · 기준선 {gun.gun_threshold.toFixed(2)}</span>}
        </dd>
        <dt className="text-[13.5px] text-faint">드리프트</dt>
        <dd>{gun.drift_warning.length ? gun.drift_warning.join(' · ') : <span className="text-sub">경고 없음</span>}</dd>
      </dl>
      <div className="h-px bg-line-soft" />
      <span className="text-[13px] font-medium text-sub">열린 고장</span>
      {gun.open_cases.length === 0 && <p className="text-sm text-sub">없습니다.</p>}
      {gun.open_cases.map((c) => (
        <div key={c.case_id} className="flex items-center gap-3 rounded-lg border border-line px-3.5 py-3">
          <div className="min-w-0 grow">
            <b className="font-semibold">{c.event.trigger.rule_code ?? '코드 없음'}</b>
            <div className="flex items-center gap-2 text-[13.5px] text-sub">
              {formatKst(c.event.trigger.rule_trigger_time ?? c.event.detected_at)} · <StatusTag status={c.status} />
            </div>
          </div>
          <Link to={`/cases/${c.case_id}`} className={`${btn} min-h-9 px-3 text-[13.5px]`}>
            열기
          </Link>
        </div>
      ))}
      <Link to={`/history?gun=${gun.gun_id}`} className={`${btn} mt-1 self-start`}>
        이 설비 정비 이력
      </Link>
    </aside>
  )
}
