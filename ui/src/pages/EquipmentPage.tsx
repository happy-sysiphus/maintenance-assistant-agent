import { useQuery } from '@tanstack/react-query'
import { Fragment, useState } from 'react'
import { Link } from 'react-router'
import { ChevronIcon } from '../components/Icons'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { DotTag, StatusTag } from '../components/Tag'
import { PageHeader, SearchBox } from '../components/PageHeader'
import { btn, card, page } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst } from '../lib/time'
import type { GunStatus, GunsResponse } from '../types/case'

// 설비 (wireframe/boards/V2Equip.dc.html): "지금 이 설비를 판단할 수 있나"를 한 줄씩.
// 줄을 누르면 그 줄 바로 아래로 요약과 열린 고장이 펼쳐진다 (한 번에 한 줄).
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
  const [openId, setOpenId] = useState<string | null>(null)
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['guns'],
    queryFn: () => apiGet<GunsResponse>('/guns'),
  })

  const all = data?.guns ?? []
  const k = keyword.trim().toLowerCase()
  const match = (g: GunStatus) =>
    (filter === 'all' || (filter === 'open' ? g.open_cases.length > 0 : g.gun_norm === 'warming_up')) && (!k || g.gun_id.toLowerCase().includes(k))
  const shown = all.filter(match)
  const filters: { key: Filter; label: string; n: number }[] = [
    { key: 'all', label: '전체', n: all.length },
    {
      key: 'open',
      label: '열린 고장 있음',
      n: all.filter((g) => g.open_cases.length > 0).length,
    },
    {
      key: 'warming_up',
      label: '기준 수집 중',
      n: all.filter((g) => g.gun_norm === 'warming_up').length,
    },
  ]

  return (
    <>
      <div className={`${page()} flex grow flex-col gap-4 pt-7 pb-7`}>
        <PageHeader title="설비" description="설비마다 지금 판단할 수 있는지와 열린 고장을 보여줍니다.">
          <SearchBox value={keyword} onChange={setKeyword} placeholder="설비 검색" label="설비 검색" />
        </PageHeader>

        <div className="flex items-center gap-2">
          {filters.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              onClick={() => setFilter(f.key)}
              className={`min-h-10 shrink-0 rounded-lg border px-3.5 text-sm whitespace-nowrap ${
                filter === f.key ? 'border-ink bg-ink font-semibold text-white' : 'border-field bg-white font-medium text-ink-2'
              }`}
            >
              {f.label} {data ? f.n : ''}
            </button>
          ))}
        </div>

        <section className={`${card} overflow-x-auto`}>
          {isPending ? (
            <LoadingRows label="설비 목록을 불러오는 중" />
          ) : isError ? (
            <ErrorState onRetry={() => refetch()} />
          ) : shown.length === 0 ? (
            <EmptyState title="조건에 맞는 설비가 없습니다" />
          ) : (
            <GunTable guns={shown} openId={openId} onToggle={(id) => setOpenId((cur) => (cur === id ? null : id))} />
          )}
        </section>
      </div>
    </>
  )
}

const COLUMNS = [
  { label: '설비', width: 'w-[18%]' },
  { label: '상태', width: 'w-[20%]' },
  { label: '최근 고장 (데이터 시각)', width: '' },
  { label: '판단 기준', width: 'w-[20%]' },
  { label: '', width: 'w-16' },
]

function OpenTag({ gun }: { gun: GunStatus }) {
  const n = gun.open_cases.length
  if (n === 0) return <DotTag color="gray">열린 고장 없음</DotTag>
  const critical = gun.open_cases.some((c) => c.severity === 'critical')
  return (
    <span className={`inline-flex items-center gap-[7px] text-sm font-medium whitespace-nowrap ${critical ? 'text-alarm' : 'text-warn'}`}>
      <span className={`size-[7px] rounded-full ${critical ? 'bg-alarm-dot' : 'bg-warn-dot'}`} />
      열린 고장 {n}
    </span>
  )
}

function GunTable({ guns, openId, onToggle }: { guns: GunStatus[]; openId: string | null; onToggle: (id: string) => void }) {
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-base'
  return (
    <table className="w-full min-w-[760px] table-fixed border-collapse text-left">
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
        {guns.map((g) => {
          const on = g.gun_id === openId
          const latest = g.open_cases[0]
          return (
            <Fragment key={g.gun_id}>
              <tr className={`cursor-pointer ${on ? 'bg-primary-soft' : 'hover:bg-canvas'}`} onClick={() => onToggle(g.gun_id)}>
                <td className={`${td} font-semibold`}>
                  {/* 키보드로도 펼칠 수 있게 첫 칸을 버튼으로 */}
                  <button
                    type="button"
                    aria-expanded={on}
                    aria-controls={`gun-${g.gun_id}`}
                    onClick={(e) => (e.stopPropagation(), onToggle(g.gun_id))}
                  >
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
                      <span className="text-sm text-sub">{formatKst(latest.event.trigger.rule_trigger_time ?? latest.event.detected_at)}</span>
                    </>
                  ) : (
                    <span className="text-sub">{EMPTY}</span>
                  )}
                </td>
                <td className={td}>{NORM[g.gun_norm]}</td>
                <td className={`${td} text-right text-sub`}>
                  <ChevronIcon size={18} className={`inline transition-transform ${on ? 'rotate-180' : ''}`} />
                </td>
              </tr>
              {on && (
                <tr id={`gun-${g.gun_id}`}>
                  <td colSpan={COLUMNS.length} className="border-t border-line-soft bg-head px-5 py-5">
                    <GunDetail gun={g} />
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

/** 펼친 칸: 왼쪽 요약, 오른쪽 열린 고장(가로로 나열), 아래 정비 이력 버튼 */
function GunDetail({ gun }: { gun: GunStatus }) {
  return (
    <div className="grid animate-enter grid-cols-[300px_minmax(0,1fr)] gap-6" aria-label={`${gun.gun_id} 상세`}>
      <div className="flex flex-col gap-4">
        <dl className="grid grid-cols-[88px_minmax(0,1fr)] items-baseline gap-x-3 gap-y-[9px] text-sm">
          <dt className="text-sm text-faint">판단 기준</dt>
          <dd>
            {NORM[gun.gun_norm]}
            {gun.gun_norm === 'warming_up' && (
              <span className="text-sm text-sub"> · 수집 {gun.warmup_rows.toLocaleString()}행 · 그동안은 공통 기준으로만 판단</span>
            )}
          </dd>
          <dt className="text-sm text-faint">마지막 점수</dt>
          <dd>
            {gun.last_score === null ? EMPTY : gun.last_score.toFixed(2)}
            {gun.gun_threshold !== null && <span className="text-sm text-sub"> · 기준선 {gun.gun_threshold.toFixed(2)}</span>}
          </dd>
          <dt className="text-sm text-faint">드리프트</dt>
          <dd>{gun.drift_warning.length ? gun.drift_warning.join(' · ') : <span className="text-sub">경고 없음</span>}</dd>
        </dl>
        <Link to={`/history?gun=${gun.gun_id}`} className={`${btn} self-start`}>
          이 설비 정비 이력
        </Link>
      </div>
      <div className="flex min-w-0 flex-col gap-2.5">
        <span className="text-xs font-medium text-sub">열린 고장 {gun.open_cases.length}</span>
        {gun.open_cases.length === 0 ? (
          <p className="text-sm text-sub">없습니다.</p>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-2.5">
            {gun.open_cases.map((c) => (
              <div key={c.case_id} className="flex items-center gap-3 rounded-lg border border-line bg-white px-3.5 py-3">
                <div className="min-w-0 grow">
                  <b className="font-semibold">{c.event.trigger.rule_code ?? '코드 없음'}</b>
                  <div className="flex flex-wrap items-center gap-x-2 text-sm text-sub">
                    {formatKst(c.event.trigger.rule_trigger_time ?? c.event.detected_at)} <StatusTag status={c.status} />
                  </div>
                </div>
                <Link to={`/cases/${c.case_id}`} className={`${btn} min-h-9 px-3 text-sm`}>
                  열기
                </Link>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
