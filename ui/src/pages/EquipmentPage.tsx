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
// 줄을 누르면 그 줄 바로 아래로 설비 상태와 처리할 고장이 펼쳐진다 (한 번에 한 줄).
// 값은 ML GET /guns 그대로 + 처리할 고장(해결 종료 · 오알람 전인 케이스). 고장 통계 · 평균 고장 간격은 근거 데이터가 없어 넣지 않는다.
// 설비 단위 "판단 보류" 값은 ML /guns에 없어서 칩을 두지 않는다 (판단 보류는 고장 알림마다 온다).

const EMPTY = '—'
// 설비 상태를 쉬운 말 하나로 (ML 값: gun_norm · last_score · gun_threshold · drift_warning)
// - 데이터 수집 중: 켜고 처음 몇 시간은 이 설비의 평소값을 모으는 중이라 알림을 덜 믿어야 한다
// - 이상 신호: 마지막 점수가 이 설비 기준선 이상
// - 알림이 잦음: ML 드리프트 경고 (정상인데 알람이 며칠째 잦음 → 기준 재점검 필요)
type GunState = { label: string; note: string; color: 'gray' | 'red' | 'amber' }
function gunState(g: GunStatus): GunState {
  if (g.gun_norm === 'warming_up') return { label: '데이터 수집 중', note: '데이터를 수집 중입니다', color: 'amber' }
  if (g.last_score !== null && g.gun_threshold !== null && g.last_score >= g.gun_threshold)
    return { label: '이상 신호', note: '평소 범위를 벗어났어요', color: 'red' }
  if (g.drift_warning.length) return { label: '알림이 잦음', note: '기준을 다시 잡아야 할 수 있어요', color: 'amber' }
  return { label: '정상', note: '평소 범위 안이에요', color: 'gray' }
}
const STATE_DOT = { gray: 'bg-dot-gray', red: 'bg-alarm-dot', amber: 'bg-warn-dot' }
const STATE_TEXT = { gray: 'text-ink', red: 'text-alarm', amber: 'text-warn' }

function StateTag({ gun }: { gun: GunStatus }) {
  const st = gunState(gun)
  const score =
    gun.last_score !== null && gun.gun_threshold !== null ? ` (점수 ${gun.last_score.toFixed(2)} / 기준 ${gun.gun_threshold.toFixed(2)})` : ''
  // 마우스를 올리면 짧은 설명과 점수를 보여준다 (화면에는 상태 이름만)
  const title = `${st.note}${score}`
  return (
    <span className={`inline-flex items-center gap-[7px] text-sm font-medium whitespace-nowrap ${STATE_TEXT[st.color]}`} title={title}>
      <span className={`size-[7px] rounded-full ${STATE_DOT[st.color]}`} />
      {st.label}
    </span>
  )
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
      label: '처리할 고장',
      n: all.filter((g) => g.open_cases.length > 0).length,
    },
    {
      key: 'warming_up',
      label: '데이터 수집 중',
      n: all.filter((g) => g.gun_norm === 'warming_up').length,
    },
  ]

  return (
    <>
      <div className={`${page()} flex grow flex-col gap-4 pt-7 pb-7`}>
        <PageHeader title="설비 현황" description="설비마다 지금 상태와 처리할 고장을 보여줍니다.">
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
  { label: '처리할 고장', width: 'w-[20%]' },
  { label: '최근 고장 (데이터 시각)', width: '' },
  { label: '설비 상태', width: 'w-[20%]' },
  { label: '', width: 'w-16' },
]

function OpenTag({ gun }: { gun: GunStatus }) {
  const n = gun.open_cases.length
  if (n === 0) return <DotTag color="gray">없음</DotTag>
  const critical = gun.open_cases.some((c) => c.severity === 'critical')
  return (
    <span className={`inline-flex items-center gap-[7px] text-sm font-medium whitespace-nowrap ${critical ? 'text-alarm' : 'text-warn'}`}>
      <span className={`size-[7px] rounded-full ${critical ? 'bg-alarm-dot' : 'bg-warn-dot'}`} />
      {n}건
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
                <td className={td}>
                  <StateTag gun={g} />
                </td>
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

/** 펼친 칸: 처리할 고장 목록 + 이 설비 정비 이력 링크 */
function GunDetail({ gun }: { gun: GunStatus }) {
  return (
    <div className="flex animate-enter flex-col gap-2.5" aria-label={`${gun.gun_id} 상세`}>
      {/* 설비 상태는 위 줄에 이미 있어서 펼친 칸에는 처리할 고장 목록과 정비 이력 링크만 둔다 */}
      <div className="flex items-center">
        <span className="grow text-xs font-medium text-sub">처리할 고장 {gun.open_cases.length}</span>
        <Link
          to={`/history?gun=${gun.gun_id}`}
          className="inline-flex min-h-8 items-center px-1 text-sm font-medium text-primary-ink hover:underline"
        >
          이 설비 정비 이력 ›
        </Link>
      </div>
      {gun.open_cases.length === 0 ? (
        <p className="text-sm text-sub">없습니다.</p>
      ) : (
        // 한 줄 = 고장 한 건 (작업함과 같은 순서: 코드 · 현상 · 발생 · 상태 · 열기), 위에서 아래로
        <ul className="overflow-hidden rounded-lg border border-line bg-white">
          {gun.open_cases.map((c) => (
            <li key={c.case_id} className="flex items-center gap-4 border-t border-line-soft px-4 py-2.5 first:border-t-0">
              <b className="w-20 shrink-0 font-semibold">
                {c.event.trigger.rule_code ?? <span className="text-sm font-medium text-sub">코드 없음</span>}
              </b>
              <span className="min-w-0 grow truncate text-sm">{c.event.fault_class?.name_ko ?? <span className="text-sub">{EMPTY}</span>}</span>
              <span className="shrink-0 text-sm text-sub">{formatKst(c.event.trigger.rule_trigger_time ?? c.event.detected_at)}</span>
              <span className="w-24 shrink-0">
                <StatusTag status={c.status} />
              </span>
              <Link to={`/cases/${c.case_id}`} className={`${btn} min-h-9 shrink-0 px-3 text-sm`}>
                열기
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
