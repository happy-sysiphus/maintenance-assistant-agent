import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router'
import { PageHeader, SearchBox } from '../components/PageHeader'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { SeverityTag, StatusTag } from '../components/Tag'
import { btn, btnPrimary, card, page } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst, formatUtc } from '../lib/time'
import type { CaseListResponse, CaseStatus, CaseSummary } from '../types/case'

// 작업함 (wireframe/boards/V2Main.dc.html, 10/23 수정)
// 위: 제목 영역 + 상태별 건수 카드(누르면 그 상태만 보기). 아래: 고장 목록.
// 상태 칸은 처리 단계만 쓴다. 코드 없이 모델 신호만 있는 고장은 고장 코드 칸에 점선 배지로 표시한다.
// 없는 값은 지어내지 않고 "—"로 둔다.

const EMPTY = '—'

// 짧은 칸은 고정 폭, 글이 긴 "현상" 칸만 남는 폭을 차지한다 (본문 최대 폭이 있어 지나치게 벌어지지 않음)
const COLUMNS: { label: string; width: string }[] = [
  { label: '경보', width: 'w-24' },
  { label: '설비', width: 'w-28' },
  { label: '고장 코드', width: 'w-40' },
  { label: '현상 (예상)', width: '' },
  { label: '발생 (데이터 시각)', width: 'w-48' },
  { label: '상태', width: 'w-32' },
  { label: '', width: 'w-56' },
]

/** 건수 카드: 한 카드가 여러 상태를 묶을 수 있다 (진행 중 = 점검 중 + 일지 작성 + 일지 승인됨) */
const GROUPS: { key: string; label: string; dot: string; statuses: CaseStatus[] }[] = [
  { key: 'new', label: '새 고장', dot: 'bg-alarm-dot', statuses: ['new'] },
  { key: 'doing', label: '진행 중', dot: 'bg-primary', statuses: ['in_progress', 'logging', 'log_approved'] },
  { key: 'unresolved', label: '미해결', dot: 'bg-warn-dot', statuses: ['unresolved'] },
  { key: 'handed_over', label: '도움 요청', dot: 'bg-warn-dot', statuses: ['handed_over'] },
]

export default function InboxPage() {
  const [keyword, setKeyword] = useState('')
  const [group, setGroup] = useState<string | null>(null)
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['cases'],
    queryFn: () => apiGet<CaseListResponse>('/cases'),
  })

  // 우선순위 순: 긴급 먼저 → 아직 아무도 안 연 고장 먼저 → 최근 발생 먼저
  const all = [...(data?.cases ?? [])].sort(byPriority)
  const picked = GROUPS.find((g) => g.key === group)
  const shown = all.filter((c) => (!picked || picked.statuses.includes(c.status)) && (!keyword.trim() || matches(c, keyword.trim())))
  const filtered = Boolean(picked || keyword.trim())

  return (
    <>
      <div className={`${page()} flex flex-col gap-5 pt-7 pb-7`}>
        <PageHeader title="작업함" description="확인이 필요한 설비 알림을 우선순위 순으로 모았습니다.">
          <SearchBox value={keyword} onChange={setKeyword} placeholder="설비 · 코드 검색" />
        </PageHeader>

        <div role="group" aria-label="상태별 건수 · 누르면 그 상태만 보기" className="grid grid-cols-4 gap-3">
          {GROUPS.map((g) => {
            const inGroup = all.filter((c) => g.statuses.includes(c.status))
            const n = inGroup.length
            // 긴급이 있을 때만 작은 빨간 건수를 붙인다 (카드 전체를 빨갛게 하지 않음)
            const urgent = inGroup.filter((c) => c.severity === 'critical').length
            const on = group === g.key
            return (
              <button
                key={g.key}
                type="button"
                aria-pressed={on}
                onClick={() => setGroup(on ? null : g.key)}
                className={`flex flex-col gap-2 rounded-xl border bg-white px-5 pt-4 pb-3.5 text-left shadow-[0_1px_2px_rgba(16,24,40,0.04)] hover:border-primary ${
                  on ? 'border-primary ring-1 ring-primary' : 'border-line'
                }`}
              >
                <span className="flex w-full items-center gap-2 text-sm font-semibold text-ink-2">
                  <span className={`size-2 rounded-full ${g.dot}`} />
                  {g.label}
                  {urgent > 0 && (
                    <span className="ml-auto inline-flex items-center gap-1.5 text-xs font-semibold text-alarm">
                      <span className="size-1.5 rounded-full bg-alarm-dot" />
                      긴급 {urgent}
                    </span>
                  )}
                </span>
                <span className={`text-[36px] leading-none font-bold tracking-[-0.02em] ${n === 0 ? 'text-dot-gray' : ''}`}>{data ? n : EMPTY}</span>
              </button>
            )
          })}
        </div>

        <section className={`${card} overflow-hidden`}>
          {isPending ? (
            <LoadingRows label="고장 목록을 불러오는 중" />
          ) : isError ? (
            <ErrorState onRetry={() => refetch()} />
          ) : shown.length === 0 ? (
            filtered ? (
              <EmptyState
                title="조건에 맞는 고장이 없습니다"
                note="설비가 모두 정상이라는 뜻은 아닙니다."
                action={
                  <button
                    type="button"
                    onClick={() => {
                      setKeyword('')
                      setGroup(null)
                    }}
                    className={`${btn} mt-2`}
                  >
                    조건 지우기
                  </button>
                }
              />
            ) : (
              <EmptyState title="처리할 고장이 없습니다" note="설비가 모두 정상이라는 뜻은 아닙니다." />
            )
          ) : (
            <table className="w-full table-fixed border-collapse text-left">
              <thead>
                <tr>
                  {COLUMNS.map((c, i) => (
                    <th
                      key={i}
                      scope="col"
                      className={`border-b border-line bg-head px-5 py-[11px] text-xs font-medium whitespace-nowrap text-sub ${c.width}`}
                    >
                      {c.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shown.map((c, i) => (
                  <CaseRow key={c.case_id} item={c} first={i === 0} />
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </>
  )
}

const STATUS_ORDER: CaseStatus[] = ['new', 'in_progress', 'handed_over', 'unresolved', 'logging', 'log_approved', 'resolved']

/** 작업함 정렬: 긴급 → 처리 단계(새 고장 먼저) → 발생 시각 최근 */
function byPriority(a: CaseSummary, b: CaseSummary) {
  const sev = (c: CaseSummary) => (c.severity === 'critical' ? 0 : 1)
  const at = (c: CaseSummary) => c.event.trigger.rule_trigger_time ?? c.event.detected_at
  return sev(a) - sev(b) || STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status) || at(b).localeCompare(at(a))
}

/** 설비 이름이나 고장 코드에 검색어가 들어 있는지 (대소문자 무시) */
function matches(c: CaseSummary, keyword: string) {
  const k = keyword.toLowerCase()
  return [c.event.gun_id, c.event.trigger.rule_code].some((v) => v?.toLowerCase().includes(k))
}

/** 줄 끝 버튼: 지금 할 일을 동사로 */
function action(item: CaseSummary): { to: string; label: string } {
  const base = `/cases/${item.case_id}`
  switch (item.status) {
    case 'new':
      return { to: base, label: '점검 시작' }
    case 'logging':
    case 'log_approved':
      // 일지를 쓰던 케이스는 정비일지로 바로 연다
      return { to: `${base}/log`, label: '일지 계속' }
    default:
      return { to: base, label: '점검 계속' }
  }
}

/** first: 우선순위 맨 위 줄. 그 줄이 새 고장이면 버튼을 파랗게 강조한다 */
function CaseRow({ item, first }: { item: CaseSummary; first: boolean }) {
  const { event } = item
  // 발생 시각: 종료 코드가 뜬 시각이 있으면 그것, 없으면 ML 판정 시각
  const occurredAt = event.trigger.rule_trigger_time ?? event.detected_at
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-base'
  const next = action(item)
  const top = first && item.status === 'new'

  return (
    <tr>
      <td className={td}>
        <SeverityTag severity={item.severity} />
      </td>
      <td className={`${td} font-semibold`}>{event.gun_id}</td>
      <td className={td}>
        {event.trigger.rule_code ? (
          <span className="font-semibold">{event.trigger.rule_code}</span>
        ) : (
          // 점선 = 아직 확정 아님. 제어기 코드 없이 모델 신호만 있어 실제 이상인지 먼저 확인해야 한다
          <span
            className="inline-flex items-center rounded-md border border-dashed border-sub px-2 py-0.5 text-xs font-medium text-ink-2"
            title="고장 코드 없이 모델 신호만 있습니다. 실제 이상인지 먼저 확인하세요."
          >
            모델 신호만
          </span>
        )}
      </td>
      <td className={td} title={event.fault_class ? `데이터셋 클래스 ${event.fault_class.code} 기준 추정` : undefined}>
        {event.fault_class?.name_ko ?? <span className="text-sub">{EMPTY}</span>}
      </td>
      <td className={td} title={formatUtc(occurredAt) ?? undefined}>
        {formatKst(occurredAt) ?? <span className="text-sub">{EMPTY}</span>}
      </td>
      <td className={td}>
        <StatusTag status={item.status} />
      </td>
      <td className={td}>
        <div className="flex items-center justify-end gap-2.5">
          <Link to={next.to} className={`${top ? btnPrimary : btn} w-[112px]`}>
            {next.label}
          </Link>
        </div>
      </td>
    </tr>
  )
}
