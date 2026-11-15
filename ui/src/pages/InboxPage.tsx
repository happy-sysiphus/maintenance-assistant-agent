import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router'
import { PageHeader, SearchBox } from '../components/PageHeader'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { SeverityTag, StatusTag } from '../components/Tag'
import { btn, card, page } from '../components/ui'
import { apiGet } from '../lib/api'
import { formatKst, formatUtc } from '../lib/time'
import { isEarly, type CaseListResponse, type CaseStatus, type CaseSummary } from '../types/case'

// 작업함 (wireframe/boards/V2Main.dc.html, 10/23 수정)
// 위: 제목 영역 + 상태별 건수 카드(누르면 그 상태만 보기). 아래: 고장 목록.
// 상태 칸은 처리 단계만 쓴다. 코드 없이 모델 신호만 있는 고장은 고장 코드 칸에 점선 배지로 표시한다.
// 없는 값은 지어내지 않고 "—"로 둔다.

const EMPTY = '—'

// 짧은 칸은 고정 폭, 글이 긴 "현상" 칸만 남는 폭을 차지한다 (본문 최대 폭이 있어 지나치게 벌어지지 않음).
// 창이 좁으면 표에 최소 폭을 두고 가로로 넘겨 본다 (글자가 한 자씩 줄바꿈되지 않게)
const COLUMNS: { label: string; width: string }[] = [
  { label: '경보', width: 'w-24' },
  { label: '설비', width: 'w-28' },
  { label: '고장 코드', width: 'w-40' },
  { label: '현상 (예상)', width: '' },
  { label: '발생 (데이터 시각)', width: 'w-48' },
  { label: '상태', width: 'w-32' },
  { label: '', width: 'w-56' },
]

/** 건수 카드: 한 카드가 여러 상태를 묶을 수 있다 (진행 중 = 점검 중 + 일지 작성 + 일지 승인됨).
 *  미해결 · 도움 요청은 1건 이상일 때만 파란 점, 없으면 회색. 주황은 조기 경보(주의) 탭과 겹쳐서 쓰지 않는다 (11/12) */
const GROUPS: { key: string; label: string; dot: string; onlyWhenAny?: boolean; statuses: CaseStatus[] }[] = [
  { key: 'new', label: '새 고장', dot: 'bg-alarm-dot', statuses: ['new'] },
  { key: 'doing', label: '진행 중', dot: 'bg-primary', statuses: ['in_progress', 'logging', 'log_approved'] },
  { key: 'unresolved', label: '미해결', dot: 'bg-primary', onlyWhenAny: true, statuses: ['unresolved'] },
  { key: 'handed_over', label: '도움 요청', dot: 'bg-primary', onlyWhenAny: true, statuses: ['handed_over'] },
]

export default function InboxPage() {
  const [keyword, setKeyword] = useState('')
  const [group, setGroup] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()
  // 탭은 주소에 둔다 (?tab=early): 케이스를 열었다가 "이전"으로 오면 보던 탭으로
  const tab: Tab = params.get('tab') === 'early' ? 'early' : 'critical'
  const setTab = (t: Tab) => {
    setGroup(null)
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (t === 'early') next.set('tab', 'early')
        else next.delete('tab')
        return next
      },
      { replace: true },
    )
  }
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['cases'],
    queryFn: () => apiGet<CaseListResponse>('/cases'),
    // 새로 온 알림을 탭 숫자로 알려 주려면 화면이 스스로 다시 읽어야 한다
    refetchInterval: 30_000,
  })

  const cases = data?.cases ?? []
  const ofTab = (t: Tab) => {
    const list = cases.filter((c) => tabOf(c) === t)
    // 긴급: 처리 단계 → 최근 발생. 조기 경보: 같은 설비 반복이 한 줄에 쌓이므로 마지막으로 울린 순서
    return t === 'early' ? list.sort((a, b) => lastAt(b).localeCompare(lastAt(a))) : list.sort(byPriority)
  }
  const all = ofTab(tab)
  const picked = GROUPS.find((g) => g.key === group)
  const shown = all.filter((c) => (!picked || picked.statuses.includes(c.status)) && (!keyword.trim() || matches(c, keyword.trim())))
  const filtered = Boolean(picked || keyword.trim())

  // "새" = 그 탭을 마지막으로 본 뒤 새로 생겼거나 다시 울려 횟수가 는 줄. 탭을 연 순간의 기록과 비교해 줄에 표시하고,
  // 보고 있는 동안 받은 것까지 본 것으로 저장한다 (이 브라우저에만, 사람마다 다름)
  const [opened, setOpened] = useState(() => ({ tab, seen: readSeen(tab) }))
  if (opened.tab !== tab) setOpened({ tab, seen: readSeen(tab) })
  useEffect(() => {
    if (!data) return
    for (const t of TABS) if (t.key === tab || readSeen(t.key) === null) writeSeen(t.key, data.cases.filter((c) => tabOf(c) === t.key))
  }, [data, tab])

  return (
    <>
      <div className={`${page()} flex flex-col gap-5 pt-7 pb-7`}>
        <PageHeader title="작업함" description="확인이 필요한 설비 알림을 우선순위 순으로 모았습니다.">
          <SearchBox value={keyword} onChange={setKeyword} placeholder="설비 · 코드 검색" />
        </PageHeader>

        <div role="group" aria-label="상태별 건수 · 누르면 그 상태만 보기" className="grid grid-cols-4 gap-3">
          {GROUPS.map((g) => {
            // 카드는 탭과 상관없이 전체 (긴급 + 조기 경보). 누르면 지금 탭 목록을 그 상태로 거른다
            const inGroup = cases.filter((c) => g.statuses.includes(c.status))
            const n = inGroup.length
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
                  <span className={`size-2 rounded-full ${g.onlyWhenAny && n === 0 ? 'bg-dot-gray' : g.dot}`} />
                  {g.label}
                </span>
                <span className={`text-[36px] leading-none font-bold tracking-[-0.02em] ${n === 0 ? 'text-dot-gray' : ''}`}>{data ? n : EMPTY}</span>
              </button>
            )
          })}
        </div>

        {/* 긴급 / 조기 경보 탭 (목록 바로 위): 긴급이 많아도 조기 경보가 밀려 안 보이지 않게, 새로 온 건수를 탭에 (11/12) */}
        <div role="tablist" aria-label="알림 종류" className="-mb-2 flex gap-7 border-b border-line">
          {TABS.map((t) => {
            const on = tab === t.key
            const list = cases.filter((c) => tabOf(c) === t.key)
            const seen = readSeen(t.key)
            const fresh = on ? 0 : list.filter((c) => isNew(c, seen)).length
            return (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={on}
                onClick={() => setTab(t.key)}
                className={`-mb-px flex min-h-12 items-center gap-2 border-b-2 px-0.5 text-lg ${
                  on ? 'border-primary font-semibold text-ink' : 'border-transparent font-medium text-sub hover:text-ink'
                }`}
              >
                <span className={`size-2 rounded-full ${t.dot}`} />
                {t.label}
                <span className="tabular-nums">{data ? list.length : EMPTY}</span>
                {fresh > 0 && (
                  <span className={`min-w-6 rounded-full px-1.5 py-0.5 text-center text-xs font-semibold tabular-nums ${t.fresh}`} aria-label={`새로 ${fresh}건`}>
                    +{fresh}
                  </span>
                )}
              </button>
            )
          })}
        </div>

        <section className={`${card} overflow-x-auto`}>
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
            <table className="w-full min-w-[1080px] table-fixed border-collapse text-left">
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
                {shown.map((c) => (
                  <CaseRow key={c.case_id} item={c} fresh={isNew(c, opened.seen)} />
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </>
  )
}

type Tab = 'critical' | 'early'
const TABS: { key: Tab; label: string; dot: string; fresh: string }[] = [
  { key: 'critical', label: '긴급', dot: 'bg-alarm-dot', fresh: 'bg-alarm-soft text-alarm' },
  { key: 'early', label: '조기 경보', dot: 'bg-warn-dot', fresh: 'bg-warn-soft text-warn' },
]
const tabOf = (c: CaseSummary): Tab => (isEarly(c.event) ? 'early' : 'critical')

/** 조기 경보가 마지막으로 울린 시각 (반복이 없으면 처음 시각) */
const lastAt = (c: CaseSummary) => c.last_detected_at ?? c.event.detected_at

/** 탭마다 "본 줄" 기록: 케이스 번호 → 그때 반복 수. 브라우저 저장소가 막혀 있으면 "새" 표시 없이 동작 */
type Seen = Record<string, number>
function readSeen(t: Tab): Seen | null {
  try {
    const v = localStorage.getItem(`inbox.seen.${t}`)
    return v ? (JSON.parse(v) as Seen) : null
  } catch {
    return null
  }
}
function writeSeen(t: Tab, list: CaseSummary[]) {
  try {
    localStorage.setItem(`inbox.seen.${t}`, JSON.stringify(Object.fromEntries(list.map((c) => [c.case_id, c.repeat_count ?? 0]))))
  } catch {
    /* 저장 못 해도 목록은 그대로 */
  }
}
/** 처음 쓰는 브라우저(기록 없음)에서는 전부 "새"로 띄우지 않는다 */
const isNew = (c: CaseSummary, seen: Seen | null) => seen !== null && (seen[c.case_id] === undefined || (c.repeat_count ?? 0) > seen[c.case_id])

const STATUS_ORDER: CaseStatus[] = ['new', 'in_progress', 'handed_over', 'unresolved', 'logging', 'log_approved', 'resolved', 'false_alarm']

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
      // 조기 경보는 대부분 고장이 아니라 먼저 "확인" (11/08)
      return { to: base, label: isEarly(item.event) ? '확인하기' : '점검 시작' }
    case 'logging':
    case 'log_approved':
      // 일지를 쓰던 케이스는 정비일지로 바로 연다
      return { to: `${base}/log`, label: '일지 계속' }
    default:
      return { to: base, label: '점검 계속' }
  }
}

/** 버튼은 모두 흰색, 줄에 마우스를 올리거나 키보드로 고르면 그 줄 버튼만 파랗게 */
function CaseRow({ item, fresh = false }: { item: CaseSummary; fresh?: boolean }) {
  const { event } = item
  // 발생 시각: 종료 코드가 뜬 시각이 있으면 그것, 없으면 ML 판정 시각
  const occurredAt = event.trigger.rule_trigger_time ?? event.detected_at
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-base'
  const next = action(item)
  const early = isEarly(event)
  // 조기 경보는 같은 설비 반복이 한 줄에 쌓인다: 처음 ~ 마지막 시각, N회
  const times = (item.repeat_count ?? 0) + 1
  const last = early && item.last_detected_at ? formatKst(item.last_detected_at)?.slice(11) : null

  return (
    <tr className="group transition-colors duration-150 hover:bg-canvas">
      <td className={td}>
        <SeverityTag severity={item.severity} />
      </td>
      <td className={`${td} font-semibold whitespace-nowrap`}>
        {/* 이 탭을 마지막으로 본 뒤 새로 왔거나 다시 울린 줄: 이름 옆 작은 점 */}
        <span className="inline-flex items-center gap-1.5">
          {event.gun_id}
          {fresh && <span className="size-1.5 shrink-0 rounded-full bg-sub" role="img" aria-label="새로 온 알림" title="새로 온 알림" />}
        </span>
      </td>
      <td className={td}>
        {event.trigger.rule_code ? (
          <span className="font-semibold">{event.trigger.rule_code}</span>
        ) : early ? (
          // 조기 경보: 주황 점선 = 고장 코드 전, 확정 아님
          <span
            className="inline-flex items-center rounded-md border border-dashed border-warn-dot px-2 py-0.5 text-xs font-medium text-warn"
            title="고장 코드가 뜨기 전 조기 경보입니다. 수십 분 안 고장 가능성이며 확정이 아닙니다."
          >
            조기 경보
          </span>
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
      {early ? (
        <td className={`${td} whitespace-nowrap`}>
          고장 가능성
          {times > 1 && <span className="ml-2 font-semibold text-warn">{times}회</span>}
        </td>
      ) : (
        <td className={td} title={event.fault_class ? `데이터셋 클래스 ${event.fault_class.code} 기준 추정` : undefined}>
          {event.fault_class?.name_ko ?? <span className="text-sub">{EMPTY}</span>}
        </td>
      )}
      <td className={td} title={formatUtc(occurredAt) ?? undefined}>
        {formatKst(occurredAt) ?? <span className="text-sub">{EMPTY}</span>}
        {last && <span className="text-sub"> ~ {last}</span>}
      </td>
      <td className={td}>
        <StatusTag status={item.status} />
      </td>
      <td className={td}>
        <div className="flex items-center justify-end gap-2.5">
          <Link
            to={next.to}
            className={`${btn} w-[112px] group-hover:border-primary! group-hover:bg-primary! group-hover:font-semibold group-hover:text-white! focus-visible:border-primary! focus-visible:bg-primary! focus-visible:text-white!`}
          >
            {next.label}
          </Link>
        </div>
      </td>
    </tr>
  )
}
