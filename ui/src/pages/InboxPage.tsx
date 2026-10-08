import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router'
import { SearchIcon } from '../components/Icons'
import { EmptyState, ErrorState, LoadingRows } from '../components/StateViews'
import { SeverityTag, StatusTag } from '../components/Tag'
import { TopBar } from '../components/TopBar'
import { apiGet } from '../lib/api'
import { formatKst, formatUtc } from '../lib/time'
import type { CaseListResponse, CaseSummary } from '../types/case'

// 작업함 (wireframe/boards/V2Main.dc.html 기준)
// 없는 값은 지어내지 않고 "—"로 둔다. 상태별 필터 칩은 지금은 만들지 않는다.

const EMPTY = '—'

// 칸 폭은 비율로 나눈다. 한 칸만 늘어나면 화면이 넓을 때 그 칸만 휑하게 벌어진다.
const COLUMNS: { label: string; width: string }[] = [
  { label: '경보', width: 'w-[9%]' },
  { label: '설비', width: 'w-[11%]' },
  { label: '고장 코드', width: 'w-[11%]' },
  { label: '예상 유형', width: 'w-[20%]' },
  { label: '발생 (데이터 시각)', width: 'w-[17%]' },
  { label: '상태', width: 'w-[12%]' },
  { label: '담당', width: 'w-[8%]' },
  { label: '', width: 'w-[12%]' },
]

export default function InboxPage() {
  const [keyword, setKeyword] = useState('')
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['cases'],
    queryFn: () => apiGet<CaseListResponse>('/cases'),
  })

  const all = data?.cases ?? []
  const shown = keyword.trim() ? all.filter((c) => matches(c, keyword.trim())) : all

  return (
    <>
      <TopBar>
        <h1 className="text-[19px] font-bold tracking-[-0.02em]">작업함</h1>
        {data && <span className="text-[13.5px] text-sub">처리할 고장 {all.length}건</span>}
        <div className="grow" />
        <label className="relative w-[300px]">
          <span className="sr-only">검색</span>
          <span className="pointer-events-none absolute top-[13px] left-3.5 text-sub">
            <SearchIcon size={18} />
          </span>
          <input
            type="search"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            placeholder="설비 · 코드 검색"
            className="min-h-11 w-full rounded-lg border border-field bg-white py-2.5 pr-3 pl-[42px] text-[15px] outline-none focus:border-primary"
          />
        </label>
      </TopBar>

      <div className="px-8 pt-6 pb-7">
        <section className="overflow-hidden rounded-[10px] border border-line bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
          {isPending ? (
            <LoadingRows label="고장 목록을 불러오는 중" />
          ) : isError ? (
            <ErrorState onRetry={() => refetch()} />
          ) : shown.length === 0 ? (
            keyword.trim() ? (
              <EmptyState
                title="조건에 맞는 고장이 없습니다"
                note="설비가 모두 정상이라는 뜻은 아닙니다. 검색어를 확인해 주세요."
                action={
                  <button
                    type="button"
                    onClick={() => setKeyword('')}
                    className="mt-2 min-h-11 rounded-lg border border-field bg-white px-4 text-[14.5px] font-medium hover:bg-canvas"
                  >
                    검색 지우기
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
                      className={`border-b border-line bg-head px-5 py-[11px] text-[12.5px] font-medium whitespace-nowrap text-faint ${c.width}`}
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

/** 설비 이름이나 고장 코드에 검색어가 들어 있는지 (대소문자 무시) */
function matches(c: CaseSummary, keyword: string) {
  const k = keyword.toLowerCase()
  return [c.event.gun_id, c.event.trigger.rule_code].some((v) => v?.toLowerCase().includes(k))
}

/** first: 목록 맨 위(가장 최근) 줄. 그 줄의 "열기"만 파랗게 강조한다 (와이어프레임 V2Main) */
function CaseRow({ item, first }: { item: CaseSummary; first: boolean }) {
  const { event } = item
  // 발생 시각: 종료 코드가 뜬 시각이 있으면 그것, 없으면 ML 판정 시각
  const occurredAt = event.trigger.rule_trigger_time ?? event.detected_at
  const td = 'border-t border-line-soft px-5 py-3.5 align-middle text-[15px]'
  const resume = item.status !== 'new'
  // 일지를 쓰던 케이스는 정비일지로 바로 연다
  const to = item.status === 'logging' || item.status === 'log_approved' ? `/cases/${item.case_id}/log` : `/cases/${item.case_id}`

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
          <span className="text-[13px] font-medium text-sub">코드 없음</span>
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
      <td className={`${td} text-[13.5px] text-sub`}>{item.assignee ?? EMPTY}</td>
      <td className={`${td} text-right`}>
        <Link
          to={to}
          className={`inline-flex min-h-11 w-[88px] items-center justify-center rounded-lg border text-[14.5px] whitespace-nowrap ${
            first && !resume
              ? 'border-primary bg-primary font-semibold text-white hover:bg-primary-ink'
              : 'border-field bg-white font-medium hover:bg-canvas'
          }`}
        >
          {resume ? '이어서' : '열기'}
        </Link>
      </td>
    </tr>
  )
}
