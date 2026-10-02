import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { EmptyState, ErrorState, LoadingState } from '../components/StateViews'
import { apiGet } from '../lib/api'
import { formatKst, formatUtc } from '../lib/time'
import type { CaseListResponse, CaseStatus, CaseSummary, Severity } from '../types/case'

// ① 작업함 (wireframe/boards/Main.dc.html 기준)
// 없는 값은 지어내지 않고 "—"로 둔다.

const EMPTY = '—'

const SEVERITY_STYLE: Record<Severity, string> = {
  critical: 'bg-red-50 text-red-700',
  warning: 'bg-orange-50 text-orange-700',
}

const STATUS_LABEL: Record<CaseStatus, { label: string; style: string }> = {
  new: { label: '신규', style: 'bg-slate-100 text-slate-700' },
  in_progress: { label: '원인 찾는 중', style: 'bg-blue-50 text-blue-700' },
  handed_over: { label: '인계됨', style: 'bg-violet-50 text-violet-700' },
}

const COLUMNS = ['경보', '설비', '트리거 (컨트롤러 코드)', '추정 고장 유형', '진행 상태', '발생 (데이터 시각 · KST)', '담당', '']

export default function InboxPage() {
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ['cases'],
    queryFn: () => apiGet<CaseListResponse>('/cases'),
  })

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="text-3xl font-bold">작업함</h1>

      <section className="mt-6 overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
        {isPending ? (
          <LoadingState label="케이스 목록을 불러오는 중" />
        ) : isError ? (
          <ErrorState what="케이스 목록" onRetry={() => refetch()} />
        ) : data.cases.length === 0 ? (
          <EmptyState
            title="처리할 케이스 없음"
            note={'"모든 설비 정상"이라는 뜻이 아닙니다. 재생 상태나 설비 상태를 확인하세요.'}
          />
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs font-semibold text-slate-500">
              <tr>
                {COLUMNS.map((c) => (
                  <th key={c} scope="col" className="px-4 py-3">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.cases.map((c) => (
                <CaseRow key={c.case_id} item={c} />
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}

function CaseRow({ item }: { item: CaseSummary }) {
  const { event } = item
  const status = STATUS_LABEL[item.status]
  // 발생 시각: 종료 코드가 뜬 시각이 있으면 그것, 없으면 ML 판정 시각
  const occurredAt = event.trigger.rule_trigger_time ?? event.detected_at

  return (
    <tr className="align-middle">
      <td className="px-4 py-4">
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${SEVERITY_STYLE[item.severity]}`}>
          {item.severity}
        </span>
      </td>
      <td className="px-4 py-4 font-semibold">{event.gun_id}</td>
      <td className="px-4 py-4">
        {event.trigger.rule_code ? (
          `종료 코드 ${event.trigger.rule_code}`
        ) : (
          <>
            <span className="text-slate-400">{EMPTY}</span>
            {event.trigger.source === 'model' && (
              <span className="ml-2 text-xs text-slate-500">모델 이상 신호만</span>
            )}
          </>
        )}
      </td>
      <td className="px-4 py-4">
        {event.fault_class ? (
          <span className="flex flex-wrap items-center gap-2">
            {event.fault_class.code} {event.fault_class.name_ko}
            <span
              title="데이터셋 클래스 기준 추정. 원인 확정이 아님"
              className="rounded-full border border-dashed border-slate-400 px-2 py-0.5 text-xs text-slate-600"
            >
              추정
            </span>
          </span>
        ) : (
          <span className="text-slate-400">{EMPTY}</span>
        )}
      </td>
      <td className="px-4 py-4">
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${status.style}`}>{status.label}</span>
      </td>
      <td className="px-4 py-4 tabular-nums" title={formatUtc(occurredAt) ?? undefined}>
        {formatKst(occurredAt) ?? <span className="text-slate-400">{EMPTY}</span>}
      </td>
      <td className="px-4 py-4">{item.assignee ?? <span className="text-slate-400">{EMPTY}</span>}</td>
      <td className="px-4 py-4 text-right">
        <Link
          to={`/cases/${item.case_id}`}
          className="inline-block rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold hover:bg-slate-50"
        >
          열기
        </Link>
      </td>
    </tr>
  )
}
