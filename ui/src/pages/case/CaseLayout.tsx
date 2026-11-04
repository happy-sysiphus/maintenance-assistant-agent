import { Link, Navigate, Outlet, useLocation } from 'react-router'
import { BackIcon } from '../../components/Icons'
import { ErrorState, LoadingRows, SlowNote } from '../../components/StateViews'
import { StatusTag } from '../../components/Tag'
import { TopBar } from '../../components/TopBar'
import { btn, card, page } from '../../components/ui'
import { formatKst } from '../../lib/time'
import { useCase } from '../../lib/useCase'
import { isClosed, type CaseDetail } from '../../types/case'

/**
 * 닫힌 케이스(해결 종료 · 오알람)는 읽기 전용 (11/04, hitl-loop.md ④ 요청 4).
 * 다시 닫으면 마지막 결과가 이전 기록(현장 기억) 라벨을 덮어써서 막는다. api도 409로 막는다.
 * 보기 화면: 해결 = 상황 · 정비일지, 오알람 = 상황만. 나머지 단계 주소로 오면 보기 화면으로 보낸다.
 */
function closedRedirect(d: CaseDetail, pathname: string): string | null {
  if (!isClosed(d.status)) return null
  const step = pathname.split('/')[3] ?? ''
  if (step === '' || (step === 'log' && d.status === 'resolved')) return null
  return d.status === 'resolved' ? `/cases/${d.case_id}/log` : `/cases/${d.case_id}`
}

function ClosedBanner({ d }: { d: CaseDetail }) {
  const c = d.records.closure
  const what = d.status === 'resolved' ? '해결 종료된 고장이에요' : '오알람으로 닫힌 고장이에요'
  return (
    <p role="status" className="flex flex-wrap items-baseline gap-x-2 rounded-lg bg-fill px-4 py-3 text-sm text-ink-2">
      <b className="font-semibold text-ink">{what}</b>
      {c && <span>· {formatKst(c.at)}</span>}
      {c?.reason && <span>· {c.reason}</span>}
      {c?.note && <span>· {c.note}</span>}
      <span className="text-sub">보기만 할 수 있어요.</span>
    </p>
  )
}

// 케이스 화면 공통 틀: 상단 바(뒤로 · 설비 · 상태 · 현장 확인 입력 · 도움 요청) + 본문
export default function CaseLayout() {
  const { data, isPending, isError, refetch } = useCase()
  // 매뉴얼 PDF가 열려도 최대 폭은 그대로 둔다 (10/29): 폭을 풀면 단계를 넘길 때마다 화면 비율이 바뀌었다.
  // PDF 칸만 380 → 540px로 넓어진다 (TwoColumns wide). 더 크게는 PDF의 "새 창"
  const wide = false
  const { pathname } = useLocation()

  return (
    <>
      <TopBar wide={wide}>
        <Link to="/" className="inline-flex min-h-11 print:hidden items-center gap-1 text-sm font-medium text-ink-2">
          <BackIcon size={18} />
          작업함
        </Link>
        {data && (
          <>
            {/* 어떤 고장을 처리 중인지 모든 단계에서 보이게. 설비 이름은 화면에서 빼고 읽어 주기용으로만 남긴다 (10/23) */}
            <h1 className={`text-base font-semibold ${data.event.trigger.rule_code ? 'text-alarm' : 'text-sub'}`}>
              <span className="sr-only">{data.event.gun_id} </span>
              {data.event.trigger.rule_code ?? '코드 없음'}
            </h1>
            <StatusTag status={data.status} />
            <div className="grow" />
            {!isClosed(data.status) && (
              <Link to="handover" className={`${btn} print:hidden`}>
                도움 요청
              </Link>
            )}
          </>
        )}
      </TopBar>
      <div className={`${page(wide)} flex grow flex-col gap-4 pt-6 pb-7`}>
        {isPending ? (
          <section className={card}>
            <LoadingRows label="고장 정보를 불러오는 중" />
            {/* 처음 여는 케이스는 api가 매뉴얼 근거로 안내를 만든다 (RAG, 5초~1분). 한 번 만들면 빠르다 */}
            <SlowNote>매뉴얼에서 이 고장의 안내를 만드는 중이에요. 처음 열 때는 1분 가까이 걸릴 수 있어요.</SlowNote>
          </section>
        ) : isError ? (
          <section className={card}>
            <ErrorState onRetry={() => refetch()} />
          </section>
        ) : closedRedirect(data, pathname) ? (
          <Navigate to={closedRedirect(data, pathname)!} replace />
        ) : (
          // 단계(점검 → 판단 …)를 넘어가면 내용이 살짝 떠오르며 나타난다
          <div key={pathname} className="flex grow animate-enter flex-col gap-4">
            {isClosed(data.status) && <ClosedBanner d={data} />}
            <Outlet context={data} />
          </div>
        )}
      </div>
    </>
  )
}
