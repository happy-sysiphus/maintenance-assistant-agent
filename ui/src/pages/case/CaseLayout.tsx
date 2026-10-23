import { Link, Outlet, useSearchParams } from 'react-router'
import { BackIcon } from '../../components/Icons'
import { ErrorState, LoadingRows } from '../../components/StateViews'
import { StatusTag } from '../../components/Tag'
import { TopBar } from '../../components/TopBar'
import { btn, card, page } from '../../components/ui'
import { useCase } from '../../lib/useCase'

// 케이스 화면 공통 틀: 상단 바(뒤로 · 설비 · 상태 · 현장 확인 입력 · 도움 요청) + 본문
export default function CaseLayout() {
  const { data, isPending, isError, refetch } = useCase()
  // 매뉴얼 PDF가 열리면(?manual=쪽) 최대 폭을 풀어 PDF를 크게 보여준다
  const [params] = useSearchParams()
  const wide = params.has('manual')

  return (
    <>
      <TopBar wide={wide}>
        <Link to="/" className="inline-flex min-h-11 print:hidden items-center gap-1 text-sm font-medium text-ink-2">
          <BackIcon size={18} />
          작업함
        </Link>
        {data && (
          <>
            <h1 className="text-xl font-bold tracking-[-0.02em]">{data.event.gun_id}</h1>
            {/* 어떤 고장을 처리 중인지 모든 단계에서 보이게 (와이어프레임 V2Action 상단) */}
            {data.event.trigger.rule_code ? (
              <span className="text-base font-semibold text-alarm">{data.event.trigger.rule_code}</span>
            ) : (
              <span className="text-sm font-medium text-sub">코드 없음</span>
            )}
            <StatusTag status={data.status} />
            <div className="grow" />
            <Link to="field" className={`${btn} print:hidden`}>
              현장 확인 입력
            </Link>
            <Link to="handover" className={`${btn} print:hidden`}>
              도움 요청
            </Link>
          </>
        )}
      </TopBar>
      <div className={`${page(wide)} flex grow flex-col gap-4 pt-6 pb-7`}>
        {isPending ? (
          <section className={card}>
            <LoadingRows label="고장 정보를 불러오는 중" />
          </section>
        ) : isError ? (
          <section className={card}>
            <ErrorState onRetry={() => refetch()} />
          </section>
        ) : (
          <Outlet context={data} />
        )}
      </div>
    </>
  )
}
