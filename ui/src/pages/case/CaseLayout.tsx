import { Link, Outlet } from 'react-router'
import { BackIcon } from '../../components/Icons'
import { ErrorState, LoadingRows } from '../../components/StateViews'
import { StatusTag } from '../../components/Tag'
import { TopBar } from '../../components/TopBar'
import { btn, card } from '../../components/ui'
import { useCase } from '../../lib/useCase'

// 케이스 화면 공통 틀: 상단 바(뒤로 · 설비 · 상태 · 현장 확인 입력 · 도움 요청) + 본문
export default function CaseLayout() {
  const { data, isPending, isError, refetch } = useCase()

  return (
    <>
      <TopBar>
        <Link to="/" className="inline-flex min-h-11 print:hidden items-center gap-1 text-sm font-medium text-[#4e5968]">
          <BackIcon size={18} />
          작업함
        </Link>
        {data && (
          <>
            <h1 className="text-[19px] font-bold tracking-[-0.02em]">{data.event.gun_id}</h1>
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
      <div className="flex grow flex-col gap-4 px-8 pt-6 pb-7">
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
