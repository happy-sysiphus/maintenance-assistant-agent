import { useSyncExternalStore, type ReactNode } from 'react'
import { NavLink, Outlet, useMatch, useSearchParams } from 'react-router'
import { ChatIcon, ClockIcon, InboxIcon, PulseIcon } from './components/Icons'

// 화면 공통 틀: 왼쪽 어두운 메뉴 + 본문.
// 메뉴는 평소 220px(아이콘 + 이름 가로), 창이 좁거나(1280px 미만) 매뉴얼 PDF가 열리면(?manual=) 96px 아이콘형으로 접는다.
const NARROW = '(max-width: 1279px)'
// 자동 접기 켜고 끄기. 10/23: PDF를 열었을 때 넓은 메뉴로 본문이 얼마나 좁아지는지 보려고 잠시 끔
const AUTO_COMPACT = false

function useNarrow() {
  return useSyncExternalStore(
    (onChange) => {
      const mq = window.matchMedia(NARROW)
      mq.addEventListener('change', onChange)
      return () => mq.removeEventListener('change', onChange)
    },
    () => window.matchMedia(NARROW).matches,
  )
}

export default function App() {
  // 케이스 화면(/cases/:id)은 작업함에서 열리므로 "작업함" 메뉴를 같이 켠다
  const inCase = useMatch('/cases/*') !== null
  const [params] = useSearchParams()
  const narrow = useNarrow()
  const compact = AUTO_COMPACT && (narrow || params.has('manual'))

  const items: { to: string; label: string; icon: (size: number) => ReactNode; on?: boolean }[] = [
    { to: '/', label: '작업함', icon: (s) => <InboxIcon size={s} />, on: inCase },
    { to: '/equipment', label: '설비', icon: (s) => <PulseIcon size={s} /> },
    { to: '/history', label: '이력', icon: (s) => <ClockIcon size={s} /> },
    { to: '/assistant', label: 'AI 챗봇', icon: (s) => <ChatIcon size={s} /> },
  ]

  return (
    <div className="flex min-h-screen bg-canvas">
      <nav
        aria-label="주 메뉴"
        className={`sticky top-0 flex h-screen shrink-0 flex-col gap-1 bg-side py-5 print:hidden ${compact ? 'w-24 items-center' : 'w-[220px] px-3'}`}
      >
        <div title="RSW 정비 지원" className={`mb-5 flex items-baseline gap-2 py-1.5 ${compact ? '' : 'px-3'}`}>
          <span className={`font-extrabold tracking-[0.02em] text-white ${compact ? 'text-[17px]' : 'text-[19px]'}`}>
            RSW<span className="text-primary">.</span>
          </span>
          {!compact && <span className="text-sm font-medium text-side-ink">정비 지원</span>}
        </div>
        {items.map((m) => (
          <NavLink key={m.to} to={m.to} end={m.to === '/'} className={({ isActive }) => navClass(isActive || Boolean(m.on), compact)}>
            {m.icon(compact ? 24 : 20)}
            {m.label}
          </NavLink>
        ))}
      </nav>
      <div className="flex min-w-0 grow flex-col">
        <Outlet />
      </div>
    </div>
  )
}

function navClass(active: boolean, compact: boolean) {
  const shape = compact ? 'min-h-16 w-20 flex-col justify-center gap-1.5 text-xs' : 'min-h-12 gap-3 px-3 text-base'
  return `flex items-center rounded-lg font-medium ${shape} ${active ? 'bg-side-on text-white' : 'text-side-ink hover:text-white'}`
}
