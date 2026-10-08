import { NavLink, Outlet, useMatch } from 'react-router'
import { ClockIcon, InboxIcon, PulseIcon } from './components/Icons'

// 화면 공통 틀 (와이어프레임 v2): 왼쪽 어두운 아이콘 메뉴 + 본문.
export default function App() {
  // 케이스 화면(/cases/:id)은 작업함에서 열리므로 "작업함" 메뉴를 같이 켠다
  const inCase = useMatch('/cases/*') !== null

  return (
    <div className="flex min-h-screen bg-canvas">
      <nav aria-label="주 메뉴" className="flex print:hidden w-20 shrink-0 flex-col items-center gap-1 bg-side py-4">
        <div className="mb-4 flex size-10 items-center justify-center rounded-lg bg-primary text-[13px] font-bold tracking-wide text-white">
          RSW
        </div>
        <NavLink
          to="/"
          end
          className={({ isActive }) => navClass(isActive || inCase)}
        >
          <InboxIcon />
          작업함
        </NavLink>
        <NavLink to="/equipment" className={({ isActive }) => navClass(isActive)}>
          <PulseIcon />
          설비
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => navClass(isActive)}>
          <ClockIcon />
          이력
        </NavLink>
      </nav>
      <div className="flex min-w-0 grow flex-col">
        <Outlet />
      </div>
    </div>
  )
}

function navClass(active: boolean) {
  return `flex min-h-15 w-16 flex-col items-center justify-center gap-1.5 rounded-lg text-xs font-medium ${
    active ? 'bg-side-on text-white' : 'text-side-ink hover:text-white'
  }`
}
