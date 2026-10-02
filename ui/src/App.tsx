import { NavLink, Outlet, useMatch } from 'react-router'

// 화면 공통 틀: 왼쪽 메뉴 + 본문. 메뉴 중 아직 없는 화면은 비활성으로 둔다.
const PLANNED = ['설비 상태', '정비 이력', '재생 콘솔']

export default function App() {
  // 케이스 화면(/cases/:id)은 작업함에서 열리므로 "작업함" 메뉴를 같이 켠다
  const inCase = useMatch('/cases/*') !== null

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-900">
      <aside className="w-56 shrink-0 border-r border-slate-200 bg-white px-4 py-6">
        <p className="px-2 text-xl font-bold">
          RSW <span className="text-blue-600">정비 지원</span>
        </p>
        <nav className="mt-8 flex flex-col gap-1 text-sm font-semibold">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `rounded-lg px-3 py-2.5 ${isActive || inCase ? 'bg-blue-600 text-white' : 'hover:bg-slate-100'}`
            }
          >
            작업함
          </NavLink>
          {PLANNED.map((name) => (
            <span key={name} aria-disabled="true" className="cursor-not-allowed rounded-lg px-3 py-2.5 text-slate-400">
              {name}
            </span>
          ))}
        </nav>
      </aside>
      <main className="flex-1 px-8 py-8">
        <Outlet />
      </main>
    </div>
  )
}
