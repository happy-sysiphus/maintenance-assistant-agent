import type { ReactNode } from 'react'
import { page } from './ui'

// 화면 맨 위 흰 띠 (와이어프레임 v2의 .top). 흰 띠는 끝까지, 안의 내용은 본문과 같은 너비에 맞춘다.
export function TopBar({ children, wide = false }: { children: ReactNode; wide?: boolean }) {
  return (
    <header className="h-16 shrink-0 border-b border-line bg-white">
      <div className={`${page(wide)} flex h-full items-center gap-3`}>{children}</div>
    </header>
  )
}
