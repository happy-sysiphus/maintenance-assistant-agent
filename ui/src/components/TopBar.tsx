import type { ReactNode } from 'react'

// 화면 맨 위 흰 띠 (와이어프레임 v2의 .top). 왼쪽은 제목, 오른쪽은 children.
export function TopBar({ children }: { children: ReactNode }) {
  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b border-line bg-white px-8">{children}</header>
  )
}
