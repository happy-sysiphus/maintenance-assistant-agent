import type { ReactNode } from 'react'
import { SearchIcon } from './Icons'
import { input } from './ui'

// 메뉴 첫 화면(작업함 · 설비 · 이력) 제목 영역. 흰 띠(TopBar)는 한 건을 처리하는 케이스 화면에서만 쓴다.
// 큰 제목 / 회색 설명 한 줄, 오른쪽에 검색 · 버튼 자리.
export function PageHeader({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children?: ReactNode
}) {
  return (
    <div className="flex items-end gap-4 pt-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-[-0.03em]">{title}</h1>
        <p className="mt-2 text-base text-sub">{description}</p>
      </div>
      <div className="grow" />
      {children && <div className="flex items-center gap-3">{children}</div>}
    </div>
  )
}

export function SearchBox({ value, onChange, placeholder, label = '검색' }: { value: string; onChange: (v: string) => void; placeholder: string; label?: string }) {
  return (
    <label className="relative w-[300px]">
      <span className="sr-only">{label}</span>
      <span className="pointer-events-none absolute top-[13px] left-3.5 text-sub">
        <SearchIcon size={18} />
      </span>
      <input type="search" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className={`${input} pl-[42px]`} />
    </label>
  )
}
