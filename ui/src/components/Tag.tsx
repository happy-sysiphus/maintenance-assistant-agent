// 점 + 글자 표시 (와이어프레임 v2의 .tag / .alarm)
import type { ReactNode } from 'react'
import type { CaseStatus, Severity } from '../types/case'

const DOT = {
  gray: 'bg-dot-gray',
  blue: 'bg-dot-blue',
  purple: 'bg-dot-purple',
  green: 'bg-dot-green',
  amber: 'bg-warn-dot',
} as const

export function DotTag({ color, children }: { color: keyof typeof DOT; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-[7px] whitespace-nowrap text-[13.5px] font-medium text-ink-2">
      <span className={`size-[7px] shrink-0 rounded-full ${DOT[color]}`} />
      {children}
    </span>
  )
}

const STATUS: Record<CaseStatus, { label: string; color: keyof typeof DOT }> = {
  new: { label: '새 고장', color: 'gray' },
  in_progress: { label: '점검 중', color: 'blue' },
  logging: { label: '일지 작성', color: 'green' },
  log_approved: { label: '일지 승인됨', color: 'green' },
  unresolved: { label: '미해결', color: 'amber' },
  handed_over: { label: '도움 요청', color: 'purple' },
  resolved: { label: '해결', color: 'green' },
}

export function StatusTag({ status }: { status: CaseStatus }) {
  const s = STATUS[status]
  return <DotTag color={s.color}>{s.label}</DotTag>
}

// 경보 수준: critical → 긴급, warning → 주의. 빨강 · 주황은 여기에만 쓴다.
const SEVERITY: Record<Severity, { label: string; text: string; dot: string }> = {
  critical: { label: '긴급', text: 'text-alarm', dot: 'bg-alarm-dot' },
  warning: { label: '주의', text: 'text-warn', dot: 'bg-warn-dot' },
}

export function SeverityTag({ severity }: { severity: Severity }) {
  const s = SEVERITY[severity]
  return (
    <span className={`inline-flex items-center gap-[7px] whitespace-nowrap text-sm font-semibold ${s.text}`} title={severity}>
      <span className={`size-2 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  )
}
