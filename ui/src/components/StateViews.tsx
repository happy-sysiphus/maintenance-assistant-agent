// 공통 상태 표시 (wireframe/boards/V2States.dc.html 기준).
// 원칙: 화면 틀은 먼저 보이고, 실패해도 입력한 내용은 지우지 않는다.
import type { ReactNode } from 'react'

/** 불러오는 중: 글자 대신 회색 줄 */
export function LoadingRows({ rows = 3, label = '불러오는 중' }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-label={label} className="flex flex-col">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-6 border-t border-line-soft px-5 py-5 first:border-t-0">
          <span className="h-3 w-14 rounded bg-[#eef0f3]" />
          <span className="h-3 w-20 rounded bg-[#eef0f3]" />
          <span className="h-3 w-40 rounded bg-[#eef0f3]" />
          <span className="h-3 grow rounded bg-[#eef0f3]" />
        </div>
      ))}
    </div>
  )
}

export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center gap-2 px-6 py-14 text-center">
      <p className="text-base font-semibold">불러오지 못했습니다</p>
      <p className="text-[13.5px] text-sub">잠시 후 다시 시도해 주세요. 계속되면 관리자에게 알려주세요.</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 min-h-11 rounded-lg border border-field bg-white px-4 text-[14.5px] font-medium hover:bg-canvas"
        >
          다시 시도
        </button>
      )}
    </div>
  )
}

export function EmptyState({ title, note, action }: { title: string; note?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-14 text-center">
      <p className="text-base font-semibold">{title}</p>
      {note && <p className="text-[13.5px] text-sub">{note}</p>}
      {action}
    </div>
  )
}
