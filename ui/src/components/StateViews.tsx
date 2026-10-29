// 공통 상태 표시 (wireframe/boards/V2States.dc.html 기준).
// 원칙: 화면 틀은 먼저 보이고, 실패해도 입력한 내용은 지우지 않는다.
import { useEffect, useState, type ReactNode } from 'react'

/** 불러오는 중: 글자 대신 회색 줄 */
export function LoadingRows({ rows = 3, label = '불러오는 중' }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-label={label} className="flex flex-col">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-6 border-t border-line-soft px-5 py-5 first:border-t-0">
          <span className="h-3 w-14 rounded bg-fill" />
          <span className="h-3 w-20 rounded bg-fill" />
          <span className="h-3 w-40 rounded bg-fill" />
          <span className="h-3 grow rounded bg-fill" />
        </div>
      ))}
    </div>
  )
}

/** 오래 걸리는 불러오기에서 몇 초 뒤에만 보이는 안내 한 줄 (빨리 끝나면 깜빡이지 않게) */
export function SlowNote({ after = 3000, children }: { after?: number; children: ReactNode }) {
  const [show, setShow] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setShow(true), after)
    return () => clearTimeout(t)
  }, [after])
  return show ? <p className="px-5 pb-5 text-sm text-sub">{children}</p> : null
}

export function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center gap-2 px-6 py-14 text-center">
      <p className="text-base font-semibold">불러오지 못했습니다</p>
      <p className="text-sm text-sub">잠시 후 다시 시도해 주세요. 계속되면 관리자에게 알려주세요.</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 min-h-11 rounded-lg border border-field bg-white px-4 text-sm font-medium hover:bg-canvas"
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
      {note && <p className="text-sm text-sub">{note}</p>}
      {action}
    </div>
  )
}
