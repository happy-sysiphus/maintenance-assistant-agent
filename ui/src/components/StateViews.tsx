// 공통 상태 표시 (wireframe/boards/States.dc.html 기준).
// 로딩 · 조회 실패 · 빈 목록을 모든 화면이 같은 모양으로 쓴다.

export function LoadingState({ label = '불러오는 중' }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-3 px-6 py-10 text-sm text-slate-500">
      <span className="size-4 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600" />
      {label}
    </div>
  )
}

export function ErrorState({ what, onRetry }: { what: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-2 px-6 py-10">
      <p className="font-semibold text-slate-900">불러오지 못했습니다</p>
      <p className="text-sm text-slate-500">조회 실패 · {what}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-1 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold hover:bg-slate-50"
        >
          다시 시도
        </button>
      )}
    </div>
  )
}

export function EmptyState({ title, note }: { title: string; note?: string }) {
  return (
    <div className="flex flex-col items-start gap-1 px-6 py-10">
      <p className="font-semibold text-slate-900">{title}</p>
      {note && <p className="text-sm text-slate-500">{note}</p>}
    </div>
  )
}
