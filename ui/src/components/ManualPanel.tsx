import { manualUrl } from '../lib/api'
import { DocIcon } from './Icons'
import { card } from './ui'

// 매뉴얼 원본 보기 (wireframe/boards/V2Manual.dc.html): 오른쪽 패널 자리에 PDF 쪽이 열리고 점검은 그대로 계속한다.
// PDF는 브라우저 기본 뷰어로 그린다 (확대 · 검색은 뷰어 도구 사용). 매뉴얼은 영어 원문이다.

const small = 'inline-flex min-h-9 items-center justify-center rounded-lg border border-field bg-white px-3 text-[13.5px] font-medium hover:bg-canvas disabled:opacity-40'

/** "매뉴얼 90쪽" 버튼 */
export function DocButton({ page, label, on, title, onOpen }: { page: number; label?: string; on: boolean; title?: string; onOpen: (page: number) => void }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      title={title}
      onClick={() => onOpen(page)}
      className={`inline-flex min-h-10 shrink-0 items-center gap-1.5 rounded-lg border px-3 text-[13.5px] font-medium whitespace-nowrap text-primary-ink ${
        on ? 'border-primary bg-[#eef3ff]' : 'border-field bg-white hover:bg-canvas'
      }`}
    >
      <DocIcon size={18} />
      {label ?? `매뉴얼 ${page}쪽`}
    </button>
  )
}

export function ManualPanel({ page, onPage, onClose }: { page: number; onPage: (page: number) => void; onClose: () => void }) {
  const url = manualUrl(page)
  return (
    <aside className={`${card} sticky top-4 flex h-[calc(100vh-120px)] min-h-[560px] flex-col overflow-hidden bg-[#edeff3]`} aria-label="매뉴얼 원본">
      <div className="flex items-center gap-2 border-b border-line bg-white px-4 py-3">
        <b className="text-base font-semibold">Festo 매뉴얼</b>
        <span className="text-[13.5px] text-sub">{page}쪽 · 영어 원문</span>
        <div className="grow" />
        <button type="button" className={small} aria-label="이전 쪽" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          ‹
        </button>
        <button type="button" className={small} aria-label="다음 쪽" onClick={() => onPage(page + 1)}>
          ›
        </button>
        <a className={small} href={url} target="_blank" rel="noreferrer">
          새 창
        </a>
        <button type="button" className={small} onClick={onClose}>
          닫기
        </button>
      </div>
      {/* 쪽이 바뀌면 key로 다시 그린다 (같은 PDF의 #page만 바꾸면 뷰어가 이동하지 않는 브라우저가 있다) */}
      <iframe key={page} title={`Festo 매뉴얼 ${page}쪽`} src={`${url}&view=FitH&navpanes=0`} className="w-full grow border-0" />
    </aside>
  )
}
