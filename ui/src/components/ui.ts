// 와이어프레임 v2의 공통 모양 (.card / .btn / .big ...)을 클래스 문자열로 모은 것

/** 화면 본문 틀: 넓은 창에서도 1440px(+ 양옆 32px)에서 멈추고 가운데 둔다. wide = 매뉴얼 PDF가 열려 넓게 쓸 때 */
export const page = (wide = false) => `mx-auto w-full px-8 ${wide ? '' : 'max-w-[1504px]'}`

export const card = 'rounded-xl border border-line bg-white shadow-[0_1px_2px_rgba(16,24,40,0.04)]'

const btnBase =
  'inline-flex min-h-11 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg border px-4 text-sm disabled:pointer-events-none disabled:opacity-40'
export const btn = `${btnBase} border-field bg-white font-medium text-ink hover:bg-canvas`
export const btnPrimary = `${btnBase} border-primary bg-primary font-semibold text-white hover:bg-primary-ink`
export const btnLarge = 'min-h-12 px-[26px] text-base'
export const btnQuiet =
  'inline-flex min-h-9 items-center px-2 text-sm font-medium text-primary-ink hover:underline disabled:opacity-40'

/** 판단 · 결과의 큰 선택 카드 */
export const bigChoice =
  'flex min-h-[92px] flex-col items-start justify-center gap-1 rounded-xl border border-field bg-white px-5 text-left hover:border-primary disabled:pointer-events-none disabled:opacity-40'

export const input =
  'min-h-11 w-full rounded-lg border border-field bg-white px-3 py-2.5 text-base outline-none focus:border-primary'
export const label = 'text-xs font-medium text-sub'
