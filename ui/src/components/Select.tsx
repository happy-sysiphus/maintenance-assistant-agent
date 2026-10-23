import { useEffect, useId, useRef, useState } from 'react'
import { ChevronIcon } from './Icons'

// 고르기 상자. 브라우저 기본 <select>는 펼친 목록을 운영체제가 그려서 화면 모양과 맞지 않아(mac 회색 메뉴 등) 직접 그린다.
// 목록은 상자 바로 아래, 상자와 같은 폭으로 열린다. 키보드: ↑↓ 이동 · Enter/Space 고르기 · Esc 닫기.

export interface SelectOption {
  value: string
  label: string
}

export function Select({
  value,
  onChange,
  options,
  placeholder = '선택',
  ariaLabel,
  className = '',
  onBlur,
}: {
  value: string
  onChange: (value: string) => void
  options: SelectOption[]
  placeholder?: string
  ariaLabel?: string
  /** 상자에 더할 클래스 (읽기 전용 회색 등) */
  className?: string
  onBlur?: () => void
}) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const listId = useId()
  const current = options.find((o) => o.value === value)

  // 바깥을 누르면 닫는다
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) {
        setOpen(false)
        onBlur?.()
      }
    }
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [open, onBlur])

  const show = () => {
    setActive(
      Math.max(
        0,
        options.findIndex((o) => o.value === value),
      ),
    )
    setOpen(true)
  }
  const pick = (v: string) => {
    onChange(v)
    setOpen(false)
    onBlur?.()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && open) {
      e.preventDefault()
      setOpen(false)
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      if (!open) return show()
      const step = e.key === 'ArrowDown' ? 1 : -1
      setActive((i) => (i + step + options.length) % options.length)
    } else if ((e.key === 'Enter' || e.key === ' ') && open) {
      e.preventDefault()
      if (options[active]) pick(options[active].value)
    } else if (e.key === 'Tab' && open) {
      setOpen(false)
    }
  }

  return (
    <div ref={root} className="relative w-full">
      <button
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-controls={listId}
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={onKeyDown}
        className={`flex min-h-11 w-full items-center gap-2 rounded-lg border bg-white px-3 py-2.5 text-left text-base outline-none disabled:cursor-default ${
          open ? 'border-primary' : 'border-field focus:border-primary'
        } ${className}`}
      >
        <span className={`grow truncate ${current ? '' : 'text-sub'}`}>{current?.label ?? placeholder}</span>
        <ChevronIcon size={16} className={`shrink-0 text-sub transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <ul
          id={listId}
          role="listbox"
          aria-label={ariaLabel}
          className="absolute top-[calc(100%+4px)] right-0 left-0 z-20 animate-enter max-h-64 overflow-auto rounded-lg border border-line bg-white py-1 shadow-[0_8px_24px_rgba(16,24,40,0.12)]"
        >
          {options.map((o, i) => {
            const on = o.value === value
            return (
              <li
                key={o.value}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={on}
                onPointerEnter={() => setActive(i)}
                // 버튼에서 포커스가 빠지지 않게 mousedown 기본 동작을 막고 고른다
                onMouseDown={(e) => e.preventDefault()}
                // <label> 안에 있을 때 클릭이 label까지 올라가면 상자가 다시 열려서 막는다
                onClick={(e) => {
                  e.stopPropagation()
                  pick(o.value)
                }}
                className={`flex min-h-10 cursor-pointer items-center gap-2 px-3 text-base ${i === active ? 'bg-canvas' : ''} ${on ? 'font-semibold text-primary-ink' : ''}`}
              >
                <span className="grow truncate">{o.label}</span>
                {on && (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden>
                    <path d="m5 12 5 5 9-10" />
                  </svg>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
