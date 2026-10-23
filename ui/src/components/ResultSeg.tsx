import type { CheckResult } from '../types/case'

// 점검 결과 고르기 (와이어프레임 v2의 .seg3): 정상 / 이상 / 건너뜀. 점검 · 현장 확인 · 수동 모드가 같이 쓴다.
// 이상은 빨간 점, 정상은 초록 체크, 건너뜀은 회색 점.

const OPTIONS: { value: CheckResult; label: string; on: string; mark: string }[] = [
  { value: 'normal', label: '정상', on: 'text-ink', mark: '✓' },
  { value: 'abnormal', label: '이상', on: 'text-alarm', mark: 'dot-alarm' },
  { value: 'skipped', label: '건너뜀', on: 'text-ink-2', mark: 'dot-gray' },
]

export function ResultSeg<T extends CheckResult>({
  label,
  value,
  onChange,
  options = ['normal', 'abnormal', 'skipped'] as T[],
}: {
  label: string
  value: T | undefined
  onChange: (v: T) => void
  options?: T[]
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex shrink-0 gap-0.5 rounded-lg bg-fill p-[3px]">
      {OPTIONS.filter((o) => options.includes(o.value as T)).map((r) => {
        const on = value === r.value
        return (
          <button
            key={r.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(r.value as T)}
            className={`flex min-h-[38px] min-w-[68px] items-center justify-center gap-1.5 rounded-md px-1 text-sm ${
              on ? `bg-white font-semibold shadow-[0_1px_2px_rgba(16,24,40,0.1),0_0_0_1px_rgba(16,24,40,0.04)] ${r.on}` : 'font-medium text-sub'
            }`}
          >
            {on &&
              (r.mark === '✓' ? (
                <span aria-hidden="true" className="text-xs font-bold text-ok">✓</span>
              ) : (
                <span className={`size-1.5 rounded-full ${r.mark === 'dot-alarm' ? 'bg-alarm-dot' : 'bg-dot-gray'}`} />
              ))}
            {r.label}
          </button>
        )
      })}
    </div>
  )
}
