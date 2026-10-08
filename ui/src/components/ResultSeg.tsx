import type { CheckResult } from '../types/case'

// 점검 결과 고르기 (와이어프레임 v2의 .seg3): 정상 / 이상 / 건너뜀. 점검 · 현장 확인 · 수동 모드가 같이 쓴다.

const OPTIONS: { value: CheckResult; label: string; on: string; dot: string }[] = [
  { value: 'normal', label: '정상', on: 'text-[#0f7b5f]', dot: 'bg-[#16a37a]' },
  { value: 'abnormal', label: '이상', on: 'text-[#c4362b]', dot: 'bg-[#e5484d]' },
  { value: 'skipped', label: '건너뜀', on: 'text-[#4e5968]', dot: 'bg-dot-gray' },
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
    <div role="radiogroup" aria-label={label} className="inline-flex shrink-0 gap-0.5 rounded-[9px] bg-[#f2f4f6] p-[3px]">
      {OPTIONS.filter((o) => options.includes(o.value as T)).map((r) => {
        const on = value === r.value
        return (
          <button
            key={r.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(r.value as T)}
            className={`flex min-h-[38px] min-w-[68px] items-center justify-center gap-1.5 rounded-[7px] px-1 text-sm ${
              on ? `bg-white font-semibold shadow-[0_1px_2px_rgba(16,24,40,0.1),0_0_0_1px_rgba(16,24,40,0.04)] ${r.on}` : 'font-medium text-sub'
            }`}
          >
            {on && <span className={`size-1.5 rounded-full ${r.dot}`} />}
            {r.label}
          </button>
        )
      })}
    </div>
  )
}
