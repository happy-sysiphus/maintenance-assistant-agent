// 데이터 시각 변환.
//
// 가정: ML 응답의 시각 문자열은 끝에 Z가 없어도(예: "2021-09-03T09:30:59") 전부 UTC다.
// 근거는 ml/README와 docs/04-repo-status.md("시각은 Z가 없는 UTC 문자열")이며,
// ML 담당에게 확인 요청 중이다(docs/05 ML 요청 4). 확인 결과가 다르면 이 파일만 고친다.
//
// 데이터 시각은 2019~2021년 공개 데이터의 시각이라 "방금 전" 같은 상대 표시는 쓰지 않는다.

const HAS_TIMEZONE = /(Z|[+-]\d{2}:?\d{2})$/i

/** 시각 문자열을 Date로. 시간대 표시가 없으면 UTC로 간주한다. 잘못된 값이면 null */
export function parseDataTime(value: string | null | undefined): Date | null {
  if (!value) return null
  const iso = HAS_TIMEZONE.test(value) ? value : `${value}Z`
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? null : date
}

const kstFormatter = new Intl.DateTimeFormat('sv-SE', {
  timeZone: 'Asia/Seoul',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
})

/** 한국 시간 "YYYY-MM-DD HH:mm". 값이 없으면 null */
export function formatKst(value: string | null | undefined): string | null {
  const date = parseDataTime(value)
  return date ? kstFormatter.format(date) : null
}

/** 원래 UTC 시각 "YYYY-MM-DD HH:mm:ss UTC" (툴팁 등 원본 확인용). 값이 없으면 null */
export function formatUtc(value: string | null | undefined): string | null {
  const date = parseDataTime(value)
  return date ? `${date.toISOString().slice(0, 19).replace('T', ' ')} UTC` : null
}
