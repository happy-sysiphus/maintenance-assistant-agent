import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { signalLabel } from '../lib/caseFlow'
import { parseDataTime } from '../lib/time'
import type { ErrorSpan, ScorePoint, SensorTrend } from '../types/case'

// 고장 전후 30분 그래프 (wireframe/boards/V2Case.dc.html).
// 값은 센서의 실제 크기가 아니라 "이 설비 평소 대비 편차(z)"다. ±1 띠가 평소 범위.
// 데이터가 빈 분은 null로 채워 선을 끊는다 (0이나 보간으로 채우지 않음).

const MINUTE = 60_000
const BLUE = '#2f55c8'
const GRAY = '#b8c0cc'
const RED = '#d92d20'

/** 1분 창 시각을 창 시작(초 0)으로 맞춘 epoch ms. sensor_trend는 창 끝(:59), trace는 창 시작(:00)이라 맞춰야 한다 */
function minuteMs(t: string) {
  const d = parseDataTime(t)
  return d ? Math.floor(d.getTime() / MINUTE) * MINUTE : NaN
}

const kstTime = new Intl.DateTimeFormat('ko-KR', { timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', hour12: false })
const kstTimeSec = new Intl.DateTimeFormat('ko-KR', {
  timeZone: 'Asia/Seoul',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})

type Row = { x: number; score?: number | null; [feature: string]: number | null | undefined }

function buildRows(trends: SensorTrend[], score: ScorePoint[]) {
  const map = new Map<number, Row>()
  const row = (x: number) => {
    let r = map.get(x)
    if (!r) map.set(x, (r = { x }))
    return r
  }
  for (const tr of trends) for (const p of tr.points) row(minuteMs(p.t))[tr.feature] = p.z
  for (const p of score) row(minuteMs(p.t)).score = p.score
  const xs = [...map.keys()].filter(Number.isFinite).sort((a, b) => a - b)
  if (xs.length === 0) return []
  // 빈 분 채우기: 키가 없는 분은 모든 값이 null → 선이 끊긴다
  const rows: Row[] = []
  for (let x = xs[0]; x <= xs[xs.length - 1]; x += MINUTE) {
    const r = map.get(x) ?? { x }
    for (const tr of trends) r[tr.feature] ??= null
    rows.push(r)
  }
  return rows
}

interface Props {
  trends: SensorTrend[]
  selected: string
  score?: ScorePoint[]
  codes?: ErrorSpan[]
}

export function SignalChart({ trends, selected, score = [], codes = [] }: Props) {
  const rows = buildRows(trends, score)
  if (rows.length === 0) {
    return <div className="flex h-40 items-center justify-center text-sm text-sub">그래프 데이터가 없습니다</div>
  }
  const x0 = rows[0].x
  const x1 = rows[rows.length - 1].x
  const spans = codes
    .map((c) => ({ ...c, s: parseDataTime(c.start)?.getTime() ?? NaN, e: parseDataTime(c.end)?.getTime() ?? NaN }))
    .filter((c) => c.e >= x0 && c.s <= x1 + MINUTE)
    // 그래프 범위 밖에서 시작 · 끝난 구간은 범위 안으로 자른다 (밖으로 나가면 그래프 전체가 가려진다)
    .map((c) => ({ ...c, inside: c.s >= x0, s0: Math.max(c.s, x0), e0: Math.min(Math.max(c.e, c.s + 1000), x1) }))
  const stopped = score.filter((p) => p.non_welding_share > 0.5).map((p) => minuteMs(p.t))
  const threshold = score[0]?.threshold
  const ticks = rows.filter((r) => Math.round((r.x - x0) / MINUTE) % 5 === 0).map((r) => r.x)
  const ordered = [...trends].sort((a) => (a.feature === selected ? 1 : -1)) // 선택한 선을 맨 위에 그린다
  const showScore = score.length > 0 && threshold !== undefined
  const yDomain: [(n: number) => number, (n: number) => number] = [
    (min) => Math.min(-2, Math.floor(min)),
    (max) => Math.max(2, Math.ceil(max)),
  ]

  const deviation = (
    <LineChart data={rows} margin={{ top: 34, right: 16, bottom: 0, left: 0 }} syncId="signal">
      <ReferenceArea y1={-2} y2={2} fill="#f2f4f8" ifOverflow="hidden" />
      <ReferenceArea y1={-1} y2={1} fill="#e6ebf3" ifOverflow="hidden" />
      {stopped.map((x) => (
        <ReferenceArea key={x} x1={x} x2={x + MINUTE} fill="#dde1ea" fillOpacity={0.8} />
      ))}
      <CartesianGrid vertical={false} stroke="transparent" />
      <XAxis dataKey="x" type="number" domain={[x0, x1]} ticks={ticks} hide={showScore} tickFormatter={(v: number) => kstTime.format(v)} tick={{ fontSize: 12, fill: '#6b7684' }} />
      <YAxis
        domain={yDomain}
        ticks={[-2, -1, 0, 1, 2]}
        width={44}
        tickFormatter={(v: number) => (v === 0 ? '평소' : v > 0 ? `+${v}` : `−${-v}`)}
        tick={{ fontSize: 12, fill: '#8b95a1' }}
        axisLine={false}
        tickLine={false}
      />
      <ReferenceLine y={0} stroke="#c9cfda" strokeDasharray="3 4" />
      {spans.map((c) => (
        <ReferenceArea key={`a${c.start}`} x1={c.s0} x2={c.e0} fill={RED} fillOpacity={0.1} />
      ))}
      {spans.filter((c) => c.inside).map((c) => (
        <ReferenceLine
          key={`l${c.start}`}
          x={c.s}
          stroke={RED}
          strokeWidth={2.5}
          label={{ value: `${c.code} ${kstTimeSec.format(c.s)} · ${c.duration_s}초`, position: 'top', fill: RED, fontSize: 13, fontWeight: 700 }}
        />
      ))}
      {ordered.map((t) => (
        <Line
          key={t.feature}
          dataKey={t.feature}
          name={signalLabel(t)}
          type="linear"
          dot={false}
          isAnimationActive={false}
          connectNulls={false}
          stroke={t.feature === selected ? BLUE : GRAY}
          strokeWidth={t.feature === selected ? 3 : 1.6}
        />
      ))}
      <Tooltip
        labelFormatter={(v) => kstTime.format(Number(v))}
        formatter={(v, name) => [v == null ? '데이터 없음' : Number(v).toFixed(2), name]}
      />
    </LineChart>
  )

  return (
    <div className="rounded-xl bg-[#fafbfd] px-2 pt-2 pb-1" role="img" aria-label="고장 전후 30분, 이 설비 평소 대비 신호 편차와 이상 점수">
      <div className={showScore ? 'h-[270px]' : 'h-[300px]'}>
        <ResponsiveContainer width="100%" height="100%">
          {deviation}
        </ResponsiveContainer>
      </div>
      {showScore && (
        <div>
          <div className="pl-11 text-xs font-semibold text-sub">이상 점수 · 0~1, 고장 확률 아님 · 점선 = 기준선 {threshold!.toFixed(2)}</div>
          <ResponsiveContainer width="100%" height={96}>
            <LineChart data={rows} margin={{ top: 6, right: 16, bottom: 0, left: 0 }} syncId="signal">
              <XAxis dataKey="x" type="number" domain={[x0, x1]} ticks={ticks} tickFormatter={(v: number) => kstTime.format(v)} tick={{ fontSize: 12, fill: '#6b7684' }} />
              <YAxis domain={[0, 1]} width={44} tick={false} axisLine={false} tickLine={false} />
              <ReferenceLine
                y={threshold}
                stroke="#8b95a1"
                strokeDasharray="4 3"
              />
              {spans.filter((c) => c.inside).map((c) => (
                <ReferenceLine key={c.start} x={c.s} stroke={RED} strokeWidth={2.5} />
              ))}
              <Line
                dataKey="score"
                name="이상 점수"
                dot={(p: { cx?: number; cy?: number; value?: number | null; index?: number }) =>
                  p.value != null && p.value > threshold! ? <circle key={p.index} cx={p.cx} cy={p.cy} r={4} fill={RED} /> : <g key={p.index} />
                }
                isAnimationActive={false}
                connectNulls={false}
                stroke="#6b7684"
                strokeWidth={1.8}
              />
              <Tooltip labelFormatter={(v) => kstTime.format(Number(v))} formatter={(v) => [v == null ? '데이터 없음' : Number(v).toFixed(3), '이상 점수']} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
