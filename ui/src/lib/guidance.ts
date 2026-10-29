// RAG 안내(10/28, docs/09)를 화면의 원인 카드 모양으로 바꾼다.
// RAG는 고장 코드 하나에 카드 하나(candidates[0])를 주고, 원인 후보는 그 안의 causes[]에 담는다.
// 화면은 원인 카드 여러 장(원인 1 → 다음 원인)으로 만들어져 있으므로 causes[] 하나를 카드 하나로 푼다.
// causes가 없는 응답(옛 api · mock)은 그대로 둔다.
import { MANUAL_NAME, type ManualDoc } from './api'
import type { Candidate, CaseDetail, Guidance, RagCite } from '../types/case'

/** 본문에 섞인 인용 표시 "[manual82 p.44]"를 지운다 (화면은 근거를 쪽 버튼으로 따로 보여준다) */
export function stripCites(text: string) {
  return text
    .replace(/\s*\[[a-z0-9]+ p\.\d+\]/g, '')
    .replace(/\s+([.,])/g, '$1')
    .trim()
}

/** 첫 문장만 ("~다." 까지). 문장 끝을 못 찾으면 전체 */
export function firstSentence(text: string) {
  return text.match(/^.*?다\./)?.[0] ?? text
}

const isDoc = (doc: string): doc is ManualDoc => doc in MANUAL_NAME

function citesIn(text: string): RagCite[] {
  return [...text.matchAll(/\[([a-z0-9]+) p\.(\d+)\]/g)].map((m) => ({ doc: m[1], page: Number(m[2]) }))
}

function expand(g: Guidance): Guidance {
  const base = g.candidates[0]
  if (!base || !g.causes?.length) return g
  const known = new Map(base.checks.map((c) => [c.id, c]))
  const candidates: Candidate[] = g.causes.map((cause, i) => {
    // 근거 쪽: 이 원인의 점검 · 조치 · 설명에 달린 인용 (같은 문서의 같은 쪽은 한 번만)
    const seen = new Set<string>()
    const evidence: Candidate['evidence'] = []
    const add = (cite: RagCite | undefined, rationale: string) => {
      if (!cite || !isDoc(cite.doc) || seen.has(`${cite.doc}:${cite.page}`)) return
      seen.add(`${cite.doc}:${cite.page}`)
      evidence.push({ doc: cite.doc, page: cite.page, quote: cite.quote ?? null, rationale })
    }
    for (const c of cause.checks) add(c.cite, c.text)
    for (const a of cause.actions ?? []) add(a.cite, a.text)
    for (const c of citesIn(cause.why)) add(c, cause.name)
    return {
      ...base,
      // 판단 · 결과 기록을 원인별로 남기려고 코드 뒤에 원인 번호를 붙인다 (E016-1, E016-2)
      situation_id: `${base.situation_id}-${i + 1}`,
      rank: i + 1,
      name: cause.name,
      why: stripCites(cause.why),
      checks: cause.checks.map((c) => known.get(c.id) ?? { id: c.id, title: c.text }),
      actions: cause.actions ?? [],
      evidence: evidence.length ? evidence : base.evidence,
    }
  })
  return { ...g, candidates }
}

/** 매뉴얼에 점검 · 조치가 없는 원인에서 챗봇으로 넘길 주소 (질문을 채워 둔다). 원인 카드 id는 "E012-2" 꼴 */
export function askAboutCause(c: Candidate, what: '확인' | '조치') {
  const code = c.situation_id.split('-')[0]
  const q = `${code}에서 '${c.name}' 원인은 ${what === '확인' ? '어떻게 확인하나요?' : '어떻게 조치하나요?'}`
  return `/assistant?q=${encodeURIComponent(q)}`
}

/** useQuery의 select로 쓴다: 서버 응답(캐시)은 그대로 두고 화면에 줄 모양만 바꾼다 */
export function expandCauses(d: CaseDetail): CaseDetail {
  return { ...d, guidance: expand(d.guidance) }
}
