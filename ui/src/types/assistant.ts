// AI 챗봇 응답 타입. FE 제안이고 api · RAG 담당과 정하면 여기를 고친다 (docs/05 "AI 챗봇" 절).
// RAG answer_question(situation_ids, question)은 마크다운 글 + evidence를 주므로,
// api가 인용 표시([Festo PDF p.N], [RSW PDF p.N])를 cite로 나눠서 내려주는 것을 가정한다.
import type { ManualDoc } from '../lib/api'
import type { Photo } from './case'

/** 상황 ID 하나 (S01~S10). 챗봇은 고른 상황의 근거로만 답한다 */
export interface Situation {
  situation_id: string
  name: string
  definition: string
  /** 상황 정의서에서 정의가 있는 쪽 */
  definition_page: number
  check_sequence: string[]
  /** 상황 정의서에서 점검 순서가 있는 쪽 */
  check_source_page: number
  coverage_note: string | null
  evidence: { page: number; rationale: string | null }[]
  diagnostics: {
    number: number
    priority: string | null
    rationale: string | null
    manual?: { name: string; page: number; text: string }
  }[]
}

export interface SituationsResponse {
  situations: Situation[]
}

export interface Cite {
  doc: ManualDoc
  page: number
}

/** 답변 문장 조각: 글자, 굵은 글자, 인용(쪽 칩) */
export type AnswerPart = { text: string; bold?: boolean } | { cite: Cite }

export type AnswerBlock = { type: 'p'; parts: AnswerPart[] } | { type: 'ol'; items: string[] }

/** RAG status와 같은 값을 쓴다 (docs/06) */
export type AnswerStatus = 'ok' | 'retrieval_only' | 'no_evidence' | 'needs_citation_review' | 'unknown_situation_id' | 'error'

export interface AssistantAnswer {
  status: AnswerStatus
  /** 질문에 맞춰 api가 고른 상황 ID (답의 근거 범위). 못 고르면 빈 배열 */
  situation_ids: string[]
  blocks: AnswerBlock[]
  /** "확인 필요" 한 줄 (매뉴얼 기준과 설비 기준이 다를 때 등). 없으면 null */
  caution: string | null
  evidence: (Cite & { title: string })[]
}

/** 질문만 보낸다. 어떤 상황(S01~S10)의 근거를 쓸지는 api가 질문 · 앞 대화를 보고 고른다 */
export interface AskInput {
  question: string
  photos: Photo[]
  /** 앞 대화. RAG 함수는 대화를 기억하지 않아서 api가 같이 넘긴다 */
  history: { role: 'user' | 'assistant'; text: string }[]
}
