import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'
import { CameraIcon, ChatIcon, DocIcon, PlusIcon, SendIcon } from '../components/Icons'
import { ManualPanel } from '../components/ManualPanel'
import { btn } from '../components/ui'
import { apiGet, apiSend, MANUAL_NAME, MANUAL_SHORT } from '../lib/api'
import { addPhotos, MAX_PHOTOS } from '../lib/photo'
import type { AnswerPart, AskInput, AssistantAnswer, Cite, SituationsResponse } from '../types/assistant'
import type { Photo } from '../types/case'

// AI 챗봇 (wireframe 시안 V2Chat): ChatGPT처럼 따로 여는 질문 화면.
// 왼쪽 대화 목록 / 가운데 대화 / 아래 입력창. 그냥 질문하면 api가 질문에 맞는 상황(S01~S10)을 골라
// 그 매뉴얼 근거를 바탕으로 답한다 (근거가 없으면 지어내지 않고 못 찾았다고 답함).
// 사진은 붙여 보낼 수 있지만 AI가 사진 속 상태를 판단하지는 않는다 (멀티모달 연결 전).

type UserMsg = { role: 'user'; text: string; photos: Photo[] }
type BotMsg = {
  role: 'assistant'
  state: 'pending' | 'done' | 'error'
  answer?: AssistantAnswer
  feedback?: 'up' | 'down'
  ask: AskInput
}
type Msg = UserMsg | BotMsg
interface Conversation {
  id: string
  title: string
  /** 첫 답이 고른 상황 ID. 대화 목록에 작게 보여준다 */
  topic: string | null
  at: string
  messages: Msg[]
}

// 화면을 옮겨 다녀도 대화가 남도록 모듈에 둔다 (새로고침하면 사라짐 · mock과 같은 수준)
let saved: Conversation[] = []

export default function AssistantPage() {
  const { data } = useQuery({ queryKey: ['situations'], queryFn: () => apiGet<SituationsResponse>('/situations') })
  const situations = data?.situations ?? []
  const [convs, setConvs] = useState<Conversation[]>(saved)
  // 케이스 화면의 "AI 챗봇에게 물어보기"는 ?q=질문 으로 온다: 새 대화에 질문을 채워 둔다 (보내기는 사람이)
  const [params] = useSearchParams()
  const preset = params.get('q') ?? ''
  const [activeId, setActiveId] = useState<string | null>(preset ? null : (saved[0]?.id ?? null))
  const [draft, setDraft] = useState(preset)
  const [photos, setPhotos] = useState<Photo[]>([])
  const [pdf, setPdf] = useState<Cite | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  const active = convs.find((c) => c.id === activeId) ?? null
  const nameOf = (sid: string | null) => (sid ? (situations.find((s) => s.situation_id === sid)?.name ?? sid) : '일반 질문')

  useEffect(() => {
    saved = convs
  }, [convs])
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [active?.messages.length, activeId])

  const patch = (id: string, fn: (c: Conversation) => Conversation) => setConvs((prev) => prev.map((c) => (c.id === id ? fn(c) : c)))

  const request = async (convId: string, index: number, ask: AskInput) => {
    try {
      const answer = await apiSend<AssistantAnswer>('POST', '/assistant/ask', ask)
      patch(convId, (c) => ({
        ...c,
        topic: c.topic ?? answer.situation_ids[0] ?? null,
        messages: c.messages.map((m, i) => (i === index ? { ...(m as BotMsg), state: 'done', answer } : m)),
      }))
    } catch {
      patch(convId, (c) => ({ ...c, messages: c.messages.map((m, i) => (i === index ? { ...(m as BotMsg), state: 'error' } : m)) }))
    }
  }

  const send = (text: string) => {
    const question = text.trim()
    if (!question) return
    const before = active?.messages ?? []
    const history = before.flatMap<AskInput['history'][number]>((m) =>
      m.role === 'user' ? [{ role: 'user', text: m.text }] : m.answer ? [{ role: 'assistant', text: plain(m.answer) }] : [],
    )
    const ask: AskInput = { question, photos, history }
    const added: Msg[] = [
      { role: 'user', text: question, photos },
      { role: 'assistant', state: 'pending', ask },
    ]
    const index = before.length + 1
    const id = active?.id ?? crypto.randomUUID()
    if (active) {
      patch(id, (c) => ({ ...c, messages: [...c.messages, ...added] }))
    } else {
      setConvs((prev) => [{ id, title: question.slice(0, 28), topic: null, at: new Date().toISOString(), messages: added }, ...prev])
      setActiveId(id)
    }
    setDraft('')
    setPhotos([])
    void request(id, index, ask)
  }

  const retry = (index: number, ask: AskInput) => {
    if (!active) return
    patch(active.id, (c) => ({ ...c, messages: c.messages.map((m, i) => (i === index ? { ...(m as BotMsg), state: 'pending' } : m)) }))
    void request(active.id, index, ask)
  }

  const feedback = (index: number, value: 'up' | 'down') => {
    if (!active) return
    patch(active.id, (c) => ({ ...c, messages: c.messages.map((m, i) => (i === index ? { ...(m as BotMsg), feedback: value } : m)) }))
    void apiSend('POST', '/assistant/feedback', { conversation_id: active.id, message_index: index, helpful: value === 'up' }).catch(() => {})
  }

  const newChat = () => {
    setActiveId(null)
    setDraft('')
    setPhotos([])
  }

  return (
    <div className="flex h-screen min-w-0">
      <aside className="flex w-[260px] shrink-0 flex-col gap-1 border-r border-line bg-white px-3 py-4" aria-label="대화 목록">
        {/* 위 흰 띠 없이, 화면 이름은 대화 목록 맨 위에 둔다 */}
        <h1 className="px-3 pt-1 pb-3 text-lg font-semibold">AI 챗봇</h1>
        <button type="button" className={`${btn} mb-3.5 font-semibold`} onClick={newChat}>
          <PlusIcon size={16} />새 대화
        </button>
        <span className="px-3 pb-1 text-xs font-medium text-sub">최근 대화</span>
        {convs.length === 0 && <p className="px-3 text-sm text-sub">아직 대화가 없습니다.</p>}
        <ol className="flex flex-col gap-0.5 overflow-y-auto">
          {convs.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                aria-current={c.id === activeId ? 'true' : undefined}
                onClick={() => setActiveId(c.id)}
                className={`flex w-full flex-col gap-0.5 rounded-lg px-3 py-2.5 text-left text-sm hover:bg-canvas ${c.id === activeId ? 'bg-fill font-semibold' : ''}`}
              >
                <span className="truncate">{c.title}</span>
                <span className="truncate text-xs font-normal text-sub">{nameOf(c.topic)}</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="grow" />
        <p className="px-3 text-xs leading-relaxed text-sub">연결된 문서: {Object.values(MANUAL_NAME).join(' · ')}</p>
      </aside>

      <div className="flex min-w-0 grow flex-col">
        <div className="flex min-h-0 grow">
          <div className="flex min-w-0 grow flex-col">
            <div className="min-h-0 grow overflow-y-auto">
              <div className="mx-auto flex w-full max-w-[760px] flex-col gap-5 px-8 py-6">
                {!active ? (
                  <Empty onPick={send} />
                ) : (
                  active.messages.map((m, i) =>
                    m.role === 'user' ? (
                      <UserBubble key={i} msg={m} />
                    ) : (
                      <BotBubble key={i} msg={m} onCite={setPdf} onRetry={() => retry(i, m.ask)} onFeedback={(v) => feedback(i, v)} />
                    ),
                  )
                )}
                <div ref={bottom} />
              </div>
            </div>

            <Composer
              draft={draft}
              setDraft={setDraft}
              photos={photos}
              setPhotos={setPhotos}
              onSend={() => send(draft)}
            />
          </div>
          {pdf && (
            <div className="w-[540px] shrink-0 border-l border-line bg-canvas p-4">
              <ManualPanel page={pdf.page} doc={pdf.doc} onPage={(page) => setPdf({ ...pdf, page })} onClose={() => setPdf(null)} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/** 답변을 글자로 (앞 대화를 api에 넘길 때) */
function plain(a: AssistantAnswer) {
  return a.blocks
    .map((b) => (b.type === 'ol' ? b.items.map((t, i) => `${i + 1}. ${t}`).join('\n') : b.parts.map((p) => ('cite' in p ? '' : p.text)).join('')))
    .join('\n')
}

const SUGGESTIONS = ['보정 압력이 늦게 올라오면 뭐부터 확인해?', 'E029 드리프트는 어떻게 점검해?', '전극 캡 마모는 어떻게 확인해?']

function Empty({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-3 pt-52 text-center">
      <span className="flex size-12 items-center justify-center rounded-xl bg-primary-soft text-primary-ink">
        <ChatIcon size={24} />
      </span>
      <h2 className="text-xl font-semibold">무엇이 궁금한가요?</h2>
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((q) => (
          <button key={q} type="button" onClick={() => onPick(q)} className="min-h-9 rounded-full border border-field bg-white px-3.5 text-sm font-medium text-ink-2 hover:border-primary">
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

function UserBubble({ msg }: { msg: UserMsg }) {
  return (
    <div className="flex max-w-[75%] flex-col items-end gap-1.5 self-end">
      {msg.photos.length > 0 && (
        <div className="flex flex-wrap justify-end gap-1.5">
          {msg.photos.map((p, i) => (
            <img key={i} src={p.url} alt={`보낸 사진 ${i + 1}`} className="h-28 w-40 rounded-xl border border-line object-cover" />
          ))}
        </div>
      )}
      <div className="rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-base whitespace-pre-wrap text-white">{msg.text}</div>
    </div>
  )
}

function BotBubble({
  msg,
  onCite,
  onRetry,
  onFeedback,
}: {
  msg: BotMsg
  onCite: (c: Cite) => void
  onRetry: () => void
  onFeedback: (v: 'up' | 'down') => void
}) {
  const [showSrc, setShowSrc] = useState(false)
  const a = msg.answer
  const pill = (on: boolean) =>
    `inline-flex min-h-[30px] items-center rounded-full border px-2.5 text-xs font-medium ${on ? 'border-primary bg-primary-soft text-primary-ink' : 'border-line bg-white text-ink-2 hover:bg-canvas'}`
  return (
    <div className="flex items-start gap-3">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary-soft text-primary-ink">
        <ChatIcon size={18} />
      </span>
      <div className="flex min-w-0 grow flex-col gap-2.5">
        {msg.state === 'pending' && (
          <p className="pt-1 text-sm text-sub" role="status">
            근거를 찾는 중…
          </p>
        )}
        {msg.state === 'error' && (
          <div className="flex items-center gap-3 pt-0.5 text-sm" role="alert">
            <span className="text-alarm">답을 가져오지 못했습니다.</span>
            <button type="button" className="font-medium text-primary-ink hover:underline" onClick={onRetry}>
              다시 시도
            </button>
          </div>
        )}
        {a && (
          <>
            <div className={`flex flex-col gap-2 text-base leading-relaxed ${a.status === 'no_evidence' ? 'text-ink-2' : ''}`}>
              {a.blocks.map((b, i) =>
                b.type === 'ol' ? (
                  <ol key={i} className="list-decimal pl-5">
                    {b.items.map((t, j) => (
                      <li key={j}>{t}</li>
                    ))}
                  </ol>
                ) : (
                  <p key={i}>
                    {b.parts.map((p, j) => (
                      <Part key={j} part={p} onCite={onCite} />
                    ))}
                  </p>
                ),
              )}
            </div>
            {a.caution && (
              <div className="flex items-start gap-2 rounded border-l-[3px] border-hint-line bg-hint px-3 py-2 text-sm text-ink-2">
                <b className="shrink-0 font-semibold text-warn">확인 필요</b>
                <span>{a.caution}</span>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-1.5">
              {a.status !== 'no_evidence' && (
                <>
                  <button type="button" aria-pressed={msg.feedback === 'up'} className={pill(msg.feedback === 'up')} onClick={() => onFeedback('up')}>
                    도움이 됐어요
                  </button>
                  <button type="button" aria-pressed={msg.feedback === 'down'} className={pill(msg.feedback === 'down')} onClick={() => onFeedback('down')}>
                    아니에요
                  </button>
                </>
              )}
              <div className="grow" />
              {a.evidence.length > 0 && (
                <button type="button" aria-expanded={showSrc} className={pill(false)} onClick={() => setShowSrc((v) => !v)}>
                  {showSrc ? '근거 접기' : `근거 ${a.evidence.length}개 보기`}
                </button>
              )}
            </div>
            {showSrc && (
              <ul className="flex flex-col gap-1.5">
                {a.evidence.map((e, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      onClick={() => onCite(e)}
                      className="flex w-full items-center gap-2.5 rounded-lg border border-line bg-white px-3 py-2 text-left text-sm hover:bg-canvas"
                    >
                      <span className="text-primary-ink">
                        <DocIcon size={15} />
                      </span>
                      <span className="min-w-0 grow truncate">
                        {MANUAL_NAME[e.doc]} · {e.title}
                      </span>
                      <span className="shrink-0 text-sub">{e.page}쪽</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function Part({ part, onCite }: { part: AnswerPart; onCite: (c: Cite) => void }) {
  if ('cite' in part) {
    const c = part.cite
    return (
      <button
        type="button"
        onClick={() => onCite(c)}
        className="mx-0.5 inline-flex items-center rounded-md bg-primary-soft px-1.5 align-[1px] text-xs font-semibold text-primary-ink hover:underline"
        title={`${MANUAL_NAME[c.doc]} ${c.page}쪽 열기`}
      >
        {MANUAL_SHORT[c.doc] ?? c.doc} {c.page}쪽
      </button>
    )
  }
  return part.bold ? <b className="font-semibold">{part.text}</b> : <>{part.text}</>
}

function Composer({
  draft,
  setDraft,
  photos,
  setPhotos,
  onSend,
}: {
  draft: string
  setDraft: (v: string) => void
  photos: Photo[]
  setPhotos: (p: Photo[]) => void
  onSend: () => void
}) {
  return (
    <div className="shrink-0 px-8 pt-2 pb-6">
      <div className="mx-auto flex w-full max-w-[760px] flex-col gap-2.5 rounded-2xl border border-field bg-white py-3 pr-3 pl-4 shadow-[0_2px_8px_rgba(16,24,40,0.06)] focus-within:border-primary">
        {photos.length > 0 && (
          <div className="flex items-center gap-2.5">
            {photos.map((p, i) => (
              <div key={i} className="relative">
                <img src={p.url} alt={`붙인 사진 ${i + 1}`} className="size-14 rounded-lg border border-line object-cover" />
                <button
                  type="button"
                  aria-label={`사진 ${i + 1} 빼기`}
                  onClick={() => setPhotos(photos.filter((_, j) => j !== i))}
                  className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full border border-field bg-white text-xs text-ink-2"
                >
                  ✕
                </button>
              </div>
            ))}
            <span className="text-xs text-sub">
              사진 {photos.length}장 · 최대 {MAX_PHOTOS}장
            </span>
          </div>
        )}
        <textarea
          rows={2}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            // Enter = 보내기, Shift+Enter = 줄바꿈 (한글 조합 중에는 보내지 않음)
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              onSend()
            }
          }}
          placeholder="매뉴얼 · 고장 상황에 대해 물어보세요"
          aria-label="질문"
          className="w-full resize-none border-0 bg-transparent text-base outline-none"
        />
        <div className="flex items-center gap-2">
          <label
            className={`flex size-9 shrink-0 items-center justify-center rounded-full border border-field text-ink-2 hover:bg-canvas ${photos.length >= MAX_PHOTOS ? 'pointer-events-none opacity-40' : 'cursor-pointer'}`}
            title="사진 찍기 · 올리기"
          >
            <span className="sr-only">사진 찍기 · 올리기</span>
            <CameraIcon size={20} />
            <input
              type="file"
              accept="image/*"
              capture="environment"
              multiple
              className="sr-only"
              disabled={photos.length >= MAX_PHOTOS}
              onChange={async (e) => {
                const files = [...(e.target.files ?? [])]
                e.target.value = ''
                setPhotos(await addPhotos(photos, files))
              }}
            />
          </label>
          <div className="grow" />
          <button
            type="button"
            aria-label="보내기"
            disabled={!draft.trim()}
            onClick={onSend}
            className="flex size-10 items-center justify-center rounded-xl bg-primary text-white hover:bg-primary-ink disabled:opacity-40"
          >
            <SendIcon size={18} />
          </button>
        </div>
      </div>
    </div>
  )
}
