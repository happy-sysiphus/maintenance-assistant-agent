// 화면 코드는 이 함수들로만 api 서버를 부른다.
// mock(MSW)이든 진짜 서버든 주소 앞부분만 VITE_API_BASE_URL로 바뀐다.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// mock 모드에서 MSW를 다시 연결하는 함수 (main.tsx가 넣는다). 진짜 api 모드에서는 null.
let reconnectMock: (() => Promise<unknown>) | null = null
export function setMockReconnect(fn: () => Promise<unknown>) {
  reconnectMock = fn
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const send = () =>
    fetch(`${BASE_URL}${path}`, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  let res = await send()
  // mock 모드인데 JSON이 아닌 응답(개발 서버의 프록시 오류 · 404 페이지)이면 요청이 MSW를 거치지 않은 것이다.
  // 브라우저가 쉬는 서비스 워커를 끄면 MSW가 이 탭을 잊어서 생긴다 → 다시 연결하고 한 번만 다시 보낸다.
  if (reconnectMock && !res.headers.get('Content-Type')?.includes('application/json')) {
    await reconnectMock()
    res = await send()
  }
  if (!res.ok) throw new ApiError(res.status, `${method} ${path} 실패 (${res.status})`)
  return (await res.json()) as T
}

/** 매뉴얼 PDF 주소. page는 PDF 파일의 쪽 순서(1부터)이고 RAG의 page_number와 같다 */
export function manualUrl(page: number) {
  return `${BASE_URL}/manuals/festo.pdf#page=${page}`
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>('GET', path)
}

export function apiSend<T>(method: 'POST' | 'PUT', path: string, body: unknown): Promise<T> {
  return request<T>(method, path, body)
}
