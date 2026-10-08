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

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) throw new ApiError(res.status, `${method} ${path} 실패 (${res.status})`)
  return (await res.json()) as T
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>('GET', path)
}

export function apiSend<T>(method: 'POST' | 'PUT', path: string, body: unknown): Promise<T> {
  return request<T>(method, path, body)
}
