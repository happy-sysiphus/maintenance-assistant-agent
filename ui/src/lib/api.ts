// 화면 코드는 이 함수로만 api 서버를 부른다.
// mock(MSW)이든 진짜 서버든 주소 앞부분만 VITE_API_BASE_URL로 바뀐다.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) throw new ApiError(res.status, `GET ${path} 실패 (${res.status})`)
  return (await res.json()) as T
}
