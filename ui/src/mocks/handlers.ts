import { delay, http, HttpResponse } from 'msw'
import type { CaseListResponse } from '../types/case'
import cases from './fixtures/cases.json'
import health from './fixtures/health.json'

// 고정 JSON은 src/mocks/fixtures/에 둔다.
// ui/mock/은 깃에 안 올라가는 개인 폴더이고, data/라는 폴더 이름은 레포 루트 .gitignore에 걸린다.
// 경로는 '*/...'로 써서 VITE_API_BASE_URL 앞부분과 상관없이 잡히게 한다.

// 화면 상태 확인용: 브라우저 주소에 ?mock=empty | error | slow 를 붙인다 (핸들러는 페이지에서 돌아서 location을 읽을 수 있다)
function mockMode() {
  return new URLSearchParams(window.location.search).get('mock')
}

export const handlers = [
  http.get('*/api/health', () => HttpResponse.json(health)),

  http.get('*/api/cases', async () => {
    const mode = mockMode()
    if (mode === 'slow') await delay(3000)
    if (mode === 'error') return HttpResponse.json({ error: 'mock error' }, { status: 500 })
    if (mode === 'empty') return HttpResponse.json<CaseListResponse>({ cases: [] })
    return HttpResponse.json(cases as CaseListResponse)
  }),
]
