import { http, HttpResponse } from 'msw'
import health from './fixtures/health.json'

// 고정 JSON은 src/mocks/fixtures/에 둔다.
// ui/mock/은 깃에 안 올라가는 개인 폴더이고, data/라는 폴더 이름은 레포 루트 .gitignore에 걸린다.
// 경로는 '*/...'로 써서 VITE_API_BASE_URL 앞부분과 상관없이 잡히게 한다.
export const handlers = [
  http.get('*/api/health', () => HttpResponse.json(health)),
]
