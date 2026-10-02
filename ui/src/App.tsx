import { useQuery } from '@tanstack/react-query'
import { apiGet } from './lib/api'

type Health = { status: string; source: string }

// 빈 화면 확인용. 화면 구현을 시작하면 라우트별 페이지로 바꾼다.
export default function App() {
  const { data, isPending, isError } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<Health>('/health'),
  })

  return (
    <main className="min-h-screen bg-slate-50 p-8 text-slate-900">
      <h1 className="text-2xl font-bold">RSW 정비 지원</h1>
      <p className="mt-2 text-sm text-slate-500">
        api 연결: {isPending ? '확인 중' : isError ? '실패' : `${data.status} (${data.source})`}
      </p>
    </main>
  )
}
