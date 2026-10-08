import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import './index.css'
import App from './App.tsx'
import { setMockReconnect } from './lib/api.ts'
import EquipmentPage from './pages/EquipmentPage.tsx'
import HistoryPage from './pages/HistoryPage.tsx'
import InboxPage from './pages/InboxPage.tsx'
import ActionPage from './pages/case/ActionPage.tsx'
import CaseLayout from './pages/case/CaseLayout.tsx'
import CheckPage from './pages/case/CheckPage.tsx'
import JudgePage from './pages/case/JudgePage.tsx'
import FieldPage from './pages/case/FieldPage.tsx'
import HandoverPage from './pages/case/HandoverPage.tsx'
import LogPage from './pages/case/LogPage.tsx'
import ManualModePage from './pages/case/ManualModePage.tsx'
import NoCausePage from './pages/case/NoCausePage.tsx'
import OverviewPage from './pages/case/OverviewPage.tsx'
import ResultPage from './pages/case/ResultPage.tsx'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <InboxPage /> },
      { path: 'history', element: <HistoryPage /> },
      { path: 'equipment', element: <EquipmentPage /> },
      {
        path: 'cases/:caseId',
        element: <CaseLayout />,
        children: [
          { index: true, element: <OverviewPage /> },
          { path: 'check', element: <CheckPage /> },
          { path: 'judge', element: <JudgePage /> },
          { path: 'action', element: <ActionPage /> },
          { path: 'result', element: <ResultPage /> },
          { path: 'no-cause', element: <NoCausePage /> },
          { path: 'log', element: <LogPage /> },
          { path: 'manual', element: <ManualModePage /> },
          { path: 'field', element: <FieldPage /> },
          { path: 'handover', element: <HandoverPage /> },
        ],
      },
    ],
  },
])

// 실패 시 재시도는 1번만 (기본 3번이면 오류 표시까지 7초 넘게 걸림)
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1 } } })

// VITE_USE_MOCK=true면 MSW를 먼저 켜고 나서 화면을 그린다 (첫 요청부터 mock이 받도록)
async function enableMocking() {
  if (import.meta.env.VITE_USE_MOCK !== 'true') return
  const { worker } = await import('./mocks/browser')
  await worker.start({ onUnhandledFrame: 'bypass' })

  // 탭을 오래 두면 브라우저가 쉬는 서비스 워커를 끄고, 다시 켜진 워커는 이 탭을 모른다(요청이 진짜 서버로 샌다).
  // 탭으로 돌아올 때와, 요청이 샌 것이 보일 때(lib/api.ts) 다시 연결한다. 저장한 mock 데이터는 페이지에 있어 그대로다.
  let pending: Promise<unknown> | null = null
  const reconnect = () =>
    (pending ??= (async () => {
      await worker.stop()
      await worker.start({ onUnhandledFrame: 'bypass', quiet: true })
    })().finally(() => {
      pending = null
    }))
  setMockReconnect(reconnect)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void reconnect()
  })
}

enableMocking().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>,
  )
})
