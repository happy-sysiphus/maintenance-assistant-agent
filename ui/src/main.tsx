import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import './index.css'
import App from './App.tsx'
import InboxPage from './pages/InboxPage.tsx'
import ActionPage from './pages/case/ActionPage.tsx'
import CaseLayout from './pages/case/CaseLayout.tsx'
import CheckPage from './pages/case/CheckPage.tsx'
import JudgePage from './pages/case/JudgePage.tsx'
import LaterPage from './pages/case/LaterPage.tsx'
import NoCausePage from './pages/case/NoCausePage.tsx'
import OverviewPage from './pages/case/OverviewPage.tsx'
import ResultPage from './pages/case/ResultPage.tsx'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <InboxPage /> },
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
          // 다음 묶음에서 만들 화면
          { path: 'log', element: <LaterPage title="정비일지" board="V2Log.dc.html" /> },
          { path: 'manual', element: <LaterPage title="직접 점검 (수동 모드)" board="V2NoGuide.dc.html" /> },
          { path: 'field', element: <LaterPage title="현장 확인 입력" board="V2Field.dc.html" /> },
          { path: 'handover', element: <LaterPage title="도움 요청" board="V2Handover.dc.html" /> },
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
