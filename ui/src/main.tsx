import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter } from 'react-router'
import { RouterProvider } from 'react-router/dom'
import './index.css'
import App from './App.tsx'
import CasePage from './pages/CasePage.tsx'
import InboxPage from './pages/InboxPage.tsx'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <InboxPage /> },
      { path: 'cases/:caseId', element: <CasePage /> },
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
