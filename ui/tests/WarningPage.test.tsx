// @vitest-environment jsdom
import { afterAll, afterEach, beforeAll, expect, test, vi } from 'vitest'
import { spawn, type ChildProcess } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
import { render, screen, waitFor, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider, Outlet } from 'react-router'
import WarningPage from '../src/pages/case/WarningPage'

const base = 'http://127.0.0.1:19017/api'
vi.mock('../src/lib/api', async (original) => {
  const actual = await original<typeof import('../src/lib/api')>()
  const send = async (method: string, path: string, body?: unknown) => {
    const response = await fetch(`http://127.0.0.1:19017/api${path}`, { method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body) })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response.json()
  }
  return { ...actual, apiGet: (path: string) => send('GET', path), apiSend: send }
})
let server: ChildProcess
let serverLog = ''
beforeAll(async () => {
  server = spawn(process.env.WARNING_TEST_PYTHON ?? 'python', ['../api/tests/warning_demo_server.py'], { env: { ...process.env, WARNING_TEST_PORT: '19017' }, stdio: ['ignore', 'pipe', 'pipe'] })
  server.stderr?.on('data', data => { serverLog += String(data) })
  for (let i = 0; i < 100; i++) {
    try { const response = await fetch(`${base}/cases/CASE-0001`); if (response.ok) return } catch { /* server starting */ }
    if (server.exitCode !== null) throw new Error(serverLog)
    await delay(100)
  }
  throw new Error(`API test server unavailable: ${serverLog}`)
}, 20000)
afterAll(() => server?.kill())
afterEach(cleanup)

async function mount() {
  const d = await (await fetch(`${base}/cases/CASE-0001`)).json()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter([{ path: '/', element: <Outlet context={d} />, children: [{ index: true, element: <WarningPage /> }] }])
  return render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>)
}

test('real API: observations → candidate → selection → draft reload → manual source → confirmed resolution', async () => {
  const user = userEvent.setup()
  let view = await mount()
  await screen.findByText('이력 재사용 꺼짐 · 해결 기록은 저장됩니다.')
  expect(screen.getAllByText(/관측 60초/)).toHaveLength(3)
  expect((screen.getByRole('button', { name: '해결 결과 저장·종료' }) as HTMLButtonElement).disabled).toBe(true)
  await user.click(screen.getByRole('button', { name: '원인 후보 보기', exact: true }))
  await user.click(await screen.findByRole('button', { name: '이 사례 선택', exact: true }))
  await screen.findByRole('button', { name: '선택됨', exact: true })
  await user.type(screen.getByLabelText('작업자', { exact: true }), '테스트 작업자')
  await user.type(screen.getByLabelText('실제 수행한 조치', { exact: true }), '공급 계통 점검 후 정상 복귀를 확인함')
  await user.click(screen.getByRole('button', { name: '점검·조치 임시 저장', exact: true }))
  await waitFor(async () => {
    const state = await (await fetch(`${base}/cases/CASE-0001/warning`)).json()
    expect(state.draft.actual_action).toBe('공급 계통 점검 후 정상 복귀를 확인함')
  })
  view.unmount()
  view = await mount()
  await waitFor(() => expect((screen.getByLabelText('작업자', { exact: true }) as HTMLInputElement).value).toBe('테스트 작업자'))
  await user.type(screen.getByLabelText('직접 매뉴얼 검색', { exact: true }), '공급 계통을 어떻게 점검하나요?')
  await user.click(screen.getByRole('button', { name: '검색', exact: true }))
  const source = await screen.findByRole('button', { name: '이 근거 선택', exact: true })
  await user.click(source)
  await waitFor(async () => {
    const state = await (await fetch(`${base}/cases/CASE-0001/warning`)).json()
    expect(state.selection.manual_source.evidence_id).toBe('demo-proof')
  })
  await user.click(screen.getByLabelText('실제 조치 후 해결됐음을 확인했습니다.', { exact: true }))
  await user.click(screen.getByLabelText('위 정비일지를 확인하고 승인합니다.', { exact: true }))
  await user.click(screen.getByRole('button', { name: '해결 결과 저장·종료', exact: true }))
  await waitFor(async () => {
    const closed = await (await fetch(`${base}/cases/CASE-0001`)).json()
    expect(closed.status).toBe('resolved')
    expect(closed.warning_resolution_status).toBe('pending_review')
  })
  view.unmount()
  await mount()
  await screen.findByText('종료된 경보입니다. 기록은 읽기 전용입니다.')
  expect((screen.getByLabelText('실제 수행한 조치', { exact: true }) as HTMLTextAreaElement).disabled).toBe(true)
}, 20000)
