// @vitest-environment jsdom
// 조기 경보 화면 흐름 (11/11 흐름도, 11/12 화면): 실제 api(합성 서버, ../api/tests/warning_demo_server.py)를 띄우고 누른다.
// 검색 · Claude · Aura 이력만 테스트 대역. 실행: WARNING_TEST_PYTHON=<api 가상환경 python> npm run test:warning
import { afterAll, afterEach, beforeAll, expect, test, vi } from 'vitest'
import { spawn, type ChildProcess } from 'node:child_process'
import { setTimeout as delay } from 'node:timers/promises'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createMemoryRouter, RouterProvider } from 'react-router'
import CaseLayout from '../src/pages/case/CaseLayout'
import OverviewPage from '../src/pages/case/OverviewPage'
import CheckPage from '../src/pages/case/CheckPage'
import ActionPage from '../src/pages/case/ActionPage'
import LogPage from '../src/pages/case/LogPage'
import ManualModePage from '../src/pages/case/ManualModePage'

const PORT = '19018'
const base = `http://127.0.0.1:${PORT}/api`
vi.mock('../src/lib/api', async (original) => {
  const actual = await original<typeof import('../src/lib/api')>()
  const send = async (method: string, path: string, body?: unknown) => {
    const response = await fetch(`http://127.0.0.1:19018/api${path}`, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
    if (!response.ok) throw new actual.ApiError(response.status, `${method} ${path} 실패 (${response.status})`)
    return response.json()
  }
  return { ...actual, apiGet: (path: string) => send('GET', path), apiSend: send }
})
// 그래프는 이 테스트의 대상이 아니다 (jsdom에는 크기 측정이 없다)
vi.mock('../src/components/SignalChart', () => ({ SignalChart: () => null }))

let server: ChildProcess
let serverLog = ''
beforeAll(async () => {
  server = spawn(process.env.WARNING_TEST_PYTHON ?? 'python', ['../api/tests/warning_demo_server.py'], {
    env: { ...process.env, WARNING_TEST_PORT: PORT },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  server.stderr?.on('data', (data) => (serverLog += String(data)))
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch(`${base}/cases/CASE-0001`)).ok) return
    } catch {
      /* 서버 켜지는 중 */
    }
    if (server.exitCode !== null) throw new Error(serverLog)
    await delay(100)
  }
  throw new Error(`api 테스트 서버가 안 켜짐: ${serverLog}`)
}, 20000)
afterAll(() => server?.kill())
afterEach(cleanup)

function mount(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const router = createMemoryRouter(
    [
      {
        path: '/cases/:caseId',
        element: <CaseLayout />,
        children: [
          { index: true, element: <OverviewPage /> },
          { path: 'check', element: <CheckPage /> },
          { path: 'action', element: <ActionPage /> },
          { path: 'log', element: <LogPage /> },
          { path: 'manual', element: <ManualModePage /> },
        ],
      },
      { path: '/', element: <p>작업함</p> },
    ],
    { initialEntries: [path] },
  )
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

const state = async (cid: string) => (await fetch(`${base}/cases/${cid}`)).json()

test('첫 화면 → 원인 후보 → 점검 → 조치 → 정비일지 → 해결 종료', async () => {
  const user = userEvent.setup()
  mount('/cases/CASE-0001')
  // B. 첫 화면: 할 일은 원인 후보 보기 하나 (상세 조회만으로 후보를 만들지 않는다)
  await user.click(await screen.findByRole('button', { name: '원인 후보 보기' }))
  // C. 후보 목록 → 점검
  await user.click(await screen.findByRole('button', { name: '1번 원인으로 점검' }))
  await screen.findByRole('navigation', { name: '진행 단계' })
  expect(screen.queryByText('판단')).toBeNull()
  // 모두 정상이면 조치로 가지 않고 "이 원인은 아닌 것 같아요" (판단 단계 대신)
  await user.click(screen.getAllByRole('radio', { name: /정상/ })[0])
  await user.click(screen.getByRole('button', { name: '다음' }))
  await screen.findByText('점검한 항목에 이상이 없어요')
  // 후보가 이것 하나뿐이라 다음 원인 대신 바로 수동 모드 (흐름도 "맞는 후보 없음 → 수동 모드")
  expect(screen.getByRole('button', { name: '매뉴얼에서 직접 찾기' })).toBeTruthy()
  // 이상이 있으면 조치로
  await user.click(screen.getAllByRole('radio', { name: /이상/ })[0])
  await user.click(screen.getByRole('button', { name: '다음' }))
  // 조치
  await user.click(await screen.findByRole('button', { name: '교체' }))
  await user.type(screen.getByPlaceholderText('실제로 한 작업을 적어 주세요'), '공급 계통 점검 후 케이블 교체')
  await user.type(screen.getByPlaceholderText('이름'), '테스트 작업자')
  await user.click(screen.getByRole('button', { name: '다음' }))
  // 정비일지: 해결 확인 체크 전에는 닫을 수 없다
  const close = await screen.findByRole('button', { name: '일지 승인하고 해결 종료' })
  expect((close as HTMLButtonElement).disabled).toBe(true)
  await screen.findByText('교체 — 공급 계통 점검 후 케이블 교체')
  await user.click(screen.getByLabelText('조치 후 정상으로 돌아온 것을 확인했어요'))
  await user.click(close)
  await waitFor(async () => expect((await state('CASE-0001')).status).toBe('resolved'))
  // 닫히면 상황 화면에서 해결 기록(피드백 루프)을 보여 준다
  await screen.findByText('✓ 해결로 닫았어요')
}, 30000)

test('맞는 원인이 없으면 수동 모드에서 근거를 골라 점검으로', async () => {
  const user = userEvent.setup()
  // 같은 경보 원본을 다른 설비 · 번호로 하나 더 넣는다
  const first = await state('CASE-0001')
  const event = { ...first.event, event_id: 'EW-DEMO-0002', gun_id: 'DEMO-GUN-02' }
  expect((await fetch(`${base}/ml/handoffs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(event) })).ok).toBe(true)
  mount('/cases/CASE-0002')
  await user.click(await screen.findByRole('button', { name: '원인 후보 보기' }))
  await user.click(await screen.findByRole('link', { name: '맞는 원인이 없어요 · 직접 찾기' }))
  await user.type(await screen.findByLabelText('질문'), '공급 계통을 어떻게 점검하나요?{Enter}')
  await screen.findByText('검색된 매뉴얼의 공급 계통 점검 항목을 확인하세요.')
  await user.click(screen.getByRole('button', { name: '이 근거로 정비' }))
  // 직접 찾은 근거로 정비할 때는 원인을 적어야 다음으로 간다 (정비일지 · 이력의 원인)
  const cause = await screen.findByLabelText('원인')
  expect((screen.getByRole('button', { name: '다음' }) as HTMLButtonElement).disabled).toBe(true)
  await user.type(cause, 'MPYD 케이블 접촉 불량')
  expect((screen.getByRole('button', { name: '다음' }) as HTMLButtonElement).disabled).toBe(false)
  const s = await (await fetch(`${base}/cases/CASE-0002/warning?event_id=EW-DEMO-0002`)).json()
  expect(s.selection.manual_source.evidence_id).toBe('demo-proof')
}, 30000)
