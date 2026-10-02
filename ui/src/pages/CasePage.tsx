import { Link, useParams } from 'react-router'

// ② 원인 찾기 · 조치 자리. 기준 화면은 wireframe/boards/Resolve1.dc.html (아직 구현 전)
export default function CasePage() {
  const { caseId } = useParams()

  return (
    <div className="mx-auto max-w-6xl">
      <Link to="/" className="text-sm font-semibold text-blue-700">
        ← 작업함
      </Link>
      <h1 className="mt-2 text-2xl font-bold">{caseId}</h1>
      <p className="mt-2 text-sm text-slate-500">원인 찾기 · 조치 화면은 아직 만들지 않았습니다.</p>
    </div>
  )
}
