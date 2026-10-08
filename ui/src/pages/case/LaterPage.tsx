import { Link } from 'react-router'
import { btn, card } from '../../components/ui'

// 다음 묶음에서 만들 화면의 자리 (수동 모드 · 현장 확인 입력 · 도움 요청)
export default function LaterPage({ title, board }: { title: string; board: string }) {
  return (
    <section className={`${card} flex flex-col items-start gap-3 p-[30px]`}>
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="text-sm text-sub">
        이 화면은 다음 묶음에서 만듭니다. 기준 화면: <code>wireframe/boards/{board}</code>
      </p>
      <Link to=".." relative="path" className={btn}>
        상황으로 돌아가기
      </Link>
    </section>
  )
}
