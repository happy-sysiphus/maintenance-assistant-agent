import { useOutletContext } from 'react-router'
import type { CaseDetail } from '../../types/case'

/** 케이스 화면 안에서 불러온 케이스 (CaseLayout이 Outlet context로 넘긴다) */
export function useCaseDetail() {
  return useOutletContext<CaseDetail>()
}
