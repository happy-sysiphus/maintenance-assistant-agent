import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router'
import type { CaseDetail } from '../types/case'
import { apiGet, apiSend } from './api'
import { expandCauses } from './guidance'

export function useCaseId() {
  const { caseId } = useParams()
  return caseId ?? ''
}

export function useCaseQuery(id: string) {
  // RAG 원인 목록(causes)을 원인 카드로 푼다. 캐시에는 서버 응답 그대로 둔다
  return useQuery({ queryKey: ['case', id], queryFn: () => apiGet<CaseDetail>(`/cases/${id}`), select: expandCauses })
}

/** 주소의 :caseId 케이스 */
export function useCase() {
  return useCaseQuery(useCaseId())
}

/** 케이스에 무언가를 저장하고, 서버가 돌려준 최신 케이스로 화면을 갱신한다 */
export function useCaseMutation<T>(method: 'POST' | 'PUT', path: string) {
  const id = useCaseId()
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: T) => apiSend<CaseDetail>(method, `/cases/${id}${path}`, body),
    onSuccess: (d) => {
      qc.setQueryData(['case', id], d)
      qc.invalidateQueries({ queryKey: ['cases'] })
      qc.invalidateQueries({ queryKey: ['history'] })
      qc.invalidateQueries({ queryKey: ['guns'] })
    },
  })
}
