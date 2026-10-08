import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router'
import type { CaseDetail } from '../types/case'
import { apiGet, apiSend } from './api'

export function useCaseId() {
  const { caseId } = useParams()
  return caseId ?? ''
}

export function useCase() {
  const id = useCaseId()
  return useQuery({ queryKey: ['case', id], queryFn: () => apiGet<CaseDetail>(`/cases/${id}`) })
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
    },
  })
}
