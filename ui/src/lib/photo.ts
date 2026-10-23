import type { Photo } from '../types/case'

// 사진 첨부 공용: 현장 확인 · AI 챗봇가 같이 쓴다.
// 업로드 api가 없어 지금은 data URL로 바꿔 요청에 같이 실어 보낸다 (api가 생기면 업로드한 파일 주소로 바꾼다).

export const MAX_PHOTOS = 4

export function readPhoto(file: File): Promise<Photo> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve({ name: file.name, url: String(reader.result) })
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

/** 고른 파일을 남은 자리만큼 읽어서 붙인다. input을 비우기 전에 배열로 복사해서 넘길 것 (비우면 FileList도 비워진다) */
export async function addPhotos(prev: Photo[], files: File[] | FileList | null): Promise<Photo[]> {
  if (!files) return prev
  const added = await Promise.all([...files].slice(0, MAX_PHOTOS - prev.length).map(readPhoto))
  return [...prev, ...added].slice(0, MAX_PHOTOS)
}
