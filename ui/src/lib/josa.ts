// 한국어 조사 고르기: 앞 글자의 받침을 보고 "이/가", "을/를" 등을 붙인다.
// 마지막 글자가 한글이 아니면(영문 · 숫자 · 괄호) 받침을 알 수 없어 "이(가)"처럼 둘 다 쓴다.

export function josa(word: string, withFinal: string, withoutFinal: string) {
  const code = word.trim().charCodeAt(word.trim().length - 1) - 0xac00
  if (code < 0 || code > 11171) return `${word}${withFinal}(${withoutFinal})`
  return `${word}${code % 28 ? withFinal : withoutFinal}`
}
