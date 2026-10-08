import { createReadStream, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

// 매뉴얼 PDF는 api 서버가 내려줄 파일이다 (docs/05). api가 생기기 전, mock 개발 중에만
// 레포의 rag/data/ 원본을 개발 서버가 그대로 내려준다. ui/로 복사하지 않고 빌드에도 넣지 않는다.
const MANUALS: Record<string, string> = {
  'festo.pdf': fileURLToPath(new URL('../rag/data/festo_manual.pdf', import.meta.url)),
}

function devManuals(): Plugin {
  return {
    name: 'dev-manuals',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/api/manuals/', (req, res) => {
        const file = MANUALS[(req.url ?? '').split('?')[0].replace(/^\//, '')]
        if (!file || !existsSync(file)) {
          res.statusCode = 404
          return res.end('manual not found')
        }
        res.setHeader('Content-Type', 'application/pdf')
        createReadStream(file).pipe(res)
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react(), tailwindcss(), ...(env.VITE_USE_MOCK === 'true' ? [devManuals()] : [])],
    server: {
      // MSW를 끄고 진짜 api 서버를 쓸 때, /api 요청을 그 서버로 넘긴다 (브라우저 CORS 회피)
      proxy: env.VITE_API_PROXY_TARGET
        ? { '/api': { target: env.VITE_API_PROXY_TARGET, changeOrigin: true } }
        : undefined,
    },
  }
})
