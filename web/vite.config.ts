import { defineConfig, loadEnv, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

function createE2ERunMarker(runId: string): Plugin {
  return {
    name: 'train-factory-e2e-run-marker',
    transformIndexHtml() {
      return [
        {
          tag: 'meta',
          attrs: { name: 'train-factory-e2e-run', content: runId },
          injectTo: 'head',
        },
      ]
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const e2eRunId = process.env.VITE_E2E_RUN_ID || env.VITE_E2E_RUN_ID

  return {
    plugins: [react(), ...(e2eRunId ? [createE2ERunMarker(e2eRunId)] : [])],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 3000,
      host: true,
      allowedHosts: true,
      proxy: {
        '/api': {
          target:
            process.env.VITE_PROXY_TARGET || env.VITE_PROXY_TARGET || 'http://localhost:18001',
          changeOrigin: true,
          configure(proxy) {
            proxy.on('proxyReq', (proxyReq, request) => {
              const remoteAddress = request.socket.remoteAddress
              if (remoteAddress) {
                proxyReq.setHeader('x-forwarded-for', remoteAddress)
              } else {
                proxyReq.removeHeader('x-forwarded-for')
              }
            })
          },
        },
      },
    },
    build: {
      manifest: true,
      // Output directory
      outDir: 'dist',
      // Enable source maps for debugging
      sourcemap: mode !== 'production',
      // Chunk size warning limit (500kb)
      chunkSizeWarningLimit: 500,
      rollupOptions: {
        output: {
          onlyExplicitManualChunks: true,
          manualChunks(id) {
            const modulePath = id.replace(/\\/g, '/')
            if (/\/node_modules\/(react|react-dom|scheduler)\//.test(modulePath)) {
              return 'react-runtime'
            }
            if (modulePath.includes('/src/i18n/locales/')) {
              return 'locales'
            }
            // Keep the renderer in the lazy chart dependency tree.
            if (modulePath.includes('/node_modules/zrender/')) {
              return 'chart-renderer'
            }
          },
          // Asset file naming
          assetFileNames: 'assets/[name]-[hash][extname]',
          chunkFileNames: 'chunks/[name]-[hash].js',
          entryFileNames: 'js/[name]-[hash].js',
        },
      },
      // Minification options
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: mode === 'production',
          drop_debugger: mode === 'production',
        },
      },
    },
  }
})
