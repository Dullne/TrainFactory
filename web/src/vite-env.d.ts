/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_TIMEOUT: string
  readonly VITE_AUTH_LOGOUT_TIMEOUT?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
