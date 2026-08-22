import { useState } from 'react'
import ReactDOM from 'react-dom/client'
import { AuthProvider, useAuth } from '@/auth/AuthContext'
import { ApiError, toApiError } from '@/services/ApiError'

function AuthContextHarness() {
  const { status, refreshUser, logout } = useAuth()
  const [refreshSettledCount, setRefreshSettledCount] = useState(0)
  const [refreshResult, setRefreshResult] = useState('idle')
  const [apiErrorIdentity] = useState(() => {
    const original = new ApiError('original-message', 'http', 409, 'E_EXISTING', '/auth/me')
    const converted = toApiError(original, 'fallback-message')
    return `${converted === original ? 'same' : 'different'}:${converted.kind}:${String(converted.status ?? 'none')}:${String(converted.code ?? 'none')}:${String(converted.requestPath ?? 'none')}:${converted.message}`
  })

  const handleRefresh = () => {
    setRefreshResult('pending')
    void refreshUser()
      .then(
        () => setRefreshResult('resolved'),
        (error: unknown) => {
          const shape =
            error && typeof error === 'object'
              ? (error as {
                  name?: unknown
                  kind?: unknown
                  status?: unknown
                  requestPath?: unknown
                })
              : {}
          setRefreshResult(
            `rejected:${String(shape.name ?? 'unknown')}:${String(shape.kind ?? 'unknown')}:${String(shape.status ?? 'unknown')}:${String(shape.requestPath ?? 'unknown')}`
          )
        }
      )
      .finally(() => setRefreshSettledCount((count) => count + 1))
  }

  const handleLogout = () => {
    void logout().catch(() => undefined)
  }

  return (
    <main>
      <output aria-label="Authentication status">{status}</output>
      <output aria-label="Refresh settled count">{refreshSettledCount}</output>
      <output aria-label="Refresh result">{refreshResult}</output>
      <output aria-label="ApiError identity">{apiErrorIdentity}</output>
      <button type="button" onClick={handleRefresh}>
        Refresh
      </button>
      <button type="button" onClick={handleLogout}>
        Logout
      </button>
    </main>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <AuthProvider>
    <AuthContextHarness />
  </AuthProvider>
)
