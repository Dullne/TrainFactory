import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  authApi,
  type AuthConfigResponse,
  type AuthLoginRequest,
  type AuthRegisterRequest,
  type AuthUser,
} from '@/services/api'
import { isApiError } from '@/services/ApiError'
import { offerLoginCredentialForSaving } from '@/auth/passwordCredential'
import { clearWorkspaceSession } from '@/auth/clearWorkspaceSession'

export type { AuthUser } from '@/services/api'

export type AuthStatus = 'checking' | 'authenticated' | 'unauthenticated'

export interface AuthConfig extends AuthConfigResponse {
  apiAvailable: boolean
}

export interface AuthContextValue {
  user: AuthUser | null
  status: AuthStatus
  config: AuthConfig
  login(values: AuthLoginRequest): Promise<void>
  register(values: AuthRegisterRequest): Promise<void>
  logout(): Promise<void>
  clearLocalSession(): void
  refreshUser(): Promise<void>
}

const DEFAULT_CONFIG: AuthConfig = {
  self_registration_enabled: false,
  direct_storage_registration_enabled: false,
  apiAvailable: false,
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error('Authentication request failed')
}

function loadInitialAuth() {
  return Promise.allSettled([
    authApi.config(),
    authApi.me({ silentUnauthorizedProbe: true }),
  ] as const)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [status, setStatus] = useState<AuthStatus>('checking')
  const [config, setConfig] = useState<AuthConfig>(DEFAULT_CONFIG)
  const mountedRef = useRef(false)
  const initializationRef = useRef<ReturnType<typeof loadInitialAuth> | null>(null)
  const operationGenerationRef = useRef(0)

  const beginOperation = useCallback(() => {
    operationGenerationRef.current += 1
    return operationGenerationRef.current
  }, [])

  const isCurrentOperation = useCallback((generation: number) => {
    return mountedRef.current && operationGenerationRef.current === generation
  }, [])

  const setUnauthenticatedForOperation = useCallback(
    (generation: number, endSession = false) => {
      if (!isCurrentOperation(generation)) return
      if (endSession) clearWorkspaceSession()
      setUser(null)
      setStatus('unauthenticated')
    },
    [isCurrentOperation]
  )

  const fetchUserForOperation = useCallback(
    async (generation: number) => {
      const nextUser = await authApi.me()
      if (isCurrentOperation(generation)) {
        setUser(nextUser)
        setStatus('authenticated')
      }
    },
    [isCurrentOperation]
  )

  const refreshUser = useCallback(async () => {
    const generation = beginOperation()
    try {
      await fetchUserForOperation(generation)
    } catch (error) {
      // 仅会话失效（401）才清除身份；网络抖动/5xx 不清，避免误踢已登录用户
      if (isApiError(error) && error.status === 401) {
        setUnauthenticatedForOperation(generation, true)
      }
      throw toError(error)
    }
  }, [beginOperation, fetchUserForOperation, setUnauthenticatedForOperation])

  const login = useCallback(
    async (values: AuthLoginRequest) => {
      const generation = beginOperation()
      try {
        await authApi.login(values)
        if (!isCurrentOperation(generation)) return
        const nextUser = await authApi.me()
        if (!isCurrentOperation(generation)) return
        await offerLoginCredentialForSaving(values)
        if (!isCurrentOperation(generation)) return
        setUser(nextUser)
        setStatus('authenticated')
      } catch (error) {
        setUnauthenticatedForOperation(generation)
        throw toError(error)
      }
    },
    [beginOperation, isCurrentOperation, setUnauthenticatedForOperation]
  )

  const register = useCallback(
    async (values: AuthRegisterRequest) => {
      const generation = beginOperation()
      try {
        await authApi.register(values)
        if (!isCurrentOperation(generation)) return
        await fetchUserForOperation(generation)
      } catch (error) {
        setUnauthenticatedForOperation(generation)
        throw toError(error)
      }
    },
    [beginOperation, fetchUserForOperation, isCurrentOperation, setUnauthenticatedForOperation]
  )

  const logout = useCallback(async () => {
    const generation = beginOperation()
    try {
      await authApi.logout()
    } catch (error) {
      if (!isApiError(error) || error.status !== 401) throw error

      // token 已失效（401）时后端无法正常登出，但本地会话仍应清除——
      // 登出动作不应因服务端 401 而静默失败（残留 httpOnly cookie 无害，
      // 所有请求已 401）。
    }
    setUnauthenticatedForOperation(generation, true)
  }, [beginOperation, setUnauthenticatedForOperation])

  const clearLocalSession = useCallback(() => {
    const generation = beginOperation()
    setUnauthenticatedForOperation(generation, true)
  }, [beginOperation, setUnauthenticatedForOperation])

  useEffect(() => {
    mountedRef.current = true
    let active = true
    const initialOperationGeneration = operationGenerationRef.current

    if (!initializationRef.current) {
      initializationRef.current = loadInitialAuth()
    }

    void initializationRef.current.then(([configResult, userResult]) => {
      if (!active) return

      setConfig(
        configResult.status === 'fulfilled'
          ? {
              self_registration_enabled: configResult.value.self_registration_enabled,
              direct_storage_registration_enabled:
                configResult.value.direct_storage_registration_enabled ?? false,
              apiAvailable: true,
            }
          : DEFAULT_CONFIG
      )

      if (operationGenerationRef.current !== initialOperationGeneration) return

      if (userResult.status === 'fulfilled') {
        setUser(userResult.value)
        setStatus('authenticated')
      } else {
        if (isApiError(userResult.reason) && userResult.reason.status === 401) {
          clearWorkspaceSession()
        }
        setUser(null)
        setStatus('unauthenticated')
      }
    })

    return () => {
      active = false
      mountedRef.current = false
    }
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      status,
      config,
      login,
      register,
      logout,
      clearLocalSession,
      refreshUser,
    }),
    [user, status, config, login, register, logout, clearLocalSession, refreshUser]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
