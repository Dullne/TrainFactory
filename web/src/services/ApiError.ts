import axios from 'axios'

export type ApiErrorKind = 'http' | 'network' | 'timeout' | 'unknown'

export class ApiError extends Error {
  readonly name = 'ApiError'

  constructor(
    message: string,
    readonly kind: ApiErrorKind,
    readonly status?: number,
    readonly code?: string,
    readonly requestPath?: string
  ) {
    super(message)
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError
}

export function toApiError(error: unknown, fallback: string): ApiError {
  if (isApiError(error)) return error
  if (!axios.isAxiosError(error)) return new ApiError(fallback, 'unknown')

  const requestPath = error.config?.url?.split(/[?#]/, 1)[0] || undefined
  const status = error.response?.status
  const detail = error.response?.data?.detail
  const message = typeof detail === 'string' ? detail : fallback

  if (status !== undefined) {
    return new ApiError(message, 'http', status, error.code, requestPath)
  }
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return new ApiError(fallback, 'timeout', undefined, error.code, requestPath)
  }
  return new ApiError(fallback, 'network', undefined, error.code, requestPath)
}
