import { useState, useEffect, useCallback, useRef } from 'react'
import { cachedRequest, invalidateCacheByPrefix, CacheTTL } from '@/services/cache'

interface UseCachedQueryOptions<T> {
  /** 缓存 TTL（毫秒） */
  ttl?: number
  /** 是否自动获取数据 */
  enabled?: boolean
  /** 依赖项变化时是否强制刷新 */
  forceRefreshOnDepsChange?: boolean
  /** 初始数据 */
  initialData?: T
  /** 成功回调 */
  onSuccess?: (data: T) => void
  /** 错误回调 */
  onError?: (error: Error) => void
}

interface UseCachedQueryResult<T> {
  /** 数据 */
  data: T | undefined
  /** 是否加载中 */
  loading: boolean
  /** 错误信息 */
  error: Error | null
  /** 手动刷新（使用缓存） */
  refetch: () => Promise<void>
  /** 强制刷新（忽略缓存） */
  forceRefetch: () => Promise<void>
  /** 使相关缓存失效 */
  invalidate: () => void
}

/**
 * 带缓存的数据查询 Hook
 *
 * @example
 * const { data, loading, refetch } = useCachedQuery(
 *   'models-list',
 *   () => modelApi.list({ status: 'available' }),
 *   { ttl: CacheTTL.MEDIUM }
 * )
 */
export function useCachedQuery<T>(
  key: string,
  fetcher: () => Promise<T>,
  options: UseCachedQueryOptions<T> = {}
): UseCachedQueryResult<T> {
  const {
    ttl = CacheTTL.MEDIUM,
    enabled = true,
    forceRefreshOnDepsChange = false,
    initialData,
    onSuccess,
    onError,
  } = options

  const [data, setData] = useState<T | undefined>(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const fetcherRef = useRef(fetcher)
  const onSuccessRef = useRef(onSuccess)
  const onErrorRef = useRef(onError)

  // 用于追踪组件是否已卸载
  const mountedRef = useRef(true)
  // 用于追踪是否是首次加载
  const initialLoadRef = useRef(true)

  useEffect(() => {
    fetcherRef.current = fetcher
  }, [fetcher])

  useEffect(() => {
    onSuccessRef.current = onSuccess
  }, [onSuccess])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  const fetchData = useCallback(
    async (forceRefresh = false) => {
      if (!enabled) return

      setLoading(true)
      setError(null)

      try {
        const result = await cachedRequest(key, fetcherRef.current, {
          ttl,
          forceRefresh,
        })

        if (mountedRef.current) {
          setData(result)
          onSuccessRef.current?.(result)
        }
      } catch (err) {
        if (mountedRef.current) {
          const error = err instanceof Error ? err : new Error(String(err))
          setError(error)
          onErrorRef.current?.(error)
        }
      } finally {
        if (mountedRef.current) {
          setLoading(false)
        }
      }
    },
    [key, ttl, enabled]
  )

  // 初始加载和依赖变化时获取数据
  useEffect(() => {
    if (enabled) {
      fetchData(forceRefreshOnDepsChange && !initialLoadRef.current)
      initialLoadRef.current = false
    }
  }, [enabled, fetchData, forceRefreshOnDepsChange])

  // 清理
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const refetch = useCallback(async () => {
    await fetchData(false)
  }, [fetchData])

  const forceRefetch = useCallback(async () => {
    await fetchData(true)
  }, [fetchData])

  const invalidate = useCallback(() => {
    invalidateCacheByPrefix(key)
  }, [key])

  return {
    data,
    loading,
    error,
    refetch,
    forceRefetch,
    invalidate,
  }
}

export { CacheTTL }
