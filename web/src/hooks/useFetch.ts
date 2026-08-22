import { useState, useCallback, useEffect } from 'react'

interface UseFetchOptions<T> {
  /** Whether to fetch immediately on mount */
  immediate?: boolean
  /** Initial data value */
  initialData?: T
  /** Dependencies that trigger refetch */
  deps?: unknown[]
  /** Callback on successful fetch */
  onSuccess?: (data: T) => void
  /** Callback on error */
  onError?: (error: Error) => void
}

interface UseFetchReturn<T> {
  data: T | undefined
  loading: boolean
  error: Error | null
  fetch: () => Promise<T | undefined>
  setData: React.Dispatch<React.SetStateAction<T | undefined>>
}

/**
 * Generic hook for data fetching with loading and error states
 *
 * @param fetchFn - Async function that returns data
 * @param options - Configuration options
 * @returns Fetch state and control functions
 *
 * @example
 * const { data, loading, error, fetch } = useFetch(
 *   () => trainingApi.get(taskId),
 *   { immediate: true }
 * )
 */
export function useFetch<T>(
  fetchFn: () => Promise<T>,
  options: UseFetchOptions<T> = {}
): UseFetchReturn<T> {
  const {
    immediate = false,
    initialData,
    deps = [],
    onSuccess,
    onError,
  } = options

  const [data, setData] = useState<T | undefined>(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const fetch = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const result = await fetchFn()
      setData(result)
      onSuccess?.(result)
      return result
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      onError?.(error)
      return undefined
    } finally {
      setLoading(false)
    }
  }, [fetchFn, onSuccess, onError])

  // Auto fetch on mount if immediate is true
  useEffect(() => {
    if (immediate) {
      fetch()
    }
  }, [immediate, ...deps]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    data,
    loading,
    error,
    fetch,
    setData,
  }
}

export default useFetch
