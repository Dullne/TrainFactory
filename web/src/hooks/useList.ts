import { useState, useCallback, useEffect, useRef } from 'react'

interface ListParams {
  page?: number
  page_size?: number
  [key: string]: unknown
}

type PaginatedResult<T, E extends Record<string, unknown> = Record<string, unknown>> = {
  items: T[]
  total: number
  page: number
  page_size: number
} & E

interface UseListOptions {
  autoFetch?: boolean
  defaultPageSize?: number
  defaultParams?: Record<string, unknown>
}

interface UseListReturn<T, E extends Record<string, unknown> = Record<string, unknown>> {
  data: T[]
  loading: boolean
  error: Error | null
  hasData: boolean
  isStale: boolean
  page: number
  pageSize: number
  total: number
  extra: E | null
  setPage: (page: number) => void
  setPageSize: (size: number) => void
  fetch: (params?: Partial<ListParams>) => Promise<void>
  refresh: () => Promise<void>
}

/**
 * Generic hook for paginated list data fetching
 *
 * @param fetchFn - API function that returns paginated data
 * @param options - Configuration options
 * @returns List state and control functions
 *
 * @example
 * const { data, loading, page, setPage, fetch, extra } = useList(
 *   (params) => trainingApi.list(params),
 *   { defaultPageSize: 10 }
 * )
 * // extra.stats contains additional data from API response
 */
export function useList<T, E extends Record<string, unknown> = Record<string, unknown>>(
  fetchFn: (params: ListParams) => Promise<PaginatedResult<T, E>>,
  options: UseListOptions = {}
): UseListReturn<T, E> {
  const { autoFetch = true, defaultPageSize = 10, defaultParams = {} } = options

  // Pagination, rows and aggregates always describe one successful response.
  const [result, setResult] = useState({
    data: [] as T[],
    page: 1,
    pageSize: defaultPageSize,
    total: 0,
    extra: null as E | null,
    hasData: false,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [isStale, setIsStale] = useState(false)
  const queryRef = useRef<ListParams>({ ...defaultParams, page: 1, page_size: defaultPageSize })
  const successfulQueryRef = useRef<string | null>(null)
  const fetchFnRef = useRef(fetchFn)
  fetchFnRef.current = fetchFn

  // Monotonic request id: every fetch increments it and captures its own id.
  // A response that returns after a newer fetch started is stale and is
  // dropped (no setData/setTotal/setExtra) so it can't clobber the newer page.
  const reqIdRef = useRef(0)

  const fetch = useCallback(
    async (params?: Partial<ListParams>) => {
      const requestId = ++reqIdRef.current
      let query = { ...queryRef.current, ...params }
      query.page = Math.max(1, query.page ?? 1)
      query.page_size = Math.max(1, query.page_size ?? defaultPageSize)
      queryRef.current = query
      const queryKey = (value: ListParams) =>
        JSON.stringify(
          Object.keys(value)
            .sort()
            .filter((key) => value[key] !== undefined)
            .map((key) => [key, value[key]])
        )
      setIsStale(
        successfulQueryRef.current !== null && successfulQueryRef.current !== queryKey(query)
      )
      setLoading(true)
      setError(null)
      try {
        // A shrinking total can invalidate the current offset. Fetch the valid
        // page before publishing anything, including its page number and total.
        while (requestId === reqIdRef.current) {
          const response = await fetchFnRef.current(query)
          if (requestId !== reqIdRef.current) return
          const lastPage = Math.max(1, Math.ceil(response.total / query.page_size!))
          if (query.page! > lastPage) {
            query = { ...query, page: lastPage }
            queryRef.current = query
            setIsStale(successfulQueryRef.current !== null)
            continue
          }
          const extraFields = { ...(response as Record<string, unknown>) }
          delete extraFields.items
          delete extraFields.total
          delete extraFields.page
          delete extraFields.page_size
          successfulQueryRef.current = queryKey(query)
          setResult({
            data: response.items,
            total: response.total,
            page: query.page!,
            pageSize: query.page_size!,
            extra: extraFields as E,
            hasData: true,
          })
          setIsStale(false)
          break
        }
      } catch (err) {
        // Keep the last successful data/extra so a transient polling failure
        // (e.g. during 5s background refresh) doesn't blank the list and the
        // stats cards. Only surface the error if this is still the latest req.
        if (requestId === reqIdRef.current) {
          setError(err instanceof Error ? err : new Error(String(err)))
        }
      } finally {
        if (requestId === reqIdRef.current) {
          setLoading(false)
        }
      }
    },
    [defaultPageSize]
  )

  // Retry the latest requested query, even when its previous request failed.
  const refresh = useCallback(() => fetch(), [fetch])
  const setPage = useCallback(
    (page: number) => {
      void fetch({ page })
    },
    [fetch]
  )
  const setPageSize = useCallback(
    (size: number) => {
      void fetch({ page: 1, page_size: size })
    },
    [fetch]
  )

  useEffect(() => {
    const requestSequence = reqIdRef
    if (autoFetch) void fetch()
    return () => {
      ++requestSequence.current
    }
  }, [autoFetch, fetch])

  return {
    ...result,
    loading,
    error,
    isStale,
    setPage,
    setPageSize,
    fetch,
    refresh,
  }
}

export default useList
