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

  const [data, setData] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(defaultPageSize)
  const [total, setTotal] = useState(0)
  const [extra, setExtra] = useState<E | null>(null)
  const [extraParams, setExtraParams] = useState<Record<string, unknown>>(defaultParams)

  // Monotonic request id: every fetch increments it and captures its own id.
  // A response that returns after a newer fetch started is stale and is
  // dropped (no setData/setTotal/setExtra) so it can't clobber the newer page.
  const reqIdRef = useRef(0)

  const fetch = useCallback(
    async (params?: Partial<ListParams>) => {
      const requestId = ++reqIdRef.current
      setLoading(true)
      setError(null)
      try {
        // Merge extra params if provided
        if (params) {
          const { page: newPage, page_size: newPageSize, ...rest } = params
          if (newPage !== undefined) setPage(newPage)
          if (newPageSize !== undefined) setPageSize(newPageSize)
          if (Object.keys(rest).length > 0) {
            setExtraParams((prev) => ({ ...prev, ...rest }))
          }
        }

        const result = await fetchFn({
          page: params?.page ?? page,
          page_size: params?.page_size ?? pageSize,
          ...extraParams,
          ...params,
        })

        // A newer fetch (page change / filter / refresh) superseded this one.
        if (requestId !== reqIdRef.current) return

        setData(result.items)
        setTotal(result.total)

        // Extract extra fields (everything except items, total, page, page_size)
        const extraFields = { ...(result as Record<string, unknown>) }
        delete extraFields.items
        delete extraFields.total
        delete extraFields.page
        delete extraFields.page_size
        setExtra(extraFields as E)
      } catch (err) {
        console.error('useList fetch error:', err)
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
    [fetchFn, page, pageSize, extraParams]
  )

  const refresh = useCallback(() => {
    return fetch({ page, page_size: pageSize, ...extraParams })
  }, [fetch, page, pageSize, extraParams])

  // Auto fetch on mount and when page/pageSize changes
  useEffect(() => {
    if (autoFetch) {
      fetch()
    }
  }, [page, pageSize]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    data,
    loading,
    error,
    page,
    pageSize,
    total,
    extra,
    setPage,
    setPageSize,
    fetch,
    refresh,
  }
}

export default useList
