import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useSearchParams } from 'react-router-dom'
import { useAuth } from '@/auth/AuthContext'
import { useList } from './useList'

type Filters = Record<string, readonly string[]>
interface WorkspaceOptions {
  filters?: Filters
  prefix?: string
  defaultPageSize?: number
  pageSizes?: readonly number[]
}
const EMPTY_FILTERS: Filters = {}
const PAGE_SIZES = [10, 20, 50, 100]

function positiveInteger(value: string | null, fallback: number) {
  if (!value || !/^[1-9]\d{0,5}$/.test(value)) return fallback
  return Number(value)
}

/** URL state belongs to browser history; only scroll positions use account-scoped storage. */
export function useListWorkspace({
  filters = EMPTY_FILTERS,
  prefix = '',
  defaultPageSize = 10,
  pageSizes = PAGE_SIZES,
}: WorkspaceOptions = {}) {
  const [searchParams, setSearchParams] = useSearchParams()
  const current = useRef({ searchParams, setSearchParams })
  current.current = { searchParams, setSearchParams }
  const page = positiveInteger(searchParams.get(`${prefix}page`), 1)
  const requestedSize = positiveInteger(searchParams.get(`${prefix}page_size`), defaultPageSize)
  const pageSize = pageSizes.includes(requestedSize) ? requestedSize : defaultPageSize
  const values = Object.fromEntries(
    Object.entries(filters).map(([key, allowed]) => {
      const value = searchParams.get(`${prefix}${key}`)
      return [key, value !== null && allowed.includes(value) ? value : undefined]
    })
  ) as Record<string, string | undefined>
  const valuesKey = JSON.stringify(values)
  const stableValues = useMemo<Record<string, string | undefined>>(
    () => JSON.parse(valuesKey),
    [valuesKey]
  )

  const update = useCallback(
    (patch: Record<string, string | number | undefined>, replace = false) => {
      const { searchParams: previous, setSearchParams: write } = current.current
      const next = new URLSearchParams(previous)
      for (const [key, value] of Object.entries(patch)) {
        const isDefault =
          (key === 'page' && value === 1) || (key === 'page_size' && value === defaultPageSize)
        if (value === undefined || isDefault) next.delete(`${prefix}${key}`)
        else next.set(`${prefix}${key}`, String(value))
      }
      if (next.toString() !== previous.toString()) {
        // Support multiple updates in one event without dropping the previous patch.
        current.current.searchParams = next
        write(next, { replace, preventScrollReset: true })
      }
    },
    [defaultPageSize, prefix]
  )

  // Do not forward arbitrary URL input to APIs. Preserve unrelated deep links.
  useEffect(() => {
    const invalid: Record<string, undefined> = {}
    for (const [key, allowed] of Object.entries(filters)) {
      const value = searchParams.get(`${prefix}${key}`)
      if (value !== null && !allowed.includes(value)) invalid[key] = undefined
    }
    for (const [key, value] of [
      ['page', page],
      ['page_size', pageSize],
    ] as const) {
      const raw = searchParams.get(`${prefix}${key}`)
      if (raw !== null && raw !== String(value)) invalid[key] = undefined
    }
    if (Object.keys(invalid).length) update(invalid, true)
  }, [filters, page, pageSize, prefix, searchParams, update])

  const setFilter = useCallback(
    (key: string, value: string | undefined) => update({ [key]: value, page: 1 }),
    [update]
  )
  const setPagination = useCallback(
    (nextPage: number, nextSize = pageSize, replace = false) => {
      update({ page: nextPage, page_size: nextSize }, replace)
    },
    [pageSize, update]
  )
  return { values: stableValues, page, pageSize, setFilter, setPagination, update }
}

export function useListScroll(ready: boolean) {
  const { user } = useAuth()
  const { pathname, search } = useLocation()
  const key = user ? `tf_list_workspace:v1:${user.user_id}:${pathname}${search}` : null
  useEffect(() => {
    if (!ready || !key) return
    const container = document.querySelector<HTMLElement>('.main-layout-content')
    let saved: { window: number; content: number } | null = null
    try {
      const value = JSON.parse(sessionStorage.getItem(key) || 'null')
      if (
        value &&
        Number.isFinite(value.window) &&
        value.window >= 0 &&
        Number.isFinite(value.content) &&
        value.content >= 0
      )
        saved = value
    } catch {
      /* Restricted storage must not prevent list navigation. */
    }
    let active = true
    const save = () => {
      // A route can shrink the document before effect cleanup; ignore its scroll event.
      if (!active || window.location.pathname !== pathname || window.location.search !== search)
        return
      try {
        sessionStorage.setItem(
          key,
          JSON.stringify({ window: window.scrollY, content: container?.scrollTop ?? 0 })
        )
      } catch {
        /* Scroll restoration is best effort. */
      }
    }
    const clear = () => {
      active = false
    }
    const frame = requestAnimationFrame(() => {
      if (saved && active) {
        window.scrollTo({ top: saved.window, behavior: 'instant' })
        container?.scrollTo({ top: saved.content, behavior: 'instant' })
      }
      window.addEventListener('scroll', save, { passive: true })
      container?.addEventListener('scroll', save, { passive: true })
      window.addEventListener('pagehide', save)
    })
    window.addEventListener('tf:workspace-session-cleared', clear)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', save)
      container?.removeEventListener('scroll', save)
      window.removeEventListener('pagehide', save)
      window.removeEventListener('tf:workspace-session-cleared', clear)
    }
  }, [key, ready, pathname, search])
}

/** Compose URL navigation with useList's existing committed-response and race protections. */
export function useWorkspaceList<T, E extends Record<string, unknown> = Record<string, unknown>>(
  fetchFn: Parameters<typeof useList<T, E>>[0],
  options: WorkspaceOptions & { params?: Record<string, unknown> } = {}
) {
  const workspace = useListWorkspace(options)
  const list = useList<T, E>(fetchFn, {
    autoFetch: false,
    defaultPageSize: options.defaultPageSize ?? 10,
  })
  const { fetch, data, loading, error, hasData, page, pageSize } = list
  const paramsKey = JSON.stringify(options.params ?? {})
  const params = useMemo<Record<string, unknown>>(() => JSON.parse(paramsKey), [paramsKey])
  const query = useMemo(
    () => ({ ...params, ...workspace.values, page: workspace.page, page_size: workspace.pageSize }),
    [params, workspace.values, workspace.page, workspace.pageSize]
  )
  // Include cleared filters explicitly: useList merges each request with its last query.
  const filterKeys = [
    ...Object.keys(options.filters ?? EMPTY_FILTERS),
    ...Object.keys(options.params ?? {}),
  ].join(',')
  useEffect(() => {
    const cleared = Object.fromEntries(
      filterKeys
        .split(',')
        .filter(Boolean)
        .map((key) => [key, undefined])
    )
    void fetch({ ...cleared, ...query })
  }, [fetch, query, filterKeys])
  const committedData = useRef(data)
  const { setPagination } = workspace
  useEffect(() => {
    if (data === committedData.current || loading || error) return
    committedData.current = data
    // A deletion or shrinking total may have moved useList to the last valid page.
    setPagination(page, pageSize, true)
  }, [data, loading, error, page, pageSize, setPagination])
  useListScroll(hasData)
  return { ...list, workspace }
}
