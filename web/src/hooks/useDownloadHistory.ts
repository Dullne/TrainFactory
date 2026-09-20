import { useCallback, useEffect, useRef, useState } from 'react'
import { usePolling } from '@/hooks/usePolling'

interface DownloadTask {
  status: string
  created_at?: string
}

interface UseDownloadHistoryOptions<T extends DownloadTask> {
  scopeKey: string
  getId: (task: T) => string
  isActive: (task: T) => boolean
  load: () => Promise<{ downloads: T[] }>
  onTaskSettled?: (task: T) => void
}

export function useDownloadHistory<T extends DownloadTask>({
  scopeKey,
  getId,
  isActive,
  load,
  onTaskSettled,
}: UseDownloadHistoryOptions<T>) {
  const [tasks, setTasks] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const [failed, setFailed] = useState(false)
  const requestGenerationRef = useRef(0)
  const scopeRef = useRef(scopeKey)
  const initializedScopeRef = useRef<string | null>(null)
  const inFlightRef = useRef<Promise<void> | null>(null)
  const inFlightIdRef = useRef(0)
  const activeIdsRef = useRef(new Set<string>())
  const optimisticTasksRef = useRef(new Map<string, T>())
  const tasksRevisionRef = useRef(0)
  const hasSnapshotRef = useRef(false)
  const onTaskSettledRef = useRef(onTaskSettled)
  const mountedRef = useRef(false)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  useEffect(() => {
    onTaskSettledRef.current = onTaskSettled
  }, [onTaskSettled])

  const applyDownloads = useCallback(
    (downloads: T[]) => {
      const downloadsById = new Map(downloads.map((task) => [getId(task), task]))
      for (const [taskId, task] of optimisticTasksRef.current) {
        const serverTask = downloadsById.get(taskId)
        if (serverTask) {
          if (isActive(serverTask)) optimisticTasksRef.current.set(taskId, serverTask)
          else optimisticTasksRef.current.delete(taskId)
        } else {
          downloadsById.set(taskId, task)
        }
      }
      const uniqueTasks = Array.from(downloadsById.values()).sort((left, right) =>
        (right.created_at ?? '').localeCompare(left.created_at ?? '')
      )

      if (hasSnapshotRef.current) {
        for (const task of uniqueTasks) {
          if (activeIdsRef.current.has(getId(task)) && !isActive(task)) {
            onTaskSettledRef.current?.(task)
          }
        }
      }

      activeIdsRef.current = new Set(uniqueTasks.filter(isActive).map((task) => getId(task)))
      hasSnapshotRef.current = true
      setTasks(uniqueTasks)
    },
    [getId, isActive]
  )

  const refresh = useCallback(
    (silent = false) => {
      if (!mountedRef.current || !scopeKey) return Promise.resolve()
      if (inFlightRef.current) return inFlightRef.current

      const requestGeneration = requestGenerationRef.current
      const requestScope = scopeKey
      const requestId = ++inFlightIdRef.current
      const tasksRevision = tasksRevisionRef.current
      if (!silent) setLoading(true)

      const request = (async () => {
        try {
          const response = await load()
          if (
            !mountedRef.current ||
            requestGenerationRef.current !== requestGeneration ||
            scopeRef.current !== requestScope ||
            tasksRevisionRef.current !== tasksRevision
          ) {
            return
          }
          applyDownloads(response.downloads)
          setFailed(false)
        } catch {
          if (
            mountedRef.current &&
            requestGenerationRef.current === requestGeneration &&
            scopeRef.current === requestScope
          ) {
            setFailed(true)
          }
        } finally {
          if (inFlightIdRef.current === requestId) inFlightRef.current = null
          if (
            mountedRef.current &&
            requestGenerationRef.current === requestGeneration &&
            scopeRef.current === requestScope &&
            !silent
          ) {
            setLoading(false)
          }
        }
      })()

      inFlightRef.current = request
      return request
    },
    [applyDownloads, load, scopeKey]
  )

  useEffect(() => {
    if (initializedScopeRef.current === scopeKey) {
      if (scopeKey) void refresh()
      return
    }
    initializedScopeRef.current = scopeKey
    scopeRef.current = scopeKey
    requestGenerationRef.current += 1
    inFlightIdRef.current += 1
    inFlightRef.current = null
    activeIdsRef.current.clear()
    optimisticTasksRef.current.clear()
    tasksRevisionRef.current += 1
    hasSnapshotRef.current = false
    setTasks([])
    setFailed(false)
    setLoading(false)
    if (scopeKey) void refresh()
  }, [refresh, scopeKey])

  usePolling(() => refresh(true), {
    interval: 3000,
    enabled: Boolean(scopeKey) && tasks.some(isActive),
  })

  const upsert = useCallback(
    (task: T) => {
      if (!mountedRef.current) return false
      const taskId = getId(task)
      const acceptedScope = scopeRef.current
      const pendingRequest = inFlightRef.current
      tasksRevisionRef.current += 1
      optimisticTasksRef.current.set(taskId, task)
      hasSnapshotRef.current = true
      if (isActive(task)) activeIdsRef.current.add(taskId)
      else activeIdsRef.current.delete(taskId)
      setTasks((current) => {
        const next = new Map(current.map((item) => [getId(item), item]))
        next.set(taskId, task)
        return Array.from(next.values()).sort((left, right) =>
          (right.created_at ?? '').localeCompare(left.created_at ?? '')
        )
      })

      void (async () => {
        if (pendingRequest) await pendingRequest
        if (mountedRef.current && scopeRef.current === acceptedScope) await refresh(true)
      })()
      return true
    },
    [getId, isActive, refresh]
  )

  return { tasks, loading, failed, refresh, upsert }
}
