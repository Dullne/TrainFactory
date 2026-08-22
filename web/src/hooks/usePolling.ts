import { useEffect, useRef, useCallback } from 'react'

interface UsePollingOptions {
  /** Polling interval in milliseconds */
  interval?: number
  /** Whether polling should be active */
  enabled?: boolean
  /** Whether to run immediately on enable */
  immediate?: boolean
}

/**
 * Hook for polling data at regular intervals.
 *
 * Features:
 * - Pauses when page is hidden (tab switched)
 * - Guards against request pile-up (skips if previous request still pending)
 *
 * @param callback - Function to call at each interval
 * @param options - Polling configuration
 */
export function usePolling(
  callback: () => void | Promise<void>,
  options: UsePollingOptions = {}
): void {
  const { interval = 5000, enabled = true, immediate = false } = options

  const savedCallback = useRef(callback)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fetchingRef = useRef(false)

  // Remember the latest callback
  useEffect(() => {
    savedCallback.current = callback
  }, [callback])

  // Clear interval on unmount or when disabled
  const clearPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!enabled) {
      clearPolling()
      return
    }

    const guarded = async () => {
      // Skip if page hidden or previous request still pending
      if (document.hidden || fetchingRef.current) return
      fetchingRef.current = true
      try {
        await savedCallback.current()
      } finally {
        fetchingRef.current = false
      }
    }

    // Run immediately if requested
    if (immediate) {
      guarded()
    }

    // Set up interval
    intervalRef.current = setInterval(guarded, interval)

    // Pause/resume on visibility change
    const onVisibility = () => {
      if (document.hidden) {
        clearPolling()
      } else {
        // Resume polling when page becomes visible again
        if (!intervalRef.current) {
          guarded()
          intervalRef.current = setInterval(guarded, interval)
        }
      }
    }
    document.addEventListener('visibilitychange', onVisibility)

    // Cleanup
    return () => {
      clearPolling()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [enabled, interval, immediate, clearPolling])
}

export default usePolling
