/**
 * API 请求缓存和去重机制
 */

interface CacheEntry<T> {
  data: T
  timestamp: number
  expiresAt: number
}

interface PendingRequest<T> {
  promise: Promise<T>
  timestamp: number
}

// 缓存存储
const cache = new Map<string, CacheEntry<unknown>>()

// 进行中的请求存储（用于去重）
const pendingRequests = new Map<string, PendingRequest<unknown>>()

// 默认缓存时间（毫秒）
const DEFAULT_TTL = 30000 // 30 秒

/**
 * 生成缓存键
 */
export function generateCacheKey(url: string, params?: Record<string, unknown>): string {
  const sortedParams = params
    ? Object.keys(params)
        .sort()
        .map((key) => `${key}=${JSON.stringify(params[key])}`)
        .join('&')
    : ''
  return sortedParams ? `${url}?${sortedParams}` : url
}

/**
 * 获取缓存数据
 */
export function getCache<T>(key: string): T | null {
  const entry = cache.get(key)
  if (!entry) return null

  // 检查是否过期
  if (Date.now() > entry.expiresAt) {
    cache.delete(key)
    return null
  }

  return entry.data as T
}

/**
 * 设置缓存
 */
export function setCache<T>(key: string, data: T, ttl: number = DEFAULT_TTL): void {
  const now = Date.now()
  cache.set(key, {
    data,
    timestamp: now,
    expiresAt: now + ttl,
  })
}

/**
 * 清除指定缓存
 */
export function invalidateCache(keyPattern?: string | RegExp): void {
  if (!keyPattern) {
    cache.clear()
    return
  }

  const pattern = typeof keyPattern === 'string' ? new RegExp(keyPattern) : keyPattern
  for (const key of cache.keys()) {
    if (pattern.test(key)) {
      cache.delete(key)
    }
  }
}

/**
 * 清除所有以指定前缀开头的缓存
 */
export function invalidateCacheByPrefix(prefix: string): void {
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) {
      cache.delete(key)
    }
  }
}

/**
 * 带缓存的请求函数
 */
export async function cachedRequest<T>(
  key: string,
  fetcher: () => Promise<T>,
  options: {
    ttl?: number
    forceRefresh?: boolean
  } = {}
): Promise<T> {
  const { ttl = DEFAULT_TTL, forceRefresh = false } = options

  // 如果不强制刷新，先检查缓存
  if (!forceRefresh) {
    const cached = getCache<T>(key)
    if (cached !== null) {
      return cached
    }
  }

  // 检查是否有进行中的相同请求（去重）
  const pending = pendingRequests.get(key)
  if (pending) {
    return pending.promise as Promise<T>
  }

  // 创建新请求
  const promise = fetcher()
    .then((data) => {
      setCache(key, data, ttl)
      pendingRequests.delete(key)
      return data
    })
    .catch((error) => {
      pendingRequests.delete(key)
      throw error
    })

  // 存储进行中的请求
  pendingRequests.set(key, {
    promise,
    timestamp: Date.now(),
  })

  return promise
}

/**
 * 预定义的缓存 TTL 配置
 */
export const CacheTTL = {
  /** 短期缓存：10 秒，适用于频繁变化的数据 */
  SHORT: 10000,
  /** 中期缓存：30 秒，适用于列表数据 */
  MEDIUM: 30000,
  /** 长期缓存：60 秒，适用于相对稳定的数据 */
  LONG: 60000,
  /** 资源状态：5 秒，实时性要求高 */
  RESOURCE: 5000,
} as const

/**
 * 获取缓存统计信息（用于调试）
 */
export function getCacheStats(): {
  size: number
  keys: string[]
  pendingCount: number
} {
  return {
    size: cache.size,
    keys: Array.from(cache.keys()),
    pendingCount: pendingRequests.size,
  }
}
