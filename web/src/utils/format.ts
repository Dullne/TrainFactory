/**
 * Formatting utility functions
 */
import i18n from '@/i18n'

/**
 * Format bytes to human readable string
 * @param bytes - Number of bytes
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted string (e.g., "1.5 GB")
 */
export function formatBytes(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B'
  if (!bytes || bytes < 0) return '-'

  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`
}

/**
 * Format date to localized string
 * @param date - Date string, timestamp, or Date object
 * @param options - Intl.DateTimeFormat options
 * @returns Formatted date string
 */
export function formatDate(
  date: string | number | Date | undefined | null,
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }
): string {
  if (!date) return '-'

  try {
    const d = new Date(date)
    if (isNaN(d.getTime())) return '-'
    const locale = i18n.language === 'en' ? 'en-US' : 'zh-CN'
    return d.toLocaleString(locale, options)
  } catch {
    return '-'
  }
}

/**
 * Format date to relative time (e.g., "2 hours ago")
 * @param date - Date string, timestamp, or Date object
 * @returns Relative time string
 */
export function formatRelativeTime(date: string | number | Date | undefined | null): string {
  if (!date) return '-'

  try {
    const d = new Date(date)
    if (isNaN(d.getTime())) return '-'

    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffSec = Math.floor(diffMs / 1000)
    const diffMin = Math.floor(diffSec / 60)
    const diffHour = Math.floor(diffMin / 60)
    const diffDay = Math.floor(diffHour / 24)

    if (diffSec < 60) return i18n.t('common:time.justNow')
    if (diffMin < 60) return i18n.t('common:time.minutesAgo', { count: diffMin })
    if (diffHour < 24) return i18n.t('common:time.hoursAgo', { count: diffHour })
    if (diffDay < 30) return i18n.t('common:time.daysAgo', { count: diffDay })

    return formatDate(date, { year: 'numeric', month: '2-digit', day: '2-digit' })
  } catch {
    return '-'
  }
}

/**
 * Format duration in seconds to human readable string
 * @param seconds - Duration in seconds
 * @returns Formatted duration (e.g., "2h 30m 15s")
 */
export function formatDuration(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || seconds < 0) return '-'
  if (seconds === 0) return '0s'

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  const parts: string[] = []
  if (hours > 0) parts.push(`${hours}h`)
  if (minutes > 0) parts.push(`${minutes}m`)
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`)

  return parts.join(' ')
}

/**
 * Format number with thousand separators
 * @param num - Number to format
 * @param decimals - Number of decimal places
 * @returns Formatted number string
 */
export function formatNumber(num: number | undefined | null, decimals?: number): string {
  if (num === undefined || num === null) return '-'

  const options: Intl.NumberFormatOptions = {}
  if (decimals !== undefined) {
    options.minimumFractionDigits = decimals
    options.maximumFractionDigits = decimals
  }

  return num.toLocaleString(i18n.language === 'en' ? 'en-US' : 'zh-CN', options)
}

/**
 * Format percentage
 * @param value - Value (0-100 or 0-1)
 * @param decimals - Number of decimal places (default: 1)
 * @param normalized - Whether value is 0-1 (default: false, expects 0-100)
 * @returns Formatted percentage string
 */
export function formatPercent(
  value: number | undefined | null,
  decimals = 1,
  normalized = false
): string {
  if (value === undefined || value === null) return '-'

  const percent = normalized ? value * 100 : value
  return `${percent.toFixed(decimals)}%`
}

/**
 * Copy text to clipboard with fallback for non-secure contexts (HTTP).
 * Returns true on success, false on failure.
 */
export function copyToClipboard(text: string): boolean {
  const state = window as Window & { __copiedText?: string }
  state.__copiedText = text
  // Try modern API first
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).catch(() => {
      // Silently fail, already handled by fallback below
    })
  }
  // Always also try the fallback to ensure it works on HTTP
  try {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(textarea)
    return ok
  } catch {
    return false
  }
}

/**
 * Truncate string with ellipsis
 * @param str - String to truncate
 * @param maxLength - Maximum length (default: 50)
 * @returns Truncated string
 */
export function truncate(str: string | undefined | null, maxLength = 50): string {
  if (!str) return ''
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength - 3) + '...'
}
