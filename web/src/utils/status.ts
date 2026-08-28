/**
 * Status-related utility functions and mappings
 */

import { STATUS_COLOR_MAP, STATUS_DEFAULT } from '@/theme'
import i18n from '@/i18n'

// Training status types
export type TrainingStatus =
  | 'pending'
  | 'preparing'
  | 'running'
  | 'evaluating'
  | 'succeeded'
  | 'completed'
  | 'failed'
  | 'stopped'
  | 'cancelled'

// Model status types
export type ModelStatus = 'available' | 'unavailable' | 'deploying'

// General status types
export type GeneralStatus = 'active' | 'inactive' | 'archived' | 'disabled'

// All status types
export type Status = TrainingStatus | ModelStatus | GeneralStatus | string

export type StatusSemantic =
  | 'queued'
  | 'in_progress'
  | 'succeeded'
  | 'failed'
  | 'inactive'
  | 'unknown'

export interface StatusMeta {
  color: string
  text: string
  description?: string
  semantic: StatusSemantic
  isActive: boolean
  isTerminal: boolean
  isSuccess: boolean
  isError: boolean
  isAnimated: boolean
  antdColor: 'default' | 'processing' | 'success' | 'error' | 'warning'
}

const ACTIVE_STATUSES = new Set([
  'pending',
  'preparing',
  'running',
  'degraded',
  'stopping',
  'recovering',
  'publishing',
  'restarting',
  'evaluating',
  'training',
  'deploying',
  'uploading',
  'downloading',
  'processing',
])

const ANIMATED_STATUSES = new Set([
  'preparing',
  'running',
  'stopping',
  'recovering',
  'publishing',
  'restarting',
  'evaluating',
  'training',
  'deploying',
  'syncing',
  'generating',
  'loading_adapter',
  'uploading',
  'downloading',
  'processing',
])

const TERMINAL_STATUSES = new Set([
  'succeeded',
  'completed',
  'failed',
  'stopped',
  'cancelled',
])

const SUCCESS_STATUSES = new Set([
  'succeeded',
  'completed',
  'available',
  'active',
  'ready',
  'healthy',
])

const ERROR_STATUSES = new Set([
  'failed',
  'unavailable',
  'error',
  'unhealthy',
])

function getStatusSemantic(status: Status): StatusSemantic {
  if (SUCCESS_STATUSES.has(status)) return 'succeeded'
  if (ERROR_STATUSES.has(status)) return 'failed'
  if (status === 'pending') return 'queued'
  if (ACTIVE_STATUSES.has(status) || ['syncing', 'generating', 'loading_adapter'].includes(status)) {
    return 'in_progress'
  }
  if (['stopped', 'cancelled', 'inactive', 'archived', 'disabled', 'idle', 'registered'].includes(status)) {
    return 'inactive'
  }
  return 'unknown'
}

/**
 * Get color for a status
 * @param status - Status string
 * @returns Color hex code
 */
export function getStatusColor(status: Status): string {
  return STATUS_COLOR_MAP[status] || STATUS_DEFAULT
}

/**
 * Get display text for a status
 * @param status - Status string
 * @returns Localized status text
 */
export function getStatusText(status: Status): string {
  return i18n.t(`common:status.${status}`, { defaultValue: status })
}

/**
 * Get localized description for a status.
 */
export function getStatusDescription(status: Status): string | undefined {
  const description = i18n.t(`common:statusDescription.${status}`, { defaultValue: '' }).trim()
  return description || undefined
}

/**
 * Check if status represents a running/in-progress state
 * @param status - Status string
 * @returns Whether the status is active
 */
export function isActiveStatus(status: Status): boolean {
  return ACTIVE_STATUSES.has(status)
}

/**
 * Check if status represents a terminal state
 * @param status - Status string
 * @returns Whether the status is terminal
 */
export function isTerminalStatus(status: Status): boolean {
  return TERMINAL_STATUSES.has(status)
}

/**
 * Check if status represents success
 * @param status - Status string
 * @returns Whether the status is successful
 */
export function isSuccessStatus(status: Status): boolean {
  return SUCCESS_STATUSES.has(status)
}

/**
 * Check if status represents failure/error
 * @param status - Status string
 * @returns Whether the status is error
 */
export function isErrorStatus(status: Status): boolean {
  return ERROR_STATUSES.has(status)
}

/**
 * Get Ant Design Tag color name for a status
 * (for compatibility with Ant Design's preset colors)
 * @param status - Status string
 * @returns Ant Design color name
 */
export function getAntdStatusColor(
  status: Status
): 'default' | 'processing' | 'success' | 'error' | 'warning' {
  switch (status) {
    case 'preparing':
    case 'running':
    case 'stopping':
    case 'recovering':
    case 'publishing':
    case 'restarting':
    case 'evaluating':
    case 'training':
    case 'deploying':
      return 'processing'
    case 'degraded':
      return 'warning'
    case 'succeeded':
    case 'completed':
    case 'available':
    case 'active':
    case 'ready':
    case 'healthy':
      return 'success'
    case 'failed':
    case 'unavailable':
    case 'error':
    case 'unhealthy':
      return 'error'
    case 'pending':
    case 'stopped':
    case 'cancelled':
    case 'registered':
      return 'warning'
    default:
      return 'default'
  }
}

/**
 * Get normalized meta for frontend status rendering.
 */
export function getStatusMeta(status: Status): StatusMeta {
  return {
    color: getStatusColor(status),
    text: getStatusText(status),
    description: getStatusDescription(status),
    semantic: getStatusSemantic(status),
    isActive: isActiveStatus(status),
    isTerminal: isTerminalStatus(status),
    isSuccess: isSuccessStatus(status),
    isError: isErrorStatus(status),
    isAnimated: ANIMATED_STATUSES.has(status),
    antdColor: getAntdStatusColor(status),
  }
}
