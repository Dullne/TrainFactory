/**
 * Utility functions for TrainFactory
 */

export {
  formatBytes,
  formatDate,
  formatRelativeTime,
  formatDuration,
  formatNumber,
  formatPercent,
  truncate,
  copyToClipboard,
} from './format'

export {
  getStatusColor,
  getStatusText,
  getStatusDescription,
  getStatusMeta,
  isActiveStatus,
  isTerminalStatus,
  isSuccessStatus,
  isErrorStatus,
  getAntdStatusColor,
} from './status'

export type {
  TrainingStatus,
  ModelStatus,
  GeneralStatus,
  Status,
  StatusSemantic,
  StatusMeta,
} from './status'
