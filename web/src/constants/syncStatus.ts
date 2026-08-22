import type { SyncGenerationStatus, SyncStatus, SyncTrainingStatus } from '@/types'

export const ACTIVE_SYNC_STATUSES: readonly SyncStatus[] = [
  'syncing',
  'generating',
  'training',
  'loading_adapter',
]

export const SYNC_GENERATION_TOGGLE_STATUS: SyncGenerationStatus = 'completed'

export const SYNC_TRAINING_SUCCEEDED_STATUSES: readonly SyncTrainingStatus[] = [
  'completed',
  'adapter_loaded',
  'adapter_unloaded',
  'adapter_load_failed',
  'adapter_failed',
]

export const SYNC_TRAINING_CAN_LOAD_STATUSES: readonly SyncTrainingStatus[] = [
  'adapter_load_failed',
  'adapter_failed',
  'adapter_unloaded',
  'completed',
]
