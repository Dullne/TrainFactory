import { useRef, useState } from 'react'
import type { FormInstance } from 'antd'

export interface DatasetItem {
  key: string
  path: string
  max_samples?: number | null
  split: 'train' | 'eval' | 'test'
}

let datasetSequence = 0
export function createDatasetItem(): DatasetItem {
  return { key: `${Date.now()}-${datasetSequence++}`, path: '', split: 'train' }
}

type DraftValues = Record<string, unknown>
interface TrainingDraft {
  version: 1
  accountId: string
  updatedAt: number
  values: DraftValues
  datasets: DatasetItem[]
}
interface DraftSnapshot {
  draft: TrainingDraft | null
  saved: boolean
}

const prefix = 'tf_training_draft:v1:'
// This also retains the latest edit when sessionStorage is full or disabled. The
// cache belongs to this document, so a browser reload still needs a warning.
const memory = new Map<string, DraftSnapshot>()
window.addEventListener('tf:workspace-session-cleared', () => memory.clear())
window.addEventListener('beforeunload', (event) => {
  if ([...memory.values()].some(({ draft, saved }) => draft !== null && !saved)) {
    event.preventDefault()
    event.returnValue = ''
  }
})

const stringFields = new Set([
  'task_name',
  'model_type',
  'training_method',
  'tuner_type',
  'embedding_loss_name',
  'reranker_loss_name',
  'sft_checkpoint_path',
  'mixed_precision',
  'deepspeed',
  'eval_strategy',
  'save_strategy',
  'output_dir',
])
const numberFields = new Set([
  'num_train_epochs',
  'per_device_train_batch_size',
  'learning_rate',
  'warmup_ratio',
  'gradient_accumulation_steps',
  'logging_steps',
  'max_length',
  'eval_steps',
  'save_steps',
  'lora_r',
  'lora_alpha',
  'lora_dropout',
])
const lossFields = new Set([
  'scale',
  'mini_batch_size',
  'margin',
  'distance_metric',
  'triplet_margin',
  'positive_margin',
  'negative_margin',
  'num_labels',
  'guide_model',
  'normalize_by_logc',
  'ignore_empty',
  'matryoshka_dims',
  'num_negatives',
  'k',
  'beta',
  'delta',
  'temperature',
  'sigma',
  'respect_input_order',
  'n_docs',
  'n_pos',
  'name',
  'infonce_mode',
  'ranknet_max_pairs_per_batch',
  'metric',
  'chunk_size',
])
const rlFields = new Set([
  'n_docs',
  'beta',
  'reference_free',
  'kl_coef',
  'clip_range',
  'reward_type',
  'reward_k',
  'scale_rewards',
  'num_iterations',
  'chunk_size',
  'rankings_direction',
])
const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
const isScalar = (value: unknown) =>
  value === null ||
  typeof value === 'string' ||
  typeof value === 'boolean' ||
  (typeof value === 'number' && Number.isFinite(value))

// Only training fields enter storage; unknown keys, credentials and object-shaped
// input values are not accepted from a stale or malformed browser record.
function validateValues(value: unknown): value is DraftValues {
  if (!isRecord(value)) return false
  for (const [key, field] of Object.entries(value)) {
    if (stringFields.has(key)) {
      if (field !== null && typeof field !== 'string') return false
    } else if (numberFields.has(key)) {
      if (field !== null && (typeof field !== 'number' || !Number.isFinite(field))) return false
    } else if (key === 'gradient_checkpointing') {
      if (field !== null && typeof field !== 'boolean') return false
    } else if (key === 'base_model_path') {
      if (
        field !== null &&
        typeof field !== 'string' &&
        !(Array.isArray(field) && field.every((item) => typeof item === 'string'))
      )
        return false
    } else if (key === 'gpu_ids') {
      if (
        field !== null &&
        !(Array.isArray(field) && field.every((item) => Number.isInteger(item)))
      )
        return false
    } else if (key === 'loss_config' || key === 'rl_config') {
      if (field === null) continue
      const allowed = key === 'loss_config' ? lossFields : rlFields
      if (
        !isRecord(field) ||
        !Object.entries(field).every(
          ([name, item]) =>
            allowed.has(name) &&
            (isScalar(item) ||
              (name === 'matryoshka_dims' &&
                Array.isArray(item) &&
                item.every((dim) => typeof dim === 'string' || typeof dim === 'number')))
        )
      )
        return false
    } else return false
  }
  return (
    ['embedding', 'reranker', 'decoder_reranker', 'llm'].includes(String(value.model_type)) &&
    ['sft', 'grpo', 'dapo', 'dr_grpo', 'dpo', 'orpo'].includes(String(value.training_method))
  )
}

function validateDraft(value: unknown, accountId: string): value is TrainingDraft {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    value.accountId !== accountId ||
    typeof value.updatedAt !== 'number' ||
    !Number.isFinite(value.updatedAt) ||
    !validateValues(value.values) ||
    !Array.isArray(value.datasets) ||
    value.datasets.length === 0 ||
    value.datasets.length > 1000
  )
    return false
  const keys = new Set<string>()
  return value.datasets.every((item) => {
    if (
      !isRecord(item) ||
      typeof item.key !== 'string' ||
      !item.key ||
      keys.has(item.key) ||
      typeof item.path !== 'string' ||
      !['train', 'eval', 'test'].includes(String(item.split)) ||
      (item.max_samples != null &&
        (typeof item.max_samples !== 'number' || !Number.isFinite(item.max_samples)))
    )
      return false
    keys.add(item.key)
    return true
  })
}

function readDraft(accountId: string): DraftSnapshot {
  const cached = memory.get(accountId)
  if (cached) return cached
  try {
    const raw = sessionStorage.getItem(prefix + accountId)
    if (!raw) return { draft: null, saved: true }
    const draft: unknown = JSON.parse(raw)
    if (validateDraft(draft, accountId)) return { draft, saved: true }
    sessionStorage.removeItem(prefix + accountId)
  } catch {
    // Malformed or inaccessible browser storage never prevents a fresh form.
  }
  return { draft: null, saved: false }
}

export function useTrainingDraft(accountId: string, form: FormInstance) {
  const [initial] = useState(() => readDraft(accountId))
  const [datasetItems, setDatasetItems] = useState<DatasetItem[]>(
    () => initial.draft?.datasets ?? [createDatasetItem()]
  )
  const datasetRef = useRef(datasetItems)
  const revisionRef = useRef(0)
  const [status, setStatus] = useState<'empty' | 'saved' | 'restored' | 'unavailable'>(
    initial.draft ? (initial.saved ? 'restored' : 'unavailable') : 'empty'
  )
  const [updatedAt, setUpdatedAt] = useState(initial.draft?.updatedAt)

  const save = (items = datasetRef.current) => {
    revisionRef.current += 1
    // Preserve cleared defaults as null; JSON would otherwise silently drop
    // undefined and a reload could enable a default the user had cleared.
    const values = JSON.parse(
      JSON.stringify(form.getFieldsValue(true), (_, value) => (value === undefined ? null : value))
    ) as DraftValues
    const draft: TrainingDraft = {
      version: 1,
      accountId,
      updatedAt: Date.now(),
      values,
      datasets: items,
    }
    const snapshot = { draft, saved: false }
    memory.set(accountId, snapshot)
    try {
      sessionStorage.setItem(prefix + accountId, JSON.stringify(draft))
      snapshot.saved = true
    } catch {
      // Navigation can recover from memory; beforeunload protects refresh/close.
    }
    setStatus(snapshot.saved ? 'saved' : 'unavailable')
    setUpdatedAt(draft.updatedAt)
  }

  const updateDatasets = (items: DatasetItem[]) => {
    datasetRef.current = items
    setDatasetItems(items)
    save(items)
  }

  const clear = (expectedRevision?: number) => {
    // A successful request only owns the configuration it submitted. Further
    // edits, including dataset changes, must remain recoverable.
    if (expectedRevision !== undefined && expectedRevision !== revisionRef.current) {
      return 'changed' as const
    }
    revisionRef.current += 1
    // A tombstone also prevents an older value reappearing during this document
    // session if the browser denies removing an existing storage record.
    memory.set(accountId, { draft: null, saved: true })
    let cleared = true
    try {
      sessionStorage.removeItem(prefix + accountId)
    } catch {
      try {
        // Some storage policies reject removal but permit overwriting. A null
        // tombstone invalidates the old record without retaining its values.
        sessionStorage.setItem(prefix + accountId, 'null')
      } catch {
        cleared = false
      }
    }
    setStatus('empty')
    return cleared
  }

  return {
    initialValues: initial.draft?.values ?? {},
    datasetItems,
    updateDatasets,
    save,
    clear,
    getRevision: () => revisionRef.current,
    status,
    updatedAt,
    hasDraft: status !== 'empty',
  }
}
