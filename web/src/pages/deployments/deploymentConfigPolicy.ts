export interface DeploymentConfigEditorFields {
  external_api_config_id?: string | null
  dtype?: string
  enforce_eager?: boolean
  attention_backend?: string
}

interface BuildDeploymentConfigUpdateArgs {
  originalConfig: Record<string, unknown>
  otherConfig: Record<string, unknown>
  framework: string
  fields: DeploymentConfigEditorFields
}

const RESERVED_EDITOR_KEYS = new Set([
  'external_api_config_id',
  'dtype',
  'enforce_eager',
  'attention_backend',
  'docker_cmd',
  'launch_config',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function launchConfigFrom(config: Record<string, unknown>): Record<string, unknown> | undefined {
  return isRecord(config.launch_config) ? config.launch_config : undefined
}

export function prepareDeploymentConfigEditor(
  config: Record<string, unknown>,
  framework: string,
): {
  fields: DeploymentConfigEditorFields
  otherConfig: Record<string, unknown>
} {
  const managedDtype = getManagedDeploymentConfigValue(config, 'dtype')
  const managedEnforceEager = getManagedDeploymentConfigValue(config, 'enforce_eager')
  const managedAttentionBackend = getManagedDeploymentConfigValue(
    config,
    'attention_backend',
  )
  const otherConfig: Record<string, unknown> = {}

  for (const [key, value] of Object.entries(config)) {
    if (!RESERVED_EDITOR_KEYS.has(key)) otherConfig[key] = value
  }

  return {
    fields: {
      external_api_config_id:
        typeof config.external_api_config_id === 'string'
          ? config.external_api_config_id
          : undefined,
      dtype:
        typeof managedDtype === 'string' ? managedDtype : undefined,
      enforce_eager:
        framework === 'vllm'
          ? managedEnforceEager === true
          : undefined,
      attention_backend:
        framework === 'sglang'
          ? typeof managedAttentionBackend === 'string'
            ? managedAttentionBackend
            : undefined
          : undefined,
    },
    otherConfig,
  }
}

export function buildDeploymentConfigUpdate({
  originalConfig,
  otherConfig,
  framework,
  fields,
}: BuildDeploymentConfigUpdateArgs): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(otherConfig)) {
    if (!RESERVED_EDITOR_KEYS.has(key)) result[key] = value
  }

  if (originalConfig.docker_cmd !== undefined) {
    result.docker_cmd = originalConfig.docker_cmd
  }
  result.external_api_config_id = fields.external_api_config_id || null

  const originalLaunchConfig = launchConfigFrom(originalConfig)
  if (originalLaunchConfig) {
    const launchConfig = { ...originalLaunchConfig }
    if (fields.dtype) launchConfig.dtype = fields.dtype
    if (framework === 'vllm' && fields.enforce_eager !== undefined) {
      launchConfig.enforce_eager = fields.enforce_eager
    }
    if (framework === 'sglang') {
      launchConfig.attention_backend = fields.attention_backend || null
    }
    result.launch_config = launchConfig
    return result
  }

  if (fields.dtype) result.dtype = fields.dtype
  if (framework === 'vllm' && fields.enforce_eager) result.enforce_eager = true
  if (framework === 'sglang' && fields.attention_backend) {
    result.attention_backend = fields.attention_backend
  }
  return result
}

export function getDeploymentLaunchConfig(
  config: Record<string, unknown>,
): Record<string, unknown> | undefined {
  return launchConfigFrom(config)
}

export function getManagedDeploymentConfigValue(
  config: Record<string, unknown>,
  key: string,
): unknown {
  const launchConfig = launchConfigFrom(config)
  return launchConfig ? launchConfig[key] : config[key]
}

export function isValidSglangExpertParallelSize(
  expertParallelSize: number,
  tensorParallelSize: number,
): boolean {
  return (
    Number.isInteger(expertParallelSize) &&
    Number.isInteger(tensorParallelSize) &&
    (expertParallelSize === 1 || expertParallelSize === tensorParallelSize)
  )
}
