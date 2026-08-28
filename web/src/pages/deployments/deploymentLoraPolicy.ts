const SGLANG_RERANKER_MODEL_TYPES = new Set([
  'rerank',
  'reranker',
  'decoder_reranker',
])

export function isSglangRerankerLoraUnsupported(
  framework: string,
  modelType?: string | null,
): boolean {
  return framework === 'sglang' && SGLANG_RERANKER_MODEL_TYPES.has(modelType ?? '')
}

export function supportsDeploymentLora(
  framework: string,
  modelType?: string | null,
): boolean {
  if (framework === 'vllm') return true
  if (framework === 'sglang') return modelType === 'llm'
  return false
}

export function normalizeDeploymentLoraEnabled(
  enabled: boolean,
  framework: string,
  modelType?: string | null,
): boolean {
  return enabled && supportsDeploymentLora(framework, modelType)
}
