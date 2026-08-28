import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const policyPath = new URL(
  '../src/pages/deployments/deploymentLoraPolicy.ts',
  import.meta.url,
)

async function loadPolicy() {
  assert.equal(
    existsSync(policyPath),
    true,
    'deployment LoRA policy module must exist',
  )
  const output = ts.transpileModule(readFileSync(policyPath, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`)
}

test('SGLang v0.5.17 exposes LoRA only for llm models', async () => {
  const policy = await loadPolicy()

  assert.equal(policy.supportsDeploymentLora('sglang', 'llm'), true)
  assert.equal(policy.supportsDeploymentLora('sglang', 'embedding'), false)
  assert.equal(policy.supportsDeploymentLora('vllm', 'reranker'), true)
  assert.equal(policy.supportsDeploymentLora('xinference', 'embedding'), false)
  for (const modelType of ['rerank', 'reranker', 'decoder_reranker']) {
    assert.equal(policy.supportsDeploymentLora('sglang', modelType), false)
  }
})

test('unsupported SGLang model selection clears enabled LoRA', async () => {
  const policy = await loadPolicy()

  for (const modelType of ['embedding', 'rerank', 'reranker', 'decoder_reranker']) {
    assert.equal(
      policy.normalizeDeploymentLoraEnabled(true, 'sglang', modelType),
      false,
    )
  }
  assert.equal(policy.normalizeDeploymentLoraEnabled(true, 'sglang', 'llm'), true)
})
