import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const policyPath = new URL(
  '../src/pages/deployments/deploymentConfigPolicy.ts',
  import.meta.url,
)

async function loadPolicy() {
  assert.equal(existsSync(policyPath), true, 'deployment config policy module must exist')
  const output = ts.transpileModule(readFileSync(policyPath, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`)
}

const vllmLaunchConfig = {
  framework: 'vllm',
  tensor_parallel_size: 2,
  pipeline_parallel_size: 1,
  data_parallel_size: 1,
  max_context_length: 32768,
  max_concurrent_requests: 64,
  dtype: 'bfloat16',
  quantization: 'awq',
  kv_cache_dtype: 'fp8',
  gpu_pool: [0, 1],
  replica_gpu_overrides: [{ replica_index: 0, gpu_ids: [0, 1] }],
  allow_gpu_reuse: false,
  enable_expert_parallel: true,
  enforce_eager: true,
}

test('editor reads managed values from launch_config and hides reserved keys', async () => {
  const policy = await loadPolicy()
  const original = {
    external_api_config_id: 'api-config-1',
    dtype: 'float16',
    enforce_eager: false,
    docker_cmd: 'docker run preserved',
    launch_config: vllmLaunchConfig,
    custom: { keep: true },
  }

  const prepared = policy.prepareDeploymentConfigEditor(original, 'vllm')

  assert.deepEqual(prepared.fields, {
    external_api_config_id: 'api-config-1',
    dtype: 'bfloat16',
    enforce_eager: true,
    attention_backend: undefined,
  })
  assert.deepEqual(prepared.otherConfig, { custom: { keep: true } })
})

test('update preserves the full typed launch config while managed fields win', async () => {
  const policy = await loadPolicy()
  const original = {
    docker_cmd: 'docker run preserved',
    launch_config: vllmLaunchConfig,
    custom: { old: true },
  }

  const updated = policy.buildDeploymentConfigUpdate({
    originalConfig: original,
    otherConfig: {
      custom: { edited: true },
      launch_config: { framework: 'sglang' },
      docker_cmd: 'attacker supplied',
      dtype: 'half',
    },
    framework: 'vllm',
    fields: {
      external_api_config_id: null,
      dtype: 'float32',
      enforce_eager: false,
    },
  })

  assert.deepEqual(updated, {
    custom: { edited: true },
    docker_cmd: 'docker run preserved',
    external_api_config_id: null,
    launch_config: {
      ...vllmLaunchConfig,
      dtype: 'float32',
      enforce_eager: false,
    },
  })
})

test('SGLang attention backend is nested and clearing it writes null', async () => {
  const policy = await loadPolicy()
  const original = {
    attention_backend: 'triton',
    launch_config: {
      framework: 'sglang',
      tensor_parallel_size: 2,
      pipeline_parallel_size: 1,
      data_parallel_size: 1,
      dtype: 'bfloat16',
      kv_cache_dtype: 'auto',
      gpu_pool: [0, 1],
      replica_gpu_overrides: [],
      allow_gpu_reuse: false,
      expert_parallel_size: 2,
      attention_backend: 'flashinfer',
    },
  }

  const prepared = policy.prepareDeploymentConfigEditor(original, 'sglang')
  assert.equal(prepared.fields.attention_backend, 'flashinfer')

  const updated = policy.buildDeploymentConfigUpdate({
    originalConfig: original,
    otherConfig: prepared.otherConfig,
    framework: 'sglang',
    fields: {
      external_api_config_id: null,
      dtype: 'bfloat16',
      attention_backend: undefined,
    },
  })
  assert.equal(updated.launch_config.attention_backend, null)
  assert.equal(updated.launch_config.expert_parallel_size, 2)
  assert.equal('attention_backend' in updated, false)
})

test('canonical null never falls back to a stale legacy attention backend', async () => {
  const policy = await loadPolicy()
  const original = {
    attention_backend: 'triton',
    launch_config: {
      framework: 'sglang',
      tensor_parallel_size: 1,
      pipeline_parallel_size: 1,
      data_parallel_size: 1,
      dtype: 'auto',
      kv_cache_dtype: 'auto',
      gpu_pool: [],
      replica_gpu_overrides: [],
      allow_gpu_reuse: false,
      expert_parallel_size: 1,
      attention_backend: null,
    },
  }

  const prepared = policy.prepareDeploymentConfigEditor(original, 'sglang')
  assert.equal(prepared.fields.attention_backend, undefined)
  assert.equal(
    policy.getManagedDeploymentConfigValue(original, 'attention_backend'),
    null,
  )
})

test('legacy managed fields remain a read fallback when launch_config is absent', async () => {
  const policy = await loadPolicy()
  const prepared = policy.prepareDeploymentConfigEditor(
    { dtype: 'float16', enforce_eager: true, custom: 1 },
    'vllm',
  )

  assert.equal(prepared.fields.dtype, 'float16')
  assert.equal(prepared.fields.enforce_eager, true)
  assert.deepEqual(prepared.otherConfig, { custom: 1 })
})

test('SGLang expert parallel size is one or exactly tensor parallel size', async () => {
  const policy = await loadPolicy()

  assert.equal(policy.isValidSglangExpertParallelSize(1, 2), true)
  assert.equal(policy.isValidSglangExpertParallelSize(2, 2), true)
  assert.equal(policy.isValidSglangExpertParallelSize(3, 2), false)
  assert.equal(policy.isValidSglangExpertParallelSize(2, 1), false)
  assert.equal(policy.isValidSglangExpertParallelSize(1.5, 1.5), false)
})
