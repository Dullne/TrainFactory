import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const policyPath = new URL(
  '../src/pages/sync/deploymentReplicaPolicy.ts',
  import.meta.url,
)

async function loadPolicy() {
  assert.equal(existsSync(policyPath), true, 'deployment replica policy module must exist')
  const output = ts.transpileModule(readFileSync(policyPath, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText
  return import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`)
}

function deployment({ status = 'degraded', replicas, id = 'deployment-1' }) {
  return {
    deployment_id: id,
    inference_framework: 'vllm',
    status,
    replica_instances: replicas,
  }
}

const healthy = {
  replica_id: 'healthy-1',
  status: 'running',
  health_status: 'HEALTHY',
}
const unhealthy = {
  replica_id: 'unhealthy-1',
  status: 'running',
  health_status: 'UNHEALTHY',
}

test('one healthy child makes a degraded deployment selectable and auto-selected', async () => {
  const policy = await loadPolicy()
  const target = deployment({ replicas: [healthy, unhealthy] })

  assert.deepEqual(policy.getHealthyDeploymentReplicas(target), [healthy])
  assert.equal(policy.isSelectableSyncDeployment(target), true)
  assert.equal(policy.getAutomaticHealthyReplicaId(target), 'healthy-1')
  assert.equal(
    policy.isHealthyDeploymentReplicaBinding([target], 'deployment-1', 'healthy-1'),
    true,
  )
})

test('deployment with no healthy child cannot be selected or bound', async () => {
  const policy = await loadPolicy()
  const target = deployment({ status: 'running', replicas: [unhealthy] })

  assert.equal(policy.isSelectableSyncDeployment(target), false)
  assert.equal(policy.getAutomaticHealthyReplicaId(target), undefined)
  assert.equal(
    policy.isHealthyDeploymentReplicaBinding([target], 'deployment-1', 'unhealthy-1'),
    false,
  )
})

test('multiple healthy children require an explicit healthy replica', async () => {
  const policy = await loadPolicy()
  const second = { ...healthy, replica_id: 'healthy-2' }
  const target = deployment({ replicas: [healthy, second] })

  assert.equal(policy.getAutomaticHealthyReplicaId(target), undefined)
  assert.equal(
    policy.isHealthyDeploymentReplicaBinding([target], 'deployment-1', ''),
    false,
  )
  assert.equal(
    policy.isHealthyDeploymentReplicaBinding([target], 'deployment-1', 'healthy-2'),
    true,
  )
  assert.equal(
    policy.isHealthyDeploymentReplicaBinding([target], 'other-deployment', 'healthy-2'),
    false,
  )
})

test('stopped and unsupported-framework groups remain unavailable', async () => {
  const policy = await loadPolicy()
  assert.equal(
    policy.isSelectableSyncDeployment(deployment({ status: 'stopped', replicas: [healthy] })),
    false,
  )
  assert.equal(
    policy.isSelectableSyncDeployment({
      ...deployment({ status: 'running', replicas: [healthy] }),
      inference_framework: 'xinference',
    }),
    false,
  )
})
