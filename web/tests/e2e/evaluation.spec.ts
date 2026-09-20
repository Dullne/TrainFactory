import { test, expect } from '@playwright/test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { mockApi } from './helpers/mockApi'

const deploymentBase = {
  model_id: 'model-eval',
  model_uid: 'eval-reranker',
  xinference_endpoint: 'http://127.0.0.1:8999',
  replica: 1,
  gpu_memory_utilization: 0.8,
  deploy_mode: 'container',
  inference_framework: 'vllm',
  enable_lora: false,
  max_loras: 0,
  max_lora_rank: 0,
  created_at: '2026-08-24T00:00:00Z',
  updated_at: '2026-08-24T00:00:00Z',
}

test('evaluation reopen rejects the previous deployment snapshot when the new list fails', async ({
  page,
}) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  let deploymentRequestCount = 0
  const submittedBodies: Record<string, unknown>[] = []
  await page.route('**/api/deployments**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() !== 'GET' || url.pathname !== '/api/deployments') {
      return route.fallback()
    }

    deploymentRequestCount += 1
    if (deploymentRequestCount === 1) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          deployments: [
            {
              ...deploymentBase,
              deployment_id: 'dep-cycle-one',
              deployment_name: 'cycle-one-deployment',
              xinference_endpoint: 'http://127.0.0.1:8700',
              status: 'running',
            },
          ],
          total: 1,
        }),
      })
    }

    return route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'deployment list unavailable' }),
    })
  })
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    submittedBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: `eval-cycle-${submittedBodies.length}`, message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  const modal = page.locator('.ant-modal')
  await modal.getByRole('button', { name: /Add Model$/ }).click()
  await modal.getByRole('combobox').nth(1).click()
  await page.getByText(/cycle-one-deployment/).last().click()
  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()
  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await expect.poll(() => submittedBodies.length).toBe(1)
  await expect(modal).not.toBeVisible()

  await page.getByRole('button', { name: /Create Task$/ }).click()
  await expect.poll(() => deploymentRequestCount).toBe(2)
  await expect(page.getByText('Failed to load data').last()).toBeVisible()
  await modal.getByRole('button', { name: /Add Model$/ }).click()

  const deploymentPicker = modal
    .locator('.ant-select')
    .filter({ hasText: 'Select a deployed model' })
  const deploymentSelectionDisabled = await deploymentPicker
    .getByRole('combobox')
    .isDisabled()
  const staleOption = page.getByText(/cycle-one-deployment/).last()
  if (!deploymentSelectionDisabled) {
    await deploymentPicker.locator('.ant-select-selector').click()
    await expect(staleOption).toBeVisible()
    await staleOption.click()
  }

  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()
  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await page.waitForTimeout(250)

  expect(submittedBodies).toHaveLength(1)
  expect(deploymentSelectionDisabled).toBe(true)
})

test('evaluation ignores a slow deployment response from the previous open cycle', async ({
  page,
}) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  let deploymentRequestCount = 0
  let releaseFirstResponse!: () => void
  let firstResponseCompleted = false
  const firstResponseGate = new Promise<void>((resolve) => {
    releaseFirstResponse = resolve
  })
  const submittedBodies: Record<string, unknown>[] = []
  await page.route('**/api/deployments**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() !== 'GET' || url.pathname !== '/api/deployments') {
      return route.fallback()
    }

    deploymentRequestCount += 1
    const requestNumber = deploymentRequestCount
    if (requestNumber === 1) await firstResponseGate
    await route.fulfill({
      status: 200,
      headers: { 'x-evaluation-open-cycle': String(requestNumber) },
      contentType: 'application/json',
      body: JSON.stringify({
        deployments: [
          {
            ...deploymentBase,
            deployment_id: requestNumber === 1 ? 'dep-cycle-slow' : 'dep-cycle-current',
            deployment_name:
              requestNumber === 1 ? 'cycle-one-slow' : 'cycle-two-current',
            xinference_endpoint:
              requestNumber === 1
                ? 'http://127.0.0.1:8800'
                : 'http://127.0.0.1:8801',
            status: 'running',
          },
        ],
        total: 1,
      }),
    })
    if (requestNumber === 1) firstResponseCompleted = true
  })
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    submittedBodies.push(route.request().postDataJSON() as Record<string, unknown>)
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: 'eval-stale-cycle', message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  const modal = page.locator('.ant-modal')
  await expect.poll(() => deploymentRequestCount).toBe(1)
  await modal.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(modal).not.toBeVisible()

  await page.getByRole('button', { name: /Create Task$/ }).click()
  await expect.poll(() => deploymentRequestCount).toBe(2)
  await modal.getByRole('button', { name: /Add Model$/ }).click()
  const deploymentPicker = modal
    .locator('.ant-select')
    .filter({ hasText: 'Select a deployed model' })
  await deploymentPicker.locator('.ant-select-selector').click()
  await expect(page.getByText(/cycle-two-current/).last()).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.locator('.ant-select-dropdown').last()).not.toBeVisible()

  const delayedResponse = page.waitForResponse(
    (response) => response.headers()['x-evaluation-open-cycle'] === '1'
  )
  releaseFirstResponse()
  await expect.poll(() => firstResponseCompleted).toBe(true)
  expect(await (await delayedResponse).finished()).toBeNull()
  await page.evaluate(
    () => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  )

  await deploymentPicker.locator('.ant-select-selector').click()
  const currentOption = page.getByText(/cycle-two-current/).last()
  const staleOption = page.getByText(/cycle-one-slow/).last()
  // The retained dropdown is visible during enter-prepare, then briefly scales
  // to zero in enter-start. Snapshot the options only after that motion ends.
  await expect(page.locator('.ant-select-dropdown').last()).not.toHaveClass(
    /ant-slide-up-(?:enter|appear)/
  )
  await expect
    .poll(async () =>
      (await currentOption.isVisible().catch(() => false)) ||
      (await staleOption.isVisible().catch(() => false))
    )
    .toBe(true)
  const currentOptionVisible = await currentOption.isVisible().catch(() => false)
  const staleOptionVisible = await staleOption.isVisible().catch(() => false)
  if (staleOptionVisible) {
    await staleOption.click()
  } else {
    await page.keyboard.press('Escape')
  }

  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()
  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await page.waitForTimeout(250)

  expect(submittedBodies).toHaveLength(0)
  expect(currentOptionVisible).toBe(true)
  expect(staleOptionVisible).toBe(false)
})

test('evaluation accepts a healthy replica from a degraded deployment with one trusted list request', async ({
  page,
}) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  const consoleErrors: string[] = []
  page.on('console', (entry) => {
    if (entry.type() === 'error') consoleErrors.push(entry.text())
  })
  const deploymentRequests: string[] = []
  let submittedBody: Record<string, unknown> | undefined
  await page.route('**/api/deployments**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() !== 'GET' || url.pathname !== '/api/deployments') {
      return route.fallback()
    }
    deploymentRequests.push(url.toString())
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        deployments: [
          {
            ...deploymentBase,
            deployment_id: 'dep-degraded-healthy',
            deployment_name: 'degraded-healthy-group',
            status: 'degraded',
            replica_instances: [
              {
                replica_id: 'replica-healthy-a',
                deployment_id: 'dep-degraded-healthy',
                replica_index: 0,
                endpoint: 'http://127.0.0.1:8100',
                port: 8100,
                gpu_ids: [0],
                status: 'running',
                health_status: 'HEALTHY',
              },
              {
                replica_id: 'replica-healthy-b',
                deployment_id: 'dep-degraded-healthy',
                replica_index: 1,
                endpoint: 'http://127.0.0.1:8101',
                port: 8101,
                gpu_ids: [1],
                status: 'running',
                health_status: 'HEALTHY',
              },
              {
                replica_id: 'replica-unhealthy',
                deployment_id: 'dep-degraded-healthy',
                replica_index: 2,
                endpoint: 'http://127.0.0.1:8102',
                port: 8102,
                gpu_ids: [2],
                status: 'running',
                health_status: 'UNHEALTHY',
              },
            ],
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-degraded-unhealthy',
            deployment_name: 'degraded-fully-unhealthy',
            status: 'degraded',
            replica_instances: [
              {
                replica_id: 'replica-degraded-unhealthy',
                deployment_id: 'dep-degraded-unhealthy',
                replica_index: 0,
                endpoint: 'http://127.0.0.1:8200',
                port: 8200,
                gpu_ids: [3],
                status: 'running',
                health_status: 'UNHEALTHY',
              },
            ],
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-running-unhealthy',
            deployment_name: 'running-fully-unhealthy',
            status: 'running',
            replica_instances: [
              {
                replica_id: 'replica-running-unhealthy',
                deployment_id: 'dep-running-unhealthy',
                replica_index: 0,
                endpoint: 'http://127.0.0.1:8300',
                port: 8300,
                gpu_ids: [4],
                status: 'running',
                health_status: 'UNHEALTHY',
              },
            ],
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-tampered',
            deployment_name: 'tampered-replica-owner',
            status: 'running',
            replica_instances: [
              {
                replica_id: 'replica-foreign-owner',
                deployment_id: 'dep-elsewhere',
                replica_index: 0,
                endpoint: 'http://127.0.0.1:8400',
                port: 8400,
                gpu_ids: [5],
                status: 'running',
                health_status: 'HEALTHY',
              },
            ],
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-canonical-empty',
            deployment_name: 'canonical-zero-child',
            status: 'running',
            config: { replica_schema_version: 1 },
            replica_instances: [],
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-legacy-running',
            deployment_name: 'legacy-running',
            xinference_endpoint: 'http://127.0.0.1:8500',
            status: 'running',
          },
          {
            ...deploymentBase,
            deployment_id: 'dep-legacy-stopped',
            deployment_name: 'legacy-stopped',
            status: 'stopped',
          },
        ],
        total: 7,
      }),
    })
  })
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    submittedBody = route.request().postDataJSON() as Record<string, unknown>
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: 'eval-degraded', message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  const modal = page.locator('.ant-modal')
  await modal.getByRole('button', { name: /Add Model$/ }).click()

  await modal.getByRole('combobox').nth(1).click()
  await expect(page.getByText(/degraded-healthy-group/).last()).toBeVisible()
  await expect(page.getByText(/legacy-running/).last()).toBeVisible()
  await expect(page.getByText(/degraded-fully-unhealthy/)).toHaveCount(0)
  await expect(page.getByText(/running-fully-unhealthy/)).toHaveCount(0)
  await expect(page.getByText(/tampered-replica-owner/)).toHaveCount(0)
  await expect(page.getByText(/canonical-zero-child/)).toHaveCount(0)
  await expect(page.getByText(/legacy-stopped/)).toHaveCount(0)
  await page.getByText(/degraded-healthy-group/).last().click()

  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()

  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await expect(page.getByText('Please complete all model configurations')).toBeVisible()
  expect(submittedBody).toBeUndefined()

  await modal.getByRole('combobox').nth(2).click()
  const unhealthyOption = page
    .locator('.ant-select-item-option')
    .filter({ hasText: 'http://127.0.0.1:8102' })
  await expect(unhealthyOption).toHaveClass(/ant-select-item-option-disabled/)
  await page.getByText(/http:\/\/127\.0\.0\.1:8100/).last().click()

  await page.screenshot({
    path: join(tmpdir(), 'train-factory-evaluation-degraded-replica.png'),
  })

  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await expect.poll(() => submittedBody).toBeTruthy()

  expect(deploymentRequests).toHaveLength(1)
  expect(new URL(deploymentRequests[0]).searchParams.has('status')).toBe(false)
  expect(consoleErrors.filter((entry) => !entry.startsWith('Warning: [antd:'))).toEqual([])
  const models = submittedBody?.model_configs as Array<Record<string, unknown>>
  expect(models).toEqual([
    {
      endpoint: 'http://127.0.0.1:8100',
      name: 'degraded-healthy-group',
      deployment_id: 'dep-degraded-healthy',
      deployment_replica_id: 'replica-healthy-a',
      model_name: 'eval-reranker',
      inference_framework: 'vllm',
    },
  ])
})

test('evaluation keeps legacy running deployment submission compatible', async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  let submittedBody: Record<string, unknown> | undefined
  await page.route('**/api/deployments**', async (route) => {
    const url = new URL(route.request().url())
    if (route.request().method() !== 'GET' || url.pathname !== '/api/deployments') {
      return route.fallback()
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        deployments: [
          {
            ...deploymentBase,
            deployment_id: 'dep-legacy-compatible',
            deployment_name: 'legacy-compatible',
            xinference_endpoint: 'http://127.0.0.1:8600',
            status: 'running',
          },
        ],
        total: 1,
      }),
    })
  })
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    submittedBody = route.request().postDataJSON() as Record<string, unknown>
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: 'eval-legacy', message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  const modal = page.locator('.ant-modal')
  await modal.getByRole('button', { name: /Add Model$/ }).click()
  await modal.getByRole('combobox').nth(1).click()
  await page.getByText(/legacy-compatible/).last().click()
  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()
  await modal.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await expect.poll(() => submittedBody).toBeTruthy()

  const models = submittedBody?.model_configs as Array<Record<string, unknown>>
  expect(models).toEqual([
    {
      endpoint: 'http://127.0.0.1:8600',
      name: 'legacy-compatible',
      deployment_id: 'dep-legacy-compatible',
      model_name: 'eval-reranker',
      inference_framework: 'vllm',
    },
  ])
})

test('registered evaluation submits dataset identity without client storage path', async ({
  page,
}) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  let submittedBody: Record<string, unknown> | undefined
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') {
      return route.fallback()
    }
    submittedBody = route.request().postDataJSON() as Record<string, unknown>
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: 'eval-created', message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  await expect(page.locator('.ant-modal-title')).toHaveText('Create Evaluation Task')

  await page.getByRole('button', { name: /Add Model$/ }).click()
  await page.locator('.ant-modal').getByRole('combobox').nth(1).click()
  await page
    .getByText(/xinference-deployment/)
    .last()
    .click()

  await page.getByRole('button', { name: /Registered$/ }).click()
  await page.locator('.ant-modal').getByRole('combobox').nth(2).click()
  await page
    .getByText(/eval-rerank-dataset/)
    .last()
    .click()

  await expect(page.getByRole('button', { name: /Local$/ })).toHaveCount(0)
  await page.getByRole('button', { name: 'Create Evaluation Task', exact: true }).click()
  await expect.poll(() => submittedBody).toBeTruthy()

  const datasets = submittedBody?.dataset_configs as Array<Record<string, unknown>>
  expect(datasets).toEqual([
    {
      type: 'registered',
      name: 'eval-rerank-dataset',
      dataset_id: 'ds-456',
    },
  ])
})

test('custom evaluation source drops stale deployment identity', async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))

  let submittedBody: Record<string, unknown> | undefined
  await page.route('**/api/evaluations', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    submittedBody = route.request().postDataJSON() as Record<string, unknown>
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ task_id: 'eval-custom', message: 'created' }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('button', { name: /Create Task$/ }).click()
  const modal = page.locator('.ant-modal')
  await modal.getByRole('button', { name: /Add Model$/ }).click()

  await modal.getByRole('combobox').nth(1).click()
  await page.getByText(/xinference-deployment/).last().click()
  await modal.locator('.ant-select').nth(0).locator('.ant-select-selector').click()
  await page.getByText('Custom Endpoint', { exact: true }).last().click()

  await modal.getByPlaceholder('Endpoint (e.g. http://localhost:9997)').fill('http://custom.test/v1')
  await modal.getByPlaceholder('Display Name').fill('custom-reranker')
  await modal.getByRole('button', { name: /Registered$/ }).click()
  await modal
    .locator('.ant-select')
    .filter({ hasText: 'Select registered evaluation dataset' })
    .click()
  await page.getByText(/eval-rerank-dataset/).last().click()

  await modal.locator('.ant-modal-footer').getByRole('button', {
    name: 'Create Evaluation Task',
  }).click()
  await expect.poll(() => submittedBody).toBeTruthy()

  const models = submittedBody?.model_configs as Array<Record<string, unknown>>
  expect(models).toEqual([
    {
      endpoint: 'http://custom.test/v1',
      name: 'custom-reranker',
    },
  ])
})

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('evaluation detail distinguishes namespaced dataset identities with readable labels', async ({
  page,
}) => {
  await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
  await page.route('**/api/evaluations/tasks**', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [
          {
            task_id: 'eval-namespaced-datasets',
            task_name: 'Dataset identity labels',
            eval_type: 'reranker-mixed',
            model_configs: [{ endpoint: 'http://model.test/v1', name: 'reranker' }],
            dataset_configs: [
              {
                type: 'mteb',
                name: 'T2Reranking',
                result_key: 'mteb:T2Reranking',
              },
              {
                type: 'registered',
                name: 'T2Reranking',
                dataset_id: 'dataset-1',
                result_key: 'dataset:dataset-1',
              },
            ],
            identity_schema_version: 2,
            identity_map: {
              models: { reranker: { name: 'reranker' } },
              datasets: {
                'mteb:T2Reranking': { name: 'T2Reranking', type: 'mteb' },
                'dataset:dataset-1': { name: 'T2Reranking', type: 'registered' },
              },
            },
            status: 'failed',
            progress: 75,
            results: {
              reranker: {
                'mteb:T2Reranking': { 'NDCG@10': 0.9 },
                'dataset:dataset-1': { 'NDCG@10': 0.8 },
                'local:C:\\private\\customer-a\\secret.jsonl': {
                  'NDCG@10': 0.7,
                },
              },
            },
            model_progress: {
              reranker: {
                'mteb:T2Reranking': { progress: 100, status: 'completed' },
                'dataset:dataset-1': { progress: 100, status: 'completed' },
                'local:C:\\private\\customer-a\\secret.jsonl': {
                  progress: 100,
                  status: 'completed',
                },
              },
            },
            error_message: 'One later attempt failed',
            batch_size: 50,
            workers: 8,
            model_workers: 2,
            created_at: new Date().toISOString(),
            completed_at: new Date().toISOString(),
          },
        ],
        total: 1,
      }),
    })
  })

  await page.goto('/evaluations')
  const taskRow = page.locator('tr', { has: page.getByText('Dataset identity labels') })
  await taskRow.getByRole('button', { name: 'eye' }).click()

  const modal = page.locator('.ant-modal')
  await expect(modal.getByText('T2Reranking', { exact: true })).toHaveCount(6)
  await expect(modal.getByText('MTEB', { exact: true })).toHaveCount(3)
  await expect(modal.getByText('Registered', { exact: true })).toHaveCount(3)
  await expect(modal.getByText('mteb:T2Reranking', { exact: true })).toHaveCount(0)
  await expect(modal.getByText('dataset:dataset-1', { exact: true })).toHaveCount(0)
  await expect(modal.getByText(/secret\.jsonl/)).toHaveCount(0)
  await expect(modal.getByText('Local dataset', { exact: true })).toHaveCount(2)
  await expect(modal.getByText('Local', { exact: true })).toHaveCount(2)
  await page.screenshot({ path: join(tmpdir(), 'train-factory-evaluation-identity.png') })
})

test('evaluation list shows tasks with different statuses', async ({ page }) => {
  await page.goto('/evaluations')

  // Wait for API response
  await page.waitForResponse(
    (response) => response.url().includes('/api/evaluations/tasks') && response.status() === 200
  )

  // Wait for table to render with data (task_id pattern)
  await page.waitForSelector('table tbody tr:has-text("eval-")')

  await expect(page.getByRole('heading', { name: '评估任务' })).toBeVisible()

  // Check that both succeeded and failed tasks are displayed
  await expect(page.getByText('对比评估任务')).toBeVisible()
  await expect(page.getByText('失败的评估任务')).toBeVisible()
})

test('can delete a failed evaluation task', async ({ page }) => {
  await page.goto('/evaluations')

  // Wait for API response and data render
  await page.waitForResponse(
    (response) => response.url().includes('/api/evaluations/tasks') && response.status() === 200
  )
  await page.waitForSelector('table tbody tr:has-text("eval-")')

  await expect(page.getByRole('heading', { name: '评估任务' })).toBeVisible()

  // Find the failed task row and click delete
  const failedTaskRow = page.locator('tr', { has: page.getByText('失败的评估任务') })
  await expect(failedTaskRow).toBeVisible()

  // Click the delete button (icon button with delete icon) in the failed task row
  const deleteButton = failedTaskRow.getByRole('button', { name: 'delete' })
  await deleteButton.click()

  // Confirm the deletion in the popconfirm (button text has space: "确 定")
  const confirmButton = page.getByRole('button', { name: /确.*定/ })
  await confirmButton.click()

  // Verify success message
  await expect(page.getByText('任务已删除')).toBeVisible()
})

test('failed task shows status label', async ({ page }) => {
  await page.goto('/evaluations')

  // Wait for API response and data render
  await page.waitForResponse(
    (response) => response.url().includes('/api/evaluations/tasks') && response.status() === 200
  )
  await page.waitForSelector('table tbody tr:has-text("eval-")')

  // Find the failed task row
  const failedTaskRow = page.locator('tr', { has: page.getByText('失败的评估任务') })
  await expect(failedTaskRow).toBeVisible()

  // Check that failed status label is displayed (use exact match to avoid matching task name)
  await expect(failedTaskRow.getByText('失败', { exact: true })).toBeVisible()
})
