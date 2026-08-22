import { test, expect } from '@playwright/test'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { mockApi } from './helpers/mockApi'

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
