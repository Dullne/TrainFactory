import { expect, test, type Route } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

function evaluationResponse(route: Route, total = 25) {
  const url = new URL(route.request().url())
  const offset = Number(url.searchParams.get('offset')) || 0
  const limit = Number(url.searchParams.get('limit')) || 10
  return route.fulfill({
    json: {
      items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, i) => ({
        task_id: `eval-${offset + i}`,
        task_name: `Evaluation ${offset + i}`,
        eval_type: 'single',
        model_configs: [],
        dataset_configs: [],
        status: 'succeeded',
        progress: 100,
        created_at: '2026-09-19T00:00:00',
      })),
      total,
    },
  })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

test('evaluation keeps pager with successful rows when requested page fails and refresh retries it', async ({
  page,
}) => {
  let fail = true
  let failures = 0
  await page.route('**/api/evaluations/tasks?**', (route) => {
    if (fail && new URL(route.request().url()).searchParams.get('offset') === '10') {
      failures++
      return route.fulfill({ status: 503, json: { detail: 'temporary failure' } })
    }
    return evaluationResponse(route)
  })
  await page.goto('/evaluations')
  await expect(page.getByText('Evaluation 0', { exact: true })).toBeVisible()
  await page.locator('.ant-pagination-item-2').click()
  await expect.poll(() => failures).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
  await expect(page.getByText('Evaluation 0', { exact: true })).toBeVisible()
  await expect(page.getByTestId('evaluation-stale-state')).toBeVisible()
  fail = false
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('Evaluation 10', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
})

test('evaluation refetches the valid page when the last page shrinks', async ({ page }) => {
  let total = 11
  await page.route('**/api/evaluations/tasks?**', (route) => evaluationResponse(route, total))
  await page.goto('/evaluations?page=2')
  await expect(page.getByText('Evaluation 10', { exact: true })).toBeVisible()
  total = 10
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('Evaluation 0', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
  expect(new URL(page.url()).searchParams.has('page')).toBe(false)
})

test('evaluation retries a failed corrected offset and then normalizes the URL', async ({
  page,
}) => {
  let total = 11
  let fail = false
  await page.route('**/api/evaluations/tasks?**', (route) => {
    if (fail && new URL(route.request().url()).searchParams.get('offset') === '0')
      return route.fulfill({ status: 503, json: { detail: 'correction failed' } })
    return evaluationResponse(route, total)
  })
  await page.goto('/evaluations?page=2')
  await expect(page.getByText('Evaluation 10', { exact: true })).toBeVisible()
  total = 10
  fail = true
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByTestId('evaluation-stale-state')).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
  await expect(page.getByText('Evaluation 10', { exact: true })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('Evaluation 0', { exact: true })).toBeVisible()
  await expect.poll(() => new URL(page.url()).searchParams.has('page')).toBe(false)
})

function generationResponse(route: Route) {
  const url = new URL(route.request().url())
  const offset = Number(url.searchParams.get('offset')) || 0
  const limit = Number(url.searchParams.get('limit')) || 10
  return route.fulfill({
    json: {
      tasks: [
        {
          task_id: `gen-${offset}`,
          task_name: `Generation ${offset}`,
          status: offset === 0 ? 'running' : 'completed',
          generation_mode: 'doc_to_training',
          progress: 50,
          total_docs: 10,
          processed_docs: 5,
          output_sample_count: 10,
          created_at: '2026-09-19T00:00:00',
        },
      ],
      total: 25,
      limit,
      offset,
      stats: { total: 25, pending: 0, running: 1, completed: 24, failed: 0, stopped: 0 },
    },
  })
}

test('generation polling cannot replace a pending foreground page with the previous page', async ({
  page,
}) => {
  let pending: Route | undefined
  let hold = true
  const offsets: number[] = []
  await page.route('**/api/generation/tasks?**', async (route) => {
    offsets.push(Number(new URL(route.request().url()).searchParams.get('offset')))
    if (hold && new URL(route.request().url()).searchParams.get('offset') === '10') pending = route
    else await generationResponse(route)
  })
  await page.goto('/datasets/generation')
  await expect(page.getByText('Generation 0', { exact: true })).toBeVisible()
  await page.locator('.ant-pagination-item-2').click()
  await expect.poll(() => Boolean(pending)).toBe(true)
  // Let one real background polling interval elapse while page two is in flight.
  await page.waitForTimeout(5300)
  const requestsWhilePending = [...offsets]
  hold = false
  await generationResponse(pending!)
  expect(requestsWhilePending.slice(requestsWhilePending.indexOf(10))).toEqual([10])
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
  expect(new URL(page.url()).searchParams.get('generation_page')).toBe('2')
})

test('generation refresh retries the failed URL page without relabelling old rows', async ({
  page,
}) => {
  let fail = true
  let failures = 0
  await page.route('**/api/generation/tasks?**', (route) => {
    if (fail && new URL(route.request().url()).searchParams.get('offset') === '10') {
      failures++
      return route.fulfill({ status: 503, json: { detail: 'temporary failure' } })
    }
    return generationResponse(route)
  })
  await page.goto('/datasets/generation')
  await expect(page.getByText('Generation 0', { exact: true })).toBeVisible()
  await page.locator('.ant-pagination-item-2').click()
  await expect.poll(() => failures).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
  fail = false
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
})
