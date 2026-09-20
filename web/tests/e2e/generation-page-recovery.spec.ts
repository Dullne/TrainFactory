import { expect, test, type Route } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

function respond(route: Route, total: number) {
  const url = new URL(route.request().url())
  const offset = Number(url.searchParams.get('offset'))
  const limit = Number(url.searchParams.get('limit'))
  return route.fulfill({
    json: {
      tasks: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) => ({
        task_id: `task-${offset + index}`,
        task_name: `Generation ${offset + index}`,
        status: 'completed',
        generation_mode: 'doc_to_training',
        progress: 100,
        total_docs: 10,
        processed_docs: 10,
        output_sample_count: 10,
        created_at: '2026-09-20T00:00:00',
      })),
      total,
      limit,
      offset,
      stats: { total, pending: 0, running: 0, completed: total, failed: 0, stopped: 0 },
    },
  })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

test('generation normalizes an out-of-range bookmarked page and shows existing rows', async ({
  page,
}) => {
  await page.route('**/api/generation/tasks?**', (route) => respond(route, 1))
  await page.goto('/datasets/generation?generation_page=2')
  await expect(page.getByText('Generation 0', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
  expect(new URL(page.url()).searchParams.has('generation_page')).toBe(false)
})

test('generation keeps successful rows while a shrinking-page correction fails and retries its offset', async ({
  page,
}) => {
  let total = 11
  let failCorrection = false
  const offsets: number[] = []
  await page.route('**/api/generation/tasks?**', (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get('offset'))
    offsets.push(offset)
    if (failCorrection && offset === 0)
      return route.fulfill({ status: 503, json: { detail: 'Correction unavailable' } })
    return respond(route, total)
  })
  await page.goto('/datasets/generation?generation_page=2')
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
  total = 1
  failCorrection = true
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect.poll(() => offsets.includes(0)).toBe(true)
  await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
  expect(new URL(page.url()).searchParams.get('generation_page')).toBe('2')
  failCorrection = false
  const retryStart = offsets.length
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('Generation 0', { exact: true })).toBeVisible()
  expect(offsets[retryStart]).toBe(0)
  expect(new URL(page.url()).searchParams.has('generation_page')).toBe(false)
})

test('generation discards a delayed page correction after newer navigation', async ({ page }) => {
  let total = 21
  let holdCorrection = false
  let correctionStarted = false
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/generation/tasks?**', async (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get('offset'))
    if (holdCorrection && offset === 0) {
      correctionStarted = true
      await gate
      return respond(route, 1)
    }
    return respond(route, total)
  })
  await page.goto('/datasets/generation?generation_page=3')
  await expect(page.getByText('Generation 20', { exact: true })).toBeVisible()
  total = 1
  holdCorrection = true
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect.poll(() => correctionStarted).toBe(true)
  total = 21
  await page.locator('.ant-pagination-item-2').dispatchEvent('click')
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
  release()
  await page.waitForTimeout(300)
  await expect(page.getByText('Generation 10', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
  expect(new URL(page.url()).searchParams.get('generation_page')).toBe('2')
})
