import { test, expect, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

async function interceptResources(page: Page, shouldFail: () => boolean, onFailure?: () => void) {
  await page.route('**/api/resources/**', async (route) => {
    if (!shouldFail()) return route.fallback()
    onFailure?.()
    return route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'resource monitor unavailable' }),
    })
  })
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('automatic resource polling is silent and preserves the last successful data', async ({
  page,
}, testInfo) => {
  test.setTimeout(45_000)
  let failResources = false
  let failedRequests = 0
  await interceptResources(
    page,
    () => failResources,
    () => {
      failedRequests += 1
    }
  )

  await page.goto('/resources')
  await expect(page.getByText('120/256G')).toBeVisible({ timeout: 15_000 })

  failResources = true
  await expect(page.getByTestId('resource-stale-state')).toBeVisible({ timeout: 8_000 })
  await expect(page.getByText('120/256G')).toBeVisible()

  await expect.poll(() => failedRequests, { timeout: 13_000 }).toBeGreaterThanOrEqual(6)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
  await testInfo.attach('resource-stale-state', {
    body: await page.screenshot({ fullPage: false }),
    contentType: 'image/png',
  })
})

test('a slow older resource poll cannot overwrite a newer manual refresh', async ({ page }) => {
  test.setTimeout(45_000)
  let holdOldPoll = false
  let heldRequests = 0
  let releasedRequests = 0
  let newerRequests = 0
  let releaseOldPoll!: () => void
  const oldPollGate = new Promise<void>((resolve) => {
    releaseOldPoll = resolve
  })

  await page.route('**/api/resources/**', async (route) => {
    if (holdOldPoll) {
      heldRequests += 1
      await oldPollGate
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'older failed poll' }),
      })
      releasedRequests += 1
      return
    }
    if (heldRequests > 0) newerRequests += 1
    return route.fallback()
  })

  await page.goto('/resources')
  await expect(page.getByText('120/256G')).toBeVisible({ timeout: 15_000 })

  holdOldPoll = true
  await expect.poll(() => heldRequests, { timeout: 8_000 }).toBeGreaterThanOrEqual(3)
  holdOldPoll = false
  await page.getByRole('button', { name: /Refresh|刷新/ }).click()
  await expect.poll(() => newerRequests, { timeout: 5_000 }).toBeGreaterThanOrEqual(3)

  releaseOldPoll()
  await expect.poll(() => releasedRequests, { timeout: 3_000 }).toBeGreaterThanOrEqual(3)
  await page.waitForTimeout(150)

  expect(await page.getByTestId('resource-stale-state').count()).toBe(0)
  await expect(page.getByText('120/256G')).toBeVisible()
})

test('initial resource failure emits at most one local message and an inline unavailable state', async ({
  page,
}) => {
  test.setTimeout(45_000)
  await interceptResources(page, () => true)

  await page.goto('/resources')

  await expect(page.getByTestId('resource-unavailable-state')).toBeVisible({ timeout: 15_000 })
  await expect(page.locator('.ant-message-notice')).toHaveCount(1)
  await expect(page.getByText('0/0', { exact: true })).toHaveCount(0)
  await expect(page.getByText('0C/0T', { exact: true })).toHaveCount(0)
})

test('an explicit unavailable status is not downgraded to partial or fake zeroes', async ({
  page,
}) => {
  await page.route('**/api/resources/status', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        available: false,
        partial: false,
        error_code: 'resource_monitor_unavailable',
        gpu: {
          available: false,
          partial: false,
          device_state: 'unknown',
          total_gpus: null,
          allocated_gpus: null,
          free_gpus: null,
        },
        system: {
          available: false,
          partial: false,
          cpu_percent: null,
          memory_percent: null,
          memory_used_mb: null,
          memory_total_mb: null,
          disk_usage_percent: null,
          disk_used_gb: null,
          disk_total_gb: null,
          open_files: null,
          thread_count: null,
        },
      }),
    })
  )
  await page.route('**/api/resources/gpus**', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ total: 0, gpus: [] }),
    })
  )

  await page.goto('/resources')

  await expect(page.getByTestId('resource-unavailable-state')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByTestId('resource-partial-state')).toHaveCount(0)
  await expect(page.getByText('0/0', { exact: true })).toHaveCount(0)
  await expect(page.getByText('0C/0T', { exact: true })).toHaveCount(0)
})

test('an unavailable refresh preserves the previous resource snapshot atomically', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let unavailable = false
  let unavailableRequests = 0

  await page.route('**/api/resources/**', async (route) => {
    if (!unavailable) return route.fallback()
    unavailableRequests += 1
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/status')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: false,
          partial: false,
          error_code: 'resource_monitor_unavailable',
          gpu: {
            available: false,
            partial: false,
            device_state: 'unknown',
            total_gpus: null,
            allocated_gpus: null,
            free_gpus: null,
          },
          system: {
            available: false,
            partial: false,
            cpu_percent: null,
            memory_percent: null,
            memory_used_mb: null,
            memory_total_mb: null,
            disk_usage_percent: null,
            disk_used_gb: null,
            disk_total_gb: null,
            open_files: null,
            thread_count: null,
          },
        }),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ total: 0, gpus: [] }),
    })
  })

  await page.goto('/resources')
  await expect(page.getByText('NVIDIA A100')).toBeVisible({ timeout: 15_000 })

  unavailable = true
  await expect.poll(() => unavailableRequests, { timeout: 8_000 }).toBeGreaterThanOrEqual(3)
  await expect(page.getByTestId('resource-stale-state')).toBeVisible()
  await expect(page.getByText('NVIDIA A100')).toBeVisible()
  await expect(page.getByText('120/256G')).toBeVisible()
})

test('silent polling still redirects an expired session to login', async ({ page }) => {
  test.setTimeout(45_000)
  let sessionExpired = false
  await page.route('**/api/auth/me', (route) => {
    if (!sessionExpired) return route.fallback()
    return route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'session expired' }),
    })
  })
  await page.route('**/api/resources/**', (route) => {
    if (!sessionExpired) return route.fallback()
    return route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'session expired' }),
    })
  })

  await page.goto('/resources')
  await expect(page.getByText('120/256G')).toBeVisible({ timeout: 15_000 })
  sessionExpired = true

  await expect(page).toHaveURL(/\/login\?redirect=%2Fresources/, { timeout: 8_000 })
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
})
