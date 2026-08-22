import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

function buildPaginatedSyncConfigs(prefix: string) {
  return Array.from({ length: 51 }, (_, index) => {
    const number = index + 1
    return {
      task_id: `sync-${prefix.toLowerCase()}-${number}`,
      task_name: `${prefix} Task ${String(number).padStart(2, '0')}`,
      user_id: 'user-1',
      external_api_config_id: 'apicfg-page',
      external_api_url: '',
      sync_interval_seconds: 300,
      last_sync_at: null,
      generation_threshold: 100,
      generation_mode: 'doc_to_training',
      pending_record_count: 0,
      total_record_count: 0,
      training_threshold: 100,
      pending_training_samples: 0,
      total_training_samples: 0,
      total_trainings: 0,
      current_adapter_name: null,
      current_training_id: null,
      is_active: false,
      status: 'idle',
      error_message: null,
      created_at: '2026-08-16T00:00:00',
      updated_at: '2026-08-16T00:00:00',
    }
  })
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 一、SyncConfigList 页面（/sync）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test.describe('SyncConfigList Page', () => {
  test('1.1 renders statistics cards', async ({ page }) => {
    await page.goto('/sync')
    await expect(page.getByText('总任务数')).toBeVisible()
    await expect(page.getByText('运行中').first()).toBeVisible()
    await expect(page.getByText('同步中').first()).toBeVisible()
    await expect(page.getByText('异常').first()).toBeVisible()
  })

  test('1.2 statistics card values are correct', async ({ page }) => {
    await page.goto('/sync')
    // Total: 2 configs in mock
    const totalCard = page.locator('div').filter({ hasText: /^总任务数$/ }).locator('..')
    await expect(totalCard).toBeVisible()
    // Active: sync-001 is_active=true => 1
    // Syncing: sync-001 status=syncing => 1
    // Error: 0
  })

  test('1.3 renders table column headers', async ({ page }) => {
    await page.goto('/sync')
    // Use thead to avoid Ant Design's hidden measure <th> in tbody
    const thead = page.locator('thead')
    await expect(thead.locator('th:has-text("任务 ID")')).toBeVisible()
    await expect(thead.locator('th:has-text("任务名称")')).toBeVisible()
    await expect(thead.locator('th:has-text("状态")')).toBeVisible()
    await expect(thead.locator('th:has-text("数据进度")')).toBeVisible()
    await expect(thead.locator('th:has-text("训练进度")')).toBeVisible()
    await expect(thead.locator('th:has-text("当前 Adapter")')).toBeVisible()
    await expect(thead.locator('th:has-text("上次同步")')).toBeVisible()
    // "操作" column is fixed right, may have duplicate sticky element
    await expect(thead.locator('th:has-text("操作")').first()).toBeVisible()
  })

  test('1.4 displays two sync configs', async ({ page }) => {
    await page.goto('/sync')
    await expect(page.getByText('自动同步-业务A')).toBeVisible()
    await expect(page.getByText('手动同步-测试B')).toBeVisible()
  })

  test('1.5 data progress column shows correct values', async ({ page }) => {
    await page.goto('/sync')
    // sync-001: pending_record_count=120, generation_threshold=500
    await expect(page.getByText('120/500')).toBeVisible()
  })

  test('1.6 training progress column shows correct values', async ({ page }) => {
    await page.goto('/sync')
    // sync-001 has completed two rounds with 800 total training samples
    await expect(page.getByText('已训练 800 条')).toBeVisible()
    await expect(page.getByText('第 2 轮')).toBeVisible()
  })

  test('1.7 adapter tag displayed, dash for no adapter', async ({ page }) => {
    await page.goto('/sync')
    // sync-001 has adapter
    await expect(page.getByText('sync-adapter-r2')).toBeVisible()
    // sync-002 has no adapter, shows "-"
    const rows = page.locator('tr')
    const row2 = rows.filter({ hasText: '手动同步-测试B' })
    await expect(row2.getByText('-').first()).toBeVisible()
  })

  test('1.8 active config shows pause, inactive shows play', async ({ page }) => {
    await page.goto('/sync')
    // sync-001 is_active=true => pause button
    const row1 = page.locator('tr').filter({ hasText: '自动同步-业务A' })
    await expect(row1.locator('[aria-label="pause-circle"]')).toBeVisible()

    // sync-002 is_active=false => play button
    const row2 = page.locator('tr').filter({ hasText: '手动同步-测试B' })
    await expect(row2.locator('[aria-label="play-circle"]')).toBeVisible()
  })

  test('1.9 delete shows confirmation popover', async ({ page }) => {
    await page.goto('/sync')
    // Click delete on first config
    const row = page.locator('tr').filter({ hasText: '自动同步-业务A' })
    const deleteBtn = row.locator('[aria-label="delete"]')
    await deleteBtn.click()
    // Popconfirm should appear
    await expect(page.getByText('确定删除此同步任务？')).toBeVisible()
  })

  test('1.10 create button navigates to /sync/create', async ({ page }) => {
    await page.goto('/sync')
    await page.getByRole('button', { name: '创建同步任务' }).click()
    await expect(page).toHaveURL(/\/sync\/create/)
  })

  test('1.11 config name link navigates to detail page', async ({ page }) => {
    await page.goto('/sync')
    await page.getByText('自动同步-业务A').click()
    await expect(page).toHaveURL(/\/sync\/sync-001/)
  })

  test('1.12 refresh button reloads list', async ({ page }) => {
    await page.goto('/sync')
    await expect(page.locator('h4').filter({ hasText: '数据同步' })).toBeVisible()
    const refreshBtn = page.getByRole('button', { name: '刷新' })
    await expect(refreshBtn).toBeVisible()
    await refreshBtn.click()
    // Table still visible after refresh
    await expect(page.locator('table').first()).toBeVisible()
  })

  test('1.13 pagination keeps page 2 while polling 51 sync configs', async ({ page }) => {
    const configs = Array.from({ length: 51 }, (_, index) => {
      const number = index + 1
      return {
        task_id: `sync-page-${number}`,
        task_name: `分页任务 ${String(number).padStart(2, '0')}`,
        user_id: 'user-1',
        external_api_config_id: 'apicfg-page',
        external_api_url: '',
        sync_interval_seconds: 300,
        last_sync_at: null,
        generation_threshold: 100,
        generation_mode: 'doc_to_training',
        pending_record_count: 0,
        total_record_count: 0,
        training_threshold: 100,
        pending_training_samples: 0,
        total_training_samples: 0,
        total_trainings: 0,
        current_adapter_name: null,
        current_training_id: null,
        is_active: true,
        status: 'syncing',
        error_message: null,
        created_at: '2026-08-16T00:00:00',
        updated_at: '2026-08-16T00:00:00',
      }
    })
    const listRequests: Array<{ limit: string | null; offset: string | null }> = []

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const limit = url.searchParams.get('limit')
      const offset = url.searchParams.get('offset')
      listRequests.push({ limit, offset })
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tasks: offset === '50' ? configs.slice(50) : configs.slice(0, 50),
          total: 51,
        }),
      })
    })

    const firstRequestPromise = page.waitForRequest((request) => {
      const url = new URL(request.url())
      return request.method() === 'GET' && url.pathname === '/api/sync/tasks'
    })
    await page.goto('/sync')
    const firstRequestUrl = new URL((await firstRequestPromise).url())

    expect(firstRequestUrl.searchParams.get('limit')).toBe('50')
    expect(firstRequestUrl.searchParams.get('offset')).toBe('0')
    await expect(page.getByText('分页任务 01', { exact: true })).toBeVisible()
    await expect(page.getByText('分页任务 50', { exact: true })).toBeVisible()
    await expect(page.getByText('分页任务 51', { exact: true })).toHaveCount(0)

    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByText('分页任务 51', { exact: true })).toBeVisible()
    await expect(page.getByText('分页任务 01', { exact: true })).toHaveCount(0)
    await expect(page.locator('.ant-table-tbody > tr.ant-table-row')).toHaveCount(1)

    await expect.poll(
      () => listRequests.filter(({ offset }) => offset === '50').length,
      { timeout: 7000 }
    ).toBeGreaterThanOrEqual(2)
    expect(listRequests.at(-1)).toEqual({ limit: '50', offset: '50' })
    await expect(page.locator('.ant-pagination-item-2')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('分页任务 51', { exact: true })).toBeVisible()
  })

  test('1.14 pagination ignores a stale page 1 refresh response', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Race')
    let delayNextPageOneRequest = false
    let markStaleRequestStarted = () => {}
    const staleRequestStarted = new Promise<void>((resolve) => {
      markStaleRequestStarted = resolve
    })
    let releaseStaleResponse = () => {}
    const staleResponseGate = new Promise<void>((resolve) => {
      releaseStaleResponse = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset')
      if (offset === '0') {
        if (delayNextPageOneRequest) {
          delayNextPageOneRequest = false
          markStaleRequestStarted()
          await staleResponseGate
        }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ tasks: configs.slice(0, 50), total: 51 }),
        })
        return
      }

      expect(offset).toBe('50')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tasks: configs.slice(50), total: 51 }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Race Task 01', { exact: true })).toBeVisible()

    delayNextPageOneRequest = true
    await page.getByRole('button', { name: '刷新' }).click()
    await staleRequestStarted
    await page.locator('.ant-pagination-item-2').dispatchEvent('click')
    await expect(page.getByText('Race Task 51', { exact: true })).toBeVisible()

    const staleResponseFinished = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return url.pathname === '/api/sync/tasks' && url.searchParams.get('offset') === '0'
    })
    releaseStaleResponse()
    await staleResponseFinished
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    await expect(page.locator('.ant-pagination-item-2')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Race Task 51', { exact: true })).toBeVisible()
    await expect(page.getByText('Race Task 01', { exact: true })).toHaveCount(0)
    await expect(page.locator('.ant-table-tbody > tr.ant-table-row')).toHaveCount(1)
  })

  test('1.15 pagination returns to page 1 when total shrinks below page 2', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Shrink')
    const requestOffsets: string[] = []
    let pageTwoRequestCount = 0
    let totalShrank = false
    let markShrinkResponseFinished = () => {}
    const shrinkResponseFinished = new Promise<void>((resolve) => {
      markShrinkResponseFinished = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset') || ''
      requestOffsets.push(offset)
      if (offset === '50') pageTwoRequestCount += 1
      if (offset === '50' && pageTwoRequestCount === 2) totalShrank = true
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tasks: offset === '50'
            ? (totalShrank ? [] : configs.slice(50))
            : configs.slice(0, 50),
          total: totalShrank ? 50 : 51,
        }),
      })
      if (offset === '50' && pageTwoRequestCount === 2) markShrinkResponseFinished()
    })

    await page.goto('/sync')
    await expect(page.getByText('Shrink Task 01', { exact: true })).toBeVisible()
    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByText('Shrink Task 51', { exact: true })).toBeVisible()
    const pageOneRequestsBeforeShrink = requestOffsets.filter(offset => offset === '0').length

    await page.getByRole('button', { name: '刷新' }).click()
    await shrinkResponseFinished
    await expect.poll(
      () => requestOffsets.filter(offset => offset === '0').length,
      { timeout: 3000 }
    ).toBeGreaterThan(pageOneRequestsBeforeShrink)

    await expect(page.locator('.ant-pagination-item-1')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Shrink Task 01', { exact: true })).toBeVisible()
    await expect(page.getByText('Shrink Task 50', { exact: true })).toBeVisible()
    await expect(page.getByText('Shrink Task 51', { exact: true })).toHaveCount(0)
  })

  test('1.16 deleting the last pagination row returns to the previous page', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Delete')
    const requestOffsets: string[] = []
    let deleted = false

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() === 'DELETE' && url.pathname === `/api/sync/tasks/${configs[50].task_id}`) {
        deleted = true
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Sync task deleted' }),
        })
        return
      }
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset') || ''
      requestOffsets.push(offset)
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tasks: offset === '50' ? (deleted ? [] : configs.slice(50)) : configs.slice(0, 50),
          total: deleted ? 50 : 51,
        }),
      })
    })

    await page.goto('/sync')
    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByText('Delete Task 51', { exact: true })).toBeVisible()
    const pageOneRequestsBeforeDelete = requestOffsets.filter(offset => offset === '0').length

    const row = page.locator('tr').filter({ hasText: 'Delete Task 51' })
    await row.locator('[aria-label="delete"]').click()
    const popconfirm = page.locator('.ant-popconfirm')
    await expect(popconfirm).toBeVisible()
    await popconfirm.getByRole('button', { name: '确 定' }).click()

    await expect.poll(
      () => requestOffsets.filter(offset => offset === '0').length,
      { timeout: 3000 }
    ).toBeGreaterThan(pageOneRequestsBeforeDelete)
    await expect(page.locator('.ant-pagination-item-1')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Delete Task 01', { exact: true })).toBeVisible()
    await expect(page.getByText('Delete Task 51', { exact: true })).toHaveCount(0)
  })

  test('1.17 pagination survives a delayed action refresh from the old page', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Action')
    let startRequestCount = 0
    let releaseStartResponse = () => {}
    const startResponseGate = new Promise<void>((resolve) => {
      releaseStartResponse = resolve
    })
    let startResponseReleased = false
    const postActionRefreshOffsets: Array<string | null> = []

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() === 'POST' && url.pathname.endsWith('/start')) {
        expect(url.pathname).toBe(`/api/sync/tasks/${configs[0].task_id}/start`)
        startRequestCount += 1
        await startResponseGate
        startResponseReleased = true
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Sync started', worker_status: 'running' }),
        })
        return
      }
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          tasks: offset === '50' ? configs.slice(50) : configs.slice(0, 50),
          total: 51,
        }),
      })
      if (startResponseReleased) postActionRefreshOffsets.push(offset)
    })

    await page.goto('/sync')
    await expect(page.getByText('Action Task 01', { exact: true })).toBeVisible()

    const firstRow = page.locator('tr').filter({ hasText: 'Action Task 01' })
    await firstRow.locator('[aria-label="play-circle"]').click()
    const popconfirm = page.locator('.ant-popconfirm')
    await expect(popconfirm).toBeVisible()
    await popconfirm.getByRole('button', { name: '确 定' }).click()
    await expect.poll(() => startRequestCount, { timeout: 3000 }).toBe(1)

    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByText('Action Task 51', { exact: true })).toBeVisible()
    releaseStartResponse()
    await expect.poll(
      () => postActionRefreshOffsets.length,
      { timeout: 3000 }
    ).toBeGreaterThan(0)
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    await expect(page.locator('.ant-pagination-item-2')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Action Task 51', { exact: true })).toBeVisible()
    await expect(page.getByText('Action Task 01', { exact: true })).toHaveCount(0)
  })

  test('1.18 pagination rolls back when the requested page fails', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Failed Page')
    let markPageTwoFailureFinished = () => {}
    const pageTwoFailureFinished = new Promise<void>((resolve) => {
      markPageTwoFailureFinished = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset')
      if (offset === '50') {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'page 2 unavailable' }),
        })
        markPageTwoFailureFinished()
        return
      }

      expect(offset).toBe('0')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tasks: configs.slice(0, 50), total: 51 }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Failed Page Task 01', { exact: true })).toBeVisible()

    await page.locator('.ant-pagination-item-2').click()
    await pageTwoFailureFinished
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    await expect(page.locator('.ant-pagination-item-1')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Failed Page Task 01', { exact: true })).toBeVisible()
    await expect(page.getByText('Failed Page Task 51', { exact: true })).toHaveCount(0)
    await expect(page.locator('.ant-table-tbody > tr.ant-table-row')).toHaveCount(50)
  })

  test('1.19 pagination clears an invalid last page before a failed correction', async ({ page }) => {
    const configs = buildPaginatedSyncConfigs('Failed Correction')
    let shrinkNextPageTwoResponse = false
    let failCorrectionPageOneRequest = false
    let markShrinkResponseFinished = () => {}
    const shrinkResponseFinished = new Promise<void>((resolve) => {
      markShrinkResponseFinished = resolve
    })
    let markCorrectionFailureFinished = () => {}
    const correctionFailureFinished = new Promise<void>((resolve) => {
      markCorrectionFailureFinished = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset')
      if (offset === '50') {
        if (shrinkNextPageTwoResponse) {
          shrinkNextPageTwoResponse = false
          failCorrectionPageOneRequest = true
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ tasks: [], total: 50 }),
          })
          markShrinkResponseFinished()
          return
        }

        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ tasks: configs.slice(50), total: 51 }),
        })
        return
      }

      expect(offset).toBe('0')
      if (failCorrectionPageOneRequest) {
        failCorrectionPageOneRequest = false
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'page 1 correction unavailable' }),
        })
        markCorrectionFailureFinished()
        return
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tasks: configs.slice(0, 50), total: 51 }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Failed Correction Task 01', { exact: true })).toBeVisible()
    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByText('Failed Correction Task 51', { exact: true })).toBeVisible()

    shrinkNextPageTwoResponse = true
    await page.getByRole('button', { name: '刷新' }).click()
    await shrinkResponseFinished
    await correctionFailureFinished
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    await expect(page.locator('.ant-pagination-item-1')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Failed Correction Task 51', { exact: true })).toHaveCount(0)
    await expect(page.locator('.ant-table-tbody > tr.ant-table-row')).toHaveCount(0)
  })

  test('1.20 pagination polling is silent and does not show loading', async ({ page }) => {
    const [config] = buildPaginatedSyncConfigs('Silent Poll')
    config.is_active = true
    config.status = 'syncing'
    const settledConfig = {
      ...config,
      task_name: 'Silent Poll Settled Task 01',
      is_active: false,
      status: 'idle',
    }
    let pollArmed = false
    let settleNextRequest = false
    let markPollStarted = () => {}
    const pollStarted = new Promise<void>((resolve) => {
      markPollStarted = resolve
    })
    let releasePollResponse = () => {}
    const pollResponseGate = new Promise<void>((resolve) => {
      releasePollResponse = resolve
    })
    let markPollResponseFinished = () => {}
    const pollResponseFinished = new Promise<void>((resolve) => {
      markPollResponseFinished = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      expect(url.searchParams.get('offset')).toBe('0')
      if (pollArmed) {
        pollArmed = false
        markPollStarted()
        await pollResponseGate
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'poll unavailable' }),
        })
        markPollResponseFinished()
        return
      }

      const responseConfig = settleNextRequest ? settledConfig : config
      settleNextRequest = false
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tasks: [responseConfig], total: 1 }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Silent Poll Task 01', { exact: true })).toBeVisible()
    pollArmed = true
    await pollStarted
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))
    await page.evaluate(() => {
      const trackingWindow = window as typeof window & {
        __syncPollingNoticeSeen?: boolean
        __syncPollingNoticeObserver?: MutationObserver
      }
      trackingWindow.__syncPollingNoticeSeen = false
      const observer = new MutationObserver(() => {
        if (document.querySelector('.ant-message-notice')) {
          trackingWindow.__syncPollingNoticeSeen = true
        }
      })
      observer.observe(document.body, { childList: true, subtree: true })
      trackingWindow.__syncPollingNoticeObserver = observer
    })
    const failedPollResponse = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return response.status() === 500 &&
        response.request().method() === 'GET' &&
        url.pathname === '/api/sync/tasks'
    })

    try {
      await expect(page.getByRole('button', { name: '刷新' })).not.toHaveClass(/ant-btn-loading/)
      await expect(page.locator('.ant-spin-spinning')).toHaveCount(0)
    } finally {
      releasePollResponse()
    }

    await failedPollResponse
    await pollResponseFinished
    settleNextRequest = true
    await page.getByRole('button', { name: '刷新' }).click()
    await expect(page.getByText('Silent Poll Settled Task 01', { exact: true })).toBeVisible()
    const noticeSeen = await page.evaluate(() => {
      const trackingWindow = window as typeof window & {
        __syncPollingNoticeSeen?: boolean
        __syncPollingNoticeObserver?: MutationObserver
      }
      trackingWindow.__syncPollingNoticeObserver?.disconnect()
      return trackingWindow.__syncPollingNoticeSeen
    })
    expect(noticeSeen).toBe(false)
    await expect(page.locator('.ant-message-notice')).toHaveCount(0)
  })

  test('1.21 pagination action does not refresh after the list unmounts', async ({ page }) => {
    const [config] = buildPaginatedSyncConfigs('Unmount')
    let markStartRequestStarted = () => {}
    const startRequestStarted = new Promise<void>((resolve) => {
      markStartRequestStarted = resolve
    })
    let releaseStartResponse = () => {}
    const startResponseGate = new Promise<void>((resolve) => {
      releaseStartResponse = resolve
    })
    let startResponseReleased = false
    const postUnmountListRequests: string[] = []

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (
        request.method() === 'POST' &&
        url.pathname === `/api/sync/tasks/${config.task_id}/start`
      ) {
        markStartRequestStarted()
        await startResponseGate
        startResponseReleased = true
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'Sync started', worker_status: 'running' }),
        })
        return
      }
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      if (startResponseReleased) postUnmountListRequests.push(request.url())
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ tasks: [config], total: 1 }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Unmount Task 01', { exact: true })).toBeVisible()
    const row = page.locator('tr').filter({ hasText: 'Unmount Task 01' })
    await row.locator('[aria-label="play-circle"]').click()
    const popconfirm = page.locator('.ant-popconfirm')
    await expect(popconfirm).toBeVisible()
    await popconfirm.getByRole('button', { name: '确 定' }).click()
    await startRequestStarted

    await page.getByText('Unmount Task 01', { exact: true }).click()
    await expect(page).toHaveURL(new RegExp(`/sync/${config.task_id}$`))
    await expect(row).toHaveCount(0)
    const completedStartResponse = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return response.request().method() === 'POST' &&
        url.pathname === `/api/sync/tasks/${config.task_id}/start`
    })
    releaseStartResponse()
    await completedStartResponse
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    expect(postUnmountListRequests).toEqual([])
    await expect(page.locator('.ant-message-notice')).toHaveCount(0)
  })

  test('1.22 pagination does not let polling supersede an in-flight foreground page', async ({
    page,
  }) => {
    const configs = buildPaginatedSyncConfigs('Foreground')
    configs[0].is_active = true
    configs[0].status = 'syncing'
    let pageTwoRequestCount = 0
    let markForegroundRequestStarted = () => {}
    const foregroundRequestStarted = new Promise<void>((resolve) => {
      markForegroundRequestStarted = resolve
    })
    let releaseForegroundResponse = () => {}
    const foregroundResponseGate = new Promise<void>((resolve) => {
      releaseForegroundResponse = resolve
    })

    await page.route('**/api/sync/tasks**', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (request.method() !== 'GET' || url.pathname !== '/api/sync/tasks') {
        await route.fallback()
        return
      }

      const offset = url.searchParams.get('offset')
      if (offset === '0') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ tasks: configs.slice(0, 50), total: 51 }),
        })
        return
      }

      expect(offset).toBe('50')
      pageTwoRequestCount += 1
      if (pageTwoRequestCount === 1) {
        markForegroundRequestStarted()
        await foregroundResponseGate
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ tasks: configs.slice(50), total: 51 }),
        })
        return
      }

      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'poll must not supersede foreground work' }),
      })
    })
    await page.route('**/api/sync/pagination-sentinel', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      expect(request.method()).toBe('GET')
      expect(url.pathname).toBe('/api/sync/pagination-sentinel')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      })
    })

    await page.goto('/sync')
    await expect(page.getByText('Foreground Task 01', { exact: true })).toBeVisible()
    await page.evaluate(() => new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
    }))

    await page.locator('.ant-pagination-item-2').click()
    await foregroundRequestStarted
    await page.evaluate(async () => {
      Object.defineProperty(document, 'hidden', { configurable: true, get: () => true })
      document.dispatchEvent(new Event('visibilitychange'))
      Object.defineProperty(document, 'hidden', { configurable: true, get: () => false })
      document.dispatchEvent(new Event('visibilitychange'))
      const response = await fetch('/api/sync/pagination-sentinel')
      if (!response.ok) throw new Error(`sentinel failed with ${response.status}`)
    })

    const completedForegroundResponse = page.waitForResponse((response) => {
      const url = new URL(response.url())
      return response.status() === 200 &&
        response.request().method() === 'GET' &&
        url.pathname === '/api/sync/tasks' &&
        url.searchParams.get('offset') === '50'
    })
    try {
      expect(pageTwoRequestCount).toBe(1)
    } finally {
      releaseForegroundResponse()
    }

    await completedForegroundResponse
    await expect(page.locator('.ant-pagination-item-2')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByText('Foreground Task 51', { exact: true })).toBeVisible()
    await expect(page.getByText('Foreground Task 01', { exact: true })).toHaveCount(0)
  })
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 二、ExternalApiConfigList Tab
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test.describe('ExternalApiConfigList Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/sync')
    await page.getByText('外部 API').click()
  })

  test('2.1 tab switch shows API config table', async ({ page }) => {
    // Wait for API config data to appear after tab switch
    await expect(page.getByText('业务系统A')).toBeVisible()
    await expect(page.locator('.ant-tabs-tabpane-active table')).toBeVisible()
  })

  test('2.2 table column headers correct', async ({ page }) => {
    const activePane = page.locator('.ant-tabs-tabpane-active')
    await expect(activePane.locator('th:has-text("配置名称")')).toBeVisible()
    await expect(activePane.locator('th:has-text("API 地址")')).toBeVisible()
    await expect(activePane.locator('th:has-text("描述")')).toBeVisible()
    await expect(activePane.locator('th:has-text("状态")')).toBeVisible()
    await expect(activePane.locator('th:has-text("创建时间")')).toBeVisible()
    await expect(activePane.locator('th:has-text("操作")')).toBeVisible()
  })

  test('2.3 displays two API configs', async ({ page }) => {
    await expect(page.getByText('业务系统A')).toBeVisible()
    await expect(page.getByText('测试系统B')).toBeVisible()
    await expect(page.getByText('http://business-system.example.com:9003/api/embedding_texts')).toBeVisible()
  })

  test('2.4 create button opens modal', async ({ page }) => {
    const createBtn = page.getByRole('button', { name: /创建 API 配置/ })
    await createBtn.click()
    await expect(page.locator('.ant-modal')).toBeVisible()
  })

  test('2.5 create modal contains form fields', async ({ page }) => {
    await page.getByRole('button', { name: /创建 API 配置/ }).click()
    const modal = page.locator('.ant-modal')
    await expect(modal.getByText('配置名称')).toBeVisible()
    await expect(modal.getByText('数据接口地址')).toBeVisible()
    await expect(modal.getByText('API Token')).toBeVisible()
    await expect(modal.getByText('描述')).toBeVisible()
  })

  test('2.6 delete API config shows confirmation', async ({ page }) => {
    const row = page.locator('tr').filter({ hasText: '业务系统A' })
    // Delete button is icon-only (DeleteOutlined), use aria-label to locate
    const deleteBtn = row.locator('[aria-label="delete"]')
    await deleteBtn.click()
    await expect(page.getByText('确定删除此 API 配置？')).toBeVisible()
  })
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 三、SyncConfigCreate 页面（/sync/create）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test.describe('SyncConfigCreate Page', () => {
  test('3.1 renders create page structure', async ({ page }) => {
    await page.goto('/sync/create')
    await expect(page.getByText('创建同步任务').first()).toBeVisible()
    await expect(page.locator('.ant-card').first()).toBeVisible()
    // Has submit button
    await expect(page.getByRole('button', { name: '创建配置' })).toBeVisible()
  })

  test('3.2 API config dropdown loads options', async ({ page }) => {
    await page.goto('/sync/create')
    const select = page.locator('.ant-select').first()
    await select.click()
    await expect(page.getByText('业务系统A').first()).toBeVisible()
    await expect(page.getByText('测试系统B').first()).toBeVisible()
  })

  test('3.3 selecting API config shows URL info', async ({ page }) => {
    await page.goto('/sync/create')
    const select = page.locator('.ant-select').first()
    await select.click()
    await page.getByText('业务系统A').first().click()
    await expect(page.getByText('http://business-system.example.com:9003/api/embedding_texts')).toBeVisible()
  })

  test('3.4 config name is required', async ({ page }) => {
    await page.goto('/sync/create')
    // Select API config first to avoid that error
    const select = page.locator('.ant-select').first()
    await select.click()
    await page.getByText('业务系统A').first().click()
    // Click submit without filling config name
    await page.getByRole('button', { name: '创建配置' }).click()
    // Should show validation error
    await expect(page.locator('.ant-form-item-explain-error').first()).toBeVisible()
  })

  test('3.5 sync interval defaults to 300', async ({ page }) => {
    await page.goto('/sync/create')
    const intervalInput = page.locator('.ant-form-item').filter({ hasText: '同步间隔' }).locator('input')
    await expect(intervalInput).toHaveValue('300')
  })

  test('3.6 Level 1 generation switch shows config fields', async ({ page }) => {
    await page.goto('/sync/create')
    // Initially disabled — shows hint text
    await expect(page.getByText('开启开关以配置此功能').first()).toBeVisible()
    // Click generation switch
    const genCard = page.locator('.ant-card').filter({ hasText: 'Level 1' })
    await genCard.locator('.ant-switch').click()
    // Now should show generation threshold and mode
    await expect(page.getByText('生成阈值').first()).toBeVisible()
    await expect(page.getByText('生成模式').first()).toBeVisible()
  })

  test('3.7 Level 2 training switch depends on Level 1', async ({ page }) => {
    await page.goto('/sync/create')
    // Turn on training switch
    const trainCard = page.locator('.ant-card').filter({ hasText: 'Level 2' })
    await trainCard.locator('.ant-switch').click()
    // Level 1 should also be enabled
    const genCard = page.locator('.ant-card').filter({ hasText: 'Level 1' })
    const genSwitch = genCard.locator('.ant-switch')
    await expect(genSwitch).toHaveClass(/ant-switch-checked/)
    // Training fields should be visible
    await expect(page.getByText('训练阈值').first()).toBeVisible()
    await expect(page.getByText('LoRA Rank').first()).toBeVisible()
  })

  test('3.8 full form submission sends POST request', async ({ page }) => {
    let capturedBody: Record<string, unknown> | null = null
    await page.route('**/api/sync/tasks', async (route) => {
      if (route.request().method() === 'POST') {
        capturedBody = route.request().postDataJSON()
        await route.fulfill({
          status: 201,
          body: JSON.stringify({
            message: 'Sync task created',
            task: { task_id: 'sync-new', task_name: capturedBody?.task_name || 'Test' },
          }),
        })
      } else {
        await route.continue()
      }
    })

    await page.goto('/sync/create')
    // Select API config
    const select = page.locator('.ant-select').first()
    await select.click()
    await page.getByText('业务系统A').first().click()
    // Fill task name
    const nameInput = page.locator('.ant-form-item').filter({ hasText: '任务名称' }).locator('input')
    await nameInput.fill('测试同步任务')
    // Submit
    await page.getByRole('button', { name: '创建配置' }).click()
    // Should navigate back to list
    await page.waitForURL(/\/sync$/)
    expect(capturedBody).not.toBeNull()
    expect(capturedBody!.task_name).toBe('测试同步任务')
  })
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 四、SyncConfigDetail 页面（/sync/sync-001）
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test.describe('SyncConfigDetail Page', () => {
  test('4.1 renders header with config name and status', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await expect(page.getByText('自动同步-业务A')).toBeVisible()
    // Back button
    await expect(page.getByRole('button', { name: '返回' })).toBeVisible()
  })

  test('4.2 counter cards show pending records and training samples', async ({ page }) => {
    await page.goto('/sync/sync-001')
    // pending_record_count / generation_threshold = 120 / 500
    await expect(page.getByText('120 / 500')).toBeVisible()
    // pending_training_samples / training_threshold = 50 / 1000
    await expect(page.getByText('50 / 1000')).toBeVisible()
  })

  test('4.3 history totals cards correct', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await expect(page.getByText('历史总记录')).toBeVisible()
    await expect(page.getByText('2,500').or(page.getByText('2500'))).toBeVisible()
    await expect(page.getByText('历史总训练样本')).toBeVisible()
    await expect(page.getByText('800')).toBeVisible()
    await expect(page.getByText('历史训练次数')).toBeVisible()
  })

  test('4.4 overview shows config info', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await expect(page.getByText('概览')).toBeVisible()
    await expect(page.getByText('http://business-system.example.com:9003/api/embedding_texts')).toBeVisible()
    await expect(page.getByText('300s')).toBeVisible()
    await expect(page.getByText('500').first()).toBeVisible()
    await expect(page.getByText('1000').first()).toBeVisible()
  })

  test('4.5 overview shows current adapter', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await expect(page.getByText('sync-adapter-r2')).toBeVisible()
  })

  test('4.6 action buttons displayed correctly', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await expect(page.getByRole('button', { name: '立即同步' })).toBeVisible()
    await expect(page.getByRole('button', { name: '手动生成' })).toBeVisible()
    await expect(page.getByRole('button', { name: '手动训练' })).toBeVisible()
    // is_active=true, so stop button
    await expect(page.getByRole('button', { name: '停止' })).toBeVisible()
  })

  test('4.7 sync now button calls API', async ({ page }) => {
    let called = false
    await page.route('**/api/sync/tasks/sync-001/sync-now', async (route) => {
      called = true
      await route.fulfill({ body: JSON.stringify({ message: 'Sync cycle completed' }) })
    })
    await page.goto('/sync/sync-001')
    await page.getByRole('button', { name: '立即同步' }).click()
    expect(called).toBe(true)
  })

  test('4.8 trigger generation button calls API', async ({ page }) => {
    let called = false
    await page.route('**/api/sync/tasks/sync-001/trigger-generation', async (route) => {
      called = true
      await route.fulfill({ body: JSON.stringify({ message: 'trigger-generation completed' }) })
    })
    await page.goto('/sync/sync-001')
    await page.getByRole('button', { name: '手动生成' }).click()
    expect(called).toBe(true)
  })

  test('4.9 trigger training button calls API', async ({ page }) => {
    let called = false
    await page.route('**/api/sync/tasks/sync-001/trigger-training', async (route) => {
      called = true
      await route.fulfill({ body: JSON.stringify({ message: 'trigger-training completed' }) })
    })
    await page.goto('/sync/sync-001')
    await page.getByRole('button', { name: '手动训练' }).click()
    expect(called).toBe(true)
  })

  test('4.10 stop button shows confirmation then calls API', async ({ page }) => {
    let called = false
    await page.route('**/api/sync/tasks/sync-001/stop', async (route) => {
      called = true
      await route.fulfill({ body: JSON.stringify({ message: 'Sync stopped', worker_status: 'not_found' }) })
    })
    await page.goto('/sync/sync-001')
    await page.getByRole('button', { name: '停止' }).click()
    // Popconfirm
    await expect(page.getByText('确定停止同步？')).toBeVisible()
    // Confirm
    await page.getByRole('button', { name: '确 定' }).click()
    expect(called).toBe(true)
  })

  test('4.11 batches tab shows batch data', async ({ page }) => {
    await page.goto('/sync/sync-001')
    // Default tab is batches
    await expect(page.getByText('同步批次')).toBeVisible()
    // batch-001: 85 records
    await expect(page.getByText('85')).toBeVisible()
    // batch-002: 35 records
    await expect(page.getByText('35')).toBeVisible()
  })

  test('4.12 generations tab shows generation data', async ({ page }) => {
    await page.goto('/sync/sync-001')
    // Click generations tab
    await page.getByText('生成任务').click()
    // gen-task-001: input_record_count=85, output_sample_count=420
    await expect(page.getByText('420')).toBeVisible()
  })

  test('4.13 trainings tab shows training data', async ({ page }) => {
    await page.goto('/sync/sync-001')
    // Click trainings tab
    await page.getByText('训练任务').click()
    // Round tag R2 — use exact match to avoid matching "sync-adapter-r2" etc.
    await expect(page.getByText('R2', { exact: true })).toBeVisible()
    // Adapter name
    await expect(page.getByText('sync-adapter-r2').first()).toBeVisible()
  })

  test('4.14 back button navigates to list', async ({ page }) => {
    await page.goto('/sync/sync-001')
    await page.getByRole('button', { name: '返回' }).click()
    await expect(page).toHaveURL(/\/sync$/)
  })
})
