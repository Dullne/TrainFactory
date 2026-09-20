import { test, expect, type Page, type Route } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

const runtimeErrors = new WeakMap<Page, string[]>()
test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  runtimeErrors.set(page, errors)
  page.on('pageerror', error => errors.push(error.message))
  page.on('console', message => {
    // Deliberate 503 responses exercise retry behavior, not application crashes.
    // Existing static AntD toasts also warn about the application's dynamic theme.
    const expected = message.text().includes('503 (Service Unavailable)') || message.text() ===
      "Warning: [antd: message] Static function can not consume context like dynamic theme. Please use 'App' component instead."
    if (message.type() === 'error' && !expected) {
      errors.push(message.text())
    }
  })
})
test.afterEach(async ({ page }) => {
  expect(runtimeErrors.get(page)).toEqual([])
})

const screens = [
  {
    name: 'training',
    path: '/training',
    endpoint: 'train',
    key: 'tasks',
    filter: 'Filter by Status',
    option: 'Failed',
  },
  {
    name: 'models',
    path: '/models',
    endpoint: 'models',
    key: 'models',
    filter: 'Status Filter',
    option: 'Archived',
  },
  {
    name: 'datasets',
    path: '/datasets',
    endpoint: 'datasets',
    key: 'datasets',
    filter: 'Status',
    option: 'Archived',
  },
  {
    name: 'deployments',
    path: '/deployments',
    endpoint: 'deployments',
    key: 'deployments',
    filter: 'Filter by Status',
    option: 'Failed',
  },
] as const
type Screen = (typeof screens)[number]

function item(screen: Screen, index: number, revision = 'original') {
  const name = `${screen.name}-${revision}-${index}`
  const common = { created_at: '2026-09-19T00:00:00', updated_at: '2026-09-19T00:00:00' }
  switch (screen.name) {
    case 'training':
      return {
        ...common,
        task_id: `task-${index}`,
        task_name: name,
        model_type: 'embedding',
        training_method: 'sft',
        base_model_path: '/models/base',
        train_dataset_path: '/datasets/train.jsonl',
        output_dir: '/output',
        status: 'succeeded',
        progress: 100,
        gpu_ids: [],
      }
    case 'models':
      return {
        ...common,
        model_id: `model-${index}`,
        model_name: name,
        model_type: 'embedding',
        model_path: `/models/${index}`,
        source_type: 'downloaded',
        status: 'available',
        tags: [],
        metrics: {},
      }
    case 'datasets':
      return {
        ...common,
        dataset_id: `dataset-${index}`,
        dataset_name: name,
        dataset_type: 'embedding_pair',
        usage: 'train',
        model_type: ['embedding'],
        source_type: 'local',
        storage_path: '/datasets/train.jsonl',
        file_format: 'jsonl',
        file_size: 100,
        num_rows: 10,
        columns: [],
        status: 'ready',
      }
    case 'deployments':
      return {
        ...common,
        deployment_id: `deployment-${index}`,
        deployment_name: name,
        model_id: 'model-001',
        model_uid: `model-${index}`,
        xinference_endpoint: 'http://example.invalid:9997',
        replica: 1,
        gpu_memory_utilization: 0.8,
        deploy_mode: 'container',
        inference_framework: 'xinference',
        enable_lora: false,
        status: 'stopped',
      }
  }
}

async function setup(
  page: Page,
  screen: Screen,
  respond: (route: Route, url: URL) => Promise<void>
) {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
  await page.route(`**/api/${screen.endpoint}?**`, async (route) => {
    const url = new URL(route.request().url())
    // Model metadata is loaded separately by the existing model/deployment pages.
    if (Number(url.searchParams.get('limit')) > 100) return route.fallback()
    await respond(route, url)
  })
  await page.goto(screen.path)
  await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
  await expect(page).toHaveURL(new RegExp(screen.path))
  await expect(page).toHaveTitle(/TrainFactory/)
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
}

function fulfill(route: Route, screen: Screen, url: URL, total = 25, revision = 'original') {
  const offset = Number(url.searchParams.get('offset'))
  const limit = Number(url.searchParams.get('limit')) || 10
  return route.fulfill({
    json: {
      [screen.key]: Array.from({ length: total }, (_, index) =>
        item(screen, index, revision)
      ).slice(offset, offset + limit),
      total,
    },
  })
}

async function selectFilter(page: Page, screen: Screen) {
  await page
    .locator('.ant-select')
    .filter({ hasText: new RegExp(`^${screen.filter}$`) })
    .click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ hasText: new RegExp(`^${screen.option}$`) })
    .click()
}

for (const screen of screens) {
  test.describe(`${screen.name} list consistency`, () => {
    test('failed page navigation keeps the page bound to its rows and retries the requested page', async ({
      page,
    }) => {
      let fail = true
      let failures = 0
      await setup(page, screen, async (route, url) => {
        if (fail && Number(url.searchParams.get('offset')) > 0) {
          failures++
          await route.fulfill({ status: 503, json: { detail: 'Intentional list failure' } })
        } else await fulfill(route, screen, url)
      })
      await page.locator('.ant-pagination-item-2').click()
      await expect.poll(() => failures).toBeGreaterThan(0)
      await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
      await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
      await expect(page.getByRole('alert').filter({ hasText: 'previous results' })).toBeVisible()
      fail = false
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toBeVisible()
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
    })

    test('failed filter labels preserved rows as previous results, then refresh uses the filter', async ({
      page,
    }) => {
      let fail = true
      await setup(page, screen, async (route, url) => {
        if (url.searchParams.has('status')) {
          if (fail)
            await route.fulfill({ status: 503, json: { detail: 'Intentional filter failure' } })
          else await fulfill(route, screen, url, 1, 'filtered')
        } else await fulfill(route, screen, url)
      })
      await selectFilter(page, screen)
      await expect(page.getByRole('alert').filter({ hasText: 'previous results' })).toBeVisible()
      await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
      fail = false
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByText(`${screen.name}-filtered-0`, { exact: true })).toBeVisible()
      await expect(page.getByRole('alert').filter({ hasText: 'previous results' })).toHaveCount(0)
    })

    test('refresh after the last page shrinks requests and renders the valid page', async ({
      page,
    }) => {
      let total = 11
      const offsets: number[] = []
      await setup(page, screen, async (route, url) => {
        offsets.push(Number(url.searchParams.get('offset')))
        await fulfill(route, screen, url, total)
      })
      await page.locator('.ant-pagination-item-2').click()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toBeVisible()
      total = 10
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
      expect(offsets.slice(-2)).toEqual([10, 0])
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
      expect(offsets.at(-1)).toBe(0)
    })

    test('same-query refresh failures preserve successful rows', async ({ page }) => {
      let fail = false
      await setup(page, screen, async (route, url) => {
        if (fail)
          await route.fulfill({ status: 503, json: { detail: 'Intentional refresh failure' } })
        else await fulfill(route, screen, url)
      })
      fail = true
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByRole('alert').filter({ hasText: 'previous results' })).toBeVisible()
      await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
    })

    test('deleting the last row on a page returns to existing rows', async ({ page }) => {
      let total = 11
      const offsets: number[] = []
      await setup(page, screen, async (route, url) => {
        offsets.push(Number(url.searchParams.get('offset')))
        await fulfill(route, screen, url, total)
      })
      await page.route(`**/api/${screen.endpoint}/*`, async (route) => {
        if (route.request().method() !== 'DELETE') return route.fallback()
        total = 10
        await route.fulfill({ status: 204 })
      })
      await page.locator('.ant-pagination-item-2').click()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toBeVisible()
      await page.getByRole('button', { name: /^delete$/i }).click()
      await page.locator('.ant-popconfirm').getByRole('button', { name: 'OK', exact: true }).click()
      await expect(page.getByText(`${screen.name}-original-0`, { exact: true })).toBeVisible()
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
      expect(offsets.slice(-2)).toEqual([10, 0])
    })

    test('changing page size commits the requested rows and refresh keeps that size', async ({
      page,
    }) => {
      const requests: URL[] = []
      await setup(page, screen, async (route, url) => {
        requests.push(url)
        await fulfill(route, screen, url, 45)
      })
      await page.locator('.ant-pagination-item-2').click()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toBeVisible()
      await page.locator('.ant-pagination-options-size-changer').click()
      await page
        .locator('.ant-select-dropdown:visible .ant-select-item-option')
        .filter({ hasText: /^20 \/ page$/ })
        .click()
      await expect(page.getByText(`${screen.name}-original-20`, { exact: true })).toBeVisible()
      await expect(page.getByText(`${screen.name}-original-39`, { exact: true })).toBeVisible()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toHaveCount(0)
      await page.getByRole('button', { name: /Refresh$/ }).click()
      await expect(page.getByRole('button', { name: /Refresh$/ })).toBeEnabled()
      expect(requests.at(-1)!.searchParams.get('limit')).toBe('20')
      expect(requests.at(-1)!.searchParams.get('offset')).toBe('20')
    })

    test('an older response cannot overwrite a newer filtered result', async ({ page }) => {
      let pending: Route | undefined
      await setup(page, screen, async (route, url) => {
        if (url.searchParams.has('status')) await fulfill(route, screen, url, 1, 'filtered')
        else if (Number(url.searchParams.get('offset')) > 0) pending = route
        else await fulfill(route, screen, url)
      })
      await page.locator('.ant-pagination-item-2').click()
      await expect.poll(() => Boolean(pending)).toBeTruthy()
      await selectFilter(page, screen)
      await expect(page.getByText(`${screen.name}-filtered-0`, { exact: true })).toBeVisible()
      const oldResponse = page.waitForResponse(
        (response) => response.url() === pending!.request().url()
      )
      await fulfill(pending!, screen, new URL(pending!.request().url()))
      await oldResponse
      await expect(page.getByText(`${screen.name}-filtered-0`, { exact: true })).toBeVisible()
      await expect(page.getByText(`${screen.name}-original-10`, { exact: true })).toHaveCount(0)
      await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
    })
  })
}

test('deployment statistics use all matching server results and survive page changes', async ({
  page,
}, testInfo) => {
  const screen = screens[3]
  await setup(page, screen, async (route, url) => {
    const offset = Number(url.searchParams.get('offset'))
    await route.fulfill({
      json: {
        deployments: Array.from({ length: 10 }, (_, i) => item(screen, offset + i)),
        total: 25,
        stats: {
          total: 25,
          by_status: { running: 9, restarting: 2, degraded: 3, stopped: 7, failed: 4 },
        },
      },
    })
  })
  const values = page.locator('.stat-card .ant-statistic-content-value')
  await expect(values).toHaveText(['25', '11', '3', '7', '4'])
  await page.locator('.ant-pagination-item-2').click()
  await expect(page.getByText('deployments-original-10', { exact: true })).toBeVisible()
  await expect(values).toHaveText(['25', '11', '3', '7', '4'])
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
  await page.locator('.page-stats').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('deployment-page-two-stats.png'), fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(values).toHaveText(['25', '11', '3', '7', '4'])
  await page.locator('.page-stats').scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('deployment-page-two-stats-mobile.png'), fullPage: true })
})

test('legacy deployment responses show total and unknown status totals without fetching every page', async ({
  page,
}) => {
  const requestedLimits: number[] = []
  await setup(page, screens[3], async (route, url) => {
    requestedLimits.push(Number(url.searchParams.get('limit')))
    await fulfill(route, screens[3], url)
  })
  await expect(page.locator('.stat-card .ant-statistic-content-value')).toHaveText([
    '25',
    '—',
    '—',
    '—',
    '—',
  ])
  expect(requestedLimits.every((limit) => limit === 10)).toBeTruthy()
})

test('failed fallback keeps its previous page and refresh retries the corrected offset', async ({
  page,
}) => {
  let total = 11
  let fail = true
  const offsets: number[] = []
  await setup(page, screens[3], async (route, url) => {
    const offset = Number(url.searchParams.get('offset'))
    offsets.push(offset)
    if (total === 10 && offset === 0 && fail)
      await route.fulfill({ status: 503, json: { detail: 'Intentional fallback failure' } })
    else await fulfill(route, screens[3], url, total)
  })
  await page.locator('.ant-pagination-item-2').click()
  await expect(page.getByText('deployments-original-10', { exact: true })).toBeVisible()
  total = 10
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'Could not load' })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
  await expect(page.getByText('deployments-original-10', { exact: true })).toBeVisible()
  fail = false
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByText('deployments-original-0', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('1')
  expect(offsets.slice(-3)).toEqual([10, 0, 0])
})

test('deployment statistics follow filters and remain unchanged on a failed refresh', async ({
  page,
}) => {
  let fail = false
  await setup(page, screens[3], async (route, url) => {
    if (fail) return route.fulfill({ status: 503, json: { detail: 'Intentional stats failure' } })
    const filtered = url.searchParams.get('status') === 'failed'
    await route.fulfill({
      json: {
        deployments: [item(screens[3], 0, filtered ? 'filtered' : 'original')],
        total: filtered ? 4 : 25,
        stats: filtered
          ? { total: 4, by_status: { failed: 4 } }
          : { total: 25, by_status: { stopped: 21, failed: 4 } },
      },
    })
  })
  await selectFilter(page, screens[3])
  await expect(page.getByText('deployments-filtered-0', { exact: true })).toBeVisible()
  const values = page.locator('.stat-card .ant-statistic-content-value')
  await expect(values).toHaveText(['4', '0', '0', '0', '4'])
  fail = true
  await page.getByRole('button', { name: /Refresh$/ }).click()
  await expect(page.getByRole('alert').filter({ hasText: 'previous results' })).toBeVisible()
  await expect(values).toHaveText(['4', '0', '0', '0', '4'])
})
