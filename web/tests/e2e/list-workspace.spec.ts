import { test, expect, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

const runtimeErrors = new WeakMap<Page, string[]>()
test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  runtimeErrors.set(page, errors)
  page.on('pageerror', (error) => errors.push(error.message))
})
test.afterEach(async ({ page }) => {
  expect(runtimeErrors.get(page)).toEqual([])
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
})

async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
  await page.route('**/api/generation/tasks?**', async (route) => {
    const url = new URL(route.request().url())
    const offset = Number(url.searchParams.get('offset')) || 0
    const limit = Number(url.searchParams.get('limit')) || 10
    await route.fulfill({
      json: {
        tasks: Array.from({ length: Math.min(limit, 45 - offset) }, (_, i) => ({
          task_id: `generated-${offset + i}`,
          task_name: `Generated ${offset + i}`,
          status: 'completed',
          generation_mode: 'doc_to_training',
          progress: 100,
          total_docs: 10,
          processed_docs: 10,
          output_sample_count: 10,
          created_at: '2026-09-19T00:00:00',
        })),
        total: 45,
        limit,
        offset,
        stats: { total: 45, pending: 0, running: 0, completed: 45, failed: 0, stopped: 0 },
      },
    })
  })
  await page.route('**/api/train?**', async (route) => {
    const url = new URL(route.request().url())
    const offset = Number(url.searchParams.get('offset')) || 0
    const limit = Number(url.searchParams.get('limit')) || 10
    await route.fulfill({
      json: {
        tasks: Array.from({ length: Math.min(limit, 45 - offset) }, (_, i) => ({
          task_id: `task-${offset + i}`,
          task_name: `Workspace task ${offset + i}`,
          status: url.searchParams.get('status') || 'running',
          model_type: 'embedding',
          training_method: 'sft',
          base_model_path: '/models/base',
          progress: 35,
          current_step: 35,
          total_steps: 100,
          train_loss: 0.1234,
          eval_loss: 0.2345,
          created_at: '2026-09-19T00:00:00',
          gpu_ids: [],
        })),
        total: 45,
      },
    })
  })
}

test('training filter and pagination survive navigation, back and reload', async ({ page }) => {
  await setup(page)
  await page.goto('/training')
  await page.locator('.training-list-status-filter').click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ hasText: /^Failed$/ })
    .click()
  await page.locator('.ant-pagination-item-2').click()
  await expect(page.getByText('Workspace task 10', { exact: true })).toBeVisible()
  await page.getByRole('menuitem', { name: /Model Registry/ }).click()
  await expect(page).toHaveURL(/\/models/)
  await expect(page.getByRole('heading', { name: /Model/ }).first()).toBeVisible()
  await page.goBack()
  await expect(page.locator('.training-list-status-filter')).toContainText('Failed')
  await expect(page.getByText('Workspace task 10', { exact: true })).toBeVisible()
  await page.reload()
  await expect(page.locator('.training-list-status-filter')).toContainText('Failed')
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
})

test('a shared training URL restores a validated page size and back changes the filter', async ({
  page,
}) => {
  await setup(page)
  await page.goto('/training?status=failed&page=2&page_size=20')
  await expect(page.getByText('Workspace task 20', { exact: true })).toBeVisible()
  await expect(page.getByText('Workspace task 39', { exact: true })).toBeVisible()
  await page.locator('.training-list-status-filter').click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ hasText: /^Training$/ })
    .click()
  await expect(page.getByText('Workspace task 0', { exact: true })).toBeVisible()
  await page.goBack()
  await expect(page.locator('.training-list-status-filter')).toContainText('Failed')
  await expect(page.getByText('Workspace task 20', { exact: true })).toBeVisible()
})

test('invalid list URL values are normalized without removing deployment deep links', async ({
  page,
}) => {
  await setup(page)
  await page.goto('/deployments?status=invalid&page=-3&page_size=999999&model_id=model-001')
  await expect(page.getByText('xinference-deployment', { exact: true })).toBeVisible()
  await expect.poll(() => new URL(page.url()).searchParams.has('status')).toBe(false)
  expect(new URL(page.url()).searchParams.get('model_id')).toBe('model-001')
  expect(new URL(page.url()).searchParams.has('page')).toBe(false)
  expect(new URL(page.url()).searchParams.has('page_size')).toBe(false)
})

test('mobile training cards show progress, losses and confirmed actions without horizontal scroll', async ({
  page,
}, testInfo) => {
  await setup(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/training')
  const card = page.getByRole('article').filter({ hasText: 'Workspace task 0' })
  await expect(card).toBeVisible()
  await expect(card).toContainText('35%')
  await expect(card).toContainText('0.1234')
  await card.getByRole('button', { name: 'Stop', exact: true }).click()
  await expect(page.locator('.ant-popconfirm')).toBeVisible()
  await page.locator('.ant-popconfirm').getByRole('button', { name: 'Cancel' }).click()
  await expect(page.locator('.ant-popconfirm')).not.toBeVisible()
  await expect(page.locator('.ant-table')).toHaveCount(0)
  expect(
    await page
      .locator('.main-layout-content')
      .evaluate((el) => el.scrollWidth <= el.clientWidth + 1)
  ).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('training-mobile-cards.png'), fullPage: true })
})

test('mobile deployments keep replica controls, endpoints and confirmation', async ({
  page,
}, testInfo) => {
  await setup(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/deployments')
  const card = page.getByRole('article').filter({ hasText: 'vllm-lora-deployment' }).first()
  await expect(card).toBeVisible()
  await card.getByRole('button', { name: 'Expand replicas' }).click()
  await expect(card.getByRole('button', { name: 'Recreate replica' }).first()).toBeVisible()
  await card.getByRole('button', { name: 'Recreate replica' }).first().click()
  await expect(page.locator('.ant-popconfirm')).toBeVisible()
  await page.locator('.ant-popconfirm').getByRole('button', { name: 'Cancel' }).click()
  await expect(page.locator('.ant-popconfirm')).not.toBeVisible()
  await expect(page.locator('.ant-table')).toHaveCount(0)
  expect(
    await page
      .locator('.main-layout-content')
      .evaluate((el) => el.scrollWidth <= el.clientWidth + 1)
  ).toBe(true)
  await page.screenshot({
    path: testInfo.outputPath('deployment-mobile-cards.png'),
    fullPage: true,
  })
})

test('returning to training restores account-scoped scroll after rows load', async ({ page }) => {
  await setup(page)
  await page.goto('/training?page_size=20')
  await expect(page.getByText('Workspace task 19', { exact: true })).toBeVisible()
  await page.evaluate(() => window.scrollTo(0, 700))
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(600)
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          JSON.parse(
            sessionStorage.getItem('tf_list_workspace:v1:user-1:/training?page_size=20') || '{}'
          ).window
      )
    )
    .toBeGreaterThan(600)
  await page
    .locator('tr[data-row-key="task-10"]')
    .getByRole('button', { name: 'Detail', exact: true })
    .click()
  await expect(page).toHaveURL(/\/training\/task-10$/)
  await expect(page.locator('.training-detail-heading')).toBeVisible()
  await page.goBack()
  await expect(page.getByText('Workspace task 19', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(600)
  const keys = await page.evaluate(() =>
    Object.keys(sessionStorage).filter((key) => key.startsWith('tf_list_workspace:'))
  )
  expect(keys.some((key) => key.startsWith('tf_list_workspace:v1:user-1:/training'))).toBe(true)
  await page.reload()
  await expect(page.getByText('Workspace task 19', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(600)
})

test('model view and filters remain selected after reload', async ({ page }) => {
  await setup(page)
  await page.goto('/models?status=archived&model_type=embedding&view=table')
  await expect(page.locator('.ant-table')).toBeVisible()
  await expect(
    page.locator('.ant-select-selection-item').filter({ hasText: /^Archived$/ })
  ).toBeVisible()
  await page.reload()
  await expect(page.locator('.ant-table')).toBeVisible()
  await expect(
    page.locator('.ant-select-selection-item').filter({ hasText: /^Archived$/ })
  ).toBeVisible()
})

test('dataset and generation tabs preserve independent pagination and filters', async ({
  page,
}) => {
  await setup(page)
  await page.goto(
    '/datasets?status=ready&usage=train&tab=generation&generation_page=2&generation_page_size=20'
  )
  await expect(page.getByRole('tab', { name: 'Data Generation' })).toHaveAttribute(
    'aria-selected',
    'true'
  )
  await expect(page.getByText('Generated 20', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Dataset List', exact: true }).click()
  await expect(
    page.locator('.ant-select-selection-item').filter({ hasText: /^Ready$/ })
  ).toBeVisible()
  await page.getByRole('tab', { name: 'Data Generation' }).click()
  await expect(page.getByText('Generated 20', { exact: true })).toBeVisible()
  expect(new URL(page.url()).searchParams.get('generation_page')).toBe('2')
  expect(new URL(page.url()).searchParams.get('generation_page_size')).toBe('20')
  expect(new URL(page.url()).searchParams.get('status')).toBe('ready')
})

for (const screen of [
  { path: '/evaluations?tab=online-test', tab: 'Online Test' },
  { path: '/sync?tab=api', tab: 'External API' },
]) {
  test(`${screen.path} restores the selected workspace tab`, async ({ page }) => {
    await setup(page)
    await page.goto(screen.path)
    await expect(page.getByRole('tab', { name: new RegExp(`${screen.tab}$`) })).toHaveAttribute(
      'aria-selected',
      'true'
    )
    await page.reload()
    await expect(page.getByRole('tab', { name: new RegExp(`${screen.tab}$`) })).toHaveAttribute(
      'aria-selected',
      'true'
    )
  })
}

test('evaluation pagination restores server results from a shared URL and reload', async ({
  page,
}) => {
  await setup(page)
  await page.route('**/api/evaluations/tasks?**', async (route) => {
    const url = new URL(route.request().url())
    const offset = Number(url.searchParams.get('offset')) || 0
    const limit = Number(url.searchParams.get('limit')) || 10
    await route.fulfill({
      json: {
        items: [
          {
            task_id: `evaluation-${offset}`,
            task_name: `Evaluation offset ${offset}`,
            eval_type: 'single',
            model_configs: [],
            dataset_configs: [],
            status: 'succeeded',
            progress: 100,
            created_at: '2026-09-19T00:00:00',
          },
        ],
        total: 45,
        limit,
        offset,
      },
    })
  })
  await page.goto('/evaluations?page=2&page_size=20')
  await expect(page.getByText('Evaluation offset 20', { exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByText('Evaluation offset 20', { exact: true })).toBeVisible()
  await expect(page.locator('.ant-pagination-item-active')).toHaveText('2')
})

test('config source group expansion survives reload', async ({ page }) => {
  await setup(page)
  await page.goto('/configs?groups=external')
  const internal = page
    .locator('.ant-collapse-header')
    .filter({ hasText: 'Internal Deployment' })
    .first()
  await expect(internal).toHaveAttribute('aria-expanded', 'false')
  await internal.click()
  await expect(internal).toHaveAttribute('aria-expanded', 'true')
  await page.reload()
  await expect(internal).toHaveAttribute('aria-expanded', 'true')
})

test('scroll state from another account is ignored', async ({ page }) => {
  await setup(page)
  await page.addInitScript(() => {
    sessionStorage.setItem(
      'tf_list_workspace:v1:other-user:/training?page_size=20',
      JSON.stringify({ window: 700, content: 0 })
    )
  })
  await page.goto('/training?page_size=20')
  await expect(page.getByText('Workspace task 19', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
})
