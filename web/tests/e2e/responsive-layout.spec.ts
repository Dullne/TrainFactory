import { test, expect, type Locator, type Page } from '@playwright/test'
import { mockApi, mockAuth } from './helpers/mockApi'

const scenarios = [
  { width: 390, language: 'zh', mode: 'light' },
  { width: 390, language: 'en', mode: 'dark' },
  { width: 768, language: 'zh', mode: 'light' },
  { width: 1440, language: 'en', mode: 'dark' },
] as const

async function reachable(page: Page, control: Locator) {
  await control.scrollIntoViewIfNeeded()
  await expect(control).toBeInViewport({ ratio: 0.99 })
  const bounds = await control.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width + 1)
  await control.click({ trial: true })
}

async function noHorizontalOverflow(page: Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
    page.viewportSize()!.width
  )
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
}

for (const scenario of scenarios) {
  test.describe(`${scenario.width}px ${scenario.language} ${scenario.mode}`, () => {
    const runtimeErrors: string[] = []

    test.beforeEach(async ({ page }) => {
      runtimeErrors.length = 0
      page.on('pageerror', error => runtimeErrors.push(error.message))
      await page.setViewportSize({ width: scenario.width, height: 960 })
      await page.addInitScript(({ language, mode }) => {
        localStorage.setItem('tf_language', language)
        localStorage.setItem('tf_appearance:v1', JSON.stringify({ mode, accent: 'blue' }))
      }, scenario)
      await mockApi(page)
      await mockAuth(page, { authenticated: true, directStorageRegistrationEnabled: true })
    })

    test.afterEach(() => expect(runtimeErrors).toEqual([]))

    test('generation method labels fit inside their controls after wrapping', async ({ page }, testInfo) => {
      await page.route('**/api/generation/formats', route => route.fulfill({ json: {
        output_formats: [{ name: 'universal', description: 'Universal' }],
        source_types: ['jsonl'], length_types: ['short'],
      } }))
      await page.goto('/datasets/generation/create')
      const methods = page.locator('.generation-create-page .ant-radio-button-wrapper')
      await expect(methods.first()).toBeVisible()
      const clipped = await methods.evaluateAll(labels => labels
        .filter(label => label.clientHeight && label.scrollHeight > label.clientHeight + 1)
        .map(label => label.textContent))
      expect(clipped).toEqual([])
      const method = methods.filter({ hasText: /纯 LLM|Pure LLM/ })
      await reachable(page, method)
      await method.click()
      await expect(method.getByRole('radio')).toBeChecked()
      await noHorizontalOverflow(page)
      await page.screenshot({ path: testInfo.outputPath('generation-method.png') })
    })

    test('list controls remain reachable and statistics share rows', async ({ page }, testInfo) => {
      for (const route of ['/models', '/configs', '/deployments', '/evaluations']) {
        await page.goto(route)
        const toolbar = page.locator('.page-toolbar')
        await expect(toolbar.getByRole('heading')).toBeVisible()
        const controls = page.locator('.page-toolbar button, .page-toolbar .ant-select, .model-list-filters .ant-select')
        for (const control of await controls.all()) await reachable(page, control)
        const cards = page.locator('.page-stats .stat-card')
        const first = await cards.nth(0).boundingBox()
        const second = await cards.nth(1).boundingBox()
        expect(first!.y).toBeCloseTo(second!.y, 0)
        expect(first!.width).toBeGreaterThan(130)
        const refresh = toolbar.getByRole('button', { name: /刷新|Refresh/ })
        await refresh.click()
        await expect(refresh).not.toHaveClass(/ant-btn-loading/)
        await noHorizontalOverflow(page)
        await page.screenshot({ path: testInfo.outputPath(`${route.slice(1)}.png`), fullPage: true })
      }
      await page.goto('/configs')
      const edit = page.locator('.config-card-actions').first().getByRole('button', { name: /编辑|Edit/ })
      await reachable(page, edit)
      await edit.click()
      await expect(page.getByRole('dialog')).toBeVisible()
    })

    test('training details keep metrics and dates readable', async ({ page }, testInfo) => {
      await page.goto('/training/task-12345678')
      await expect(page.locator('.training-detail-heading')).toBeVisible()
      for (const control of await page.locator('.page-toolbar button').all()) {
        await reachable(page, control)
      }
      const metrics = page.locator('.training-detail-metrics .ant-statistic')
      await expect(metrics).toHaveCount(4)
      for (const metric of await metrics.all()) {
        const box = await metric.boundingBox()
        expect(box!.width).toBeGreaterThan(120)
      }
      const timeCard = page.locator('.ant-card').filter({
        has: page.locator('.ant-card-head-title').filter({ hasText: /时间信息|Time Info/ }),
      })
      await expect(timeCard).toBeVisible()
      expect((await timeCard.boundingBox())!.width).toBeGreaterThan(280)
      await page.locator('.page-toolbar').getByRole('button', { name: /刷新|Refresh/ }).click()
      await expect(page.locator('.training-detail-heading')).toBeVisible()
      await noHorizontalOverflow(page)
      await page.screenshot({ path: testInfo.outputPath('training-detail.png'), fullPage: true })
    })

    test('evaluation dataset rows keep selectors and delete buttons usable', async ({ page }, testInfo) => {
      await page.goto('/evaluations')
      await page.locator('.page-toolbar').getByRole('button', { name: /创建任务|Create Task/ }).click()
      const modal = page.getByRole('dialog')
      await modal.getByRole('button', { name: /已注册|Registered/ }).click()
      await modal.getByRole('button', { name: /MTEB/ }).click()
      await modal.getByRole('button', { name: /本地|Local/ }).click()
      const rows = modal.locator('.evaluation-dataset-row')
      await expect(rows).toHaveCount(3)
      for (const row of await rows.all()) {
        const control = row.locator('.evaluation-dataset-control')
        await control.scrollIntoViewIfNeeded()
        expect((await control.boundingBox())!.width).toBeGreaterThan(220)
        await reachable(page, row.getByRole('button', { name: /删除|Delete/ }))
      }
      const local = rows.last()
      await local.getByRole('textbox').first().fill('layout-test')
      await local.getByRole('textbox').last().fill('/data/layout-test.jsonl')
      await expect(local.getByRole('textbox').first()).toHaveValue('layout-test')
      await page.screenshot({ path: testInfo.outputPath('evaluation-datasets.png') })
      await rows.first().getByRole('button', { name: /删除|Delete/ }).click()
      await expect(rows).toHaveCount(2)
      await expect(rows.last().getByRole('textbox').first()).toHaveValue('layout-test')
      await noHorizontalOverflow(page)
      await modal.getByRole('button', { name: /取\s*消|Cancel/ }).click()
      await expect(modal).not.toBeVisible()
    })

    test('deployment model filters leave enough room for model selection', async ({ page }, testInfo) => {
      await page.goto('/deployments')
      await page.locator('.page-toolbar').getByRole('button', { name: /创建部署|Create Deployment/ }).click()
      const modal = page.getByRole('dialog')
      const selectors = modal.locator('.deployment-model-selection .ant-select')
      await expect(selectors).toHaveCount(2)
      for (const control of await selectors.all()) await reachable(page, control)
      expect((await selectors.last().boundingBox())!.width).toBeGreaterThan(220)
      await selectors.last().click()
      await expect(page.locator('.ant-select-dropdown:visible')).toBeVisible()
      await page.keyboard.press('Escape')
      await noHorizontalOverflow(page)
      await page.screenshot({ path: testInfo.outputPath('deployment-create.png') })
    })
  })
}
