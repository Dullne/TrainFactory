import { test, expect, type Page, type Route } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

const draftKey = 'tf_training_draft:v1:user-1'

async function openForm(page: Page, width = 1440) {
  await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 })
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
  await page.goto('/training/create')
  await expect(
    page.getByRole('heading', { name: 'Create Training Task', exact: true })
  ).toBeVisible()
}

async function choose(page: Page, id: string, label: string) {
  await page
    .locator('.ant-select')
    .filter({ has: page.locator(`#${id}`) })
    .click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({
      has: page.getByText(label, { exact: true }),
    })
    .click()
}

async function path(page: Page, id: string, value: string) {
  await page.locator(id).fill(value)
  await page.keyboard.press('Enter')
  await page.keyboard.press('Escape')
}

async function leaveAndReturn(page: Page) {
  await page.getByRole('menuitem', { name: /Model Registry/ }).click()
  await expect(page).toHaveURL(/\/models/)
  await expect(page.locator('#task_name')).toHaveCount(0)
  await page.goBack()
  await expect(page.locator('#task_name')).toBeVisible()
}

async function submitWithHeldResponse(page: Page) {
  let capture!: (route: Route) => void
  const heldRequest = new Promise<Route>((resolve) => {
    capture = resolve
  })
  await page.route('**/api/train', (route) => {
    if (route.request().method() === 'POST') capture(route)
    else return route.fallback()
  })
  await page.locator('#task_name').fill('Submitted configuration A')
  await path(page, '#base_model_path', '/models/pending-creation')
  await path(page, '[id^="dataset-path-"]', '/datasets/pending-creation.jsonl')
  await page.getByRole('button', { name: 'Create Task', exact: true }).click()
  const held = await heldRequest
  return async () => {
    const response = page.waitForResponse(
      (result) =>
        new URL(result.url()).pathname === '/api/train' && result.request().method() === 'POST'
    )
    await held.fulfill({ json: { task_id: 'old-submission-a', status: 'pending' } })
    await (await response).finished()
    // Let the response callback and React commit run before asserting that a
    // stale completion did not navigate or change the current form.
    await page.evaluate(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
        })
    )
  }
}

test('late submission cannot erase a new draft after leaving and returning', async ({ page }) => {
  await openForm(page)
  const finish = await submitWithHeldResponse(page)
  await leaveAndReturn(page)
  await page.locator('#task_name').fill('New configuration B')
  await finish()
  await expect(page).toHaveURL(/\/training\/create$/)
  await expect(page.locator('#task_name')).toHaveValue('New configuration B')
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('New configuration B')
})

test('edits made while submitting remain a recoverable draft when creation succeeds', async ({
  page,
}) => {
  await openForm(page)
  const finish = await submitWithHeldResponse(page)
  await page.locator('#task_name').fill('Edited configuration B')
  await page.getByRole('button', { name: /Add Dataset$/ }).click()
  await finish()
  await expect(page).toHaveURL(/\/training\/create$/)
  await expect(
    page.getByText('Task old-submission-a created. Your newer edits are kept in the draft.', {
      exact: true,
    })
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create Task', exact: true })).toBeEnabled()
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('Edited configuration B')
  await expect(page.locator('.training-create-dataset-row')).toHaveCount(2)
})

test('late submission cannot redirect the page after the account signs out', async ({ page }) => {
  await openForm(page)
  const finish = await submitWithHeldResponse(page)
  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Sign out', { exact: true }).click()
  await page
    .getByRole('dialog', { name: 'Sign out?' })
    .getByRole('button', { name: 'Confirm sign out', exact: true })
    .click()
  await expect(page).toHaveURL(/\/login$/)
  await finish()
  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByText('Training task created successfully', { exact: true })).toHaveCount(0)
  expect(await page.evaluate((key) => sessionStorage.getItem(key), draftKey)).toBeNull()
})

test('draft restores model-specific values and dynamic datasets after leaving and reloading', async ({
  page,
}) => {
  await openForm(page)
  await page.locator('#task_name').fill('Recover preference experiment')
  await choose(page, 'model_type', 'LLM')
  await choose(page, 'training_method', 'DPO')
  await path(page, '#base_model_path', '/models/draft-llm')
  await path(page, '[id^="dataset-path-"]', '/datasets/draft-train.jsonl')
  await page.getByRole('button', { name: /Add Dataset$/ }).click()
  await path(
    page,
    '.training-create-dataset-row:nth-child(2) [id^="dataset-path-"]',
    '/datasets/draft-eval.jsonl'
  )
  const second = page.locator('.training-create-dataset-row').nth(1)
  await second
    .locator('.ant-select')
    .filter({ has: page.getByRole('combobox', { name: 'Dataset split', exact: true }) })
    .click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ hasText: /^Eval$/ })
    .click()
  await second.getByRole('spinbutton', { name: 'Sample count', exact: true }).fill('73')
  await page.getByRole('button', { name: 'Expand all', exact: true }).click()
  await page.locator('#rl_config_beta').fill('0.35')
  await page.locator('#num_train_epochs').fill('9')
  await expect(page.getByRole('status')).toContainText('Draft saved')
  await leaveAndReturn(page)
  await expect(page.locator('#task_name')).toHaveValue('Recover preference experiment')
  await expect(page.getByRole('status')).toContainText('Draft restored')
  await page.reload()
  await page.getByRole('button', { name: 'Expand all', exact: true }).click()
  await expect(page.locator('#rl_config_beta')).toHaveValue('0.35')
  await expect(page.locator('#num_train_epochs')).toHaveValue('9')
  await expect(page.locator('#per_device_train_batch_size')).toHaveValue('1')
  await expect(second).toContainText('/datasets/draft-eval.jsonl')
  await expect(second).toContainText('Eval')
  await expect(second.getByRole('spinbutton', { name: 'Sample count', exact: true })).toHaveValue(
    '73'
  )
})

test('discard removes the saved values and does not restore them on reload', async ({ page }) => {
  await openForm(page)
  await page.locator('#task_name').fill('Discard this experiment')
  await page.getByRole('button', { name: 'Discard draft', exact: true }).click()
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(page.locator('#task_name')).toHaveValue('')
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('')
  expect(await page.evaluate((key) => sessionStorage.getItem(key), draftKey)).toBeNull()
})

test('a different authenticated account never restores the previous account draft', async ({
  page,
}) => {
  await openForm(page)
  await page.locator('#task_name').fill('Private account A experiment')
  await expect(page.getByRole('status')).toContainText('Draft saved')
  await page.route('**/api/auth/me', (route) =>
    route.fulfill({
      json: {
        user_id: 'user-2',
        username: 'second-account',
        email: null,
        is_active: true,
        is_admin: false,
        created_at: '2026-08-08T12:00:00',
        updated_at: '2026-08-08T12:00:00',
      },
    })
  )
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('')
  await expect(page.getByText('Private account A experiment', { exact: true })).toHaveCount(0)
})

test('malformed draft is ignored and a fresh edit can be recovered', async ({ page }) => {
  await openForm(page)
  await page.evaluate(
    (key) =>
      sessionStorage.setItem(
        key,
        JSON.stringify({
          version: 1,
          accountId: 'user-1',
          values: { task_name: { unexpected: true }, model_type: 'unknown' },
          datasets: [{ split: 'invalid' }],
        })
      ),
    draftKey
  )
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('')
  await page.locator('#task_name').fill('Recovered after corrupt data')
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('Recovered after corrupt data')
})

test('blocked storage warns before closing and still recovers in-app navigation', async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.setItem
    Storage.prototype.setItem = function (key, value) {
      if (key.startsWith('tf_training_draft:'))
        throw new DOMException('Denied', 'QuotaExceededError')
      original.call(this, key, value)
    }
  })
  await openForm(page)
  await page.locator('#task_name').fill('Keep while storage is blocked')
  await expect(page.getByRole('status')).toContainText('Keep this tab open')
  await leaveAndReturn(page)
  await expect(page.locator('#task_name')).toHaveValue('Keep while storage is blocked')
  const dialogPromise = page.waitForEvent('dialog')
  await page.evaluate(() => {
    setTimeout(() => window.location.reload(), 0)
  })
  const dialog = await dialogPromise
  expect(dialog.type()).toBe('beforeunload')
  await dialog.dismiss()
  await expect(page.locator('#task_name')).toHaveValue('Keep while storage is blocked')
})

test('successful submission removes the draft but failed submission keeps it', async ({ page }) => {
  await openForm(page)
  await page.locator('#task_name').fill('Submit draft')
  await path(page, '#base_model_path', '/models/draft-submit')
  await path(page, '[id^="dataset-path-"]', '/datasets/draft-submit.jsonl')
  await page.route('**/api/train', (route) =>
    route.fulfill({ status: 500, json: { detail: 'Try again' } })
  )
  await page.getByRole('button', { name: 'Create Task', exact: true }).click()
  await expect(page.getByText('Try again', { exact: true })).toBeVisible()
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('Submit draft')
  await page.route('**/api/train', (route) =>
    route.fulfill({ json: { task_id: 'draft-submitted', status: 'pending' } })
  )
  await page.getByRole('button', { name: 'Create Task', exact: true }).click()
  await expect(page).toHaveURL(/\/training\/draft-submitted$/)
  await page.goto('/training/create')
  await expect(page.locator('#task_name')).toHaveValue('')
  expect(await page.evaluate((key) => sessionStorage.getItem(key), draftKey)).toBeNull()
})

test('390px common fields lead, advanced summaries stay readable, and actions stay in viewport', async ({
  page,
}, testInfo) => {
  await openForm(page, 390)
  await expect(page.locator('#task_name')).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Evaluation & Saving', exact: true })
  ).toHaveAttribute('aria-expanded', 'false')
  await expect(
    page.getByRole('button', { name: 'Fine-tuning Method', exact: true })
  ).toHaveAttribute('aria-expanded', 'false')
  await expect(page.locator('[data-training-section="tuner"]')).toContainText('LoRA')
  const submit = page.getByRole('button', { name: 'Create Task', exact: true })
  await expect(submit).toBeInViewport({ ratio: 1 })
  await page.screenshot({ path: testInfo.outputPath('training-common-mobile.png') })
  await page.getByRole('button', { name: 'Expand all', exact: true }).click()
  await page.locator('#output_dir').scrollIntoViewIfNeeded()
  await expect(submit).toBeInViewport({ ratio: 1 })
  await page.locator('#output_dir').click({ trial: true })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
})

test('dataset rows work when the browser does not expose secure-context UUID generation', async ({
  page,
}) => {
  await page.addInitScript(() => Object.defineProperty(crypto, 'randomUUID', { value: undefined }))
  await openForm(page)
  await page.getByRole('button', { name: /Add Dataset$/ }).click()
  await expect(page.locator('.training-create-dataset-row')).toHaveCount(2)
})

test('discard invalidates a saved draft even if storage removal is denied', async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.removeItem
    Storage.prototype.removeItem = function (key) {
      if (key.startsWith('tf_training_draft:')) throw new DOMException('Denied', 'SecurityError')
      original.call(this, key)
    }
  })
  await openForm(page)
  await page.locator('#task_name').fill('Must not come back after discard')
  await page.getByRole('button', { name: 'Discard draft', exact: true }).click()
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(page.locator('#task_name')).toHaveValue('')
  await page.reload()
  await expect(page.locator('#task_name')).toHaveValue('')
})
