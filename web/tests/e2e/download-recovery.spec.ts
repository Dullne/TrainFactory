import { expect, test, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

async function openDownloadDialog(page: Page, buttonName: string) {
  await page.getByRole('button', { name: new RegExp(`${buttonName}$`) }).click()
  await expect(page.getByTestId('download-task-history')).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

for (const resource of ['model', 'dataset'] as const) {
  test(`${resource} history discovers tasks created elsewhere when reopened after an empty snapshot`, async ({
    page,
  }) => {
    const plural = `${resource}s`
    const buttonName = resource === 'model' ? 'Download Model' : 'Download Dataset'
    let createdElsewhere = false
    let listRequests = 0
    await page.route(`**/api/${plural}/downloads`, (route) => {
      listRequests += 1
      const task =
        resource === 'model'
          ? {
              registry_id: 'external-task',
              model_name: 'External model',
              status: 'downloading',
              progress: 35,
            }
          : {
              dataset_id: 'external-task',
              dataset_name: 'External dataset',
              status: 'downloading',
              progress: 35,
            }
      return route.fulfill({
        json: { downloads: createdElsewhere ? [task] : [], total: createdElsewhere ? 1 : 0 },
      })
    })
    await page.goto(`/${plural}`)
    await openDownloadDialog(page, buttonName)
    await expect(page.getByTestId('download-task-history')).toContainText('No download tasks')
    await expect(
      page.getByTestId('download-task-history').locator('.ant-spin-spinning')
    ).toHaveCount(0)
    const initialRequests = listRequests
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()
    createdElsewhere = true
    await openDownloadDialog(page, buttonName)
    await expect(page.getByTestId('download-task-external-task')).toContainText('35%')
    expect(listRequests).toBeGreaterThan(initialRequests)
  })
}

test('model download progress survives close, reopen, and reload without duplicates', async ({
  page,
}) => {
  let listRequests = 0
  const task = {
    registry_id: 'model-download-1',
    model_name: 'org/recoverable-model',
    display_name: 'Recoverable model',
    download_source: 'huggingface',
    remote_repo: 'org/recoverable-model',
    model_path: '/models/recoverable-model',
    status: 'downloading',
    progress: 42,
    created_at: '2026-09-19T09:00:00Z',
  }

  await page.route('**/api/models/downloads', async (route) => {
    listRequests += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ downloads: [task, task], total: 2 }),
    })
  })

  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-model-download-1')).toContainText(
    'Recoverable model'
  )
  await expect(page.getByTestId('download-task-model-download-1')).toContainText('42%')
  await expect(page.getByTestId('download-task-model-download-1')).toHaveCount(1)

  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-model-download-1')).toBeVisible()

  await page.reload()
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-model-download-1')).toBeVisible()
  expect(listRequests).toBeGreaterThanOrEqual(2)
})

test('dataset download progress survives close, reopen, and reload without duplicates', async ({
  page,
}) => {
  let listRequests = 0
  const task = {
    dataset_id: 'dataset-download-1',
    dataset_name: 'Recoverable dataset',
    source_type: 'huggingface',
    remote_repo: 'org/recoverable-dataset',
    storage_path: '/datasets/recoverable-dataset',
    status: 'downloading',
    progress: 67,
    created_at: '2026-09-19T09:00:00Z',
  }

  await page.route('**/api/datasets/downloads', async (route) => {
    listRequests += 1
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ downloads: [task, task], total: 2 }),
    })
  })

  await page.goto('/datasets')
  await openDownloadDialog(page, 'Download Dataset')
  await expect(page.getByTestId('download-task-dataset-download-1')).toContainText(
    'Recoverable dataset'
  )
  await expect(page.getByTestId('download-task-dataset-download-1')).toContainText('67%')
  await expect(page.getByTestId('download-task-dataset-download-1')).toHaveCount(1)

  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await openDownloadDialog(page, 'Download Dataset')
  await expect(page.getByTestId('download-task-dataset-download-1')).toBeVisible()

  await page.reload()
  await openDownloadDialog(page, 'Download Dataset')
  await expect(page.getByTestId('download-task-dataset-download-1')).toBeVisible()
  expect(listRequests).toBeGreaterThanOrEqual(2)
})

test('background model tracking stays single-flight and refreshes after completion', async ({
  page,
}) => {
  test.setTimeout(30_000)
  let listRequests = 0
  let modelListRequests = 0
  let releaseCompletion!: () => void
  const completionGate = new Promise<void>((resolve) => {
    releaseCompletion = resolve
  })

  page.on('request', (request) => {
    const path = new URL(request.url()).pathname
    if (request.method() === 'GET' && path.endsWith('/api/models')) modelListRequests += 1
  })
  await page.route('**/api/models/downloads', async (route) => {
    listRequests += 1
    const requestNumber = listRequests
    if (requestNumber === 2) await completionGate
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        downloads: [
          {
            registry_id: 'background-model',
            model_name: 'Background model',
            display_name: 'Background model',
            status: requestNumber === 1 ? 'downloading' : 'available',
            progress: requestNumber === 1 ? 50 : 100,
            created_at: '2026-09-19T09:00:00Z',
          },
        ],
        total: 1,
      }),
    })
  })

  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-background-model')).toContainText('50%')
  const modelRequestsBeforeCompletion = modelListRequests
  await expect.poll(() => listRequests, { timeout: 6_000 }).toBe(2)

  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await openDownloadDialog(page, 'Download Model')
  await page.waitForTimeout(300)
  expect(listRequests).toBe(2)
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()

  releaseCompletion()
  await expect(page.getByText('Model download complete!', { exact: true })).toBeVisible()
  await expect.poll(() => modelListRequests).toBeGreaterThan(modelRequestsBeforeCompletion)
})

test('a stale pre-submit snapshot cannot discard a newly accepted download', async ({ page }) => {
  test.setTimeout(30_000)
  let listRequests = 0
  let releaseInitialList!: () => void
  const initialListGate = new Promise<void>((resolve) => {
    releaseInitialList = resolve
  })

  await page.route('**/api/models/download', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        registry_id: 'newly-accepted-model',
        model_name: 'org/newly-accepted-model',
        display_name: 'Newly accepted model',
        status: 'downloading',
        model_path: '/models/newly-accepted-model',
      }),
    })
  })
  await page.route('**/api/models/downloads', async (route) => {
    listRequests += 1
    const requestNumber = listRequests
    if (requestNumber === 1) await initialListGate
    const downloads =
      requestNumber === 1
        ? []
        : [
            {
              registry_id: 'newly-accepted-model',
              model_name: 'org/newly-accepted-model',
              display_name: 'Newly accepted model',
              status: requestNumber === 2 ? 'downloading' : 'available',
              progress: requestNumber === 2 ? 10 : 100,
              created_at: '2026-09-19T09:00:00Z',
            },
          ]
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ downloads, total: downloads.length }),
    })
  })

  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await page.getByLabel('Repository').fill('org/newly-accepted-model')
  await page.getByRole('button', { name: 'Start Download', exact: true }).click()
  await expect(page.getByTestId('download-task-newly-accepted-model')).toBeVisible()

  releaseInitialList()
  await expect.poll(() => listRequests).toBeGreaterThanOrEqual(2)
  await expect(page.getByTestId('download-task-newly-accepted-model')).toContainText('10%')
  await expect(page.getByText('Model download complete!', { exact: true })).toBeVisible({
    timeout: 6_000,
  })
  expect(listRequests).toBeGreaterThanOrEqual(3)
})

test('closing during submit still tracks the accepted download', async ({ page }) => {
  let listRequests = 0
  let releaseSubmit!: () => void
  let markSubmitStarted!: () => void
  const submitGate = new Promise<void>((resolve) => {
    releaseSubmit = resolve
  })
  const submitStarted = new Promise<void>((resolve) => {
    markSubmitStarted = resolve
  })

  await page.route('**/api/models/download', async (route) => {
    if (route.request().method() !== 'POST') return route.fallback()
    markSubmitStarted()
    await submitGate
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        registry_id: 'closed-submit-model',
        model_name: 'org/closed-submit-model',
        display_name: 'Closed submit model',
        status: 'downloading',
        model_path: '/models/closed-submit-model',
      }),
    })
  })
  await page.route('**/api/models/downloads', async (route) => {
    listRequests += 1
    const downloads =
      listRequests === 1
        ? []
        : [
            {
              registry_id: 'closed-submit-model',
              model_name: 'org/closed-submit-model',
              display_name: 'Closed submit model',
              status: 'downloading',
              progress: 20,
              created_at: '2026-09-19T09:00:00Z',
            },
          ]
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ downloads, total: downloads.length }),
    })
  })

  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await page.getByLabel('Repository').fill('org/closed-submit-model')
  await page.getByRole('button', { name: 'Start Download', exact: true }).click()
  await submitStarted
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()

  releaseSubmit()
  await expect(
    page.getByText('Download task created, downloading...', { exact: true })
  ).toBeVisible()
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-closed-submit-model')).toContainText('20%')
})

test('an in-flight download list from the previous account cannot replace the new account', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let account = 'account-a'
  let authenticated = true
  let releaseAccountA!: () => void
  let markAccountARequested!: () => void
  const accountAGate = new Promise<void>((resolve) => {
    releaseAccountA = resolve
  })
  const accountARequested = new Promise<void>((resolve) => {
    markAccountARequested = resolve
  })

  await page.route('**/api/auth/**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() === 'GET' && path.endsWith('/auth/config')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          self_registration_enabled: false,
          direct_storage_registration_enabled: false,
        }),
      })
    }
    if (request.method() === 'GET' && path.endsWith('/auth/me')) {
      return route.fulfill({
        status: authenticated ? 200 : 401,
        contentType: 'application/json',
        body: JSON.stringify(
          authenticated
            ? {
                user_id: `${account}-id`,
                username: account,
                email: null,
                is_active: true,
                is_admin: false,
                created_at: '2026-09-19T09:00:00Z',
                updated_at: '2026-09-19T09:00:00Z',
              }
            : { detail: 'Not authenticated' }
        ),
      })
    }
    if (request.method() === 'POST' && path.endsWith('/auth/logout')) {
      authenticated = false
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Signed out' }),
      })
    }
    if (request.method() === 'POST' && path.endsWith('/auth/login')) {
      account = 'account-b'
      authenticated = true
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: { 'set-cookie': 'access_token=account-b-token; Path=/; HttpOnly; SameSite=Lax' },
        body: JSON.stringify({ access_token: 'account-b-token', token_type: 'bearer' }),
      })
    }
    return route.fallback()
  })

  await page.route('**/api/models/downloads', async (route) => {
    const requestAccount = account
    if (requestAccount === 'account-a') {
      markAccountARequested()
      await accountAGate
    }
    const taskId = `${requestAccount}-download`
    try {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          downloads: [
            {
              registry_id: taskId,
              model_name: `${requestAccount} model`,
              display_name: `${requestAccount} model`,
              status: 'downloading',
              progress: requestAccount === 'account-a' ? 25 : 75,
              created_at: '2026-09-19T09:00:00Z',
            },
          ],
          total: 1,
        }),
      })
    } catch {
      // Navigating through sign-out may abort account A's obsolete request.
    }
  })

  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await accountARequested
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await page.getByRole('button', { name: /Open account menu for account-a/ }).click()
  await page.getByText('Sign out', { exact: true }).click()
  await page.getByRole('button', { name: 'Confirm sign out' }).click()
  await expect(page).toHaveURL(/\/login/)

  await page.getByLabel('Username').fill('account-b')
  await page.getByLabel('Password', { exact: true }).fill('account-b-password')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page).toHaveURL(/\/training$/)
  await page.goto('/models')
  await openDownloadDialog(page, 'Download Model')
  await expect(page.getByTestId('download-task-account-b-download')).toContainText('75%')

  releaseAccountA()
  await page.waitForTimeout(300)
  await expect(page.getByTestId('download-task-account-a-download')).toHaveCount(0)
  await expect(page.getByTestId('download-task-account-b-download')).toBeVisible()
})
