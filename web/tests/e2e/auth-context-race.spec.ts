import { test, expect, type Page } from '@playwright/test'

const authenticatedUser = {
  user_id: 'user-1',
  username: 'runtime_verify',
  email: null,
  is_active: true,
  is_admin: false,
  created_at: '2026-08-08T12:00:00',
  updated_at: '2026-08-08T12:00:00',
}

async function mockRefreshResult(page: Page, refreshStatus: number) {
  let meRequestCount = 0

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
      meRequestCount += 1
      return route.fulfill({
        status: meRequestCount === 1 ? 200 : refreshStatus,
        contentType: 'application/json',
        body: JSON.stringify(
          meRequestCount === 1 ? authenticatedUser : { detail: 'Refresh failed' }
        ),
      })
    }

    return route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: `Unhandled auth endpoint: ${request.method()} ${path}` }),
    })
  })
}

test('a refresh started before logout cannot restore the authenticated state', async ({ page }) => {
  let meRequestCount = 0
  let markRefreshPending!: () => void
  let releaseRefresh!: () => void
  const refreshPending = new Promise<void>((resolve) => {
    markRefreshPending = resolve
  })
  const refreshReleased = new Promise<void>((resolve) => {
    releaseRefresh = resolve
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
      meRequestCount += 1
      if (meRequestCount === 2) {
        markRefreshPending()
        await refreshReleased
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(authenticatedUser),
      })
    }

    if (request.method() === 'POST' && path.endsWith('/auth/logout')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Logged out successfully' }),
      })
    }

    return route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ detail: `Unhandled auth endpoint: ${request.method()} ${path}` }),
    })
  })

  await page.goto('/tests/e2e/fixtures/auth-context.html')
  const status = page.getByLabel('Authentication status')
  await expect(status).toHaveText('authenticated')

  await page.getByRole('button', { name: 'Refresh' }).click()
  await refreshPending
  await page.getByRole('button', { name: 'Logout' }).click()
  await expect(status).toHaveText('unauthenticated')

  releaseRefresh()
  await expect(page.getByLabel('Refresh settled count')).toHaveText('1')
  await expect(status).toHaveText('unauthenticated')
})

test('refresh 401 rejects with a typed error and clears the local session', async ({ page }) => {
  let markLoginNavigationStarted!: () => void
  let loginNavigationRequests = 0
  let loginNavigationWasDocument = false
  const loginNavigationStarted = new Promise<void>((resolve) => {
    markLoginNavigationStarted = resolve
  })
  await page.route('**/login?**', async (route) => {
    loginNavigationRequests += 1
    loginNavigationWasDocument = route.request().isNavigationRequest()
    markLoginNavigationStarted()
    await route.fulfill({ status: 204 })
  })
  await mockRefreshResult(page, 401)
  await page.goto('/tests/e2e/fixtures/auth-context.html')

  const status = page.getByLabel('Authentication status')
  await expect(status).toHaveText('authenticated')
  await page.getByRole('button', { name: 'Refresh' }).click()

  await loginNavigationStarted
  await expect(page.getByLabel('Refresh result')).toHaveText(
    'rejected:ApiError:http:401:/auth/me'
  )
  await expect(page.getByLabel('Refresh settled count')).toHaveText('1')
  await expect(status).toHaveText('unauthenticated')
  expect(loginNavigationRequests).toBe(1)
  expect(loginNavigationWasDocument).toBe(true)
})

test('refresh 503 rejects with a typed error without clearing the authenticated user', async ({
  page,
}) => {
  await mockRefreshResult(page, 503)
  await page.goto('/tests/e2e/fixtures/auth-context.html')

  const status = page.getByLabel('Authentication status')
  await expect(status).toHaveText('authenticated')
  await page.getByRole('button', { name: 'Refresh' }).click()

  await expect(page.getByLabel('Refresh result')).toHaveText(
    'rejected:ApiError:http:503:/auth/me'
  )
  await expect(page.getByLabel('Refresh settled count')).toHaveText('1')
  await expect(status).toHaveText('authenticated')
})

test('toApiError preserves an existing ApiError instance and all diagnostic fields', async ({
  page,
}) => {
  await mockRefreshResult(page, 200)
  await page.goto('/tests/e2e/fixtures/auth-context.html')

  await expect(page.getByLabel('ApiError identity')).toHaveText(
    'same:http:409:E_EXISTING:/auth/me:original-message'
  )
})
