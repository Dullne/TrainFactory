import { test, expect, type Page } from '@playwright/test'
import { mockApi, mockAuth } from './helpers/mockApi'

const workflowAccessibleName =
  'TrainFactory workflow: data, generation, training, evaluation feedback, deployment, and service'

const signOutFailureMessage = 'Sign out failed. Your session is still active. Please retry.'

const credentialStoreEventsKey = 'tf_test_credential_store_events'

interface CredentialStoreEvent {
  id: string
  password: string
  pathname: string
  formPresent: boolean
}

async function installCredentialStoreMock(
  page: Page,
  outcome: 'resolve' | 'reject' | 'unsupported' = 'resolve'
) {
  await page.addInitScript(
    ({ storageKey, storeOutcome }) => {
      if (window.sessionStorage.getItem(storageKey) === null) {
        window.sessionStorage.setItem(storageKey, '[]')
      }

      if (storeOutcome === 'unsupported') {
        Object.defineProperty(window, 'PasswordCredential', {
          configurable: true,
          value: undefined,
        })
        Object.defineProperty(navigator, 'credentials', {
          configurable: true,
          value: {},
        })
        return
      }

      class TestPasswordCredential {
        readonly id: string
        readonly password: string

        constructor({ id, password }: { id: string; password: string }) {
          this.id = id
          this.password = password
        }
      }

      Object.defineProperty(window, 'PasswordCredential', {
        configurable: true,
        value: TestPasswordCredential,
      })
      Object.defineProperty(navigator, 'credentials', {
        configurable: true,
        value: {
          store: async (credential: TestPasswordCredential) => {
            const events = JSON.parse(
              window.sessionStorage.getItem(storageKey) ?? '[]'
            ) as CredentialStoreEvent[]
            events.push({
              id: credential.id,
              password: credential.password,
              pathname: window.location.pathname,
              formPresent: document.querySelector('form') !== null,
            })
            window.sessionStorage.setItem(storageKey, JSON.stringify(events))
            if (storeOutcome === 'reject') {
              throw new DOMException('Credential storage declined', 'NotAllowedError')
            }
            return credential
          },
        },
      })
    },
    { storageKey: credentialStoreEventsKey, storeOutcome: outcome }
  )
}

async function readCredentialStoreEvents(page: Page): Promise<CredentialStoreEvent[]> {
  return page.evaluate((storageKey) => {
    return JSON.parse(window.sessionStorage.getItem(storageKey) ?? '[]') as CredentialStoreEvent[]
  }, credentialStoreEventsKey)
}

async function openSignOutDialog(page: Page) {
  await page.goto('/training')
  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Sign out', { exact: true }).click()
  await expect(page.getByRole('dialog', { name: 'Sign out?' })).toBeVisible()
}

async function expectRetriableSignOutFailure(page: Page, timeout = 5000) {
  await expect(page).toHaveURL(/\/training$/)
  const dialog = page.getByRole('dialog', { name: 'Sign out?' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('alert')).toHaveCount(1, { timeout })
  await expect(dialog.getByRole('alert')).toHaveText(signOutFailureMessage)
  await expect(
    page.getByRole('button', {
      name: 'Open account menu for runtime_verify, Regular account',
    })
  ).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Confirm sign out' })).toBeEnabled()
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
}

async function expectLoginRedirect(page: Page, returnTo: string) {
  await expect
    .poll(() => {
      const current = new URL(page.url())
      return {
        pathname: current.pathname,
        redirect: current.searchParams.get('redirect'),
      }
    })
    .toEqual({ pathname: '/login', redirect: returnTo })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

test('unauthenticated user is redirected to login', async ({ page }) => {
  await mockAuth(page, { authenticated: false })

  await page.goto('/training')

  await expectLoginRedirect(page, '/training')
  await expect(page.getByRole('heading', { name: 'Model Training Workspace' })).toBeVisible()
})

test('denied legacy token storage does not prevent the app from mounting', async ({ page }) => {
  await page.addInitScript(() => {
    const originalGetItem = Storage.prototype.getItem
    const originalRemoveItem = Storage.prototype.removeItem

    Storage.prototype.getItem = function (key: string) {
      if (key === 'auth_token') throw new DOMException('Storage access denied', 'SecurityError')
      return originalGetItem.call(this, key)
    }
    Storage.prototype.removeItem = function (key: string) {
      if (key === 'auth_token') throw new DOMException('Storage access denied', 'SecurityError')
      return originalRemoveItem.call(this, key)
    }
  })
  await mockAuth(page, { authenticated: false })
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))

  await page.goto('/login')

  await expect(page.getByRole('heading', { name: 'Model Training Workspace' })).toBeVisible()
  expect(pageErrors).not.toContain('Storage access denied')
})

test('authenticated user cannot remain on login', async ({ page }) => {
  await mockAuth(page, { authenticated: true })

  await page.goto('/login')

  await expect(page).toHaveURL(/\/training$/)
})

test('initial unauthorized auth probe stays silent on the login page', async ({ page }) => {
  let meRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/me')) meRequests += 1
  })
  await mockAuth(page, { authenticated: false })

  await page.goto('/login')

  await expect.poll(() => meRequests).toBe(1)
  await expect(page.getByRole('heading', { name: 'Model Training Workspace' })).toBeVisible()
  await expect(page.getByText(/Session expired/i)).toHaveCount(0)
})

test('official brand replaces the standalone TF placeholder on desktop login', async ({ page }) => {
  await mockAuth(page, { authenticated: false })

  await page.goto('/login')

  const wordmark = page.locator('.trainfactory-wordmark[role="img"]')
  await expect.soft(wordmark).toHaveCount(1)
  await expect.soft(wordmark).toHaveAccessibleName('TrainFactory')
  await expect.soft(wordmark).toBeVisible()
  await expect.soft(page.getByText('TF', { exact: true })).toHaveCount(0)
})

test('official brand shows the complete TrainFactory workflow on desktop login', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false })

  await page.goto('/login')

  const workflow = page.getByRole('img', { name: workflowAccessibleName, exact: true })
  await expect.soft(workflow).toBeVisible()
  await expect
    .soft(workflow)
    .toHaveAttribute('src', /trainfactory-workflow(?:-[\w-]+)?\.svg(?:[?#].*)?$/)
  await expect.soft(workflow).toHaveJSProperty('complete', true)
  await expect.soft(workflow).toHaveJSProperty('naturalWidth', 960)
  await expect.soft(page.locator('.auth-workflow-step')).toHaveCount(0)
})

test('desktop brand content stays centered and scales across wide viewports', async ({ page }) => {
  await mockAuth(page, { authenticated: false, registrationEnabled: false })
  await page.goto('/login')

  for (const { viewportWidth, expectedContentWidth } of [
    { viewportWidth: 1280, expectedContentWidth: 520 },
    { viewportWidth: 1920, expectedContentWidth: 600 },
    { viewportWidth: 2048, expectedContentWidth: 640 },
  ]) {
    await page.setViewportSize({ width: viewportWidth, height: 900 })

    const identityBounds = await page.locator('.auth-identity').boundingBox()
    const identityContent = page.locator('.auth-identity-content')
    const contentBounds = await identityContent.boundingBox()
    const workflowBounds = await identityContent.locator('.auth-workflow-art').boundingBox()

    expect(identityBounds).not.toBeNull()
    expect(contentBounds).not.toBeNull()
    expect(workflowBounds).not.toBeNull()

    const identityCenter = identityBounds!.x + identityBounds!.width / 2
    const contentCenter = contentBounds!.x + contentBounds!.width / 2
    expect(Math.abs(contentCenter - identityCenter)).toBeLessThanOrEqual(1)
    expect(contentBounds!.width).toBeCloseTo(expectedContentWidth, 0)
    expect(workflowBounds!.x).toBeCloseTo(contentBounds!.x, 0)
    expect(workflowBounds!.width).toBeCloseTo(contentBounds!.width, 0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      viewportWidth
    )
  }

  for (const viewportWidth of [821, 980, 981]) {
    await page.setViewportSize({ width: viewportWidth, height: 900 })

    const identityBounds = await page.locator('.auth-identity').boundingBox()
    const contentBounds = await page.locator('.auth-identity-content').boundingBox()
    expect(identityBounds).not.toBeNull()
    expect(contentBounds).not.toBeNull()

    const identityCenter = identityBounds!.x + identityBounds!.width / 2
    const contentCenter = contentBounds!.x + contentBounds!.width / 2
    expect(Math.abs(contentCenter - identityCenter)).toBeLessThanOrEqual(1)
    expect(contentBounds!.x).toBeGreaterThanOrEqual(identityBounds!.x)
    expect(contentBounds!.x + contentBounds!.width).toBeLessThanOrEqual(
      identityBounds!.x + identityBounds!.width
    )
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      viewportWidth
    )
  }
})

test('desktop login panel preserves the intentional column divider', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/login')

  await expect(page.locator('.auth-panel')).toHaveCSS('border-left', '1px solid rgb(37, 45, 55)')
})

test('desktop login panel preserves its vertical scroll containment', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/login')

  await expect(page.locator('.auth-panel')).toHaveCSS('overflow-y', 'auto')
})

test('favicon serves the TrainFactory SVG mark', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  const favicon = page.locator('link[rel~="icon"]')
  await expect(favicon).toHaveCount(1)
  await expect(favicon).toHaveAttribute('href', '/trainfactory-mark.svg')

  const faviconHref = await favicon.getAttribute('href')
  expect(faviconHref).not.toBeNull()
  const response = await page.context().request.get(new URL(faviconHref!, page.url()).toString())

  expect(response.ok()).toBe(true)
  expect(response.headers()['content-type']).toMatch(/^image\/svg\+xml(?:;|$)/i)
})

test('registration control is completely hidden when self-registration is disabled', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false, registrationEnabled: false })

  await page.goto('/login')

  await expect(page.getByText('Register', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Create account' })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Sign in to workspace' })).toBeVisible()
})

test('login and registration expose standard password-manager field semantics', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  const form = page.locator('form')
  await expect(form).toHaveAttribute('autocomplete', 'on')
  await expect(form).toHaveAttribute('id', 'login-form')
  await expect(form).toHaveAttribute('name', 'login')
  await expect(page.getByLabel('Username')).toHaveAttribute('name', 'username')
  await expect(page.getByLabel('Username')).toHaveAttribute('autocomplete', 'username')
  await expect(page.getByLabel('Username')).toHaveAttribute('required', '')
  await expect(page.getByLabel('Password', { exact: true })).toHaveAttribute('name', 'password')
  await expect(page.getByLabel('Password', { exact: true })).toHaveAttribute(
    'autocomplete',
    'current-password'
  )
  await expect(page.getByLabel('Password', { exact: true })).toHaveAttribute('required', '')

  await page.getByText('Register', { exact: true }).click()
  await expect(form).toHaveAttribute('id', 'registration-form')
  await expect(form).toHaveAttribute('name', 'register')
  await expect(page.getByLabel('Email')).toHaveAttribute('name', 'email')
  await expect(page.getByLabel('Email')).toHaveAttribute('autocomplete', 'email')
  await expect(page.getByLabel('Password', { exact: true })).toHaveAttribute('name', 'password')
  await expect(page.getByLabel('Password', { exact: true })).toHaveAttribute(
    'autocomplete',
    'new-password'
  )
  await expect(page.getByLabel('Confirm password')).toHaveAttribute('name', 'confirmPassword')
  await expect(page.getByLabel('Confirm password')).toHaveAttribute('autocomplete', 'new-password')
  await expect(page.getByLabel('Confirm password')).toHaveAttribute('required', '')
})

test('native required semantics keep Ant form validation in control', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page.locator('#username_help')).toContainText('Required')
  await expect(page.locator('#password_help')).toContainText('Required')
})

test('enabled registration creates an account and refreshes the current user', async ({ page }) => {
  let meRequests = 0
  let documentLoginNavigations = 0
  let leakedCredentialQuery = false
  let registerBody: unknown = null
  page.on('request', (request) => {
    const url = new URL(request.url())
    const path = url.pathname
    if (path.endsWith('/api/auth/me')) meRequests += 1
    if (path.endsWith('/api/auth/register')) registerBody = request.postDataJSON()
    if (path === '/login' && request.resourceType() === 'document') {
      documentLoginNavigations += 1
      leakedCredentialQuery ||= ['username', 'email', 'password', 'confirmPassword'].some((name) =>
        url.searchParams.has(name)
      )
    }
  })
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('new-user')
  await page.getByLabel('Email').fill('new@example.com')
  await page.getByLabel('Password', { exact: true }).fill('LongPassword-2026')
  await page.getByLabel('Confirm password').fill('LongPassword-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page).toHaveURL(/\/training$/)
  // Initial probe + post-registration refresh + full-page redirect probe.
  await expect.poll(() => meRequests).toBe(3)
  expect(documentLoginNavigations).toBe(1)
  expect(leakedCredentialQuery).toBe(false)
  await expect
    .poll(async () =>
      (await page.context().cookies()).find((cookie) => cookie.name === 'access_token')
    )
    .toMatchObject({ value: 'mock-access-token', httpOnly: true })
  expect(registerBody).toEqual({
    username: 'new-user',
    email: 'new@example.com',
    password: 'LongPassword-2026',
  })
})

test('mismatched registration passwords are rejected before an API request', async ({ page }) => {
  let registerRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/register')) {
      registerRequests += 1
    }
  })
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('new-user')
  await page.getByLabel('Password', { exact: true }).fill('LongPassword-2026')
  await page.getByLabel('Confirm password').fill('DifferentPassword-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page.getByText('Passwords do not match')).toBeVisible()
  expect(registerRequests).toBe(0)
})

test('registration validates the trimmed username before sending a request', async ({ page }) => {
  let registerRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/register')) {
      registerRequests += 1
    }
  })
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('  a')
  await page.getByLabel('Password', { exact: true }).fill('LongPassword-2026')
  await page.getByLabel('Confirm password').fill('LongPassword-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page.getByText('Use 3 to 64 characters')).toBeVisible()
  expect(registerRequests).toBe(0)
})

test('duplicate username registration error stays on the username field', async ({ page }) => {
  await mockAuth(page, {
    authenticated: false,
    registrationEnabled: true,
    registerStatus: 409,
  })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('existing-user')
  await page.getByLabel('Password', { exact: true }).fill('LongPassword-2026')
  await page.getByLabel('Confirm password').fill('LongPassword-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.locator('#username_help')).toContainText('Username or email already exists')
  await expect(page.locator('.auth-error')).toHaveCount(0)
})

test('duplicate email registration error stays on submitted identity fields', async ({ page }) => {
  await mockAuth(page, {
    authenticated: false,
    registrationEnabled: true,
    registerStatus: 409,
  })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('new-user')
  await page.getByLabel('Email').fill('existing@example.com')
  await page.getByLabel('Password', { exact: true }).fill('LongPassword-2026')
  await page.getByLabel('Confirm password').fill('LongPassword-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.locator('#username_help')).toContainText('Username or email already exists')
  await expect(page.locator('#email_help')).toContainText('Username or email already exists')
  await expect(page.locator('.auth-error')).toHaveCount(0)
})

test('mobile login uses stable margins without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockAuth(page, { authenticated: false, registrationEnabled: false })
  await page.goto('/login')

  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  const workflow = page.getByRole('img', { name: workflowAccessibleName, exact: true })
  await expect(workflow).toBeInViewport()
  const workflowBounds = await workflow.boundingBox()
  expect(workflowBounds).not.toBeNull()
  expect(workflowBounds!.x).toBeGreaterThanOrEqual(0)
  expect(workflowBounds!.x + workflowBounds!.width).toBeLessThanOrEqual(390)

  const authForm = page.getByTestId('auth-form')
  await expect(authForm).toBeInViewport()
  const formBounds = await authForm.boundingBox()
  expect(formBounds).not.toBeNull()
  expect(formBounds!.x).toBeGreaterThanOrEqual(24)
  expect(formBounds!.x + formBounds!.width).toBeLessThanOrEqual(366)

  for (const control of await page.locator('.auth-language .ant-btn').all()) {
    const bounds = await control.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.height).toBeGreaterThanOrEqual(44)
  }

  for (const control of await page.locator('.auth-form-shell .ant-input-affix-wrapper').all()) {
    const bounds = await control.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.height).toBeGreaterThanOrEqual(44)
  }
})

test('320px registration keeps the official workflow and primary action usable', async ({
  page,
}) => {
  await page.setViewportSize({ width: 320, height: 568 })
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.evaluate(() => window.scrollTo(0, 0))

  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(320)
  const workflow = page.getByRole('img', { name: workflowAccessibleName, exact: true })
  await expect(workflow).toBeInViewport()
  const workflowBounds = await workflow.boundingBox()
  expect(workflowBounds).not.toBeNull()
  expect(workflowBounds!.x).toBeGreaterThanOrEqual(0)
  expect(workflowBounds!.x + workflowBounds!.width).toBeLessThanOrEqual(320)

  const primaryAction = page.getByRole('button', { name: 'Create account' })
  await primaryAction.scrollIntoViewIfNeeded()
  await expect(primaryAction).toBeVisible()
  await expect(primaryAction).toBeInViewport()
  const primaryActionBounds = await primaryAction.boundingBox()
  expect(primaryActionBounds).not.toBeNull()
  expect(primaryActionBounds!.x).toBeGreaterThanOrEqual(0)
  expect(primaryActionBounds!.x + primaryActionBounds!.width).toBeLessThanOrEqual(320)
})

test('mobile registration mode controls provide 44px touch targets', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  const modeItems = page.locator('.auth-mode-switch .ant-segmented-item')
  await expect(modeItems).toHaveCount(2)
  for (const modeItem of await modeItems.all()) {
    const bounds = await modeItem.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.height).toBeGreaterThanOrEqual(44)
  }
})

test('password visibility can be operated from the keyboard', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  const password = page.getByLabel('Password', { exact: true })
  await password.fill('keyboard-secret')
  await expect(password).toHaveAttribute('type', 'password')

  const visibilityToggle = page.getByRole('button', { name: /password/ })
  await visibilityToggle.focus()
  await expect(visibilityToggle).toBeFocused()
  await visibilityToggle.press('Enter')
  await expect(password).toHaveAttribute('type', 'text')
  await expect(visibilityToggle).toHaveAttribute('aria-pressed', 'true')

  await visibilityToggle.press('Space')
  await expect(password).toHaveAttribute('type', 'password')
  await expect(visibilityToggle).toHaveAttribute('aria-pressed', 'false')
})

test('configuration failure shows unavailable state and keeps registration hidden', async ({
  page,
}) => {
  await mockAuth(page, {
    authenticated: false,
    registrationEnabled: true,
    configStatus: 503,
  })

  await page.goto('/login')

  await expect(page.getByText('Service unavailable', { exact: true })).toBeVisible()
  await expect(page.getByText('Register', { exact: true })).toHaveCount(0)
})

test('login failure shows the backend error inline without a session-expired message', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false, loginStatus: 401 })
  await page.goto('/login')

  await page.getByLabel('Username').fill('alice')
  await page.getByLabel('Password', { exact: true }).fill('wrong-password')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole('alert')).toContainText('Invalid username or password')
  await expect(page.getByText(/Session expired/i)).toHaveCount(0)
})

test('failed user refresh after login uses normal session-expired handling', async ({ page }) => {
  let meRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/me')) meRequests += 1
  })
  await mockAuth(page, { authenticated: false, meStatuses: [401, 401] })
  await page.goto('/login')

  await page.getByLabel('Username').fill('alice')
  await page.getByLabel('Password', { exact: true }).fill('correct-password')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect.poll(() => meRequests).toBe(2)
  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByText('Session expired, please login again')).toBeVisible()
})

test('successful login refreshes the current user before entering training', async ({ page }) => {
  let meRequests = 0
  let documentLoginNavigations = 0
  let leakedCredentialQuery = false
  let loginBody: unknown = null
  page.on('request', (request) => {
    const url = new URL(request.url())
    const path = url.pathname
    if (path.endsWith('/api/auth/me')) meRequests += 1
    if (path.endsWith('/api/auth/login')) loginBody = request.postDataJSON()
    if (path === '/login' && request.resourceType() === 'document') {
      documentLoginNavigations += 1
      leakedCredentialQuery ||= ['username', 'password'].some((name) => url.searchParams.has(name))
    }
  })
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await page.getByLabel('Username').fill('alice')
  await page.getByLabel('Password', { exact: true }).fill('correct-password')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/training$/)
  await expect(page.getByRole('heading', { name: 'Training Tasks' })).toBeVisible()
  // Initial probe + post-login refresh + full-page redirect probe.
  expect(meRequests).toBe(3)
  expect(documentLoginNavigations).toBe(1)
  expect(leakedCredentialQuery).toBe(false)
  expect(loginBody).toEqual({ username: 'alice', password: 'correct-password' })
  await expect
    .poll(async () =>
      (await page.context().cookies()).find((cookie) => cookie.name === 'access_token')
    )
    .toMatchObject({ value: 'mock-access-token', httpOnly: true })
  await expect.poll(() => page.evaluate(() => localStorage.getItem('auth_token'))).toBeNull()
})

test('successful login replaces the login history entry', async ({ page }) => {
  await page.route('**/history-sentinel', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'text/html',
      body: '<!doctype html><title>History sentinel</title><p>History sentinel</p>',
    })
  })
  await installCredentialStoreMock(page)
  await mockAuth(page, { authenticated: false })

  await page.goto('/history-sentinel')
  await page.goto('/login')
  await page.getByLabel('Username').fill('history-user')
  await page.getByLabel('Password', { exact: true }).fill('HistoryPass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/training$/)
  await page.goBack({ waitUntil: 'domcontentloaded' })
  await expect(page).toHaveURL(/\/history-sentinel$/)
})

test('successful login offers the submitted credential before leaving the login document', async ({
  page,
}) => {
  await installCredentialStoreMock(page)
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await page.getByLabel('Username').fill('credential-user')
  await page.getByLabel('Password', { exact: true }).fill('CredentialStorePass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/training$/)
  expect(await readCredentialStoreEvents(page)).toEqual([
    {
      id: 'credential-user',
      password: 'CredentialStorePass-2026',
      pathname: '/login',
      formPresent: true,
    },
  ])
})

test('failed login does not offer credentials for storage', async ({ page }) => {
  await installCredentialStoreMock(page)
  await mockAuth(page, { authenticated: false, loginStatus: 401 })
  await page.goto('/login')

  await page.getByLabel('Username').fill('credential-user')
  await page.getByLabel('Password', { exact: true }).fill('RejectedLoginPass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole('alert')).toContainText('Invalid username or password')
  expect(await readCredentialStoreEvents(page)).toEqual([])
})

test('failed session confirmation does not offer credentials for storage', async ({ page }) => {
  await installCredentialStoreMock(page)
  await mockAuth(page, { authenticated: false, meStatuses: [401, 401] })
  await page.goto('/login')

  await page.getByLabel('Username').fill('credential-user')
  await page.getByLabel('Password', { exact: true }).fill('SessionRejectedPass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole('alert')).toContainText('Not authenticated')
  expect(await readCredentialStoreEvents(page)).toEqual([])
})

test('declining credential storage does not block a successful login', async ({ page }) => {
  await installCredentialStoreMock(page, 'reject')
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await page.getByLabel('Username').fill('credential-user')
  await page.getByLabel('Password', { exact: true }).fill('DeclinedStorePass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/training$/)
  await expect(page.getByRole('heading', { name: 'Training Tasks' })).toBeVisible()
  expect(await readCredentialStoreEvents(page)).toHaveLength(1)
})

test('a browser without credential storage support still completes login', async ({ page }) => {
  await installCredentialStoreMock(page, 'unsupported')
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await page.getByLabel('Username').fill('credential-user')
  await page.getByLabel('Password', { exact: true }).fill('UnsupportedStorePass-2026')
  await page.getByRole('button', { name: 'Sign in' }).click()

  await expect(page).toHaveURL(/\/training$/)
  expect(await readCredentialStoreEvents(page)).toEqual([])
})

test('registration never offers the new password for login credential storage', async ({
  page,
}) => {
  await installCredentialStoreMock(page)
  await mockAuth(page, { authenticated: false, registrationEnabled: true })
  await page.goto('/login')

  await page.getByText('Register', { exact: true }).click()
  await page.getByLabel('Username').fill('registration-user')
  await page.getByLabel('Password', { exact: true }).fill('RegistrationPass-2026')
  await page.getByLabel('Confirm password').fill('RegistrationPass-2026')
  await page.getByRole('button', { name: 'Create account' }).click()

  await expect(page).toHaveURL(/\/training$/)
  expect(await readCredentialStoreEvents(page)).toEqual([])
})

test('administrator account menu exposes identity and requires confirmed sign out', async ({
  page,
}) => {
  let logoutRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/logout')) {
      logoutRequests += 1
    }
  })
  await mockAuth(page, { authenticated: true, isAdmin: true })
  await page.goto('/training')

  const accountMenu = page.getByRole('button', { name: /admin/i })
  await expect(accountMenu).toBeVisible()
  await expect(accountMenu).toHaveAccessibleName(
    'Open account menu for admin, Administrator account'
  )
  await expect(accountMenu).toContainText('Administrator account')
  await accountMenu.click()
  await expect(
    page.locator('.ant-dropdown-menu').getByText('Administrator account', { exact: true })
  ).toBeVisible()
  await expect(page.getByText('User management')).toBeVisible()
  await expect(page.getByText('Change password')).toBeVisible()
  await expect(page.getByText('Sign out')).toBeVisible()

  await page.getByText('Sign out').click()
  await expect(page.getByRole('dialog', { name: 'Sign out?' })).toBeVisible()
  await page.getByRole('button', { name: 'Cancel' }).click()
  await expect(page).toHaveURL(/\/training$/)
  expect(logoutRequests).toBe(0)

  await accountMenu.click()
  await page.getByText('Sign out').click()
  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expect(page).toHaveURL(/\/login$/)
  expect(logoutRequests).toBe(1)
})

test('regular account menu does not expose user management', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training')

  const accountMenu = page.getByRole('button', { name: /runtime_verify/i })
  await expect(accountMenu).toHaveAccessibleName(
    'Open account menu for runtime_verify, Regular account'
  )
  await expect(accountMenu).toContainText('Regular account')
  await accountMenu.click()
  await expect(
    page.locator('.ant-dropdown-menu').getByText('Regular account', { exact: true })
  ).toBeVisible()
  await expect(page.getByText('Change password')).toBeVisible()
  await expect(page.getByText('User management')).toHaveCount(0)
})

test('desktop navigation collapse control is keyboard accessible', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training')

  const collapseNavigation = page.getByRole('button', { name: 'Collapse navigation' })
  await collapseNavigation.focus()
  await expect(collapseNavigation).toBeFocused()
  await collapseNavigation.press('Enter')

  await expect(page.getByRole('button', { name: 'Expand navigation' })).toBeVisible()
  await expect(page.locator('.ant-layout-sider')).toHaveClass(/ant-layout-sider-collapsed/)
})

test('desktop brand is a keyboard-operable link to the training workspace', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training/create')

  const brandLink = page.getByRole('link', { name: 'TrainFactory' })
  await expect(brandLink).toHaveAttribute('href', '/training')
  await brandLink.focus()
  await expect(brandLink).toBeFocused()
  await brandLink.press('Enter')

  await expect(page).toHaveURL(/\/training$/)
})

test('desktop breadcrumb navigation exposes native keyboard-operable links', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training/create')

  const homeLink = page.getByRole('link', { name: 'Home' })
  const trainingLink = page.getByRole('link', { name: 'Training' })
  await expect(homeLink).toHaveAttribute('href', '/training')
  await expect(trainingLink).toHaveAttribute('href', '/training')
  await trainingLink.focus()
  await expect(trainingLink).toBeFocused()
  await trainingLink.press('Enter')

  await expect(page).toHaveURL(/\/training$/)
})

test('administrator section breadcrumb target resolves to user management', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: true })
  await page.goto('/admin')

  await expect(page).toHaveURL(/\/admin\/users$/)
  await expect(page.getByRole('heading', { name: 'User management' })).toBeVisible()
})

test('768px administrator layout keeps a long account identity inside the header', async ({
  page,
}) => {
  await page.setViewportSize({ width: 768, height: 844 })
  await mockAuth(page, {
    authenticated: true,
    isAdmin: true,
    username: 'administrator_with_a_long_name',
  })
  await page.goto('/admin/users')

  await expect(page.getByRole('heading', { name: 'User management' })).toBeVisible()

  const header = page.locator('.main-layout-header')
  const accountMenu = page.getByRole('button', {
    name: /Open account menu for administrator_with_a_long_name/,
  })
  const breadcrumb = page.locator('.main-layout-breadcrumb')
  const headerBounds = await header.boundingBox()
  const accountBounds = await accountMenu.boundingBox()
  expect(headerBounds).not.toBeNull()
  expect(accountBounds).not.toBeNull()
  expect(accountBounds!.y).toBeGreaterThanOrEqual(headerBounds!.y)
  expect(accountBounds!.y + accountBounds!.height).toBeLessThanOrEqual(
    headerBounds!.y + headerBounds!.height
  )
  if (await breadcrumb.isVisible()) {
    const breadcrumbBounds = await breadcrumb.boundingBox()
    expect(breadcrumbBounds).not.toBeNull()
    expect(breadcrumbBounds!.y).toBeGreaterThanOrEqual(headerBounds!.y)
    expect(breadcrumbBounds!.y + breadcrumbBounds!.height).toBeLessThanOrEqual(
      headerBounds!.y + headerBounds!.height
    )
  }
  await expect(page.getByRole('button', { name: 'Open navigation' })).toBeVisible()
  await expect(page.locator('.ant-layout-sider')).toHaveCount(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(768)
})

for (const width of [320, 390, 768]) {
  test(`training list toolbar remains readable at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 })
    await mockAuth(page, { authenticated: true, isAdmin: false })
    await page.goto('/training')

    const title = page.getByRole('heading', { name: 'Training Tasks' })
    const statusFilter = page.locator('.main-layout-content .ant-select').first()
    const refreshButton = page.getByRole('button', { name: 'Refresh' })
    const createButton = page.getByRole('button', { name: 'Create Task' })
    const titleBounds = await title.boundingBox()
    const filterBounds = await statusFilter.boundingBox()
    const refreshBounds = await refreshButton.boundingBox()
    const createBounds = await createButton.boundingBox()

    expect(titleBounds).not.toBeNull()
    expect(titleBounds!.width).toBeGreaterThanOrEqual(120)
    expect(titleBounds!.height).toBeLessThanOrEqual(40)
    expect(filterBounds).not.toBeNull()
    expect(refreshBounds).not.toBeNull()
    expect(createBounds).not.toBeNull()
    expect(filterBounds!.width).toBeGreaterThanOrEqual(width - 64)
    expect(filterBounds!.y).toBeGreaterThanOrEqual(titleBounds!.y + titleBounds!.height + 8)
    expect(refreshBounds!.y).toBeGreaterThanOrEqual(filterBounds!.y + filterBounds!.height + 8)
    expect(createBounds!.y).toBe(refreshBounds!.y)
    expect(refreshBounds!.x + refreshBounds!.width).toBeLessThanOrEqual(createBounds!.x)
    expect(createBounds!.x + createBounds!.width).toBeLessThanOrEqual(width - 20)

    const statWidths = await page
      .locator('.training-list-stats > .ant-col')
      .evaluateAll((columns) => columns.map((column) => column.getBoundingClientRect().width))
    expect(statWidths).toHaveLength(4)
    expect(Math.min(...statWidths)).toBeGreaterThanOrEqual(width === 320 ? 130 : 160)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width
    )
  })
}

test('training list toolbar stays on one row at desktop width', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 })
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training')

  const title = page.getByRole('heading', { name: 'Training Tasks' })
  const statusFilter = page.locator('.main-layout-content .ant-select').first()
  const createButton = page.getByRole('button', { name: 'Create Task' })
  const titleBounds = await title.boundingBox()
  const filterBounds = await statusFilter.boundingBox()
  const createBounds = await createButton.boundingBox()
  expect(titleBounds).not.toBeNull()
  expect(titleBounds!.width).toBeGreaterThanOrEqual(120)
  expect(filterBounds).not.toBeNull()
  expect(createBounds).not.toBeNull()
  expect(Math.abs(titleBounds!.y - filterBounds!.y)).toBeLessThanOrEqual(8)
  expect(createBounds!.x + createBounds!.width).toBeLessThanOrEqual(1244)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1280)
})

test('mobile actions use cards and fixed table actions mask horizontally scrolled content', async ({
  page,
}) => {
  const taskName = 'qwen3-embedding-0.6b-post-deploy-validation'
  const now = '2026-08-08T12:00:00'

  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.route('**/api/train*', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    if (request.method() !== 'GET' || !path.endsWith('/api/train')) {
      await route.fallback()
      return
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: [
          {
            task_id: 'd0e9f783-1234-5678-90ab-cdef12345678',
            task_name: taskName,
            model_type: 'embedding',
            training_method: 'sft',
            base_model_path: 'Qwen/Qwen3-Embedding-0.6B',
            status: 'succeeded',
            progress: 100,
            train_loss: 0.0004,
            gpu_ids: [0],
            created_at: now,
            updated_at: now,
          },
        ],
        total: 1,
        stats: {
          total: 1,
          pending: 0,
          running: 0,
          succeeded: 1,
          failed: 0,
          stopped: 0,
        },
      }),
    })
  })

  for (const width of [320, 390, 768, 1280]) {
    await page.setViewportSize({ width, height: 844 })
    await page.goto('/training')

    if (width < 576) {
      const card = page.locator('.list-workspace-cards article').filter({ hasText: taskName })
      await expect(card).toBeVisible()
      await expect(card.getByRole('button').first()).toBeVisible()
      await expect(page.locator('.ant-table')).toHaveCount(0)
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        width
      )
      continue
    }

    const row = page.locator('.ant-table-tbody > tr.ant-table-row').filter({ hasText: taskName })
    await expect(row).toBeVisible()
    await row
      .locator('td')
      .first()
      .hover({ position: { x: 4, y: 4 } })

    const overlap = await row.evaluate((element) => {
      const actionCell = element.querySelector<HTMLElement>('td.ant-table-cell-fix-right')
      if (!actionCell) throw new Error('Expected a fixed Actions cell')

      const actionRect = actionCell.getBoundingClientRect()
      const actionBackground = getComputedStyle(actionCell).backgroundColor
      const alphaMatch = actionBackground.match(/^rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\)$/)
      const actionBackgroundAlpha = alphaMatch ? Number(alphaMatch[1]) : 1
      const actionButtons = Array.from(actionCell.querySelectorAll<HTMLElement>('button')).map(
        (button) => button.getBoundingClientRect()
      )
      const candidates = Array.from(
        element.querySelectorAll<HTMLElement>(
          'td:not(.ant-table-cell-fix-right) .ant-tag, td:not(.ant-table-cell-fix-right) .ant-typography'
        )
      )

      const intersects = (first: DOMRect, second: DOMRect) =>
        Math.max(first.left, second.left) < Math.min(first.right, second.right) &&
        Math.max(first.top, second.top) < Math.min(first.bottom, second.bottom)

      const overlappingLabels = candidates
        .filter((candidate) => intersects(candidate.getBoundingClientRect(), actionRect))
        .map((candidate) => candidate.textContent?.trim() || candidate.className)
      const iconOverlapLabels = candidates
        .filter((candidate) => {
          const candidateRect = candidate.getBoundingClientRect()
          return actionButtons.some((buttonRect) => intersects(candidateRect, buttonRect))
        })
        .map((candidate) => candidate.textContent?.trim() || candidate.className)

      return {
        actionBackground,
        actionBackgroundAlpha,
        actionRect: {
          left: actionRect.left,
          right: actionRect.right,
        },
        overlappingLabels,
        iconOverlapLabels,
        visibleLeaks: actionBackgroundAlpha < 1 ? overlappingLabels : [],
      }
    })

    expect(
      overlap.visibleLeaks,
      `${width}px fixed Actions cell ${JSON.stringify(overlap)}`
    ).toEqual([])
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width
    )
  }
})

for (const account of [
  { label: 'administrator', isAdmin: true, username: 'admin' },
  { label: 'regular user', isAdmin: false, username: 'runtime_verify' },
]) {
  test(`mobile ${account.label} can use navigation, account menu, and sign out`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockAuth(page, { authenticated: true, isAdmin: account.isAdmin })
    await page.goto('/training')

    await expect(page.getByRole('heading', { name: 'Training Tasks' })).toBeVisible()
    expect(
      await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
      }))
    ).toEqual({ clientWidth: 390, scrollWidth: 390 })

    const openNavigation = page.getByRole('button', { name: 'Open navigation' })
    await expect(openNavigation).toBeVisible()
    await openNavigation.click()

    const navigation = page.getByRole('dialog', { name: 'Main navigation' })
    await expect(navigation).toBeVisible()
    await navigation.getByRole('button', { name: 'Close navigation' }).click()
    await expect(navigation).toBeHidden()

    const accountMenu = page.getByRole('button', {
      name: new RegExp(`Open account menu for ${account.username}`),
    })
    await expect(accountMenu).toBeVisible()
    const accountBounds = await accountMenu.boundingBox()
    expect(accountBounds).not.toBeNull()
    expect(accountBounds!.x).toBeGreaterThanOrEqual(0)
    expect(accountBounds!.x + accountBounds!.width).toBeLessThanOrEqual(390)

    await accountMenu.click()
    const dropdown = page.locator('.ant-dropdown-menu')
    await expect(dropdown).toBeVisible()
    const dropdownBounds = await dropdown.boundingBox()
    expect(dropdownBounds).not.toBeNull()
    expect(dropdownBounds!.x).toBeGreaterThanOrEqual(0)
    expect(dropdownBounds!.x + dropdownBounds!.width).toBeLessThanOrEqual(390)

    await page.getByText('Sign out').click()
    await page.getByRole('button', { name: 'Confirm sign out' }).click()
    await expect(page).toHaveURL(/\/login$/)
  })
}

test('confirmed sign out locks duplicate confirmation events while the request is pending', async ({
  page,
}) => {
  let logoutRequests = 0
  let releaseLogout: (() => void) | undefined
  const logoutResponseGate = new Promise<void>((resolve) => {
    releaseLogout = resolve
  })
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.route('**/api/auth/logout', async (route) => {
    logoutRequests += 1
    await logoutResponseGate
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ message: 'Logged out successfully' }),
    })
  })
  await page.goto('/training')

  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Sign out').click()
  const confirmButton = page.getByRole('button', { name: 'Confirm sign out' })
  await confirmButton.evaluate((button) => {
    const confirm = () =>
      button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
    confirm()
    confirm()
  })

  await expect.poll(() => logoutRequests).toBeGreaterThan(0)
  await page.waitForTimeout(200)
  expect(logoutRequests).toBe(1)
  await expect(confirmButton).toBeDisabled()

  releaseLogout?.()
  await expect(page).toHaveURL(/\/login$/)
  expect(logoutRequests).toBe(1)
})

test('password change validates confirmation and length before requesting the API', async ({
  page,
}) => {
  let changePasswordRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/change-password')) {
      changePasswordRequests += 1
    }
  })
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.goto('/training')

  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Change password').click()
  await page.getByLabel('Current password', { exact: true }).fill('OldPassword-2026')
  await page.getByLabel('New password', { exact: true }).fill('NewPassword-2026')
  await page.getByLabel('Confirm new password', { exact: true }).fill('DifferentPassword-2026')
  await page.getByRole('button', { name: 'Update password' }).click()
  await expect(page.getByText('Passwords do not match')).toBeVisible()
  expect(changePasswordRequests).toBe(0)

  await page.getByLabel('New password', { exact: true }).fill('short-1')
  await page.getByLabel('Confirm new password', { exact: true }).fill('short-1')
  await page.getByRole('button', { name: 'Update password' }).click()
  await expect(
    page.locator('#newPassword_help').getByText('Use 10 to 128 characters')
  ).toBeVisible()
  expect(changePasswordRequests).toBe(0)

  const tooLongPassword = 'x'.repeat(129)
  await page.getByLabel('New password', { exact: true }).fill(tooLongPassword)
  await page.getByLabel('Confirm new password', { exact: true }).fill(tooLongPassword)
  await page.getByRole('button', { name: 'Update password' }).click()
  await expect(
    page.locator('#newPassword_help').getByText('Use 10 to 128 characters')
  ).toBeVisible()
  expect(changePasswordRequests).toBe(0)
})

test('successful password change synchronously locks duplicate submits and requires a fresh login', async ({
  page,
}) => {
  let changePasswordRequests = 0
  let requestBody: unknown = null
  let releaseChangePassword: (() => void) | undefined
  const changePasswordResponseGate = new Promise<void>((resolve) => {
    releaseChangePassword = resolve
  })
  await mockAuth(page, {
    authenticated: true,
    isAdmin: false,
    logoutStatus: 503,
  })
  await page.route('**/api/auth/change-password', async (route) => {
    changePasswordRequests += 1
    requestBody = route.request().postDataJSON()
    await changePasswordResponseGate
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        message: 'Password changed successfully. Please log in again.',
      }),
    })
  })
  await page.goto('/training')

  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Change password').click()
  await page.getByLabel('Current password', { exact: true }).fill('OldPassword-2026')
  await page.getByLabel('New password', { exact: true }).fill('NewPassword-2026')
  await page.getByLabel('Confirm new password', { exact: true }).fill('NewPassword-2026')
  const updateButton = page.getByRole('button', { name: 'Update password' })
  await page
    .getByRole('dialog', { name: 'Change password' })
    .locator('form')
    .evaluate((form) => {
      const submitEvent = () =>
        form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
      submitEvent()
      submitEvent()
    })

  await expect.poll(() => changePasswordRequests).toBeGreaterThan(0)
  await page.waitForTimeout(200)
  expect(changePasswordRequests).toBe(1)
  await expect(updateButton).toBeDisabled()
  await expect(page.getByLabel('Current password', { exact: true })).toBeDisabled()
  expect(requestBody).toEqual({
    old_password: 'OldPassword-2026',
    new_password: 'NewPassword-2026',
  })

  releaseChangePassword?.()
  await expect(page).toHaveURL(/\/login$/)
  expect(changePasswordRequests).toBe(1)
})

test('logout 401 clears the local session without a retry error', async ({ page }) => {
  let logoutRequests = 0
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.endsWith('/api/auth/logout')) {
      logoutRequests += 1
    }
  })
  await mockAuth(page, { authenticated: true, isAdmin: false, logoutStatus: 401 })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expect(page.getByRole('dialog', { name: 'Sign out?' })).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(
    page.getByRole('button', {
      name: 'Open account menu for runtime_verify, Regular account',
    })
  ).toHaveCount(0)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
  expect(logoutRequests).toBe(1)
})

test('logout 403 keeps the session and offers retry', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false, logoutStatus: 403 })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expectRetriableSignOutFailure(page)
})

test('logout 503 keeps the session and offers retry', async ({ page }) => {
  await mockAuth(page, { authenticated: true, isAdmin: false, logoutStatus: 503 })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expectRetriableSignOutFailure(page)
})

test('logout timeout keeps the session and offers retry', async ({ page }) => {
  test.setTimeout(45000)
  let logoutRequests = 0
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.route('**/api/auth/logout', () => {
    logoutRequests += 1
    // Stay pending for both the short dev timeout and the production default.
  })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expectRetriableSignOutFailure(page, 35000)
  expect(logoutRequests).toBe(1)
})

test('logout network abort keeps the session and offers retry', async ({ page }) => {
  let logoutRequests = 0
  await mockAuth(page, { authenticated: true, isAdmin: false })
  await page.route('**/api/auth/logout', async (route) => {
    logoutRequests += 1
    await route.abort('connectionfailed')
  })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expectRetriableSignOutFailure(page)
  expect(logoutRequests).toBe(1)
})

test('logout failure renders one localized inline error and no global toast', async ({ page }) => {
  const backendDetail = 'server-only logout diagnostic must not be rendered'
  await mockAuth(page, {
    authenticated: true,
    isAdmin: false,
    logoutStatus: 503,
    logoutErrorDetail: backendDetail,
  })
  await openSignOutDialog(page)

  await page.getByRole('button', { name: 'Confirm sign out' }).click()

  await expectRetriableSignOutFailure(page)
  await expect(page.getByText(backendDetail, { exact: true })).toHaveCount(0)
})

test('password change API errors stay inline without closing the modal', async ({ page }) => {
  await mockAuth(page, {
    authenticated: true,
    isAdmin: false,
    changePasswordStatus: 400,
  })
  await page.goto('/training')

  await page.getByRole('button', { name: /runtime_verify/i }).click()
  await page.getByText('Change password').click()
  await page.getByLabel('Current password', { exact: true }).fill('WrongPassword-2026')
  await page.getByLabel('New password', { exact: true }).fill('NewPassword-2026')
  await page.getByLabel('Confirm new password', { exact: true }).fill('NewPassword-2026')
  await page.getByRole('button', { name: 'Update password' }).click()

  await expect(page.getByRole('alert')).toContainText('Current password is incorrect')
  await expect(page.getByRole('dialog', { name: 'Change password' })).toBeVisible()
  await expect(page).toHaveURL(/\/training$/)
})

test.describe('administrator user management', () => {
  const adminAccount = {
    user_id: 'admin-1',
    username: 'admin',
    email: 'admin@example.com',
    is_active: true,
    is_admin: true,
    created_at: '2026-08-08T12:00:00',
    updated_at: '2026-08-08T12:00:00',
  }

  const operatorAccount = {
    user_id: 'user-2',
    username: 'operator',
    email: 'operator@example.com',
    is_active: true,
    is_admin: false,
    created_at: '2026-08-07T09:30:00',
    updated_at: '2026-08-07T09:30:00',
  }

  test('administrator opens the dense account table from the account menu', async ({ page }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
    })
    await page.goto('/training')

    await page.getByRole('button', { name: /admin.*administrator account/i }).click()
    await page.getByText('User management', { exact: true }).click()

    await expect(page).toHaveURL(/\/admin\/users$/)
    await expect(page.getByRole('heading', { name: 'User management' })).toBeVisible()
    await expect(page.locator('.admin-users-toolbar').getByText('2 accounts')).toBeVisible()
    for (const heading of ['Username', 'Email', 'Role', 'Status', 'Created', 'Actions']) {
      await expect(page.getByRole('columnheader', { name: heading })).toBeVisible()
    }
    await expect(page.getByRole('row').filter({ hasText: 'operator@example.com' })).toBeVisible()
    await expect(page.getByRole('button', { name: /delete/i })).toHaveCount(0)
  })

  test('regular user is redirected before the administrator page fetches accounts', async ({
    page,
  }) => {
    let adminListRequests = 0
    page.on('request', (request) => {
      const url = new URL(request.url())
      if (request.method() === 'GET' && url.pathname.endsWith('/api/auth/admin/users')) {
        adminListRequests += 1
      }
    })
    await mockAuth(page, { authenticated: true, isAdmin: false })

    await page.goto('/admin/users')

    await expect(page).toHaveURL(/\/training$/)
    await expect(page.getByRole('heading', { name: 'User management' })).toHaveCount(0)
    expect(adminListRequests).toBe(0)
  })

  test('administrator creates a regular account and sees it in the table', async ({ page }) => {
    let createBody: unknown = null
    page.on('request', (request) => {
      const url = new URL(request.url())
      if (request.method() === 'POST' && url.pathname.endsWith('/api/auth/admin/users')) {
        createBody = request.postDataJSON()
      }
    })
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount],
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Create account' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create account' })
    await dialog.getByLabel('Username').fill('operator')
    await dialog.getByLabel('Email').fill('operator@example.com')
    await dialog.getByLabel('Password', { exact: true }).fill('OperatorPass-2026')
    await dialog.getByRole('button', { name: 'Create' }).click()

    await expect(dialog).toHaveCount(0)
    await expect(page.getByRole('row').filter({ hasText: 'operator@example.com' })).toBeVisible()
    expect(createBody).toEqual({
      username: 'operator',
      email: 'operator@example.com',
      password: 'OperatorPass-2026',
      is_admin: false,
    })
  })

  test('a list response started before account creation cannot remove the new account', async ({
    page,
  }) => {
    let markListRequested!: () => void
    let releaseList!: () => void
    let staleListCompleted = false
    const listRequested = new Promise<void>((resolve) => {
      markListRequested = resolve
    })
    const listReleased = new Promise<void>((resolve) => {
      releaseList = resolve
    })
    const createdAccount = {
      ...operatorAccount,
      user_id: 'user-created',
      username: 'new-operator',
      email: 'new-operator@example.com',
    }

    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount],
    })
    await page.route('**/api/auth/admin/users*', async (route) => {
      const request = route.request()
      const path = new URL(request.url()).pathname
      if (!path.endsWith('/api/auth/admin/users')) return route.fallback()

      if (request.method() === 'GET') {
        markListRequested()
        await listReleased
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            users: [adminAccount],
            total: 1,
            limit: 20,
            offset: 0,
          }),
        })
        staleListCompleted = true
        return
      }

      if (request.method() === 'POST') {
        return route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify(createdAccount),
        })
      }

      return route.fallback()
    })
    await page.goto('/admin/users')
    await listRequested

    await page.getByRole('button', { name: 'Create account' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create account' })
    await dialog.getByLabel('Username').fill('new-operator')
    await dialog.getByLabel('Email').fill('new-operator@example.com')
    await dialog.getByLabel('Password', { exact: true }).fill('NewOperatorPass-2026')
    await dialog.getByRole('button', { name: 'Create' }).click()
    await expect(
      page.getByRole('row').filter({ hasText: 'new-operator@example.com' })
    ).toBeVisible()

    releaseList()
    await expect.poll(() => staleListCompleted).toBe(true)
    await expect(
      page.getByRole('row').filter({ hasText: 'new-operator@example.com' })
    ).toBeVisible()
  })

  test('duplicate account conflicts stay on the username and email fields', async ({ page }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount],
      adminCreateStatus: 409,
      adminActionError: 'Username or email already exists',
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Create account' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create account' })
    await dialog.getByLabel('Username').fill('admin')
    await dialog.getByLabel('Email').fill('admin@example.com')
    await dialog.getByLabel('Password', { exact: true }).fill('DuplicatePass-2026')
    await dialog.getByRole('button', { name: 'Create' }).click()

    await expect(dialog).toBeVisible()
    await expect(dialog.locator('#username_help')).toContainText('Username or email already exists')
    await expect(dialog.locator('#email_help')).toContainText('Username or email already exists')
  })

  test('administrator confirms status and role changes', async ({ page }) => {
    const patchBodies: unknown[] = []
    page.on('request', (request) => {
      const url = new URL(request.url())
      if (request.method() === 'PATCH' && /\/api\/auth\/admin\/users\/[^/]+$/.test(url.pathname)) {
        patchBodies.push(request.postDataJSON())
      }
    })
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
    })
    await page.goto('/admin/users')

    const operatorRow = page.getByRole('row').filter({ hasText: 'operator@example.com' })
    await operatorRow.getByRole('button', { name: 'Disable operator' }).click()
    await page.getByRole('button', { name: 'Confirm disable' }).click()
    await expect(operatorRow.getByText('Disabled', { exact: true })).toBeVisible()

    await operatorRow.getByRole('button', { name: 'Promote operator' }).click()
    await page.getByRole('button', { name: 'Confirm promotion' }).click()
    await expect(operatorRow.getByText('Administrator', { exact: true })).toBeVisible()
    expect(patchBodies).toEqual([{ is_active: false }, { is_admin: true }])
  })

  test('a list refresh started before an account mutation cannot undo the mutation', async ({
    page,
  }) => {
    let listRequestCount = 0
    let markStaleListRequested!: () => void
    let releaseStaleList!: () => void
    let staleListCompleted = false
    const staleListRequested = new Promise<void>((resolve) => {
      markStaleListRequested = resolve
    })
    const staleListReleased = new Promise<void>((resolve) => {
      releaseStaleList = resolve
    })
    const originalAccounts = [adminAccount, operatorAccount]

    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: originalAccounts,
    })
    await page.route('**/api/auth/admin/users*', async (route) => {
      const request = route.request()
      const path = new URL(request.url()).pathname

      if (request.method() === 'GET' && path.endsWith('/api/auth/admin/users')) {
        listRequestCount += 1
        if (listRequestCount === 2) {
          markStaleListRequested()
          await staleListReleased
        }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            users: originalAccounts,
            total: originalAccounts.length,
            limit: 20,
            offset: 0,
          }),
        })
        if (listRequestCount === 2) staleListCompleted = true
        return
      }

      if (
        request.method() === 'PATCH' &&
        path.endsWith(`/api/auth/admin/users/${operatorAccount.user_id}`)
      ) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ...operatorAccount, is_active: false }),
        })
      }

      return route.fallback()
    })
    await page.goto('/admin/users')

    const operatorRow = page.getByRole('row').filter({ hasText: 'operator@example.com' })
    await expect(operatorRow.getByText('Active', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: /Refresh/ }).click()
    await staleListRequested
    await operatorRow.getByRole('button', { name: 'Disable operator' }).dispatchEvent('click')
    await page.getByRole('button', { name: 'Confirm disable' }).click()
    await expect(operatorRow.getByText('Disabled', { exact: true })).toBeVisible()

    releaseStaleList()
    await expect.poll(() => staleListCompleted).toBe(true)
    await expect(operatorRow.getByText('Disabled', { exact: true })).toBeVisible()
  })

  test('administrator resets another account password with confirmation validation', async ({
    page,
  }) => {
    let resetBody: unknown = null
    page.on('request', (request) => {
      const url = new URL(request.url())
      if (request.method() === 'POST' && url.pathname.endsWith('/reset-password')) {
        resetBody = request.postDataJSON()
      }
    })
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Reset password for operator' }).click()
    const dialog = page.getByRole('dialog', { name: 'Reset password for operator' })
    await expect(dialog).toContainText(
      'Resetting this password signs out existing sessions for this account.'
    )
    await dialog.getByLabel('New password', { exact: true }).fill('TemporaryPass-2026')
    await dialog.getByLabel('Confirm new password', { exact: true }).fill('DifferentPass-2026')
    await dialog.getByRole('button', { name: 'Reset password' }).click()
    await expect(dialog.getByText('Passwords do not match')).toBeVisible()
    expect(resetBody).toBeNull()

    await dialog.getByLabel('Confirm new password', { exact: true }).fill('TemporaryPass-2026')
    await dialog.getByRole('button', { name: 'Reset password' }).click()
    await expect(dialog).toHaveCount(0)
    expect(resetBody).toEqual({ new_password: 'TemporaryPass-2026' })
  })

  test('password-reset API errors stay inline and keep the dialog open', async ({ page }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
      adminResetStatus: 400,
      adminActionError: 'Unable to reset password',
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Reset password for operator' }).click()
    const dialog = page.getByRole('dialog', { name: 'Reset password for operator' })
    await dialog.getByLabel('New password', { exact: true }).fill('TemporaryPass-2026')
    await dialog.getByLabel('Confirm new password', { exact: true }).fill('TemporaryPass-2026')
    await dialog.getByRole('button', { name: 'Reset password' }).click()

    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('alert')).toContainText('Unable to reset password')
  })

  test('self-demotion refreshes identity and exits the administrator page', async ({ page }) => {
    const secondAdministrator = {
      ...operatorAccount,
      user_id: 'admin-2',
      username: 'manager',
      email: 'manager@example.com',
      is_admin: true,
    }
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, secondAdministrator],
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Demote admin' }).click()
    await page.getByRole('button', { name: 'Confirm demotion' }).click()

    await expect(page).toHaveURL(/\/training$/)
    await expect(
      page.getByRole('button', {
        name: 'Open account menu for admin, Regular account',
      })
    ).toBeVisible()
  })

  test('resetting the current administrator password ends the browser session', async ({
    page,
  }) => {
    let logoutRequests = 0
    page.on('request', (request) => {
      if (new URL(request.url()).pathname.endsWith('/api/auth/logout')) {
        logoutRequests += 1
      }
    })
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount],
      logoutStatus: 503,
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Reset password for admin' }).click()
    const dialog = page.getByRole('dialog', { name: 'Reset password for admin' })
    await dialog.getByLabel('New password', { exact: true }).fill('NewAdminPass-2026')
    await dialog.getByLabel('Confirm new password', { exact: true }).fill('NewAdminPass-2026')
    await dialog.getByRole('button', { name: 'Reset password' }).click()

    await expect(page).toHaveURL(/\/login$/)
    expect(logoutRequests).toBe(1)
  })

  test('creating from a later page returns to page one and shows the new account', async ({
    page,
  }) => {
    const accounts = Array.from({ length: 21 }, (_, index) => ({
      user_id: index === 0 ? 'admin-1' : `user-${index + 1}`,
      username: index === 0 ? 'admin' : `operator-${index}`,
      email: index === 0 ? 'admin@example.com' : `operator-${index}@example.com`,
      is_active: true,
      is_admin: index === 0,
      created_at: '2026-08-08T12:00:00',
      updated_at: '2026-08-08T12:00:00',
    }))
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: accounts,
    })
    await page.goto('/admin/users')
    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByRole('row').filter({ hasText: 'operator-20@example.com' })).toBeVisible()

    await page.getByRole('button', { name: 'Create account' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create account' })
    await dialog.getByLabel('Username').fill('newest-operator')
    await dialog.getByLabel('Email').fill('newest@example.com')
    await dialog.getByLabel('Password', { exact: true }).fill('NewestPass-2026')
    await dialog.getByRole('button', { name: 'Create' }).click()

    await expect(page.locator('.ant-pagination-item-1')).toHaveClass(/ant-pagination-item-active/)
    await expect(page.getByRole('row').filter({ hasText: 'newest@example.com' })).toBeVisible()
  })

  test('password visibility controls are disabled while administrator commands are pending', async ({
    page,
  }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
      adminCreateDelayMs: 500,
      adminResetDelayMs: 500,
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Create account' }).click()
    const createDialog = page.getByRole('dialog', { name: 'Create account' })
    await createDialog.getByLabel('Username').fill('delayed-user')
    await createDialog.getByLabel('Password', { exact: true }).fill('DelayedPass-2026')
    await createDialog.getByRole('button', { name: 'Create' }).click()
    await expect(createDialog.getByRole('button', { name: 'Show password' })).toBeDisabled()
    await expect(createDialog).toHaveCount(0)

    await page.getByRole('button', { name: 'Reset password for operator' }).click()
    const resetDialog = page.getByRole('dialog', { name: 'Reset password for operator' })
    await resetDialog.getByLabel('New password', { exact: true }).fill('DelayedReset-2026')
    await resetDialog.getByLabel('Confirm new password', { exact: true }).fill('DelayedReset-2026')
    await resetDialog.getByRole('button', { name: 'Reset password' }).click()
    await expect(resetDialog.getByRole('button', { name: 'Show new password' })).toBeDisabled()
    await expect(
      resetDialog.getByRole('button', { name: 'Show confirmation password' })
    ).toBeDisabled()
  })

  test('self-deactivation is unavailable and no hard-delete control is rendered', async ({
    page,
  }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount, operatorAccount],
    })
    await page.goto('/admin/users')

    const selfRow = page.getByRole('row').filter({ hasText: 'admin@example.com' })
    await expect(selfRow.getByRole('button', { name: 'Disable admin' })).toBeDisabled()
    await expect(page.getByRole('button', { name: /delete/i })).toHaveCount(0)
  })

  test('last-administrator conflict stays inline in the open role dialog', async ({ page }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [adminAccount],
      adminPatchStatus: 409,
      adminActionError: 'Account update conflicts with safety rules',
    })
    await page.goto('/admin/users')

    await page.getByRole('button', { name: 'Demote admin' }).click()
    const dialog = page.getByRole('dialog', { name: 'Demote admin' })
    await dialog.getByRole('button', { name: 'Confirm demotion' }).click()

    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('alert')).toContainText(
      'Account update conflicts with safety rules'
    )
    await expect(page.getByRole('row').filter({ hasText: 'admin@example.com' })).toContainText(
      'Administrator'
    )
  })

  test('account list exposes loading, empty, error, and retry states', async ({ page }) => {
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: [],
      adminListStatuses: [503, 200],
      adminListDelayMs: 300,
    })
    await page.goto('/admin/users')

    await expect(page.getByRole('status', { name: 'Loading accounts' })).toBeVisible()
    await expect(page.getByRole('alert')).toContainText('Unable to load accounts')
    await page.getByRole('button', { name: 'Retry loading accounts' }).click()
    await expect(page.getByText('No user accounts')).toBeVisible()
    await expect(page.locator('.admin-users-toolbar').getByText('0 accounts')).toBeVisible()
  })

  test('server pagination sends stable limit and offset values', async ({ page }) => {
    const listQueries: Array<{ limit: string | null; offset: string | null }> = []
    const accounts = Array.from({ length: 21 }, (_, index) => ({
      user_id: index === 0 ? 'admin-1' : `user-${index + 1}`,
      username: index === 0 ? 'admin' : `operator-${index}`,
      email: index === 0 ? 'admin@example.com' : `operator-${index}@example.com`,
      is_active: true,
      is_admin: index === 0,
      created_at: `2026-08-${String(8 - Math.min(index, 7)).padStart(2, '0')}T12:00:00`,
      updated_at: '2026-08-08T12:00:00',
    }))
    page.on('request', (request) => {
      const url = new URL(request.url())
      if (request.method() === 'GET' && url.pathname.endsWith('/api/auth/admin/users')) {
        listQueries.push({
          limit: url.searchParams.get('limit'),
          offset: url.searchParams.get('offset'),
        })
      }
    })
    await mockAuth(page, {
      authenticated: true,
      isAdmin: true,
      adminUsers: accounts,
    })
    await page.goto('/admin/users')

    await expect(page.getByRole('row').filter({ hasText: 'operator-19@example.com' })).toBeVisible()
    await page.locator('.ant-pagination-item-2').click()
    await expect(page.getByRole('row').filter({ hasText: 'operator-20@example.com' })).toBeVisible()
    expect(listQueries).toContainEqual({ limit: '20', offset: '0' })
    expect(listQueries).toContainEqual({ limit: '20', offset: '20' })
  })
})
