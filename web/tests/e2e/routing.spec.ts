import { expect, test, type Page } from '@playwright/test'
import { mockApi, mockAuth } from './helpers/mockApi'

const DEEP_LINK = '/training/task-deep?tab=events#latest'
const DEFAULT_RETURN_TO = '/training'
const SENTINEL_ORIGIN = 'https://routing-sentinel.invalid'
const SENTINEL_HOSTNAME = new URL(SENTINEL_ORIGIN).hostname

function loginURLFor(rawRedirect: string) {
  return `/login?redirect=${encodeURIComponent(rawRedirect)}`
}

async function expectCurrentURL(page: Page, expected: string) {
  await expect
    .poll(() => {
      const current = new URL(page.url())
      return `${current.pathname}${current.search}${current.hash}`
    })
    .toBe(expected)
}

async function signIn(page: Page) {
  await page.getByLabel('Username').fill('alice')
  await page.getByLabel('Password', { exact: true }).fill('correct-password')
  await page.getByRole('button', { name: 'Sign in' }).click()
}

async function expectProtectedNotFound(
  page: Page,
  options: {
    heading: string
    navigationItem: string
    accountMenu: string
    backAction: string
    trainingAction: string
  }
) {
  const headings = page.getByRole('heading', { level: 1 })
  await expect(headings).toHaveCount(1)

  const heading = page.getByRole('heading', { level: 1, name: options.heading, exact: true })
  await expect(heading).toBeVisible()
  await expect(heading).toBeFocused()
  await expect(page.getByRole('img', { name: 'No Found', exact: true })).toHaveCount(0)
  await expect(page.getByRole('link', { name: 'TrainFactory', exact: true })).toBeVisible()
  const navigation = page.getByRole('menu')
  await expect(navigation).toBeVisible()
  await expect(navigation.getByText(options.navigationItem, { exact: true })).toBeVisible()
  await expect(
    page.getByRole('button', { name: options.accountMenu, exact: true })
  ).toBeVisible()
  await expect(page.getByRole('button', { name: options.backAction, exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: options.trainingAction, exact: true })).toBeVisible()

  return heading
}

async function holdTrainingListUnauthorized(page: Page) {
  let markStarted: (() => void) | undefined
  let releaseUnauthorized: (() => void) | undefined
  const started = new Promise<void>((resolve) => {
    markStarted = resolve
  })
  const responseGate = new Promise<void>((resolve) => {
    releaseUnauthorized = resolve
  })

  await page.route(
    (url) => url.pathname === '/api/train',
    async (route) => {
      markStarted?.()
      await responseGate
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Session expired' }),
      })
    },
    { times: 1 }
  )

  return {
    started,
    release: () => releaseUnauthorized?.(),
  }
}

async function expectEnglishTrainingList(page: Page) {
  await expect(
    page.getByRole('heading', { level: 4, name: 'Training Tasks', exact: true })
  ).toBeVisible()
  await expect(page.getByRole('heading', { level: 1 })).toHaveCount(0)
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

test('anonymous protected route keeps pathname, search, and hash in the login redirect', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false })

  await page.goto(DEEP_LINK)

  await expect
    .poll(() => {
      const current = new URL(page.url())
      return {
        pathname: current.pathname,
        redirect: current.searchParams.get('redirect'),
      }
    })
    .toEqual({ pathname: '/login', redirect: DEEP_LINK })
})

test('successful login returns to the original protected deep link', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto(DEEP_LINK)

  await signIn(page)

  await expectCurrentURL(page, DEEP_LINK)
})

test('authenticated login route honors a safe redirect', async ({ page }) => {
  await mockAuth(page, { authenticated: true })

  await page.goto(loginURLFor(DEEP_LINK))

  await expectCurrentURL(page, DEEP_LINK)
})

test('ordinary API 401 keeps pathname, search, and hash in the login redirect', async ({ page }) => {
  await mockAuth(page, { authenticated: true, meStatuses: [200, 401] })
  await page.route(
    (url) => url.pathname === '/api/train/task-deep',
    (route) =>
      route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Session expired' }),
      })
  )

  await page.goto(DEEP_LINK)

  await expect
    .poll(() => {
      const current = new URL(page.url())
      return {
        pathname: current.pathname,
        redirect: current.searchParams.get('redirect'),
      }
    })
    .toEqual({ pathname: '/login', redirect: DEEP_LINK })
})

for (const loginLikePath of ['/loginish?tab=events#latest', '/LOGIN%2F'] as const) {
  test(`ordinary API 401 redirects the non-login route ${JSON.stringify(loginLikePath)} with its complete URL`, async ({
    page,
  }) => {
    await mockAuth(page, { authenticated: true, meStatuses: [200, 401] })
    const unauthorized = await holdTrainingListUnauthorized(page)
    await page.goto('/training')
    await unauthorized.started
    await page.evaluate((path) => window.history.replaceState(null, '', path), loginLikePath)

    unauthorized.release()

    await expect
      .poll(() => {
        const current = new URL(page.url())
        return {
          pathname: current.pathname,
          redirect: current.searchParams.get('redirect'),
        }
      })
      .toEqual({ pathname: '/login', redirect: loginLikePath })
  })
}

for (const normalizedLoginPath of ['/login/', '/%6Cogin', '/x/%2e%2e/login'] as const) {
  test(`ordinary API 401 does not redirect the normalized login path ${JSON.stringify(normalizedLoginPath)}`, async ({
    page,
  }) => {
    const documentRequests: string[] = []
    let trackNavigation = false
    page.on('request', (request) => {
      if (
        trackNavigation &&
        request.isNavigationRequest() &&
        request.resourceType() === 'document' &&
        request.frame() === page.mainFrame()
      ) {
        documentRequests.push(request.url())
      }
    })
    await mockAuth(page, { authenticated: true })
    const unauthorized = await holdTrainingListUnauthorized(page)
    await page.goto('/training')
    await unauthorized.started
    await page.evaluate(
      (path) => window.history.replaceState(null, '', path),
      normalizedLoginPath
    )
    const expectedURL = await page.evaluate(
      () => `${window.location.pathname}${window.location.search}${window.location.hash}`
    )

    trackNavigation = true
    unauthorized.release()

    await expect(page.getByText('Session expired, please login again')).toBeVisible()
    await expectCurrentURL(page, expectedURL)
    expect(documentRequests).toEqual([])
  })
}

const unsafeRedirects = [
  `${SENTINEL_ORIGIN}/phish`,
  '//routing-sentinel.invalid/phish',
  '%2F%2Frouting-sentinel.invalid/phish',
  '/\\routing-sentinel.invalid/phish',
  '/%5Crouting-sentinel.invalid/phish',
  '/%ZZ',
  '/login/',
  '/login?redirect=%2Flogin',
  '/%6Cogin',
  '/LOGIN%2F',
  '/%2e/login',
  '/x/%2e%2e/login',
  '/%2e%2e//routing-sentinel.invalid/phish',
  '/foo/%2e%2e//routing-sentinel.invalid/phish',
  '/\t//routing-sentinel.invalid/phish',
  '/login//',
] as const

for (const rawRedirect of unsafeRedirects) {
  test(`unsafe redirect ${JSON.stringify(rawRedirect)} falls back without contacting sentinel`, async ({
    page,
  }) => {
    const sentinelRequests: string[] = []
    const postLoginDocumentRequests: string[] = []
    let trackPostLoginNavigation = false
    page.on('request', (request) => {
      if (new URL(request.url()).hostname === SENTINEL_HOSTNAME) {
        sentinelRequests.push(request.url())
      }
      if (
        trackPostLoginNavigation &&
        request.isNavigationRequest() &&
        request.resourceType() === 'document' &&
        request.frame() === page.mainFrame()
      ) {
        postLoginDocumentRequests.push(request.url())
      }
    })
    await page.route((url) => url.hostname === SENTINEL_HOSTNAME, (route) => route.abort())
    await mockAuth(page, { authenticated: false })
    await page.goto(loginURLFor(rawRedirect))
    await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible()

    trackPostLoginNavigation = true
    await signIn(page)

    await expectCurrentURL(page, DEFAULT_RETURN_TO)
    expect(sentinelRequests).toEqual([])
    // 登录成功后的整页跳转（浏览器才会弹出保存密码提示）仅允许发生在站内；
    // 外站/未知 origin 的整页导航仍然禁止。
    expect(
      postLoginDocumentRequests.every(
        (url) => new URL(url).origin === new URL(page.url()).origin
      )
    ).toBe(true)
  })
}

test('authenticated English 404 keeps the unknown URL and focuses every unknown pathname', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/unknown/first?source=routing#missing')

  await expectCurrentURL(page, '/unknown/first?source=routing#missing')
  const heading = await expectProtectedNotFound(page, {
    heading: '404 · Page not found',
    navigationItem: 'Training',
    accountMenu: 'Open account menu for runtime_verify, Regular account',
    backAction: 'Go back',
    trainingAction: 'Back to training',
  })

  await page.getByRole('button', {
    name: 'Open account menu for runtime_verify, Regular account',
  }).focus()
  await expect(heading).not.toBeFocused()
  await page.evaluate(() => {
    window.history.pushState(null, '', '/unknown/second?source=spa#missing-again')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })

  await expectCurrentURL(page, '/unknown/second?source=spa#missing-again')
  await expectProtectedNotFound(page, {
    heading: '404 · Page not found',
    navigationItem: 'Training',
    accountMenu: 'Open account menu for runtime_verify, Regular account',
    backAction: 'Go back',
    trainingAction: 'Back to training',
  })
})

test('authenticated Chinese 404 remains inside the protected application layout', async ({ page }) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/unknown/language-seed')
  await page.getByRole('button', { name: 'global 中', exact: true }).click()
  await page.evaluate(() => {
    window.history.pushState(null, '', '/unknown/zh?source=routing#missing')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })

  await expectCurrentURL(page, '/unknown/zh?source=routing#missing')
  await expectProtectedNotFound(page, {
    heading: '404 · 页面未找到',
    navigationItem: '训练管理',
    accountMenu: '打开 runtime_verify 的账户菜单，普通账户',
    backAction: '返回上一页',
    trainingAction: '返回训练列表',
  })
})

test('anonymous login returns to the complete unknown URL and its protected 404', async ({ page }) => {
  const unknownURL = '/unknown/after-login?source=auth#missing'
  await mockAuth(page, { authenticated: false })

  await page.goto(unknownURL)
  await expect
    .poll(() => {
      const current = new URL(page.url())
      return {
        pathname: current.pathname,
        redirect: current.searchParams.get('redirect'),
      }
    })
    .toEqual({ pathname: '/login', redirect: unknownURL })

  await signIn(page)

  await expectCurrentURL(page, unknownURL)
  await expectProtectedNotFound(page, {
    heading: '404 · Page not found',
    navigationItem: 'Training',
    accountMenu: 'Open account menu for runtime_verify, Regular account',
    backAction: 'Go back',
    trainingAction: 'Back to training',
  })
})

test('404 back button returns to the previous application page', async ({ page }) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/training')
  await expectCurrentURL(page, '/training')
  await page.goto('/unknown/history-entry')
  await expectProtectedNotFound(page, {
    heading: '404 · Page not found',
    navigationItem: 'Training',
    accountMenu: 'Open account menu for runtime_verify, Regular account',
    backAction: 'Go back',
    trainingAction: 'Back to training',
  })

  await page.getByRole('button', { name: 'Go back', exact: true }).click()

  await expectCurrentURL(page, '/training')
  await expectEnglishTrainingList(page)
})

test('404 training link returns to the training list', async ({ page }) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/unknown/training-link')
  await expectProtectedNotFound(page, {
    heading: '404 · Page not found',
    navigationItem: 'Training',
    accountMenu: 'Open account menu for runtime_verify, Regular account',
    backAction: 'Go back',
    trainingAction: 'Back to training',
  })

  await page.getByRole('link', { name: 'Back to training', exact: true }).click()

  await expectCurrentURL(page, '/training')
  await expectEnglishTrainingList(page)
})
