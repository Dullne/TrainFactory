import { test, expect, type Locator, type Page } from '@playwright/test'
import { mockApi, mockAuth } from './helpers/mockApi'

const appearanceStorageKey = 'tf_appearance:v1'
type ColorMode = 'dark' | 'light'
type AccentColor = 'blue' | 'purple' | 'cyan' | 'green' | 'rose'
type VisualStyle = 'classic' | 'workbench' | 'studio'

async function openAppearance(page: Page) {
  const trigger = page.getByRole('button', { name: 'Appearance', exact: true })
  await expect(trigger).toBeVisible()
  if ((await trigger.getAttribute('aria-expanded')) !== 'true') {
    await trigger.click()
  }
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')
  await expect(page.getByRole('group', { name: 'Color mode', exact: true })).toBeVisible()
  await expect(
    page.getByRole('group', { name: 'Accent color', exact: true, includeHidden: true })
  ).toBeAttached()
  for (const style of ['Classic Tech', 'Minimal Workbench', 'Soft Studio']) {
    await expect(page.getByRole('radio', { name: style, exact: true })).toBeVisible()
  }
}

async function closeAppearance(page: Page) {
  await page.keyboard.press('Escape')
  const trigger = page.getByRole('button', { name: 'Appearance', exact: true })
  await expect(trigger).toHaveAttribute('aria-expanded', 'false')
  await expect(trigger).toBeFocused()
  await expect(page.getByRole('group', { name: 'Color mode', exact: true })).toBeHidden()
  await expect(
    page.getByRole('group', { name: 'Accent color', exact: true, includeHidden: true })
  ).toBeHidden()
}

async function openAdvancedCustomization(page: Page) {
  const details = page.locator('.theme-picker details')
  if ((await details.count()) === 0) return
  if ((await details.getAttribute('open')) === null) {
    await details.locator('summary').click()
  }
  await expect(details).toHaveAttribute('open', '')
}

async function chooseStyle(page: Page, label: string) {
  await openAppearance(page)
  const radio = page.getByRole('radio', { name: label, exact: true })
  await radio.locator('xpath=ancestor::label[1]').click()
  await expect(radio).toBeChecked()
}

async function chooseAppearance(page: Page, mode: 'Dark' | 'Light', accent: string) {
  await openAppearance(page)
  await page
    .getByRole('group', { name: 'Color mode', exact: true })
    .getByRole('radio', { name: mode, exact: true })
    .check()
  await openAppearance(page)
  await openAdvancedCustomization(page)
  await page
    .getByRole('group', { name: 'Accent color', exact: true })
    .getByRole('radio', { name: accent, exact: true })
    .check()
}

async function expectAppearance(
  page: Page,
  mode: ColorMode,
  accent: AccentColor,
  style: VisualStyle = 'classic'
) {
  await expect(page.locator('html')).toHaveAttribute('data-theme', mode)
  await expect(page.locator('html')).toHaveAttribute('data-accent', accent)
  await expect(page.locator('html')).toHaveAttribute('data-style', style)
}

async function expectStoredAppearance(
  page: Page,
  mode: ColorMode,
  accent: AccentColor,
  style: VisualStyle = 'classic'
) {
  await expect
    .poll(() =>
      page.evaluate((key) => JSON.parse(localStorage.getItem(key) ?? 'null'), appearanceStorageKey)
    )
    .toEqual({ mode, accent, style })
}

interface CardVisualSignature {
  borderRadius: string
  bodyPadding: string
  backgroundColor: string
  boxShadow: string
}

async function cardVisualSignature(card: Locator): Promise<CardVisualSignature> {
  return card.evaluate((element) => {
    const cardStyle = getComputedStyle(element)
    const body = element.querySelector<HTMLElement>('.ant-card-body')
    if (!body) throw new Error('Expected the rendered card to have an .ant-card-body')
    return {
      borderRadius: cardStyle.borderRadius,
      bodyPadding: getComputedStyle(body).padding,
      backgroundColor: cardStyle.backgroundColor,
      boxShadow: cardStyle.boxShadow,
    }
  })
}

async function backgroundColor(locator: Locator) {
  return locator.evaluate((element) => getComputedStyle(element).backgroundColor)
}

async function expectWithinViewport(locator: Locator, width: number, height: number) {
  await expect(locator).toBeInViewport()
  // Popovers align to the viewport after their opening animation and measurement.
  await expect
    .poll(async () => {
      const bounds = await locator.boundingBox()
      return bounds
        ? {
            left: bounds.x >= 0,
            top: bounds.y >= 0,
            right: bounds.x + bounds.width <= width,
            bottom: bounds.y + bounds.height <= height,
          }
        : null
    })
    .toEqual({ left: true, top: true, right: true, bottom: true })
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

test('login defaults to dark blue classic and exposes every appearance option', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await expectAppearance(page, 'dark', 'blue')
  await openAppearance(page)

  const modes = page.getByRole('group', { name: 'Color mode', exact: true })
  await expect(modes.getByRole('radio')).toHaveCount(2)
  await expect(modes.getByRole('radio', { name: 'Dark', exact: true })).toBeChecked()
  await expect(modes.getByRole('radio', { name: 'Light', exact: true })).toBeVisible()

  for (const style of ['Classic Tech', 'Minimal Workbench', 'Soft Studio']) {
    await expect(page.getByRole('radio', { name: style, exact: true })).toBeVisible()
  }
  await expect(page.getByRole('radio', { name: 'Classic Tech', exact: true })).toBeChecked()

  await expect(page.getByText('Advanced customization', { exact: true })).toBeVisible()
  await openAdvancedCustomization(page)
  const accents = page.getByRole('group', { name: 'Accent color', exact: true })
  await expect(accents.getByRole('radio')).toHaveCount(5)
  await expect(accents.getByRole('radio', { name: 'Blue', exact: true })).toBeChecked()
  for (const accent of ['Purple', 'Cyan', 'Green', 'Rose']) {
    await expect(accents.getByRole('radio', { name: accent, exact: true })).toBeVisible()
  }
})

test('style presets render meaningfully different card geometry, spacing, surfaces, and shadows', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/training/create')

  const card = page.locator('.training-create-section.ant-card').first()
  await expect(card).toBeVisible()
  const signatures: Record<VisualStyle, CardVisualSignature> = {
    classic: await cardVisualSignature(card),
    workbench: {} as CardVisualSignature,
    studio: {} as CardVisualSignature,
  }

  await openAppearance(page)
  await chooseStyle(page, 'Minimal Workbench')
  await expectAppearance(page, 'dark', 'blue', 'workbench')
  signatures.workbench = await cardVisualSignature(card)

  await chooseStyle(page, 'Soft Studio')
  await expectAppearance(page, 'dark', 'blue', 'studio')
  signatures.studio = await cardVisualSignature(card)

  for (const property of ['borderRadius', 'bodyPadding', 'backgroundColor'] as const) {
    expect(
      new Set([
        signatures.classic[property],
        signatures.workbench[property],
        signatures.studio[property],
      ]).size,
      `${property} should visibly distinguish all three style presets`
    ).toBe(3)
  }
  expect(signatures.classic.boxShadow).toBe('none')
  expect(signatures.workbench.boxShadow).toBe('none')
  expect(signatures.studio.boxShadow).not.toBe('none')

  const screenshotDir = process.env.TF_THEME_SCREENSHOT_DIR
  if (screenshotDir) {
    await page.setViewportSize({ width: 1440, height: 1000 })
    const englishTrigger = page.getByRole('button', { name: 'Appearance', exact: true })
    if ((await englishTrigger.getAttribute('aria-expanded')) === 'true') {
      await page.keyboard.press('Escape')
    }
    await expect(page.locator('.ant-popover:visible')).toHaveCount(0)
    await page.getByRole('button', { name: 'global 中', exact: true }).click()
    await expect(page.getByRole('heading', { name: '创建训练任务', exact: true })).toBeVisible()
    const styles = [
      ['classic', '经典科技'],
      ['workbench', '极简工作台'],
      ['studio', '柔和工作室'],
    ] as const
    for (const [style, label] of styles) {
      const trigger = page.getByRole('button', { name: '外观', exact: true })
      if ((await trigger.getAttribute('aria-expanded')) !== 'true') {
        await trigger.click()
      }
      const styleRadio = page.getByRole('radio', { name: label, exact: true })
      await styleRadio.locator('xpath=ancestor::label[1]').click()
      await expect(styleRadio).toBeChecked()
      for (const [mode, modeLabel] of [
        ['dark', '深色'],
        ['light', '浅色'],
      ] as const) {
        if ((await trigger.getAttribute('aria-expanded')) !== 'true') {
          await trigger.click()
        }
        await page.getByRole('radio', { name: modeLabel, exact: true }).check()
        await expectAppearance(page, mode, 'blue', style)
        if ((await trigger.getAttribute('aria-expanded')) === 'true') {
          await page.keyboard.press('Escape')
        }
        await expect(page.locator('.ant-popover:visible')).toHaveCount(0)
        await page.getByRole('heading', { name: '创建训练任务', exact: true }).click()
        await page.screenshot({
          path: `${screenshotDir}/training-${style}-${mode}.png`,
          animations: 'disabled',
        })
      }
    }
  }
})

test('switching style presets preserves entered form data', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')
  const username = page.getByLabel('Username', { exact: true })
  const password = page.getByLabel('Password', { exact: true })
  await username.fill('draft-user')
  await password.fill('unsaved-password')

  await openAppearance(page)
  await chooseStyle(page, 'Minimal Workbench')
  await expectAppearance(page, 'dark', 'blue', 'workbench')
  await expect(username).toHaveValue('draft-user')
  await expect(password).toHaveValue('unsaved-password')

  await chooseStyle(page, 'Soft Studio')
  await expectAppearance(page, 'dark', 'blue', 'studio')
  await expect(username).toHaveValue('draft-user')
  await expect(password).toHaveValue('unsaved-password')
})

test('switching style preserves an in-progress training form, route, and collapsed section', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: true })
  await page.goto('/training/create')
  const taskName = page.locator('#task_name')
  const epochs = page.locator('#num_train_epochs')
  const trainingSection = page.getByRole('button', {
    name: 'Training Parameters',
    exact: true,
  })
  await taskName.fill('theme-preserves-training-draft')
  await epochs.fill('7')
  await trainingSection.click()
  await expect(trainingSection).toHaveAttribute('aria-expanded', 'false')
  await expect(epochs).toBeHidden()

  await openAppearance(page)
  await chooseStyle(page, 'Soft Studio')

  await expectAppearance(page, 'dark', 'blue', 'studio')
  await expect(page).toHaveURL(/\/training\/create$/)
  await expect(taskName).toHaveValue('theme-preserves-training-draft')
  await expect(epochs).toHaveValue('7')
  await expect(trainingSection).toHaveAttribute('aria-expanded', 'false')
  await expect(epochs).toBeHidden()
})

test('light purple changes rendered login colors immediately and survives reload', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  const surface = page.locator('.auth-page')
  const signIn = page.getByRole('button', { name: 'Sign in', exact: true })
  await expect(signIn).toBeVisible()
  const darkBackground = await backgroundColor(surface)
  const blueButton = await backgroundColor(signIn)

  await openAppearance(page)
  await chooseStyle(page, 'Soft Studio')
  await expectAppearance(page, 'dark', 'blue', 'studio')
  await openAdvancedCustomization(page)
  await page.getByRole('radio', { name: 'Purple', exact: true }).check()
  await expectAppearance(page, 'dark', 'purple', 'studio')
  await expect.poll(() => backgroundColor(signIn)).not.toBe(blueButton)
  await page.getByRole('radio', { name: 'Light', exact: true }).check()

  await expectAppearance(page, 'light', 'purple', 'studio')
  await expect.poll(() => backgroundColor(surface)).not.toBe(darkBackground)
  await expect.poll(() => backgroundColor(signIn)).not.toBe(blueButton)
  await expectStoredAppearance(page, 'light', 'purple', 'studio')

  // Both surfaces must remain visibly painted, including after a fresh document load.
  await expect(surface).not.toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
  await expect(signIn).not.toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
  await page.reload()

  await expectAppearance(page, 'light', 'purple', 'studio')
  await expect.poll(() => backgroundColor(surface)).not.toBe(darkBackground)
  await expect.poll(() => backgroundColor(signIn)).not.toBe(blueButton)
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Soft Studio', exact: true })).toBeChecked()
  await expect(page.getByRole('radio', { name: 'Light', exact: true })).toBeChecked()
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Purple', exact: true })).toBeChecked()
})

test('keyboard arrows select a style preset and Escape returns focus to the entry', async ({
  page,
}) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')
  const trigger = page.getByRole('button', { name: 'Appearance', exact: true })
  await trigger.press('Enter')
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')

  const classic = page.getByRole('radio', { name: 'Classic Tech', exact: true })
  const workbench = page.getByRole('radio', { name: 'Minimal Workbench', exact: true })
  await expect(classic).toBeFocused()
  await page.keyboard.press('ArrowRight')
  await expect(workbench).toBeChecked()
  await expect(workbench).toBeFocused()
  await expectAppearance(page, 'dark', 'blue', 'workbench')
  await expectStoredAppearance(page, 'dark', 'blue', 'workbench')
  await closeAppearance(page)
})

test('appearance is shared by login, authenticated navigation, and sign out', async ({ page }) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')
  await openAppearance(page)
  await chooseAppearance(page, 'Light', 'Purple')
  await closeAppearance(page)

  await page.getByLabel('Username', { exact: true }).fill('runtime_verify')
  await page.getByLabel('Password', { exact: true }).fill('correct-password')
  await page.getByRole('button', { name: 'Sign in', exact: true }).click()

  await expect(page).toHaveURL(/\/training$/)
  await expect(page.getByRole('heading', { name: 'Training Tasks', exact: true })).toBeVisible()
  await expectAppearance(page, 'light', 'purple')
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Light', exact: true })).toBeChecked()
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Purple', exact: true })).toBeChecked()
  await chooseAppearance(page, 'Dark', 'Cyan')
  await closeAppearance(page)

  await page.getByRole('menuitem', { name: /Datasets$/ }).click()
  await expect(page).toHaveURL(/\/datasets$/)
  await expectAppearance(page, 'dark', 'cyan')
  await expectStoredAppearance(page, 'dark', 'cyan')
  await openAppearance(page)
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Cyan', exact: true })).toBeChecked()
  await closeAppearance(page)

  await page
    .getByRole('button', { name: 'Open account menu for runtime_verify, Regular account' })
    .click()
  await page.getByText('Sign out', { exact: true }).click()
  await page.getByRole('button', { name: 'Confirm sign out', exact: true }).click()

  await expect(page).toHaveURL(/\/login$/)
  await expectAppearance(page, 'dark', 'cyan')
  await expectStoredAppearance(page, 'dark', 'cyan')
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Dark', exact: true })).toBeChecked()
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Cyan', exact: true })).toBeChecked()
})

for (const route of ['/login', '/training']) {
  test(`390px ${route} keeps the appearance entry and options inside the viewport`, async ({
    page,
  }, testInfo) => {
    const viewport = { width: 390, height: 844 }
    await page.setViewportSize(viewport)
    await mockAuth(page, { authenticated: route !== '/login' })
    await page.goto(route)

    await expectWithinViewport(
      page.getByRole('button', { name: 'Appearance', exact: true }),
      viewport.width,
      viewport.height
    )
    await openAppearance(page)
    const popover = page.locator('.ant-popover').filter({
      has: page.getByRole('group', { name: 'Color mode', exact: true }),
    })
    await expectWithinViewport(popover, viewport.width, viewport.height)
    await openAdvancedCustomization(page)
    if (route === '/training' && process.env.TF_THEME_SCREENSHOT_DIR) {
      await page.screenshot({
        path: `${process.env.TF_THEME_SCREENSHOT_DIR}/picker-mobile-en.png`,
        animations: 'disabled',
      })
      await testInfo.attach('mobile appearance picker', {
        path: `${process.env.TF_THEME_SCREENSHOT_DIR}/picker-mobile-en.png`,
        contentType: 'image/png',
      })
    }
    for (const radio of await popover.getByRole('radio').all()) {
      // The option label is the visible tap target even when the native radio is visually hidden.
      const label = radio.locator('xpath=ancestor::label[1]')
      await expectWithinViewport(label, viewport.width, viewport.height)
    }
    await chooseAppearance(page, 'Light', 'Rose')
    await expectAppearance(page, 'light', 'rose')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      viewport.width
    )
  })
}

test('Chinese mobile workspace keeps the appearance panel and all options inside the viewport', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockAuth(page, { authenticated: true })
  await page.goto('/training')
  await page.getByRole('button', { name: 'global 中', exact: true }).click()
  await page.getByRole('button', { name: '外观', exact: true }).click()
  const popover = page.locator('.ant-popover').filter({
    has: page.getByRole('group', { name: '显示模式', exact: true }),
  })
  await expectWithinViewport(popover, 390, 844)
  for (const style of ['经典科技', '极简工作台', '柔和工作室']) {
    await expect(page.getByRole('radio', { name: style, exact: true })).toBeVisible()
  }
  await page.getByText('高级选项', { exact: true }).click()
  if (process.env.TF_THEME_SCREENSHOT_DIR) {
    await page.screenshot({
      path: `${process.env.TF_THEME_SCREENSHOT_DIR}/picker-mobile-zh.png`,
      animations: 'disabled',
    })
  }
  for (const radio of await popover.getByRole('radio').all()) {
    await expectWithinViewport(radio.locator('xpath=ancestor::label[1]'), 390, 844)
  }
  await page.getByRole('radio', { name: '柔和工作室', exact: true }).check()
  await page.getByRole('radio', { name: '浅色', exact: true }).check()
  await page.getByRole('radio', { name: '玫瑰', exact: true }).check()
  await expectAppearance(page, 'light', 'rose', 'studio')
})

test('legacy mode and accent storage loads as classic without losing either setting', async ({
  page,
}) => {
  await page.addInitScript((key) => {
    localStorage.setItem(key, JSON.stringify({ mode: 'light', accent: 'purple' }))
  }, appearanceStorageKey)
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await expectAppearance(page, 'light', 'purple', 'classic')
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Classic Tech', exact: true })).toBeChecked()
  await expect(page.getByRole('radio', { name: 'Light', exact: true })).toBeChecked()
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Purple', exact: true })).toBeChecked()

  await chooseStyle(page, 'Minimal Workbench')
  await expectStoredAppearance(page, 'light', 'purple', 'workbench')
})

test('an unknown stored style falls back to classic while preserving valid mode and accent', async ({
  page,
}) => {
  await page.addInitScript((key) => {
    localStorage.setItem(
      key,
      JSON.stringify({ mode: 'light', accent: 'green', style: 'unsupported-style' })
    )
  }, appearanceStorageKey)
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await expectAppearance(page, 'light', 'green', 'classic')
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Classic Tech', exact: true })).toBeChecked()
})

for (const [description, storedValue] of [
  ['malformed JSON', '{broken'],
  ['unknown options', JSON.stringify({ mode: 'sepia', accent: 'orange' })],
  ['null', 'null'],
  ['an array', '[]'],
]) {
  test(`invalid stored appearance (${description}) recovers to usable dark blue`, async ({
    page,
  }) => {
    await page.addInitScript(({ key, value }) => localStorage.setItem(key, value), {
      key: appearanceStorageKey,
      value: storedValue,
    })
    await mockAuth(page, { authenticated: false })
    await page.goto('/login')

    await expect(page.getByRole('button', { name: 'Sign in', exact: true })).toBeVisible()
    await expectAppearance(page, 'dark', 'blue')
    await openAppearance(page)
    await chooseAppearance(page, 'Light', 'Green')
    await expectAppearance(page, 'light', 'green')
    await expectStoredAppearance(page, 'light', 'green')
  })
}

test('blocked storage writes do not prevent appearance changes in the current page', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await page.addInitScript((key) => {
    const originalSetItem = Storage.prototype.setItem
    Storage.prototype.setItem = function (storageKey: string, value: string) {
      if (this === window.localStorage && storageKey === key) {
        throw new DOMException('Storage writes blocked for this test', 'QuotaExceededError')
      }
      return originalSetItem.call(this, storageKey, value)
    }
  }, appearanceStorageKey)
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')

  await expectAppearance(page, 'dark', 'blue')
  const surface = page.locator('.auth-page')
  const signIn = page.getByRole('button', { name: 'Sign in', exact: true })
  const originalBackground = await backgroundColor(surface)
  const originalButton = await backgroundColor(signIn)
  await openAppearance(page)
  await chooseStyle(page, 'Minimal Workbench')
  await chooseAppearance(page, 'Light', 'Green')

  await expectAppearance(page, 'light', 'green', 'workbench')
  await expect.poll(() => backgroundColor(surface)).not.toBe(originalBackground)
  await expect.poll(() => backgroundColor(signIn)).not.toBe(originalButton)
  await expect(page.getByRole('radio', { name: 'Light', exact: true })).toBeChecked()
  await expect(page.getByRole('radio', { name: 'Minimal Workbench', exact: true })).toBeChecked()
  await expect(page.getByRole('radio', { name: 'Green', exact: true })).toBeChecked()
  expect(await page.evaluate((key) => localStorage.getItem(key), appearanceStorageKey)).toBeNull()
  expect(errors).toEqual([])
})

test('open tabs synchronize appearance changes in both directions without reload', async ({
  page,
  context,
}) => {
  await mockAuth(page, { authenticated: false })
  await page.goto('/login')
  const secondPage = await context.newPage()
  await mockApi(secondPage)
  await mockAuth(secondPage, { authenticated: false })
  await secondPage.goto('/login')

  await expectAppearance(page, 'dark', 'blue')
  await expectAppearance(secondPage, 'dark', 'blue')
  await openAppearance(page)
  await chooseStyle(page, 'Minimal Workbench')
  await chooseAppearance(page, 'Light', 'Purple')

  await expectAppearance(secondPage, 'light', 'purple', 'workbench')
  await openAppearance(secondPage)
  await expect(
    secondPage.getByRole('radio', { name: 'Minimal Workbench', exact: true })
  ).toBeChecked()
  await expect(secondPage.getByRole('radio', { name: 'Light', exact: true })).toBeChecked()
  await openAdvancedCustomization(secondPage)
  await expect(secondPage.getByRole('radio', { name: 'Purple', exact: true })).toBeChecked()
  await chooseStyle(secondPage, 'Soft Studio')
  await chooseAppearance(secondPage, 'Dark', 'Rose')

  await expectAppearance(page, 'dark', 'rose', 'studio')
  await openAppearance(page)
  await expect(page.getByRole('radio', { name: 'Soft Studio', exact: true })).toBeChecked()
  await expect(page.getByRole('radio', { name: 'Dark', exact: true })).toBeChecked()
  await openAdvancedCustomization(page)
  await expect(page.getByRole('radio', { name: 'Rose', exact: true })).toBeChecked()
  await expectStoredAppearance(page, 'dark', 'rose', 'studio')
})
