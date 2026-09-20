import { test, expect } from '@playwright/test'
import { mockApi, mockAuth } from './helpers/mockApi'

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('tf_language', 'en'))
  await mockApi(page)
})

for (const logoutStatus of [200, 500]) {
  test(`workspace state is ${logoutStatus === 200 ? 'cleared after confirmed logout' : 'retained when logout fails'}`, async ({
    page,
  }) => {
    await mockAuth(page, { authenticated: true, logoutStatus })
    await page.goto('/training')
    await page.getByRole('heading', { name: 'Training Tasks', exact: true }).waitFor()
    await page.evaluate(() => {
      sessionStorage.setItem('tf_training_draft:v1:user-1', 'private-draft')
      sessionStorage.setItem('tf_list_workspace:v1:user-1:/training', '420')
      sessionStorage.setItem('unrelated-preference', 'keep')
      window.addEventListener('tf:workspace-session-cleared', () => {
        sessionStorage.setItem('clear-event-observed', 'yes')
      })
    })
    await page.getByRole('button', { name: /runtime_verify/i }).click()
    await page.getByText('Sign out', { exact: true }).click()
    await page
      .getByRole('dialog', { name: 'Sign out?' })
      .getByRole('button', { name: 'Confirm sign out', exact: true })
      .click()
    if (logoutStatus === 200) {
      await expect(page).toHaveURL(/\/login$/)
      await expect
        .poll(() => page.evaluate(() => sessionStorage.getItem('tf_training_draft:v1:user-1')))
        .toBeNull()
      await expect
        .poll(() =>
          page.evaluate(() => sessionStorage.getItem('tf_list_workspace:v1:user-1:/training'))
        )
        .toBeNull()
      expect(await page.evaluate(() => sessionStorage.getItem('clear-event-observed'))).toBe('yes')
    } else {
      await expect(
        page.getByText('Sign out failed. Your session is still active. Please retry.', {
          exact: true,
        })
      ).toBeVisible()
      expect(await page.evaluate(() => sessionStorage.getItem('tf_training_draft:v1:user-1'))).toBe(
        'private-draft'
      )
      expect(await page.evaluate(() => sessionStorage.getItem('clear-event-observed'))).toBeNull()
    }
    expect(await page.evaluate(() => sessionStorage.getItem('unrelated-preference'))).toBe('keep')
  })
}

for (const failure of ['login', 'session-confirmation', 'registration'] as const) {
  test(`temporary ${failure} failure retains the account workspace`, async ({ page }) => {
    await mockAuth(page, {
      authenticated: false,
      registrationEnabled: true,
      meStatuses: [503, 503],
      loginStatus: failure === 'login' ? 503 : 200,
      registerStatus: 503,
    })
    await page.addInitScript(() => {
      sessionStorage.setItem('tf_training_draft:v1:user-1', 'recoverable-draft')
      sessionStorage.setItem('tf_list_workspace:v1:user-1:/training', '420')
    })
    await page.goto('/login')
    await page.getByRole('heading', { name: 'Sign in to workspace' }).waitFor()
    if (failure === 'registration') {
      await page.getByText('Register', { exact: true }).click()
    }
    await page.getByLabel('Username', { exact: true }).fill('workspace_user')
    await page.getByLabel('Password', { exact: true }).fill('Workspace-Test-2026')
    if (failure === 'registration') {
      await page.getByLabel('Confirm password').fill('Workspace-Test-2026')
      await page.getByRole('button', { name: 'Create account', exact: true }).click()
      await expect(page.locator('#username_help')).toContainText('Username or email already exists')
    } else {
      await page.getByRole('button', { name: 'Sign in', exact: true }).click()
      await expect(
        page
          .getByRole('alert')
          .filter({
            hasText: failure === 'login' ? 'Invalid username or password' : 'Not authenticated',
          })
          .first()
      ).toBeVisible()
    }
    expect(await page.evaluate(() => sessionStorage.getItem('tf_training_draft:v1:user-1'))).toBe(
      'recoverable-draft'
    )
    expect(
      await page.evaluate(() => sessionStorage.getItem('tf_list_workspace:v1:user-1:/training'))
    ).toBe('420')
  })
}
