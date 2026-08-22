import { defineConfig, devices } from '@playwright/test'
import { createWebServerConfig, resolvePlaywrightRuntime } from './playwright.runtime.mjs'

const runtime = resolvePlaywrightRuntime(process.env)
const webServer = createWebServerConfig(runtime)
const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH

export default defineConfig({
  testDir: './tests/e2e',
  globalSetup: './tests/e2e/global-setup.ts',
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  workers: runtime.external ? undefined : 2,
  reporter: [['list'], ['html', { open: 'never' }]],
  metadata: {
    trainFactoryE2E: {
      external: runtime.external,
      runId: runtime.runId,
      baseURL: runtime.baseURL,
    },
  },
  use: {
    baseURL: runtime.baseURL,
    locale: 'zh-CN',
    viewport: { width: 1280, height: 720 },
    launchOptions: executablePath ? { executablePath } : undefined,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  ...(webServer ? { webServer } : {}),
})
