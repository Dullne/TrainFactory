import { readFile } from 'node:fs/promises'
import { test, expect, request, type APIRequestContext } from '@playwright/test'

const API_BASE = process.env.E2E_SYNC_API_BASE || ''
const SOURCE_URL = process.env.E2E_SYNC_SOURCE_URL || ''
const RUN_SYNC_INTEGRATION = process.env.E2E_RUN_SYNC_INTEGRATION === '1'
let apiClient: APIRequestContext | undefined
let authState: Awaited<ReturnType<APIRequestContext['storageState']>> | undefined

type ApiConfig = {
  config_id: string
  config_name: string
  status: string
}

type SyncConfig = {
  task_id: string
  task_name: string
  is_active: boolean
  status: string
  error_message?: string | null
  generation_threshold?: number
  training_threshold?: number
}

const state: {
  stamp: string
  apiConfigName: string
  syncConfigName: string
  apiConfigId: string
  syncConfigId: string
} = {
  stamp: `${Date.now()}`,
  apiConfigName: '',
  syncConfigName: '',
  apiConfigId: '',
  syncConfigId: '',
}

async function apiRequest(
  method: string,
  path: string,
  body?: Record<string, unknown>,
): Promise<{ status: number; data: Record<string, unknown> }> {
  if (!apiClient) throw new Error('Sync integration API session is not initialized')
  const res = await apiClient.fetch(`${API_BASE}${path}`, {
    method,
    data: body,
  })
  const text = await res.text()
  let data: Record<string, unknown> = {}
  if (text) {
    try {
      data = JSON.parse(text) as Record<string, unknown>
    } catch {
      data = { raw: text }
    }
  }
  return { status: res.status(), data }
}

async function findApiConfigByName(name: string): Promise<ApiConfig | null> {
  const { status, data } = await apiRequest('GET', '/api-configs')
  expect(status).toBe(200)
  const configs = (data.configs || []) as ApiConfig[]
  return configs.find((c) => c.config_name === name) || null
}

async function findSyncConfigByName(name: string): Promise<SyncConfig | null> {
  const { status, data } = await apiRequest('GET', '/tasks')
  expect(status).toBe(200)
  const tasks = (data.tasks || []) as SyncConfig[]
  return tasks.find((c) => c.task_name === name) || null
}

async function cleanupByState() {
  if (!apiClient) return
  if (state.syncConfigId) {
    await apiRequest('DELETE', `/tasks/${state.syncConfigId}`)
  } else if (state.syncConfigName) {
    const sync = await findSyncConfigByName(state.syncConfigName)
    if (sync) await apiRequest('DELETE', `/tasks/${sync.task_id}`)
  }

  if (state.apiConfigId) {
    await apiRequest('DELETE', `/api-configs/${state.apiConfigId}`)
  } else if (state.apiConfigName) {
    const api = await findApiConfigByName(state.apiConfigName)
    if (api) await apiRequest('DELETE', `/api-configs/${api.config_id}`)
  }
}

test.describe('Sync Integration (real backend)', () => {
  test.skip(
    !RUN_SYNC_INTEGRATION,
    'Set E2E_RUN_SYNC_INTEGRATION=1 to run tests that modify a real backend.',
  )
  test.describe.configure({ mode: 'serial' })

  test.beforeAll(async () => {
    const credentialsFile = process.env.E2E_SYNC_CREDENTIALS_FILE
    if (!API_BASE || !SOURCE_URL || !credentialsFile) {
      throw new Error(
        'Sync integration requires E2E_SYNC_API_BASE, E2E_SYNC_SOURCE_URL and E2E_SYNC_CREDENTIALS_FILE for an isolated backend.',
      )
    }
    const credentials = JSON.parse((await readFile(credentialsFile, 'utf8')).replace(/^\uFEFF/, ''))
    if (typeof credentials.username !== 'string' || typeof credentials.password !== 'string') {
      throw new Error('Sync integration credentials must contain username and password strings')
    }
    apiClient = await request.newContext()
    const login = await apiClient.post(new URL('/api/auth/login', API_BASE).href, {
      data: { username: credentials.username, password: credentials.password },
    })
    expect(login.status()).toBe(200)
    authState = await apiClient.storageState()
    state.apiConfigName = `e2e-api-${state.stamp}`
    state.syncConfigName = `e2e-sync-${state.stamp}`
    await cleanupByState()
  })

  test.afterAll(async () => {
    try {
      await cleanupByState()
    } finally {
      await apiClient?.dispose()
      apiClient = undefined
    }
  })

  test.beforeEach(async ({ context }) => {
    if (!authState) throw new Error('Sync integration browser session is not initialized')
    await context.addCookies(authState.cookies)
    await context.addInitScript(() => localStorage.setItem('tf_language', 'zh'))
  })

  test('创建外部 API 配置与同步任务（UI）', async ({ page }) => {
    await page.goto('/sync')

    // 1) 创建外部 API 配置
    await page.getByText('外部 API').first().click()
    await page.getByRole('button', { name: '创建 API 配置' }).click()

    const modal = page.locator('.ant-modal').last()
    await expect(modal).toBeVisible()
    await modal.getByPlaceholder('输入配置名称').fill(state.apiConfigName)
    await modal
      .getByPlaceholder('http://example.com:9003/api/embedding_texts')
      .fill(SOURCE_URL)
    await modal.locator('.ant-select').click()
    await page.getByText('Bearer Token', { exact: true }).click()
    await modal.getByPlaceholder('输入 API Token').fill('playwright-token')
    await modal.getByPlaceholder('可选描述信息').fill('playwright integration test')
    await modal.locator('.ant-btn-primary').click()

    const apiRow = page.locator('tr', { hasText: state.apiConfigName }).first()
    await expect(apiRow).toBeVisible()

    const api = await findApiConfigByName(state.apiConfigName)
    expect(api).not.toBeNull()
    state.apiConfigId = api?.config_id || ''

    // 2) 创建同步任务（引用上面的 API 配置）
    await page.getByText('同步任务').first().click()
    await page.getByRole('button', { name: '创建同步任务' }).click()
    await page.waitForURL('**/sync/create')

    await page.getByPlaceholder('输入任务名称').first().fill(state.syncConfigName)

    const apiSelect = page.locator('.ant-select').first()
    await apiSelect.click()
    await page.getByText(state.apiConfigName).first().click()

    await page.getByRole('button', { name: '创建配置' }).click()
    await page.waitForURL('**/sync')

    const syncRow = page.locator('tr', { hasText: state.syncConfigName }).first()
    await expect(syncRow).toBeVisible()

    const sync = await findSyncConfigByName(state.syncConfigName)
    expect(sync).not.toBeNull()
    expect(sync?.generation_threshold).toBe(0)
    expect(sync?.training_threshold).toBe(0)
    state.syncConfigId = sync?.task_id || ''
  })

  test('同步任务操作：立即同步、停止、启动（UI）', async ({ page }) => {
    expect(state.syncConfigId).not.toBe('')
    await page.goto(`/sync/${state.syncConfigId}`)
    await expect(page.getByText(state.syncConfigName).first()).toBeVisible()

    // 立即同步（无新增数据也应完成）
    await page.getByRole('button', { name: '立即同步' }).click()
    await expect(page.locator('.ant-message')).toContainText('同步完成')
    const synced = await apiRequest('GET', `/tasks/${state.syncConfigId}`)
    expect(synced.status).toBe(200)
    const syncedTask = synced.data.task as SyncConfig
    expect(syncedTask.status).toBe('idle')
    expect(syncedTask.error_message).toBeFalsy()
    expect((await findApiConfigByName(state.apiConfigName))?.status).toBe('active')

    // 停止（含确认弹窗）
    await page.getByRole('button', { name: '停止' }).first().click()
    await expect(page.locator('.ant-popconfirm')).toBeVisible()
    await page.locator('.ant-popconfirm-buttons .ant-btn-primary').click()
    await expect(page.locator('.ant-message')).toContainText('同步已停止')

    const stopped = await apiRequest('GET', `/tasks/${state.syncConfigId}`)
    expect(stopped.status).toBe(200)
    expect((stopped.data.task as SyncConfig).is_active).toBeFalsy()

    // 启动
    await page.getByRole('button', { name: '启动' }).first().click()
    await expect(page.locator('.ant-message')).toContainText('同步已启动')

    const started = await apiRequest('GET', `/tasks/${state.syncConfigId}`)
    expect(started.status).toBe(200)
    expect((started.data.task as SyncConfig).is_active).toBeTruthy()
  })

  test('数据源认证失败显示失败，修正 Token 后可恢复同步（UI）', async ({ page }) => {
    const invalidSource = await apiRequest('PATCH', `/api-configs/${state.apiConfigId}`, {
      auth_config: { auth_method: 'bearer', token: 'invalid-playwright-token' },
    })
    expect(invalidSource.status).toBe(200)
    try {
      await page.goto(`/sync/${state.syncConfigId}`)
      const failedSync = page.waitForResponse(
        (response) => response.url().endsWith(`/tasks/${state.syncConfigId}/sync-now`)
          && response.request().method() === 'POST',
      )
      await page.getByRole('button', { name: '立即同步' }).click()
      expect((await failedSync).status()).toBe(502)
      await expect(page.locator('.ant-message')).toContainText('同步失败')
      await expect(page.locator('.ant-message')).not.toContainText('同步完成')
      const failedTask = await apiRequest('GET', `/tasks/${state.syncConfigId}`)
      expect(failedTask.status).toBe(200)
      expect((failedTask.data.task as SyncConfig).status).toBe('error')
      expect((failedTask.data.task as SyncConfig).error_message).toBeTruthy()
    } finally {
      const restoredSource = await apiRequest('PATCH', `/api-configs/${state.apiConfigId}`, {
        auth_config: { auth_method: 'bearer', token: 'playwright-token' },
      })
      expect(restoredSource.status).toBe(200)
    }

    await page.getByRole('button', { name: '立即同步' }).click()
    await expect(page.locator('.ant-message')).toContainText('同步完成')
    const recovered = await apiRequest('GET', `/tasks/${state.syncConfigId}`)
    expect(recovered.status).toBe(200)
    expect((recovered.data.task as SyncConfig).status).toBe('idle')
    expect((recovered.data.task as SyncConfig).error_message).toBeFalsy()
    expect((await findApiConfigByName(state.apiConfigName))?.status).toBe('active')
  })

  test('被同步任务引用时，外部 API 配置不可删除（UI）', async ({ page }) => {
    expect(state.apiConfigId).not.toBe('')
    await page.goto('/sync')
    await page.getByText('外部 API').first().click()

    const apiRow = page.locator('tr', { hasText: state.apiConfigName }).first()
    await expect(apiRow).toBeVisible()
    await apiRow.locator('.ant-btn-dangerous').click()
    await expect(page.locator('.ant-popconfirm')).toBeVisible()
    await page.locator('.ant-popconfirm-buttons .ant-btn-primary').click()

    // 删除应失败，行仍然存在
    await expect(page.locator('tr', { hasText: state.apiConfigName }).first()).toBeVisible()

    const stillExists = await findApiConfigByName(state.apiConfigName)
    expect(stillExists).not.toBeNull()
  })

  test('删除同步任务后可删除外部 API 配置（UI）', async ({ page }) => {
    await page.goto('/sync')

    // 先删同步任务
    const syncRow = page.locator('tr', { hasText: state.syncConfigName }).first()
    await expect(syncRow).toBeVisible()
    await syncRow.locator('.ant-btn-dangerous').click()
    await expect(page.locator('.ant-popconfirm')).toBeVisible()
    await page.locator('.ant-popconfirm-buttons .ant-btn-primary').click()
    await expect(page.locator('tr', { hasText: state.syncConfigName })).toHaveCount(0)
    state.syncConfigId = ''

    // 再删外部 API 配置
    await page.getByText('外部 API').first().click()
    const apiRow = page.locator('tr', { hasText: state.apiConfigName }).first()
    await expect(apiRow).toBeVisible()
    await apiRow.locator('.ant-btn-dangerous').click()
    await expect(page.locator('.ant-popconfirm')).toBeVisible()
    await page.locator('.ant-popconfirm-buttons .ant-btn-primary').click()
    await expect(page.locator('tr', { hasText: state.apiConfigName })).toHaveCount(0)
    state.apiConfigId = ''

    // API 再确认一次没有残留
    expect(await findSyncConfigByName(state.syncConfigName)).toBeNull()
    expect(await findApiConfigByName(state.apiConfigName)).toBeNull()
  })
})
