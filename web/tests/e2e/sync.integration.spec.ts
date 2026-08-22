import { test, expect } from '@playwright/test'

const API_BASE = process.env.E2E_SYNC_API_BASE || 'http://127.0.0.1:18000/api/sync'
const RUN_SYNC_INTEGRATION = process.env.E2E_RUN_SYNC_INTEGRATION === '1'

type ApiConfig = {
  config_id: string
  config_name: string
}

type SyncConfig = {
  task_id: string
  task_name: string
  is_active: boolean
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
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
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
  return { status: res.status, data }
}

async function findApiConfigByName(name: string): Promise<ApiConfig | null> {
  const { status, data } = await apiRequest('GET', '/api-configs')
  if (status !== 200) return null
  const configs = (data.configs || []) as ApiConfig[]
  return configs.find((c) => c.config_name === name) || null
}

async function findSyncConfigByName(name: string): Promise<SyncConfig | null> {
  const { status, data } = await apiRequest('GET', '/tasks')
  if (status !== 200) return null
  const tasks = (data.tasks || []) as SyncConfig[]
  return tasks.find((c) => c.task_name === name) || null
}

async function cleanupByState() {
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
    state.apiConfigName = `e2e-api-${state.stamp}`
    state.syncConfigName = `e2e-sync-${state.stamp}`
    await cleanupByState()
  })

  test.afterAll(async () => {
    await cleanupByState()
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
      .fill('http://127.0.0.1:18000/api/sync/tasks')
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
