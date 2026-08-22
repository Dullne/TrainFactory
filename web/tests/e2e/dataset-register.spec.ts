import { test, expect, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

async function installDatasetCreateRenderProbe(page: Page) {
  await page.addInitScript(() => {
    const markerSelector = '#storage_path, [name="storage_path"]'
    const probeWindow = window as typeof window & { __datasetCreateRendered?: boolean }
    probeWindow.__datasetCreateRendered = false

    const containsCreateForm = (node: Node) => {
      if (node.nodeType !== Node.ELEMENT_NODE) return false
      const element = node as Element
      return element.matches(markerSelector) || element.querySelector(markerSelector) !== null
    }

    const markExistingCreateForm = () => {
      if (document.querySelector(markerSelector)) {
        probeWindow.__datasetCreateRendered = true
      }
    }

    new MutationObserver((records) => {
      if (
        records.some((record) =>
          Array.from(record.addedNodes).some((node) => containsCreateForm(node))
        )
      ) {
        probeWindow.__datasetCreateRendered = true
      }
    }).observe(document, { childList: true, subtree: true })

    document.addEventListener('DOMContentLoaded', markExistingCreateForm, { once: true })
  })
}

test.describe('Dataset List Page', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page, { directStorageRegistrationEnabled: true })
    await page.goto('/datasets')
    // 等待页面关键元素渲染完成
    await expect(page.locator('h4').getByText('数据集管理')).toBeVisible()
  })

  test('renders dataset list page with action buttons', async ({ page }) => {
    // 检查页面标题
    await expect(page.locator('h4').getByText('数据集管理')).toBeVisible()

    // 检查添加数据集按钮
    await expect(page.getByRole('button', { name: /注册数据集/ })).toBeVisible()

    // 检查刷新按钮
    await expect(page.getByRole('button', { name: /刷新/ })).toBeVisible()
  })

  test('renders dataset table with correct columns', async ({ page }) => {
    // 检查表格列头
    await expect(page.getByRole('columnheader', { name: '名称' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '用途' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '适用模型' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '数据类型' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '格式' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '大小' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '样本数' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '存储路径' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '创建时间' })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '操作' })).toBeVisible()
  })

  test('displays dataset table correctly', async ({ page }) => {
    // 验证表格渲染正确
    const table = page.locator('table')
    await expect(table).toBeVisible()

    // 检查表格体存在
    const tableBody = page.locator('tbody')
    await expect(tableBody).toBeVisible()
  })

  test('type filter works correctly', async ({ page }) => {
    // 点击数据类型筛选器
    const typeFilter = page.locator('.ant-select').filter({ hasText: '数据类型' })
    await typeFilter.click()

    // 检查筛选选项存在
    const dropdown = page.locator('.ant-select-dropdown:visible')
    await expect(dropdown).toBeVisible()
  })

  test('refresh button triggers data reload', async ({ page }) => {
    // 等待页面加载完成
    await expect(page.locator('h4').getByText('数据集管理')).toBeVisible()

    // 等待表格加载
    await expect(page.locator('table')).toBeVisible()

    const refreshButton = page.getByRole('button', { name: /刷新/ })
    await expect(refreshButton).toBeVisible()

    // 点击刷新按钮
    await refreshButton.click()

    // 验证刷新后表格仍然存在
    await expect(page.locator('table')).toBeVisible()
  })

  test('add dataset button navigates to create page', async ({ page }) => {
    // 点击注册数据集按钮
    await page.getByRole('button', { name: /注册数据集/ }).click()

    // 验证导航到创建页面
    await expect(page).toHaveURL(/\/datasets\/create/)
  })
})

test.describe('Dataset List Page with direct storage registration disabled', () => {
  const roles = [
    { label: 'regular users', isAdmin: false },
    { label: 'administrators', isAdmin: true },
  ] as const

  for (const role of roles) {
    test(`hides direct registration from ${role.label} but keeps upload and download`, async ({
      page,
    }) => {
      await mockApi(page, { isAdmin: role.isAdmin })
      await page.goto('/datasets')

      await expect(page.locator('h4').getByText('数据集管理')).toBeVisible()
      await expect(page.getByRole('button', { name: /注册数据集/ })).toHaveCount(0)
      await expect(page.getByRole('button', { name: /上传数据集/ })).toBeVisible()
      await expect(page.getByRole('button', { name: /下载数据集/ })).toBeVisible()
    })
  }

  test('fails closed when auth config omits the direct storage capability', async ({ page }) => {
    await mockApi(page, { omitDirectStorageRegistrationEnabled: true })
    const configResponsePromise = page.waitForResponse(
      (response) =>
        response.request().method() === 'GET' &&
        new URL(response.url()).pathname.endsWith('/api/auth/config')
    )

    await page.goto('/models')
    const configPayload = (await configResponsePromise).json()

    await expect(configPayload).resolves.not.toHaveProperty(
      'direct_storage_registration_enabled'
    )
    await expect(page.getByRole('button', { name: /注册模型/ })).toHaveCount(0)

    await page.goto('/datasets')
    await expect(page.getByRole('button', { name: /注册数据集/ })).toHaveCount(0)
  })

  test('redirects a direct create URL without ever rendering the create form', async ({ page }) => {
    await installDatasetCreateRenderProbe(page)
    await mockApi(page)
    await page.goto('/datasets/create')

    await expect(page).toHaveURL(/\/datasets$/)
    await expect(page.locator('h4').getByText('数据集管理')).toBeVisible()
    expect(
      await page.evaluate(
        () =>
          (window as typeof window & { __datasetCreateRendered?: boolean })
            .__datasetCreateRendered ?? false
      )
    ).toBe(false)
  })
})
