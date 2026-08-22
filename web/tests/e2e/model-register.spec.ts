import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.describe('Model List Page', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page, { directStorageRegistrationEnabled: true })
    await page.goto('/models')
    // 等待页面关键元素渲染完成
    await expect(page.locator('h4').getByText('模型列表')).toBeVisible()
  })

  test('renders model list page with statistics panel', async ({ page }) => {
    // 检查页面标题
    await expect(page.locator('h4').getByText('模型列表')).toBeVisible()

    // 检查统计面板
    await expect(page.locator('text=总模型数')).toBeVisible()
    await expect(page.locator('text=可用').first()).toBeVisible()
    await expect(page.locator('text=未就绪')).toBeVisible()
    await expect(page.locator('text=已归档')).toBeVisible()
  })

  test('renders toolbar with filters and view mode toggle', async ({ page }) => {
    // 检查视图切换按钮
    await expect(page.locator('.ant-segmented')).toBeVisible()

    // 检查类型筛选器
    await expect(page.locator('.ant-select').filter({ hasText: '类型筛选' })).toBeVisible()

    // 检查状态筛选器
    await expect(page.locator('.ant-select').filter({ hasText: '状态筛选' })).toBeVisible()

    // 检查刷新按钮
    await expect(page.getByRole('button', { name: /刷新/ })).toBeVisible()

    // 无认证模式保留直接路径登记入口
    await expect(page.getByRole('button', { name: /注册模型/ })).toBeVisible()
  })

  test('opens the direct registration modal when the capability is enabled', async ({ page }) => {
    await page.getByRole('button', { name: /注册模型/ }).click()

    await expect(page.getByRole('dialog')).toContainText('注册本地模型')
  })

  test('view mode toggle works correctly', async ({ page }) => {
    // 默认是卡片视图，点击表格视图图标
    const tableViewButton = page.locator('.ant-segmented-item').last()
    await tableViewButton.click()

    // 验证表格视图显示
    await expect(page.locator('table')).toBeVisible()
  })

  test('type filter shows options', async ({ page }) => {
    // 点击类型筛选器
    const typeFilter = page.locator('.ant-select').filter({ hasText: '类型筛选' })
    await typeFilter.click()

    // 检查筛选选项
    const dropdown = page.locator('.ant-select-dropdown:visible')
    await expect(dropdown).toBeVisible()
    await expect(dropdown.getByTitle('Embedding', { exact: true })).toBeVisible()
    await expect(dropdown.getByTitle('Reranker', { exact: true })).toBeVisible()
    await expect(dropdown.getByTitle('LLM', { exact: true })).toBeVisible()
  })

  test('status filter shows options', async ({ page }) => {
    // 点击状态筛选器
    const statusFilter = page.locator('.ant-select').filter({ hasText: '状态筛选' })
    await statusFilter.click()

    // 检查筛选选项
    const dropdown = page.locator('.ant-select-dropdown:visible')
    await expect(dropdown).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '可用' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '已注册' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '已归档' })).toBeVisible()
  })
})

test.describe('Model List Page with direct storage registration disabled', () => {
  const roles = [
    { label: 'regular users', isAdmin: false },
    { label: 'administrators', isAdmin: true },
  ] as const

  for (const role of roles) {
    test(`hides direct path registration from ${role.label} but keeps remote download`, async ({
      page,
    }) => {
      await mockApi(page, { isAdmin: role.isAdmin })
      await page.goto('/models')

      await expect(page.locator('h4').getByText('模型列表')).toBeVisible()
      await expect(page.getByRole('button', { name: /注册模型/ })).toHaveCount(0)
      await expect(page.getByRole('button', { name: /下载模型/ })).toBeVisible()
    })
  }
})
