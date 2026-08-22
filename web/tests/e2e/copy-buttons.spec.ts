import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.beforeEach(async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.addInitScript(() => {
    const state = window as Window & { __copiedText?: string }
    if (!state.__copiedText) {
      state.__copiedText = ''
    }
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, 'clipboard', {
        value: {},
        configurable: true,
      })
    }
    const clipboard = navigator.clipboard as Clipboard & { _writeText?: (text: string) => Promise<void> }
    const writeText = clipboard.writeText?.bind(clipboard)
    clipboard.writeText = async (text: string) => {
      state.__copiedText = text
      if (writeText) {
        await writeText(text)
      }
    }
    clipboard.readText = async () => state.__copiedText || ''

    const originalExecCommand = document.execCommand?.bind(document)
    document.execCommand = (commandId: string) => {
      if (commandId === 'copy') {
        const active = document.activeElement as HTMLTextAreaElement | null
        if (active && 'value' in active) {
          state.__copiedText = active.value
        } else {
          const textarea = document.querySelector('textarea')
          if (textarea) {
            state.__copiedText = textarea.value
          }
        }
      }
      return originalExecCommand ? originalExecCommand(commandId) : true
    }
  })
  await mockApi(page)
})

// ==================== 1. 配置页面 (ConfigCard) ====================
test.describe('配置页面 - ConfigCard 复制按钮', () => {
  test('端点复制按钮可见可点击', async ({ page }) => {
    await page.goto('/configs')
    await expect(page.getByRole('heading', { name: '模型配置' })).toBeVisible()

    // 展开 Collapse（如果有的话），找到 CopyOutlined 按钮
    const copyBtn = page.locator('.anticon-copy').first()
    await expect(copyBtn).toBeVisible()

    await copyBtn.click({ force: true })

    // 检查成功提示
    await expect(page.locator('.ant-message-success')).toBeVisible()
  })
})

// ==================== 2. 配置页面 - 测试连接弹窗 ====================
test.describe('配置页面 - 测试连接弹窗模型列表复制', () => {
  test('测试连接弹窗显示模型列表并可复制', async ({ page }) => {
    await page.goto('/configs')
    await expect(page.getByRole('heading', { name: '模型配置' })).toBeVisible()

    // 点击 ConfigCard 上的测试连接按钮（ApiOutlined 图标，在 Collapse 面板内）
    const checkBtn = page.locator('.ant-collapse').locator('button').filter({ has: page.locator('.anticon-api') }).first()
    await expect(checkBtn).toBeVisible()
    await checkBtn.click()

    // 等待 Modal.success 弹窗出现
    const modal = page.locator('.ant-modal-confirm-success')
    await expect(modal).toBeVisible({ timeout: 5000 })

    // 弹窗应包含"可用模型"
    await expect(modal.locator('text=可用模型')).toBeVisible()

    // 点击真实按钮，避免在弹窗动画期间直接派发到图标节点。
    const copyBtn = modal.locator('button').filter({ has: page.locator('.anticon-copy') }).first()
    await expect(copyBtn).toBeVisible()

    await copyBtn.click()
    await expect
      .poll(async () => page.evaluate(() => (window as Window & { __copiedText?: string }).__copiedText || ''))
      .toMatch(/gpt-4o-mini|text-embedding-3-small/)
  })
})

// ==================== 3. 配置页面 - API 测试弹窗 ====================
test.describe('配置页面 - API 测试弹窗复制按钮', () => {
  test('开场动画期间选择的 cURL 标签不会在动画结束后被重置', async ({ page }) => {
    await page.goto('/configs')
    await expect(page.getByRole('heading', { name: '模型配置' })).toBeVisible()

    await page.addStyleTag({
      content: `
        .ant-modal.ant-zoom-appear,
        .ant-modal.ant-zoom-appear-active {
          animation-duration: 1200ms !important;
        }
      `,
    })

    const apiTestBtn = page.locator('.anticon-code').first()
    await apiTestBtn.click()

    const modal = page.getByRole('dialog', { name: /API 测试/ })
    await expect(modal).toBeVisible()
    await expect(modal).toHaveClass(/ant-zoom-appear/)

    const curlTab = modal.getByRole('tab', { name: /cURL/ })
    await curlTab.dispatchEvent('click')
    await expect(curlTab).toHaveAttribute('aria-selected', 'true')

    await expect(modal).not.toHaveClass(/ant-zoom-appear/, { timeout: 5000 })
    await expect(curlTab).toHaveAttribute('aria-selected', 'true')
  })

  test('cURL 复制按钮可见可点击', async ({ page }) => {
    await page.goto('/configs')
    await expect(page.getByRole('heading', { name: '模型配置' })).toBeVisible()

    // 点击 API 测试按钮（CodeOutlined 图标）
    const apiTestBtn = page.locator('.anticon-code').first()
    await expect(apiTestBtn).toBeVisible()
    await apiTestBtn.click()

    // 等待 API 测试弹窗出现
    const modal = page.getByRole('dialog', { name: /API 测试/ })
    await expect(modal).toBeVisible()

    // 切换到 cURL 标签
    const curlTab = modal.getByRole('tab', { name: /cURL/ })
    await curlTab.click()
    await expect(curlTab).toHaveAttribute('aria-selected', 'true')

    // 找到复制按钮
    const copyBtn = modal.getByRole('button', { name: '复制' })
    await expect(copyBtn).toBeVisible()

    await copyBtn.click()
    await expect
      .poll(async () => page.evaluate(() => (window as Window & { __copiedText?: string }).__copiedText || ''))
      .toContain('curl')
    await expect(page.locator('.ant-message-success')).toBeVisible()
  })
})

// ==================== 4. 部署页面 ====================
test.describe('部署页面 - 复制按钮', () => {
  test('端点复制按钮可见可点击', async ({ page }) => {
    await page.goto('/deployments')
    await expect(page.getByRole('heading', { name: '部署列表' })).toBeVisible()

    // 限定到容器名称单元格，找到复制按钮
    const containerCell = page.getByRole('cell', { name: /xinference-dep-001/i })
    const copyBtn = containerCell.locator('button').filter({ has: page.locator('.anticon-copy') }).first()
    await expect(copyBtn).toBeVisible()

    await copyBtn.dispatchEvent('click')
    await expect(page.locator('.ant-message-success')).toBeVisible()
  })

  test('Docker 命令复制按钮可见可点击', async ({ page }) => {
    await page.goto('/deployments')
    await expect(page.getByRole('heading', { name: '部署列表' })).toBeVisible()

    // 点击"查看参数"按钮打开部署配置详情弹窗
    const paramsBtn = page.getByRole('button', { name: '查看参数' }).first()
    await expect(paramsBtn).toBeVisible()
    await paramsBtn.click()

    // 等待弹窗出现
    const modal = page.locator('.ant-modal')
    await expect(modal).toBeVisible()

    // 找到启动命令的复制按钮
    const dockerCopyBtn = modal.locator('button', { hasText: '复制' }).first()
    await expect(dockerCopyBtn).toBeVisible({ timeout: 5000 })

    await dockerCopyBtn.click()
    await expect(page.locator('.ant-message-success')).toBeVisible()
  })
})
