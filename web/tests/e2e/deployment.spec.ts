import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test.describe('Deployment List Page', () => {
  test('renders deployment statistics panel', async ({ page }) => {
    await page.goto('/deployments')

    // 检查统计面板（使用更精确的选择器）
    await expect(page.locator('text=总部署数')).toBeVisible()
    await expect(page.locator('div').filter({ hasText: /^运行中$/ }).first()).toBeVisible()
    await expect(page.locator('div').filter({ hasText: /^已停止$/ }).first()).toBeVisible()
    await expect(page.locator('div').filter({ hasText: /^失败$/ }).first()).toBeVisible()
  })

  test('renders deployment table with correct columns', async ({ page }) => {
    await page.goto('/deployments')

    // 检查表格列头 - 与实际 UI 一致
    await expect(page.locator('th:has-text("容器名称")')).toBeVisible()
    await expect(page.locator('th:has-text("框架")')).toBeVisible()
    await expect(page.locator('th:has-text("端口")')).toBeVisible()
    await expect(page.locator('th:has-text("状态")')).toBeVisible()
    await expect(page.locator('th:has-text("模型数")')).toBeVisible()
    await expect(page.locator('th:has-text("GPU")')).toBeVisible()
    await expect(page.locator('th:has-text("显存使用")')).toBeVisible()
    await expect(page.locator('th:has-text("操作")')).toBeVisible()
  })

  test('displays multiple deployments with different frameworks', async ({ page }) => {
    await page.goto('/deployments')

    // 验证表格渲染正确
    const table = page.locator('table')
    await expect(table).toBeVisible()

    // 检查不同推理框架的部署（使用更精确的选择器，只匹配表格单元格）
    await expect(page.getByRole('cell', { name: 'XINFERENCE', exact: true })).toBeVisible()
    await expect(page.getByRole('cell', { name: 'VLLM', exact: true })).toBeVisible()
    await expect(page.getByRole('cell', { name: 'SGLANG', exact: true })).toBeVisible()
  })

  test('displays container names correctly', async ({ page }) => {
    await page.goto('/deployments')

    // 检查容器名称
    await expect(page.getByText('xinference-dep-001')).toBeVisible()
    await expect(page.getByText('vllm-dep-002')).toBeVisible()
    await expect(page.getByText('sglang-dep-003')).toBeVisible()
  })

  test('displays service endpoints', async ({ page }) => {
    await page.goto('/deployments')

    // 检查端口显示
    await expect(page.getByRole('cell', { name: '9997' })).toBeVisible()
    await expect(page.getByRole('cell', { name: '8000' })).toBeVisible()
    await expect(page.getByRole('cell', { name: '8001' })).toBeVisible()
  })

  test('status filter works correctly', async ({ page }) => {
    await page.goto('/deployments')

    // 点击状态筛选器
    const statusFilter = page.locator('.ant-select').filter({ hasText: '状态筛选' })
    await statusFilter.click()

    // 检查筛选选项
    const dropdown = page.locator('.ant-select-dropdown:visible')
    await expect(dropdown).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '等待中' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '运行中' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '已停止' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: '失败' })).toBeVisible()
  })

  test('refresh button triggers data reload', async ({ page }) => {
    await page.goto('/deployments')

    // 等待页面加载完成
    await expect(page.getByText('部署列表')).toBeVisible()

    const refreshButton = page.getByRole('button', { name: '刷新' })
    await expect(refreshButton).toBeVisible()

    // 点击刷新按钮
    await refreshButton.click()

    // 验证刷新后表格仍然存在
    await expect(page.locator('table')).toBeVisible()
  })
})

test.describe('Create Deployment Modal - Multi-Framework', () => {
  test('opens and closes correctly', async ({ page }) => {
    await page.goto('/deployments')

    // 打开模态框
    await page.getByRole('button', { name: '创建部署' }).click()
    await expect(page.locator('.ant-modal').filter({ hasText: '创建部署' })).toBeVisible()

    // 关闭模态框
    await page.locator('.ant-modal-close').click()
    await expect(page.locator('.ant-modal').filter({ hasText: '创建部署' })).not.toBeVisible()
  })

  test('form has required fields including inference framework', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 检查表单字段存在
    await expect(page.locator('.ant-form-item').filter({ hasText: '部署名称' })).toBeVisible()
    await expect(page.locator('.ant-form-item').filter({ hasText: '选择模型' })).toBeVisible()
    await expect(page.locator('.ant-form-item').filter({ hasText: '推理框架' })).toBeVisible()
    await expect(page.locator('.ant-form-item').filter({ hasText: '副本数' })).toBeVisible()
    await expect(page.locator('.ant-form-item').filter({ hasText: 'GPU 选择' })).toBeVisible()
    await expect(page.locator('.ant-form-item').filter({ hasText: 'GPU 显存利用率' })).toBeVisible()
  })

  test('inference framework selector shows all options', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 点击推理框架选择器
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()

    // 检查三种框架选项
    const dropdown = page.locator('.ant-select-dropdown:visible')
    await expect(dropdown).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: 'Xinference' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: 'vLLM' })).toBeVisible()
    await expect(dropdown.locator('.ant-select-item-option').filter({ hasText: 'SGLang' })).toBeVisible()
  })

  test('LoRA options appear when selecting vLLM framework', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 选择 vLLM 框架
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()
    await page.locator('.ant-select-dropdown:visible').locator('.ant-select-item-option').filter({ hasText: 'vLLM' }).click()

    // 检查 LoRA 热加载选项出现
    await expect(modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' })).toBeVisible()
  })

  test('LoRA options appear when selecting SGLang framework', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 选择 SGLang 框架
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()
    await page.locator('.ant-select-dropdown:visible').locator('.ant-select-item-option').filter({ hasText: 'SGLang' }).click()

    // 检查 LoRA 热加载选项出现
    await expect(modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' })).toBeVisible()
  })

  test('LoRA options hidden when selecting Xinference framework', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 默认是 Xinference，检查 LoRA 选项不可见
    await expect(modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' })).not.toBeVisible()
  })

  test('enabling LoRA shows advanced settings', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 选择 vLLM 框架
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()
    await page.locator('.ant-select-dropdown:visible').locator('.ant-select-item-option').filter({ hasText: 'vLLM' }).click()

    // 启用 LoRA 热加载
    const loraSwitch = modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' }).locator('.ant-switch')
    await loraSwitch.click()

    // 检查 LoRA 已启用提示出现
    await expect(modal.getByText('LoRA 热加载已启用')).toBeVisible()
  })

  test('form validation shows error for empty required fields', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 点击创建按钮触发验证
    await modal.locator('button.ant-btn-primary').click()

    // 检查验证错误
    await expect(page.getByText('请输入部署名称')).toBeVisible()
    await expect(page.getByText('请选择模型')).toBeVisible()
  })

  test('bind existing model flow works', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 切换到绑定已有模型
    await modal.locator('.ant-radio-button-wrapper').filter({ hasText: '绑定已有模型' }).click()

    // 填写部署名称与端点
    const deployNameInput = modal.locator('.ant-form-item').filter({ hasText: '部署名称' }).locator('input')
    await deployNameInput.fill('bind-test-deployment')
    const endpointInput = modal.locator('.ant-form-item').filter({ hasText: '推理服务端点' }).locator('input')
    await endpointInput.fill('http://localhost:9997')

    // 发现模型
    await modal.getByRole('button', { name: '发现模型' }).click()

    // 选择发现的模型（折叠栏）
    const modelPanel = modal.locator('.ant-collapse-header').filter({ hasText: 'bge-base-zh-v1' })
    await expect(modelPanel).toBeVisible()
    await modelPanel.click()
    await expect(modal.getByText('已选择: bge-base-zh-v1')).toBeVisible()

    // 点击绑定
    const bindButton = modal.getByRole('button', { name: /绑\s*定/ })
    await bindButton.click()

    // 验证绑定成功提示
    await expect(page.getByText('绑定成功')).toBeVisible()

    // 模态框应关闭
    await expect(page.locator('.ant-modal-content')).not.toBeVisible()
  })

  test('can fill deployment form with vLLM and LoRA enabled', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 填写部署名称
    const deployNameInput = modal.locator('.ant-form-item').filter({ hasText: '部署名称' }).locator('input')
    await deployNameInput.fill('vllm-test-deployment')

    // 选择 vLLM 框架
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()
    await page.locator('.ant-select-dropdown:visible').locator('.ant-select-item-option').filter({ hasText: 'vLLM' }).click()

    // 启用 LoRA 热加载
    const loraSwitch = modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' }).locator('.ant-switch')
    await loraSwitch.click()

    // 填写副本数
    const replicasInput = modal.locator('.ant-form-item').filter({ hasText: '副本数' }).locator('.ant-input-number input')
    await replicasInput.clear()
    await replicasInput.fill('1')

    // 验证值已填入
    await expect(deployNameInput).toHaveValue('vllm-test-deployment')
    await expect(modal.getByText('LoRA 热加载已启用')).toBeVisible()
  })

  test('cancel button closes modal without submitting', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 填写一些数据
    const deployNameInput = modal.locator('.ant-form-item').filter({ hasText: '部署名称' }).locator('input')
    await deployNameInput.fill('test-deployment')

    // 点击关闭按钮（右上角 X）
    await page.locator('.ant-modal-close').click()

    // 模态框应该关闭
    await expect(modal).not.toBeVisible()
  })
})

test.describe('Deployment Actions', () => {
  test('can start a stopped deployment', async ({ page }) => {
    await page.goto('/deployments')

    // 找到已停止的部署行（sglang 容器）
    const stoppedRow = page.locator('tr').filter({ hasText: 'sglang-dep-003' })
    await expect(stoppedRow).toBeVisible()

    // 检查启动按钮存在
    const startButton = stoppedRow.getByRole('button', { name: '启动' })
    await expect(startButton).toBeVisible()
  })

  test('can stop a running deployment', async ({ page }) => {
    await page.goto('/deployments')

    // 找到运行中的部署行（vllm 容器）
    const runningRow = page.locator('tr').filter({ hasText: 'vllm-dep-002' })
    await expect(runningRow).toBeVisible()

    // 检查停止按钮存在
    const stopButton = runningRow.getByRole('button', { name: '停止' })
    await expect(stopButton).toBeVisible()
  })

  test('running deployment shows restart button, stopped does not', async ({ page }) => {
    await page.goto('/deployments')

    // 运行中的部署应显示重启按钮
    const xinferenceRow = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await expect(xinferenceRow.getByRole('button', { name: '重启' })).toBeVisible()

    const vllmRow = page.locator('tr').filter({ hasText: 'vllm-dep-002' })
    await expect(vllmRow.getByRole('button', { name: '重启' })).toBeVisible()

    // 已停止的部署不应显示重启按钮
    const stoppedRow = page.locator('tr').filter({ hasText: 'sglang-dep-003' })
    await expect(stoppedRow.getByRole('button', { name: '重启' })).not.toBeVisible()
  })

  test('xinference restart modal shows all three modes', async ({ page }) => {
    await page.goto('/deployments')

    // Xinference 容器部署 → 应显示三个模式：自动、仅重载模型、重启容器
    const row = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()
    await expect(modal.getByText('重启部署')).toBeVisible()
    await expect(modal.getByText(/重启会先停止部署/)).toBeVisible()
    await expect(modal.getByText('重启模式')).toBeVisible()

    // 三个 Radio 模式选项
    const radios = modal.locator('.ant-radio-wrapper')
    await expect(radios).toHaveCount(3)
    await expect(radios.nth(0)).toContainText('自动（推荐）')
    await expect(radios.nth(1)).toContainText('仅重载模型')
    await expect(radios.nth(2)).toContainText('完全重启容器进程')

    // 默认选中 auto，GPU reset checkbox 不可见
    await expect(modal.locator('.ant-checkbox-wrapper')).not.toBeVisible()
  })

  test('vllm restart modal does not show model-only mode', async ({ page }) => {
    await page.goto('/deployments')

    // vLLM 容器部署 → 不应显示"仅重载模型"选项
    const row = page.locator('tr').filter({ hasText: 'vllm-dep-002' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 只有两个 Radio：自动、重启容器（无"仅重载模型"）
    const radios = modal.locator('.ant-radio-wrapper')
    await expect(radios).toHaveCount(2)
    await expect(radios.nth(0)).toContainText('自动（推荐）')
    await expect(radios.nth(1)).toContainText('完全重启容器进程')
  })

  test('selecting container mode reveals GPU reset checkbox', async ({ page }) => {
    await page.goto('/deployments')

    const row = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 默认 auto 模式，GPU reset checkbox 不可见
    await expect(modal.locator('.ant-checkbox-wrapper')).not.toBeVisible()

    // 切换到容器模式（第三个 radio）
    await modal.locator('.ant-radio-wrapper').nth(2).click()

    // GPU reset checkbox 出现
    await expect(modal.locator('.ant-checkbox-wrapper')).toBeVisible()
    await expect(modal.locator('.ant-checkbox-wrapper')).toContainText('清理 GPU 缓存')

    // 切回自动模式，checkbox 再次隐藏
    await modal.locator('.ant-radio-wrapper').nth(0).click()
    await expect(modal.locator('.ant-checkbox-wrapper')).not.toBeVisible()
  })

  test('restart modal can be cancelled', async ({ page }) => {
    await page.goto('/deployments')

    const row = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 点击取消按钮
    await modal.locator('.ant-modal-footer').getByRole('button').first().click()
    await expect(modal).not.toBeVisible()
  })

  test('restart with auto mode sends correct request', async ({ page }) => {
    // 拦截 restart API，验证请求参数
    let capturedBody: Record<string, unknown> | null = null
    await page.route('**/api/deployments/dep-001/restart', async (route) => {
      capturedBody = route.request().postDataJSON()
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ deployment_id: 'dep-001', status: 'running' }),
      })
    })

    await page.goto('/deployments')
    const row = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 默认 auto 模式，直接点重启
    await modal.locator('.ant-modal-footer').getByRole('button').last().click()
    await expect(page.getByText('重启请求已发送')).toBeVisible()

    // 验证请求参数
    expect(capturedBody).toEqual({ mode: 'auto', reset_gpu: false })
  })

  test('restart with container mode and GPU reset sends correct request', async ({ page }) => {
    let capturedBody: Record<string, unknown> | null = null
    await page.route('**/api/deployments/dep-001/restart', async (route) => {
      capturedBody = route.request().postDataJSON()
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ deployment_id: 'dep-001', status: 'running' }),
      })
    })

    await page.goto('/deployments')
    const row = page.locator('tr').filter({ hasText: 'xinference-dep-001' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 选择容器模式（最后一个 radio）
    await modal.locator('.ant-radio-wrapper').last().click()

    // 勾选 GPU reset
    await modal.locator('.ant-checkbox-input').check()

    // 提交
    await modal.locator('.ant-modal-footer').getByRole('button').last().click()
    await expect(page.getByText('重启请求已发送')).toBeVisible()

    // 验证请求参数
    expect(capturedBody).toEqual({ mode: 'container', reset_gpu: true })
  })

  test('restart API failure shows error message', async ({ page }) => {
    await page.route('**/api/deployments/dep-002/restart', async (route) => {
      return route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: '容器重启失败' }),
      })
    })

    await page.goto('/deployments')
    const row = page.locator('tr').filter({ hasText: 'vllm-dep-002' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    await modal.locator('.ant-modal-footer').getByRole('button').last().click()

    // 应显示错误提示
    await expect(page.getByText('重启失败')).toBeVisible()
  })
})
