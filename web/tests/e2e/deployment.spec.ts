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
    await expect(page.getByRole('columnheader', { name: '部署组', exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '框架', exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '状态', exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '健康副本', exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'GPU', exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: '操作', exact: true })).toBeVisible()
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

  test('displays deployment group names correctly', async ({ page }) => {
    await page.goto('/deployments')

    // 父行显示稳定的部署组名称，实例细节在展开行中显示。
    await expect(page.getByText('xinference-deployment')).toBeVisible()
    await expect(page.getByText('vllm-lora-deployment')).toBeVisible()
    await expect(page.getByText('sglang-deployment')).toBeVisible()
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

  test('edits nested launch config without discarding topology fields', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    let patchBody: Record<string, unknown> | undefined
    const launchConfig = {
      framework: 'vllm',
      tensor_parallel_size: 2,
      pipeline_parallel_size: 1,
      data_parallel_size: 1,
      max_context_length: 32768,
      max_concurrent_requests: 64,
      dtype: 'bfloat16',
      quantization: 'awq',
      kv_cache_dtype: 'fp8',
      gpu_pool: [0, 1],
      replica_gpu_overrides: [{ replica_index: 0, gpu_ids: [0, 1] }],
      allow_gpu_reuse: false,
      enable_expert_parallel: true,
      enforce_eager: true,
    }
    const deployment = {
      deployment_id: 'nested-config-deployment',
      model_id: 'model-001',
      model_uid: 'nested-config-model',
      deployment_name: 'nested-config-vllm',
      xinference_endpoint: 'http://127.0.0.1:8200',
      replica: 1,
      replica_instances: [
        {
          replica_id: 'nested-config-replica',
          deployment_id: 'nested-config-deployment',
          replica_index: 0,
          endpoint: 'http://127.0.0.1:8200',
          port: 8200,
          gpu_ids: [0, 1],
          status: 'running',
          health_status: 'HEALTHY',
        },
      ],
      gpu_memory_utilization: 0.8,
      deploy_mode: 'container',
      container_name: 'nested-config-vllm-r0',
      port: 8200,
      inference_framework: 'vllm',
      enable_lora: false,
      max_loras: 4,
      max_lora_rank: 64,
      status: 'running',
      health_status: 'HEALTHY',
      config: {
        dtype: 'float16',
        enforce_eager: false,
        docker_cmd: 'docker run preserved',
        launch_config: launchConfig,
        custom: { keep: true },
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    await page.route('**/api/deployments**', async (route) => {
      const request = route.request()
      const path = new URL(request.url()).pathname
      if (request.method() === 'GET' && path === '/api/deployments') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ deployments: [deployment], total: 1 }),
        })
      }
      if (
        request.method() === 'PATCH' &&
        path === '/api/deployments/nested-config-deployment/config'
      ) {
        patchBody = request.postDataJSON() as Record<string, unknown>
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(deployment),
        })
      }
      return route.fallback()
    })

    await page.goto('/deployments')
    const row = page.locator('tr').filter({ hasText: 'nested-config-vllm' })
    await row.getByRole('button', { name: 'View Parameters' }).click()
    const paramsModal = page.locator('.ant-modal').filter({
      hasText: 'Deployment Configuration Details',
    })
    await expect(
      paramsModal.locator('.ant-descriptions-row').filter({ hasText: 'Precision (dtype)' }),
    ).toContainText('bfloat16')
    await paramsModal
      .locator('.ant-modal-footer')
      .getByRole('button', { name: 'Close' })
      .click()

    await row.getByRole('button', { name: 'Edit Parameters' }).click()
    const editModal = page.locator('.ant-modal').filter({ hasText: 'Edit Deployment Config' })
    await expect(
      editModal
        .locator('.ant-form-item')
        .filter({ hasText: 'Compute Precision' })
        .locator('.ant-select-selection-item'),
    ).toContainText('bfloat16')
    const otherConfig = editModal.locator('textarea')
    await expect(otherConfig).toHaveValue(/"custom"/)
    await expect(otherConfig).not.toHaveValue(/launch_config/)

    const dtypeSelect = editModal
      .locator('.ant-form-item')
      .filter({ hasText: 'Compute Precision' })
      .locator('.ant-select')
    await dtypeSelect.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^float32$/ })
      .click()
    await editModal
      .locator('.ant-form-item')
      .filter({ hasText: 'Enforce Eager' })
      .locator('.ant-switch')
      .click()
    await editModal.getByRole('button', { name: 'Save' }).click()

    await expect.poll(() => patchBody).toBeTruthy()
    expect(patchBody).toEqual({
      config: {
        custom: { keep: true },
        docker_cmd: 'docker run preserved',
        external_api_config_id: null,
        launch_config: {
          ...launchConfig,
          dtype: 'float32',
          enforce_eager: false,
        },
      },
    })
  })
})

test.describe('Create Deployment Modal - Multi-Framework', () => {
  test('uses a fixed modal footer and clears hidden framework fields', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal.locator('.ant-modal-footer')).toBeVisible()
    await expect(modal.locator('.ant-modal-body')).toHaveCSS('overflow-y', 'auto')

    const framework = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')
    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^SGLang -/ })
      .click()

    const attention = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Attention Backend' })
      .locator('.ant-select')
    await attention.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
      .filter({ hasText: /^triton$/ })
      .click()

    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^vLLM -/ })
      .click()
    await modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Enforce Eager' })
      .locator('.ant-switch')
      .click()

    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^SGLang -/ })
      .click()
    await expect(attention.locator('.ant-select-selection-placeholder')).toBeVisible()

    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^vLLM -/ })
      .click()
    await expect(
      modal.locator('.ant-form-item').filter({ hasText: 'Enforce Eager' }).locator('.ant-switch')
    ).not.toHaveClass(/ant-switch-checked/)
  })

  test('shows the version-pinned vLLM and SGLang runtime options', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()

    const modal = page.locator('.ant-modal-content')
    const framework = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')

    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^vLLM -/ })
      .click()
    const quantization = modal
      .locator('.ant-form-item')
      .filter({ hasText: /^Quantization/ })
      .locator('.ant-select')
    await quantization.click()
    await quantization.locator('input').fill('deepseek_v4_fp8')
    await expect(
      page
        .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
        .filter({ hasText: /^deepseek_v4_fp8$/ }),
    ).toBeVisible()
    await page.keyboard.press('Escape')

    const kvCache = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'KV Cache Data Type' })
      .locator('.ant-select')
    await kvCache.click()
    await expect(
      page
        .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
        .filter({ hasText: /^float16$/ }),
    ).toBeVisible()
    await page.keyboard.press('Escape')

    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^SGLang -/ })
      .click()
    const attention = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Attention Backend' })
      .locator('.ant-select')
    await attention.click()
    await attention.locator('input').fill('hpc_ops')
    await expect(
      page
        .locator('.ant-select-dropdown:visible .ant-select-item-option-content')
        .filter({ hasText: /^hpc_ops$/ }),
    ).toBeVisible()
  })

  test('blocks invalid SGLang expert parallel size and submits EP equal to TP', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    let createBody: Record<string, unknown> | undefined
    let createCount = 0
    await page.route('**/api/deployments/container', async (route) => {
      if (route.request().method() !== 'POST') return route.fallback()
      createCount += 1
      createBody = route.request().postDataJSON() as Record<string, unknown>
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ deployment_id: 'created-sglang' }),
      })
    })

    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()
    const modal = page.locator('.ant-modal-content')
    await modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Deployment Name' })
      .locator('input')
      .fill('sglang-ep-validation')

    const modelSelect = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Select Model' })
      .locator('.ant-select')
      .nth(1)
    await modelSelect.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: 'bge-base-zh' })
      .click()

    const framework = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')
    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^SGLang -/ })
      .click()

    const tensorParallel = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Tensor Parallelism (TP)' })
      .locator('input')
    const expertParallel = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Expert Parallelism (EP)' })
      .locator('input')
    await tensorParallel.fill('2')
    await expertParallel.fill('3')
    await modal.getByRole('button', { name: 'Create' }).click()

    await expect(
      modal.getByText('Expert parallelism must be 1 or equal tensor parallelism'),
    ).toBeVisible()
    expect(createCount).toBe(0)

    await expertParallel.fill('2')
    await modal.getByRole('button', { name: 'Create' }).click()
    await expect.poll(() => createBody).toBeTruthy()
    expect(createCount).toBe(1)
    expect(createBody?.launch_config).toMatchObject({
      framework: 'sglang',
      tensor_parallel_size: 2,
      expert_parallel_size: 2,
    })
  })

  test('keeps parallel size controls inside the modal on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()

    const modal = page.locator('.ant-modal-content')
    const framework = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')
    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^vLLM -/ })
      .click()

    const grid = modal.getByTestId('parallel-size-grid')
    await expect(grid).toHaveCSS('display', 'grid')
    await grid.scrollIntoViewIfNeeded()
    const gridBox = await grid.boundingBox()
    expect(gridBox).not.toBeNull()

    const controls = [
      modal.locator('.ant-form-item').filter({ hasText: 'Tensor Parallelism (TP)' }),
      modal.locator('.ant-form-item').filter({ hasText: 'Pipeline Parallelism (PP)' }),
      modal.locator('.ant-form-item').filter({ hasText: 'Data Parallelism (DP)' }),
    ]
    const boxes = await Promise.all(controls.map((control) => control.boundingBox()))
    for (const box of boxes) {
      expect(box).not.toBeNull()
      expect(box!.x).toBeGreaterThanOrEqual(gridBox!.x - 1)
      expect(box!.x + box!.width).toBeLessThanOrEqual(gridBox!.x + gridBox!.width + 1)
    }
    expect(boxes[1]!.y).toBeGreaterThan(boxes[0]!.y)
    expect(boxes[2]!.y).toBeGreaterThan(boxes[1]!.y)
  })

  test('keeps replica GPU overrides inside the modal on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()

    const modal = page.locator('.ant-modal-content')
    const framework = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')
    await framework.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^vLLM -/ })
      .click()

    const gpuAllocation = modal.getByRole('button', { name: /GPU Allocation/ })
    if ((await gpuAllocation.getAttribute('aria-expanded')) !== 'true') {
      await gpuAllocation.focus()
      await page.keyboard.press('Enter')
    }
    await expect(gpuAllocation).toHaveAttribute('aria-expanded', 'true')
    await modal.getByRole('button', { name: 'Add Replica GPU Override' }).click()

    const override = modal.getByTestId('replica-gpu-override-0')
    await override.scrollIntoViewIfNeeded()
    const overrideBox = await override.boundingBox()
    expect(overrideBox).not.toBeNull()
    for (const control of [
      override.locator('.ant-input-number'),
      override.locator('.ant-select'),
      override.getByRole('button', { name: 'Remove replica GPU override' }),
    ]) {
      const box = await control.boundingBox()
      expect(box).not.toBeNull()
      expect(box!.x).toBeGreaterThanOrEqual(overrideBox!.x - 1)
      expect(box!.x + box!.width).toBeLessThanOrEqual(
        overrideBox!.x + overrideBox!.width + 1,
      )
    }
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(390)
  })

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

  test('LoRA options stay hidden for a non-LLM SGLang model', async ({ page }) => {
    await page.goto('/deployments')
    await page.getByRole('button', { name: '创建部署' }).click()

    // 等待模态框完全打开
    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    // 选择 SGLang 框架
    const frameworkSelect = modal.locator('.ant-form-item').filter({ hasText: '推理框架' }).locator('.ant-select')
    await frameworkSelect.click()
    await page.locator('.ant-select-dropdown:visible').locator('.ant-select-item-option').filter({ hasText: 'SGLang' }).click()

    await expect(modal.locator('.ant-form-item').filter({ hasText: '启用 LoRA 热加载' })).not.toBeVisible()
  })

  test('LoRA options appear for an LLM SGLang model', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.route('**/api/models**', async (route) => {
      const path = new URL(route.request().url()).pathname
      if (route.request().method() !== 'GET' || path !== '/api/models') {
        return route.fallback()
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: [
            {
              model_id: 'llm-model',
              model_name: 'qwen-llm',
              model_type: 'llm',
              model_path: '/models/qwen-llm',
              status: 'available',
              is_adapter: false,
            },
          ],
          total: 1,
        }),
      })
    })

    await page.goto('/deployments')
    await page.getByRole('button', { name: 'Create Deployment' }).click()
    const modal = page.locator('.ant-modal-content')
    const modelSelect = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Select Model' })
      .locator('.ant-select')
      .nth(1)
    await modelSelect.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: 'qwen-llm' })
      .click()

    const frameworkSelect = modal
      .locator('.ant-form-item')
      .filter({ hasText: 'Inference Framework' })
      .locator('.ant-select')
    await frameworkSelect.click()
    await page
      .locator('.ant-select-dropdown:visible .ant-select-item-option')
      .filter({ hasText: /^SGLang -/ })
      .click()

    await expect(
      modal.locator('.ant-form-item').filter({ hasText: 'Enable LoRA Hot-Loading' }),
    ).toBeVisible()
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
  test('replica groups can be expanded from the keyboard', async ({ page }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.goto('/deployments')

    const groupRow = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
    const expandButton = groupRow.getByRole('button', { name: 'Expand replicas' })
    await expandButton.focus()
    await page.keyboard.press('Enter')
    await expect(page.getByText('http://127.0.0.1:8002', { exact: true })).toBeVisible()
    await expect(groupRow.getByRole('button', { name: 'Collapse replicas' })).toBeVisible()
  })

  test('degraded deployments remain visible and manage adapters on a healthy replica', async ({
    page,
  }) => {
    await page.addInitScript(() => window.localStorage.setItem('tf_language', 'en'))
    await page.route('**/api/deployments**', async (route) => {
      const url = new URL(route.request().url())
      if (route.request().method() !== 'GET' || url.pathname !== '/api/deployments') {
        return route.fallback()
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          deployments: [
            {
              deployment_id: 'degraded-deployment',
              deployment_name: 'degraded-vllm',
              model_id: 'model-001',
              model_uid: 'bge-base-zh',
              xinference_endpoint: 'http://127.0.0.1:8100',
              inference_framework: 'vllm',
              deploy_mode: 'container',
              replica: 2,
              enable_lora: true,
              max_loras: 4,
              gpu_memory_utilization: 0.8,
              status: 'degraded',
              health_status: 'UNHEALTHY',
              replica_instances: [
                {
                  replica_id: 'healthy-replica',
                  deployment_id: 'degraded-deployment',
                  replica_index: 0,
                  endpoint: 'http://127.0.0.1:8100',
                  port: 8100,
                  gpu_ids: [0],
                  status: 'running',
                  health_status: 'HEALTHY',
                },
                {
                  replica_id: 'failed-replica',
                  deployment_id: 'degraded-deployment',
                  replica_index: 1,
                  endpoint: 'http://127.0.0.1:8101',
                  port: 8101,
                  gpu_ids: [1],
                  status: 'failed',
                  health_status: 'UNHEALTHY',
                },
              ],
            },
          ],
          total: 1,
        }),
      })
    })

    await page.goto('/deployments')
    await expect(page.getByText('Degraded', { exact: true }).first()).toBeVisible()
    const row = page.locator('tr').filter({ hasText: 'degraded-vllm' })
    await row.getByRole('button', { name: 'Adapter Management' }).click()

    const modal = page.locator('.ant-modal').filter({ hasText: 'LoRA Adapter Management' })
    await expect(modal.getByText('Target Replica')).toBeVisible()
    await modal.locator('.ant-select-selector').click()
    await page.locator('.ant-select-dropdown:visible .ant-select-item-option').filter({ hasText: '#0' }).click()
    await expect(modal.locator('.ant-select-selection-item')).toContainText('#0')
    await expect(modal.getByText('Deployment not running')).toHaveCount(0)
  })

  test('expands an independent replica group and targets one replica', async ({ page }) => {
    let requestedPath = ''
    page.on('request', (request) => {
      if (request.method() === 'POST' && request.url().includes('/replicas/')) {
        requestedPath = new URL(request.url()).pathname
      }
    })
    await page.goto('/deployments')

    const groupRow = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
    await groupRow.locator('.anticon-right').click()
    await expect(page.getByText('http://127.0.0.1:8000', { exact: true })).toBeVisible()
    await expect(page.getByText('http://127.0.0.1:8002', { exact: true })).toBeVisible()

    await page.getByRole('button', { name: '重启副本' }).nth(1).click()
    await expect.poll(() => requestedPath).toBe(
      '/api/deployments/dep-002/replicas/22222222-2222-4222-8222-222222222222/restart'
    )
  })

  test('can start a stopped deployment', async ({ page }) => {
    await page.goto('/deployments')

    // 找到已停止的部署行（sglang 容器）
    const stoppedRow = page.locator('tr').filter({ hasText: 'sglang-deployment' })
    await expect(stoppedRow).toBeVisible()

    // 检查启动按钮存在
    const startButton = stoppedRow.getByRole('button', { name: '启动' })
    await expect(startButton).toBeVisible()
  })

  test('can stop a running deployment', async ({ page }) => {
    await page.goto('/deployments')

    // 找到运行中的部署行（vllm 容器）
    const runningRow = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
    await expect(runningRow).toBeVisible()

    // 检查停止按钮存在
    const stopButton = runningRow.getByRole('button', { name: '停止' })
    await expect(stopButton).toBeVisible()
  })

  test('running deployment shows restart button, stopped does not', async ({ page }) => {
    await page.goto('/deployments')

    // 运行中的部署应显示重启按钮
    const xinferenceRow = page.locator('tr').filter({ hasText: 'xinference-deployment' })
    await expect(xinferenceRow.getByRole('button', { name: '重启' })).toBeVisible()

    const vllmRow = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
    await expect(vllmRow.getByRole('button', { name: '重启' })).toBeVisible()

    // 已停止的部署不应显示重启按钮
    const stoppedRow = page.locator('tr').filter({ hasText: 'sglang-deployment' })
    await expect(stoppedRow.getByRole('button', { name: '重启' })).not.toBeVisible()
  })

  test('xinference restart modal shows all three modes', async ({ page }) => {
    await page.goto('/deployments')

    // Xinference 容器部署 → 应显示三个模式：自动、仅重载模型、重启容器
    const row = page.locator('tr').filter({ hasText: 'xinference-deployment' })
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
    const row = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
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

    const row = page.locator('tr').filter({ hasText: 'xinference-deployment' })
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

    const row = page.locator('tr').filter({ hasText: 'xinference-deployment' })
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
    const row = page.locator('tr').filter({ hasText: 'xinference-deployment' })
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
    const row = page.locator('tr').filter({ hasText: 'xinference-deployment' })
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
    const row = page.locator('tr').filter({ hasText: 'vllm-lora-deployment' })
    await row.getByRole('button', { name: '重启' }).click()

    const modal = page.locator('.ant-modal-content')
    await expect(modal).toBeVisible()

    await modal.locator('.ant-modal-footer').getByRole('button').last().click()

    // 应显示错误提示
    await expect(page.getByText('重启失败')).toBeVisible()
  })
})
