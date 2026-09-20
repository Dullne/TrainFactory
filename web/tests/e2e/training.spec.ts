import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('create training form supports model type switching', async ({ page }) => {
  await page.goto('/training/create')
  await page.getByRole('button', { name: '全部展开', exact: true }).click()

  const modelTypeForm = page.locator('.ant-form-item:has-text("模型类型") .ant-select').first()
  await expect(modelTypeForm).toBeVisible()
  await modelTypeForm.click()
  await page.locator('.ant-select-item-option:has-text("Decoder Reranker")').click()

  const selectedModelType = page.locator(
    '.ant-form-item:has-text("模型类型") .ant-select-selection-item'
  )
  await expect(selectedModelType).toHaveText(/Decoder Reranker/)

  const methodSelect = page.locator('.ant-form-item:has-text("训练方法") .ant-select').first()
  await methodSelect.click()
  await page.locator('.ant-select-item-option[title="GRPO"]').click()

  const selectedMethod = page.locator(
    '.ant-form-item:has-text("训练方法") .ant-select-selection-item'
  )
  await expect(selectedMethod).toHaveText(/GRPO/)
})

test('can open deployment modal', async ({ page }) => {
  await page.goto('/deployments')
  await page.getByRole('button', { name: '创建部署' }).click()
  await expect(page.locator('.ant-modal:has-text("创建部署")')).toBeVisible()
})

test('create training form supports multiple datasets with train/eval split', async ({ page }) => {
  await page.goto('/training/create')
  await page.getByRole('button', { name: '全部展开', exact: true }).click()

  // Wait for page to load
  await expect(page.locator('text=数据集配置')).toBeVisible()

  // Verify first dataset has train split selector visible
  await expect(page.locator('.ant-select-selection-item:has-text("训练集")')).toBeVisible()

  // Initially there's only 1 delete button
  const deleteButtons = page.locator('.anticon-delete')
  await expect(deleteButtons).toHaveCount(1)

  // Add a second dataset
  await page.getByRole('button', { name: '添加数据集' }).click()
  await page.waitForTimeout(300)

  // Now we should have 2 delete buttons
  await expect(deleteButtons).toHaveCount(2)

  // Now we should have two "训练集" labels
  const trainLabels = page.locator('.ant-select-selection-item:has-text("训练集")')
  await expect(trainLabels).toHaveCount(2)

  // Click on the second "训练集" selector to change it to eval
  await trainLabels.nth(1).click()
  await page.locator('.ant-select-item-option[title="验证集"]').click()

  // Verify we now have one train and one eval
  await expect(page.locator('.ant-select-selection-item:has-text("训练集")')).toHaveCount(1)
  await expect(page.locator('.ant-select-selection-item:has-text("验证集")')).toHaveCount(1)

  // Add a third dataset
  await page.getByRole('button', { name: '添加数据集' }).click()
  await page.waitForTimeout(300)
  await expect(deleteButtons).toHaveCount(3)

  // Remove the third dataset
  await deleteButtons.nth(2).click()
  await expect(deleteButtons).toHaveCount(2)
})

test('create training form shows existing datasets in dropdown', async ({ page }) => {
  await page.goto('/training/create')
  await page.getByRole('button', { name: '全部展开', exact: true }).click()

  // Wait for datasets to load from mock API
  await page.waitForTimeout(800)

  // Find and click on the dataset path selector by placeholder text
  const pathSelector = page
    .locator('.ant-select')
    .filter({
      has: page.locator(
        '[class*="ant-select-selection-placeholder"]:has-text("选择数据集或输入路径")'
      ),
    })
  await expect(pathSelector).toBeVisible()
  await pathSelector.click()

  // Verify mock datasets appear in dropdown (format: "name (dataset_type)")
  await expect(page.locator('.ant-select-item-option:has-text("example-dataset")')).toBeVisible()
  await expect(
    page.locator('.ant-select-item-option:has-text("eval-rerank-dataset")')
  ).toBeVisible()
})

test('llm dpo and orpo expose beta and submit rl_config', async ({ page }) => {
  let capturedBody: Record<string, unknown> | null = null

  await page.route('**/api/train', async (route) => {
    capturedBody = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-llm-dpo',
        task_name: 'LLM DPO Task',
        status: 'pending',
        message: 'ok',
      }),
    })
  })

  await page.goto('/training/create')
  await page.getByRole('button', { name: '全部展开', exact: true }).click()
  await expect(page.locator('text=数据集配置')).toBeVisible()

  const modelTypeForm = page.locator('.ant-form-item:has-text("模型类型") .ant-select').first()
  await modelTypeForm.click()
  await page.locator('.ant-select-item-option:has-text("LLM")').click()

  const methodSelect = page.locator('.ant-form-item:has-text("训练方法") .ant-select').first()
  await methodSelect.click()
  await page.locator('.ant-select-item-option[title="DPO"]').click()
  await expect(page.locator('.ant-form-item-label:has-text("DPO Beta")')).toBeVisible()
  await expect(page.locator('.ant-form-item-label:has-text("DPO Beta")')).toHaveCount(1)
  await expect(page.locator('.ant-form-item-label:has-text("排序方向")')).toBeVisible()

  await methodSelect.click()
  await page.locator('.ant-select-item-option[title="ORPO"]').click()
  await expect(page.locator('.ant-form-item-label:has-text("ORPO Beta")')).toBeVisible()
  await expect(page.locator('.ant-form-item-label:has-text("ORPO Beta")')).toHaveCount(1)

  await methodSelect.click()
  await page.locator('.ant-select-item-option[title="DPO"]').click()

  const rankingsDirectionSelect = page
    .locator('.ant-form-item:has-text("排序方向") .ant-select')
    .first()
  await rankingsDirectionSelect.click()
  await page.locator('.ant-select-item-option[title="排名越小越好"]').click()

  const baseModelSelect = page.locator('.ant-form-item:has-text("基础模型") .ant-select').first()
  await baseModelSelect.click()
  await page.keyboard.type('/models/Qwen3-0.6B')
  await page.keyboard.press('Enter')

  const datasetPathSelect = page
    .locator('.ant-select')
    .filter({
      has: page.locator(
        '[class*="ant-select-selection-placeholder"]:has-text("选择数据集或输入路径")'
      ),
    })
    .first()
  await datasetPathSelect.click()
  await page.keyboard.type('/tmp/dpo.jsonl')
  await page.keyboard.press('Enter')

  await page.getByRole('button', { name: '创建任务' }).click()

  await expect.poll(() => capturedBody).not.toBeNull()
  expect(capturedBody).toMatchObject({
    model_type: 'llm',
    training_method: 'dpo',
    base_model_path: '/models/Qwen3-0.6B',
    rl_config: {
      beta: 0.1,
      rankings_direction: 'lower_is_better',
    },
    datasets: [
      {
        path: '/tmp/dpo.jsonl',
        split: 'train',
      },
    ],
  })
})

test('decoder reranker dpo exposes beta and submits rl_config', async ({ page }) => {
  let capturedBody: Record<string, unknown> | null = null

  await page.route('**/api/train', async (route) => {
    capturedBody = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-decoder-dpo',
        task_name: 'Decoder DPO Task',
        status: 'pending',
        message: 'ok',
      }),
    })
  })

  await page.goto('/training/create')
  await page.getByRole('button', { name: '全部展开', exact: true }).click()
  await expect(page.locator('text=数据集配置')).toBeVisible()

  const modelTypeForm = page.locator('.ant-form-item:has-text("模型类型") .ant-select').first()
  await modelTypeForm.click()
  await page.locator('.ant-select-item-option:has-text("Decoder Reranker")').click()

  const methodSelect = page.locator('.ant-form-item:has-text("训练方法") .ant-select').first()
  await methodSelect.click()
  await page.locator('.ant-select-item-option[title="DPO"]').click()

  await expect(page.locator('.ant-form-item-label:has-text("DPO Beta")')).toBeVisible()
  await expect(page.locator('.ant-form-item-label:has-text("DPO Beta")')).toHaveCount(1)
  await expect(page.locator('.ant-form-item-label:has-text("Reference Free")')).toBeVisible()
  await expect(page.locator('.ant-form-item-label:has-text("排序方向")')).toHaveCount(0)

  const betaInput = page.locator('.ant-form-item:has-text("DPO Beta") input').first()
  await betaInput.fill('0.25')

  const baseModelSelect = page.locator('.ant-form-item:has-text("基础模型") .ant-select').first()
  await baseModelSelect.click()
  await page.keyboard.type('/models/Qwen3-Reranker-0.6B')
  await page.keyboard.press('Enter')

  await page.getByLabel('SFT 模型路径').fill('/app/output/sft-task/checkpoint-10')

  const datasetPathSelect = page
    .locator('.ant-select')
    .filter({
      has: page.locator(
        '[class*="ant-select-selection-placeholder"]:has-text("选择数据集或输入路径")'
      ),
    })
    .first()
  await datasetPathSelect.click()
  await page.keyboard.type('/tmp/decoder-dpo.jsonl')
  await page.keyboard.press('Enter')

  await page.getByRole('button', { name: '创建任务' }).click()

  await expect.poll(() => capturedBody).not.toBeNull()
  expect(capturedBody).toMatchObject({
    model_type: 'decoder_reranker',
    training_method: 'dpo',
    base_model_path: '/models/Qwen3-Reranker-0.6B',
    sft_checkpoint_path: '/app/output/sft-task/checkpoint-10',
    rl_config: {
      beta: 0.25,
      reference_free: false,
      n_docs: 8,
    },
    datasets: [
      {
        path: '/tmp/decoder-dpo.jsonl',
        split: 'train',
      },
    ],
  })
})
