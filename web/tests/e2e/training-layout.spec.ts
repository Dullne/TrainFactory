import { test, expect, type Locator, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

const copy = {
  zh: {
    title: '创建训练任务',
    modelType: '模型类型',
    trainingMethod: '训练方法',
    baseModel: '基础模型',
    baseModelRequired: '请选择模型或输入路径',
    addDataset: '添加数据集',
    datasetSplit: '数据集用途',
    datasetPath: '数据集路径',
    sampleCount: '采样数量',
    removeDataset: '删除数据集',
    evalSplit: '验证集',
    submit: '创建任务',
    rankingDirection: '排序方向',
    lowerRanking: '排名越小越好',
    docsPerGroup: '每组文档数',
    sftModelPath: 'SFT 模型路径',
  },
  en: {
    title: 'Create Training Task',
    modelType: 'Model Type',
    trainingMethod: 'Training Method',
    baseModel: 'Base Model',
    baseModelRequired: 'Please select a model or enter a path',
    addDataset: 'Add Dataset',
    datasetSplit: 'Dataset split',
    datasetPath: 'Dataset path',
    sampleCount: 'Sample count',
    removeDataset: 'Remove dataset',
    evalSplit: 'Eval',
    submit: 'Create Task',
    rankingDirection: 'Ranking Direction',
    lowerRanking: 'Lower Is Better',
    docsPerGroup: 'Docs per Group',
    sftModelPath: 'SFT Model Path',
  },
} as const

type Language = keyof typeof copy
type Mode = 'dark' | 'light'

async function openForm(page: Page, language: Language, mode: Mode, width: number) {
  await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 })
  await page.addInitScript(
    ({ language, mode }) => {
      localStorage.setItem('tf_language', language)
      localStorage.setItem('tf_appearance:v1', JSON.stringify({ mode, accent: 'blue' }))
    },
    { language, mode }
  )
  await mockApi(page)
  await page.goto('/training/create')
  await expect(page.locator('html')).toHaveAttribute('data-theme', mode)
  await page
    .getByRole('button', { name: language === 'zh' ? '全部展开' : 'Expand all', exact: true })
    .click()
}

function selectControl(page: Page, label: string, scope: Page | Locator = page) {
  return scope.locator('.ant-select').filter({
    has: page.getByRole('combobox', { name: new RegExp(`^(?:\\*\\s*)?${label}$`) }),
  })
}

async function chooseOption(page: Page, control: Locator, option: string) {
  await expect(control).toBeVisible()
  await control.click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ has: page.getByText(option, { exact: true }) })
    .click()
  await expect(control).toContainText(option)
}

async function expectHorizontalBounds(locator: Locator, viewportWidth: number) {
  await expect(locator).toBeVisible()
  const bounds = await locator.boundingBox()
  expect(bounds).not.toBeNull()
  expect(bounds!.x).toBeGreaterThanOrEqual(0)
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewportWidth)
  return bounds!
}

async function expectReachable(page: Page, control: Locator) {
  await control.scrollIntoViewIfNeeded()
  await expect(control).toBeInViewport({ ratio: 0.99 })
  await expectHorizontalBounds(control, page.viewportSize()!.width)
  // A trial click also checks that another panel or a fixed footer does not cover the control.
  await control.click({ trial: true })
}

async function expectSectionLayout(page: Page, width: number) {
  const sections = page.locator('.training-create-section')
  await expect.poll(() => sections.count()).toBeGreaterThanOrEqual(2)
  const bounds = await sections.evaluateAll((elements) =>
    elements.map((element) => {
      const { x, y, width, height } = element.getBoundingClientRect()
      return { x, y, width, height }
    })
  )
  for (const section of bounds) {
    expect(section.x).toBeGreaterThanOrEqual(0)
    expect(section.x + section.width).toBeLessThanOrEqual(width)
  }
  if (width === 390) {
    for (const [index, section] of bounds.entries()) {
      expect(section.x).toBeCloseTo(bounds[0].x, 0)
      expect(section.width).toBeCloseTo(bounds[0].width, 0)
      if (index > 0) {
        const previous = bounds[index - 1]
        expect(section.y).toBeGreaterThanOrEqual(previous.y + previous.height - 1)
      }
    }
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
}

for (const width of [390, 1440]) {
  for (const mode of ['dark', 'light'] as const) {
    for (const language of ['zh', 'en'] as const) {
      test(`${width}px ${mode} ${language}: model, datasets, validation, and submit stay usable`, async ({
        page,
      }) => {
        const text = copy[language]
        let submissions = 0
        page.on('request', (request) => {
          if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/train') {
            submissions += 1
          }
        })
        await openForm(page, language, mode, width)

        const baseModel = selectControl(page, text.baseModel)
        const modelBounds = await expectHorizontalBounds(baseModel, width)
        // A narrow control can technically fit on screen while still being unusable on a phone.
        expect(modelBounds.width).toBeGreaterThanOrEqual(280)
        await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1)
        await expect(page.getByRole('heading', { level: 1, name: text.title })).toBeVisible()
        await expectSectionLayout(page, width)
        await expectReachable(page, baseModel)

        const submit = page.getByRole('button', { name: text.submit, exact: true })
        await expectReachable(page, submit)
        await submit.click()
        const requiredError = page.getByText(text.baseModelRequired, { exact: true })
        await expect(requiredError).toBeVisible()
        await requiredError.scrollIntoViewIfNeeded()
        await expect(requiredError).toBeInViewport({ ratio: 0.99 })
        await expectHorizontalBounds(requiredError, width)
        expect(submissions).toBe(0)

        await chooseOption(page, baseModel, 'bge-base-zh (embedding)')
        await expect(requiredError).toBeHidden()

        const rows = page.locator('.training-create-dataset-row')
        await expect(rows).toHaveCount(1)
        const addDataset = page.getByRole('button', { name: new RegExp(`${text.addDataset}$`) })
        await expectReachable(page, addDataset)
        await addDataset.click()
        await expect(rows).toHaveCount(2)

        const secondRow = rows.nth(1)
        await expectHorizontalBounds(secondRow, width)
        const datasetPath = selectControl(page, text.datasetPath, secondRow)
        const datasetSplit = selectControl(page, text.datasetSplit, secondRow)
        const sampleCount = secondRow.getByRole('spinbutton', {
          name: text.sampleCount,
          exact: true,
        })
        const removeDataset = secondRow.getByRole('button', {
          name: text.removeDataset,
          exact: true,
        })
        for (const control of [datasetPath, datasetSplit, sampleCount, removeDataset]) {
          await expectReachable(page, control)
        }
        const pathBounds = await expectHorizontalBounds(datasetPath, width)
        expect(pathBounds.width).toBeGreaterThanOrEqual(200)

        await datasetPath.click()
        await page.keyboard.type('/tmp/layout-example.jsonl')
        await page.keyboard.press('Enter')
        await page.keyboard.press('Escape')
        await expect(datasetPath).toContainText('/tmp/layout-example.jsonl')
        await chooseOption(page, datasetSplit, text.evalSplit)
        await sampleCount.fill('120')
        await sampleCount.press('Tab')
        await expect(sampleCount).toHaveValue('120')
        await removeDataset.click()
        await expect(rows).toHaveCount(1)
        await expect(rows.first().getByRole('button', { name: text.removeDataset })).toBeDisabled()

        await expectSectionLayout(page, width)
        await expectReachable(page, submit)
        expect(submissions).toBe(0)
      })
    }
  }
}

for (const width of [390, 1440]) {
  test(`${width}px: evaluation and save step fields stay reachable and submit their values`, async ({
    page,
  }) => {
    await openForm(page, 'zh', 'light', width)
    let submitted: Record<string, unknown> | null = null
    await page.route('**/api/train', async (route) => {
      submitted = route.request().postDataJSON()
      await route.fulfill({
        json: { task_id: 'layout-eval-save', status: 'pending', message: 'ok' },
      })
    })
    await chooseOption(page, selectControl(page, copy.zh.baseModel), 'bge-base-zh (embedding)')
    const datasetPath = selectControl(page, copy.zh.datasetPath)
    await datasetPath.click()
    await page.keyboard.type('/tmp/layout-train.jsonl')
    await page.keyboard.press('Enter')
    await page.keyboard.press('Escape')
    await chooseOption(page, selectControl(page, '评估策略'), '按步数评估')
    await chooseOption(page, selectControl(page, '保存策略'), '按步数保存')
    const evalSteps = page.locator('#eval_steps')
    const saveSteps = page.locator('#save_steps')
    await expectReachable(page, evalSteps)
    await evalSteps.fill('25')
    await expectReachable(page, saveSteps)
    await saveSteps.fill('50')
    await page.getByRole('button', { name: copy.zh.submit, exact: true }).click()
    await expect.poll(() => submitted).not.toBeNull()
    expect(submitted).toMatchObject({
      model_type: 'embedding',
      training_method: 'sft',
      eval_strategy: 'steps',
      eval_steps: 25,
      save_strategy: 'steps',
      save_steps: 50,
      num_train_epochs: 3,
      per_device_train_batch_size: 16,
      datasets: [{ path: '/tmp/layout-train.jsonl', split: 'train' }],
    })
  })
}

for (const language of ['zh', 'en'] as const) {
  test(`390px ${language}: LLM and Decoder Reranker conditional fields remain reachable`, async ({
    page,
  }) => {
    const text = copy[language]
    await openForm(page, language, language === 'zh' ? 'dark' : 'light', 390)
    const modelType = selectControl(page, text.modelType)
    const trainingMethod = selectControl(page, text.trainingMethod)

    await chooseOption(page, modelType, 'LLM')
    await expect(page.locator('.training-create-model-options')).toBeHidden()
    await chooseOption(page, trainingMethod, 'DPO')
    await expect(page.locator('.training-create-model-options')).toBeVisible()
    const beta = page.getByRole('spinbutton', { name: 'DPO Beta', exact: true })
    await expectReachable(page, beta)
    await beta.fill('0.25')
    await expect(beta).toHaveValue('0.25')
    const rankingDirection = selectControl(page, text.rankingDirection)
    await expectReachable(page, rankingDirection)
    await chooseOption(page, rankingDirection, text.lowerRanking)

    await chooseOption(page, modelType, 'Decoder Reranker')
    const docsPerGroup = page.getByRole('spinbutton', { name: text.docsPerGroup, exact: true })
    await expectReachable(page, docsPerGroup)
    await docsPerGroup.fill('12')
    await expect(docsPerGroup).toHaveValue('12')
    await chooseOption(page, trainingMethod, 'DPO')

    const sftPath = page.getByRole('textbox', {
      name: new RegExp(`^(?:\\*\\s*)?${text.sftModelPath}$`),
    })
    await expectReachable(page, sftPath)
    await sftPath.fill('/models/decoder-sft/final')
    await expect(sftPath).toHaveValue('/models/decoder-sft/final')
    await expectReachable(page, beta)
    await beta.fill('0.3')
    await expect(beta).toHaveValue('0.3')
    const referenceFree = page.getByRole('switch', { name: 'Reference Free', exact: true })
    await expectReachable(page, referenceFree)
    await referenceFree.click()
    await expect(referenceFree).toBeChecked()
    await expect(rankingDirection).toHaveCount(0)

    await expectSectionLayout(page, 390)
    await expectReachable(page, page.getByRole('button', { name: text.submit, exact: true }))
  })
}
