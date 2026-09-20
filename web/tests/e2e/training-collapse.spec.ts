import { test, expect, type Locator, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

async function openForm(page: Page, width = 1440, language: 'zh' | 'en' = 'zh') {
  await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 })
  await page.addInitScript((language) => {
    localStorage.setItem('tf_language', language)
    localStorage.setItem('tf_appearance:v1', JSON.stringify({ mode: 'light', accent: 'blue' }))
  }, language)
  await mockApi(page)
  await page.goto('/training/create')
  await expect(
    page.getByRole('heading', {
      level: 1,
      name: language === 'zh' ? '创建训练任务' : 'Create Training Task',
    })
  ).toBeVisible()
  await page
    .getByRole('button', { name: language === 'zh' ? '全部展开' : 'Expand all', exact: true })
    .click()
}

function selectControl(page: Page, label: string, scope: Page | Locator = page) {
  return scope.locator('.ant-select').filter({
    has: page.getByRole('combobox', { name: new RegExp(`^(?:\\*\\s*)?${label}$`) }),
  })
}

async function chooseOption(page: Page, label: string, option: string) {
  await selectControl(page, label).click()
  await page
    .locator('.ant-select-dropdown:visible .ant-select-item-option')
    .filter({ has: page.getByText(option, { exact: true }) })
    .click()
}

async function enterPath(page: Page, label: string, path: string, scope: Page | Locator = page) {
  await selectControl(page, label, scope).click()
  await page.keyboard.type(path)
  await page.keyboard.press('Enter')
  await page.keyboard.press('Escape')
}

test('a section can collapse from its heading and reopen without losing its values', async ({
  page,
}) => {
  await openForm(page)
  const title = page.getByRole('button', { name: '基础配置', exact: true })
  await expect(title).toHaveAttribute('aria-expanded', 'true')
  const taskName = page.locator('#task_name')
  await taskName.fill('保留我的训练配置')
  await title.click()
  await expect(title).toHaveAttribute('aria-expanded', 'false')
  await expect(taskName).toBeHidden()
  await expect(taskName).toHaveCount(1)
  await title.click()
  await expect(taskName).toBeVisible()
  await expect(taskName).toHaveValue('保留我的训练配置')
})

for (const width of [390, 1440]) {
  test(`${width}px: all sections collapse accessibly, retain values, and support Enter and Space`, async ({
    page,
  }, testInfo) => {
    const pageErrors: string[] = []
    page.on('pageerror', (error) => pageErrors.push(error.message))
    await openForm(page, width)
    const sectionTitles = [
      '基础配置',
      '数据集配置',
      '模型专项参数',
      '训练参数',
      '评估与保存',
      '微调方法',
      '资源配置',
    ]
    const controlledIds: string[] = []
    for (const title of sectionTitles) {
      const button = page.getByRole('button', { name: title, exact: true })
      await expect(button).toHaveAttribute('aria-expanded', 'true')
      const controlledId = await button.getAttribute('aria-controls')
      expect(controlledId).toBeTruthy()
      controlledIds.push(controlledId!)
      await expect(page.locator(`[id="${controlledId}"]`)).toBeVisible()
    }
    expect(new Set(controlledIds).size).toBe(sectionTitles.length)
    await page.locator('#task_name').fill('折叠后仍保留')
    await page.locator('#num_train_epochs').fill('7')
    await page.locator('#output_dir').fill('/tmp/collapse-output')
    await chooseOption(page, '评估策略', '按步数评估')
    await page.locator('#eval_steps').fill('33')
    await enterPath(page, '数据集路径', '/tmp/retained-train.jsonl')
    await page.getByRole('spinbutton', { name: '采样数量', exact: true }).fill('120')

    await page.getByRole('button', { name: '全部收起', exact: true }).click()
    for (const title of sectionTitles) {
      const button = page.getByRole('button', { name: title, exact: true })
      await expect(button).toBeVisible()
      await expect(button).toHaveAttribute('aria-expanded', 'false')
      const body = page.locator(`[id="${await button.getAttribute('aria-controls')}"]`)
      await expect(body).toBeHidden()
      await expect(body).toHaveCount(1)
    }
    await expect(page.locator('#num_train_epochs')).toHaveCount(1)
    await expect(page.locator('#eval_steps')).toHaveCount(1)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width
    )
    await page.screenshot({
      path: testInfo.outputPath(`all-collapsed-${width}.png`),
      fullPage: true,
    })

    const basic = page.getByRole('button', { name: '基础配置', exact: true })
    await basic.focus()
    await basic.press('Enter')
    await expect(basic).toHaveAttribute('aria-expanded', 'true')
    await expect(page.locator('#task_name')).toHaveValue('折叠后仍保留')
    await basic.press('Space')
    await expect(basic).toHaveAttribute('aria-expanded', 'false')
    await expect(basic).toBeFocused()

    await page.getByRole('button', { name: '全部展开', exact: true }).click()
    for (const title of sectionTitles) {
      await expect(page.getByRole('button', { name: title, exact: true })).toHaveAttribute(
        'aria-expanded',
        'true'
      )
    }
    await expect(page.locator('#task_name')).toHaveValue('折叠后仍保留')
    await expect(page.locator('#num_train_epochs')).toHaveValue('7')
    await expect(page.locator('#eval_steps')).toHaveValue('33')
    await expect(page.locator('#output_dir')).toHaveValue('/tmp/collapse-output')
    await expect(selectControl(page, '数据集路径')).toContainText('/tmp/retained-train.jsonl')
    await expect(page.getByRole('spinbutton', { name: '采样数量', exact: true })).toHaveValue('120')
    expect(pageErrors).toEqual([])
  })

  test(`${width}px: submitting hidden errors opens their sections and focuses the first invalid field`, async ({
    page,
  }) => {
    await openForm(page, width)
    let submissions = 0
    page.on('request', (request) => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/train') {
        submissions += 1
      }
    })
    const basic = page.getByRole('button', { name: '基础配置', exact: true })
    const training = page.getByRole('button', { name: '训练参数', exact: true })
    const datasets = page.getByRole('button', { name: '数据集配置', exact: true })
    const submit = page.getByRole('button', { name: '创建任务', exact: true })
    const epochs = page.locator('#num_train_epochs')
    await epochs.fill('')
    await page.getByRole('button', { name: '全部收起', exact: true }).click()
    await submit.click()
    await expect(basic).toHaveAttribute('aria-expanded', 'true')
    await expect(training).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByText('请选择模型或输入路径', { exact: true })).toBeVisible()
    await expect(page.locator('#base_model_path')).toBeFocused()
    await expect(datasets).toHaveAttribute('aria-expanded', 'false')
    expect(submissions).toBe(0)

    await chooseOption(page, '基础模型', 'bge-base-zh (embedding)')
    await page.getByRole('button', { name: '全部收起', exact: true }).click()
    await submit.click()
    await expect(training).toHaveAttribute('aria-expanded', 'true')
    await expect(epochs).toBeFocused()
    await expect(basic).toHaveAttribute('aria-expanded', 'false')
    expect(submissions).toBe(0)

    await epochs.fill('3')
    await page.getByRole('button', { name: '全部收起', exact: true }).click()
    await submit.click()
    await expect(datasets).toHaveAttribute('aria-expanded', 'true')
    await expect(page.getByText('请至少选择一个训练集', { exact: true })).toBeVisible()
    await expect(page.getByRole('combobox', { name: '数据集路径', exact: true })).toBeFocused()
    await expect(training).toHaveAttribute('aria-expanded', 'false')
    expect(submissions).toBe(0)
  })
}

test('390px: adding a dataset from its collapsed header expands it without clearing existing rows', async ({
  page,
}) => {
  await openForm(page, 390)
  const rows = page.locator('.training-create-dataset-row')
  await enterPath(page, '数据集路径', '/tmp/keep-first.jsonl', rows.first())
  await rows.first().getByRole('spinbutton', { name: '采样数量', exact: true }).fill('54')
  const datasets = page.getByRole('button', { name: '数据集配置', exact: true })
  await datasets.click()
  await expect(rows.first()).toBeHidden()
  await page.getByRole('button', { name: /添加数据集$/ }).click()
  await expect(datasets).toHaveAttribute('aria-expanded', 'true')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(1)).toBeVisible()
  await expect(selectControl(page, '数据集路径', rows.first())).toContainText(
    '/tmp/keep-first.jsonl'
  )
  await expect(rows.first().getByRole('spinbutton', { name: '采样数量', exact: true })).toHaveValue(
    '54'
  )
})

test('collapsed DPO settings remain in the submitted training request', async ({ page }) => {
  await openForm(page)
  let submitted: Record<string, unknown> | null = null
  await page.route('**/api/train', async (route) => {
    submitted = route.request().postDataJSON()
    await route.fulfill({
      json: { task_id: 'collapse-dpo', status: 'pending', message: 'ok' },
    })
  })
  await chooseOption(page, '模型类型', 'LLM')
  await chooseOption(page, '训练方法', 'DPO')
  await enterPath(page, '基础模型', '/models/collapse-llm')
  await enterPath(page, '数据集路径', '/tmp/collapse-dpo.jsonl')
  await page.getByRole('spinbutton', { name: 'DPO Beta', exact: true }).fill('0.35')
  await chooseOption(page, '排序方向', '排名越小越好')
  await page.locator('#num_train_epochs').fill('7')
  await page.getByRole('button', { name: '全部收起', exact: true }).click()
  await expect(page.locator('#rl_config_beta')).toHaveCount(1)
  await expect(page.locator('#rl_config_beta')).toBeHidden()
  await page.getByRole('button', { name: '创建任务', exact: true }).click()
  await expect.poll(() => submitted).not.toBeNull()
  expect(submitted).toMatchObject({
    model_type: 'llm',
    training_method: 'dpo',
    base_model_path: '/models/collapse-llm',
    num_train_epochs: 7,
    per_device_train_batch_size: 1,
    gradient_checkpointing: true,
    bf16: true,
    rl_config: { beta: 0.35, rankings_direction: 'lower_is_better' },
    datasets: [{ path: '/tmp/collapse-dpo.jsonl', split: 'train' }],
  })
})

test('390px: English expand and collapse controls remain reachable', async ({ page }) => {
  await openForm(page, 390, 'en')
  const collapse = page.getByRole('button', { name: 'Collapse all', exact: true })
  const expand = page.getByRole('button', { name: 'Expand all', exact: true })
  for (const control of [collapse, expand]) {
    const bounds = await control.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
  }
  await collapse.click()
  await expect(page.locator('#task_name')).toBeHidden()
  await expand.click()
  await expect(page.locator('#task_name')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
})
