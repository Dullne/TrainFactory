import { expect, test, type ConsoleMessage, type Locator, type Page } from '@playwright/test'
import {
  createLossChartModel,
  normalizeLossHistory,
  type LossChartLabels,
} from '../../src/components/training/LossCurveDataTable'
import { mockApi } from './helpers/mockApi'

type LossMetric = Record<string, unknown>

interface LocaleExpectations {
  language: 'zh' | 'en'
  title: string
  loading: string
  empty: string
  rangeUnsupported: string
  axisStep: string
  trainSeries: string
  evalSeries: string
  tooltipStep: string
  ariaDescription: string
  tableToggle: string
  tableCaption: string
  missing: string
}

const locales: LocaleExpectations[] = [
  {
    language: 'zh',
    title: 'Loss 曲线',
    loading: '正在加载训练指标…',
    empty: '暂无 Loss 数据',
    rangeUnsupported: '数值范围过大，无法绘制图表。请查看 Loss 数据表获取准确值。',
    axisStep: '训练步数',
    trainSeries: '训练 Loss',
    evalSeries: '验证 Loss',
    tooltipStep: '步数',
    ariaDescription: '训练 Loss 与验证 Loss 随训练步数变化的折线图',
    tableToggle: '查看 Loss 数据表',
    tableCaption: 'Loss 曲线数据',
    missing: '无',
  },
  {
    language: 'en',
    title: 'Loss Curve',
    loading: 'Loading training metrics…',
    empty: 'No loss data available',
    rangeUnsupported:
      'This numeric range cannot be plotted. View the loss data table for the exact values.',
    axisStep: 'Training step',
    trainSeries: 'Training loss',
    evalSeries: 'Evaluation loss',
    tooltipStep: 'Step',
    ariaDescription: 'Line chart of training loss and evaluation loss by training step',
    tableToggle: 'View loss data table',
    tableCaption: 'Loss curve data',
    missing: 'Not available',
  },
]

const expectedTrainPoints = [
  [1, 0.987654],
  [2, 0.5],
  [4, 0.25],
  [7, 0.07],
  [9, 0.7],
  [11, 0.11],
]

const expectedEvalPoints = [
  [1, 0.8765],
  [3, 0.333333],
  [5, 0.125],
  [8, 0.0087654],
  [9, 0.8],
  [10, 0.2],
]

const knownTrainingDetailDeprecations = new Set([
  'Warning: [antd: Progress] `strokeWidth` is deprecated. Please use `size` instead.',
  'Warning: [antd: Card] `bodyStyle` is deprecated. Please use `styles.body` instead.',
])

interface PageProblemGate {
  problems: string[]
  onConsole(message: ConsoleMessage): void
  onPageError(error: Error): void
}

const pageProblemGates = new WeakMap<Page, PageProblemGate>()

test.beforeEach(async ({ page }) => {
  const gate: PageProblemGate = {
    problems: [],
    onConsole(message) {
      if (message.type() !== 'warning' && message.type() !== 'error') return
      if (knownTrainingDetailDeprecations.has(message.text())) return
      gate.problems.push(`${message.type()}: ${message.text()}`)
    },
    onPageError(error) {
      gate.problems.push(`pageerror: ${error.message}`)
    },
  }

  if (pageProblemGates.has(page)) throw new Error('Duplicate loss-curve page problem gate')
  pageProblemGates.set(page, gate)
  page.on('console', gate.onConsole)
  page.on('pageerror', gate.onPageError)
})

test.afterEach(async ({ page }) => {
  const gate = pageProblemGates.get(page)
  if (!gate) throw new Error('Missing loss-curve page problem gate')

  try {
    if (!page.isClosed()) await waitForAnimationFrames(page)
    expect(gate.problems).toEqual([])
  } finally {
    page.off('console', gate.onConsole)
    page.off('pageerror', gate.onPageError)
    pageProblemGates.delete(page)
  }
})

const comprehensiveHistory: LossMetric[] = [
  { step: 1, train_loss: 0.987654, eval_loss: 0.8765 },
  { step: 2, train_loss: 0.5 },
  { step: 3, eval_loss: 0.333333 },
  { global_step: 4, loss: '0.25' },
  { current_step: '5', eval_validation_loss: '0.12500' },
  { step: null, global_step: 7, train_loss: '0.07' },
  { step: '', current_step: 8, eval_alpha_loss: 0.0087654 },
  { step: 9, train_loss: 0.9 },
  { step: 9, eval_loss: 0.8 },
  { step: 9, train_loss: 0.7 },
  { step: 10, eval_z_loss: 0.1, eval_a_loss: 0.2 },
  { step: 11, loss: 0.11 },
  { step: 11, train_loss: 'not-a-number' },
  { step: 12, train_loss: false, eval_loss: ' ' },
  { step: null, global_step: ' ', current_step: false, train_loss: 99 },
  { step: true, global_step: null, current_step: undefined, eval_loss: 98 },
  { step: 'not-a-number', global_step: '', current_step: null, loss: 97 },
]

const unsupportedSignedHistory: LossMetric[] = [
  { step: 1, train_loss: -Number.MAX_VALUE },
  { step: 2, train_loss: Number.MAX_VALUE },
]

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>((release) => {
    resolve = release
  })
  return { promise, resolve }
}

async function preparePage(page: Page, language: 'zh' | 'en') {
  await page.addInitScript((storedLanguage) => {
    window.localStorage.setItem('tf_language', storedLanguage)
  }, language)
  await page.addInitScript(() => {
    if (typeof window.ResizeObserver !== 'undefined') return

    class ResizeObserverShim {
      private readonly observed = new Map<Element, { width: number; height: number }>()
      private frame: number | null = null

      constructor(private readonly callback: ResizeObserverCallback) {}

      observe(target: Element) {
        const rect = target.getBoundingClientRect()
        this.observed.set(target, { width: rect.width, height: rect.height })
        if (this.frame === null) this.frame = window.requestAnimationFrame(() => this.check())
      }

      unobserve(target: Element) {
        this.observed.delete(target)
        if (this.observed.size === 0 && this.frame !== null) {
          window.cancelAnimationFrame(this.frame)
          this.frame = null
        }
      }

      disconnect() {
        this.observed.clear()
        if (this.frame !== null) window.cancelAnimationFrame(this.frame)
        this.frame = null
      }

      private check() {
        const entries: ResizeObserverEntry[] = []
        for (const [target, previous] of this.observed) {
          const rect = target.getBoundingClientRect()
          if (rect.width === previous.width && rect.height === previous.height) continue
          this.observed.set(target, { width: rect.width, height: rect.height })
          entries.push({ target, contentRect: rect } as ResizeObserverEntry)
        }
        if (entries.length > 0) {
          this.callback(entries, this as unknown as ResizeObserver)
        }
        this.frame = window.requestAnimationFrame(() => this.check())
      }
    }

    window.ResizeObserver = ResizeObserverShim as unknown as typeof ResizeObserver
  })
  await mockApi(page)
}

async function mockLossHistory(page: Page, history: LossMetric[], waitForRelease?: Promise<void>) {
  await page.route('**/api/train/*/metrics**', async (route) => {
    if (waitForRelease) await waitForRelease
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-loss-curve',
        loss_history: history,
        summary: null,
        current_metrics: null,
        has_data: history.length > 0,
      }),
    })
  })
}

function lossCard(page: Page, title: string) {
  return page
    .locator('.ant-card')
    .filter({ has: page.locator('.ant-card-head-title', { hasText: title }) })
    .first()
}

async function readTable(table: Locator) {
  return table
    .locator('tbody tr')
    .evaluateAll((rows) =>
      rows.map((row) =>
        Array.from(row.querySelectorAll('th[scope="row"], td')).map(
          (cell) => cell.textContent?.trim() ?? ''
        )
      )
    )
}

function chartLabels(locale: LocaleExpectations): LossChartLabels {
  return {
    stepAxis: locale.axisStep,
    trainSeries: locale.trainSeries,
    evalSeries: locale.evalSeries,
    tooltipStep: locale.tooltipStep,
  }
}

function expectCompleteChartModel(locale: LocaleExpectations) {
  const rows = normalizeLossHistory(comprehensiveHistory)
  const model = createLossChartModel(rows, chartLabels(locale))

  expect(model.renderable).toBe(true)
  expect(model.legend).toEqual([locale.trainSeries, locale.evalSeries])
  expect(model.xAxis.name).toBe(locale.axisStep)
  expect(model.xAxis.formatLabel(1000)).toBe('1.0k')
  expect(model.series.map(({ name, data }) => ({ name, data }))).toEqual([
    { name: locale.trainSeries, data: expectedTrainPoints },
    { name: locale.evalSeries, data: expectedEvalPoints },
  ])
  expect(
    model.formatTooltip([
      { seriesName: locale.trainSeries, value: expectedTrainPoints[0] },
      { seriesName: locale.evalSeries, value: expectedEvalPoints[0] },
    ])
  ).toBe(
    [
      `${locale.tooltipStep}: 1`,
      `${locale.trainSeries}: 0.987654`,
      `${locale.evalSeries}: 0.8765`,
    ].join('<br/>')
  )
}

async function waitForAnimationFrames(page: Page) {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve()))
      })
  )
}

const axisTestLabels: LossChartLabels = {
  stepAxis: 'Step',
  trainSeries: 'Training loss',
  evalSeries: 'Evaluation loss',
  tooltipStep: 'Step',
}

function nextUp(value: number) {
  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, value)
  view.setBigUint64(0, view.getBigUint64(0) + 1n)
  return view.getFloat64(0)
}

function expectProvidedAxisIsSafe(yAxis: { min?: number; max?: number; interval?: number }) {
  for (const value of [yAxis.min, yAxis.max, yAxis.interval]) {
    if (value !== undefined) expect(Number.isFinite(value)).toBe(true)
  }
  if (yAxis.interval !== undefined) {
    expect(yAxis.min).toBeDefined()
    expect(yAxis.max).toBeDefined()
    expect(yAxis.interval).toBeGreaterThan(0)
    expect(yAxis.min! + yAxis.interval).toBeGreaterThan(yAxis.min!)
  }
}

test('loss chart axis preserves distinct labels for tiny finite values', () => {
  const { yAxis } = createLossChartModel(
    [
      { step: 1, train: 1e-6, eval: null },
      { step: 2, train: 2e-6, eval: null },
    ],
    axisTestLabels
  )

  expectProvidedAxisIsSafe(yAxis)
  expect([yAxis.min, yAxis.max, yAxis.interval].every(Number.isFinite)).toBe(true)
  expect(yAxis.min!).toBeLessThan(yAxis.max!)
  expect(yAxis.interval).toBeGreaterThan(0)
  expect(yAxis.min).toBeLessThanOrEqual(1e-6)
  expect(yAxis.max).toBeGreaterThanOrEqual(2e-6)
  expect(yAxis.formatLabel(1e-6)).not.toBe(yAxis.formatLabel(2e-6))
  expect(yAxis.formatLabel(1e-6)).toMatch(/e/i)
  expect(yAxis.formatLabel(2e-6)).toMatch(/e/i)
})

test('loss chart axis remains finite for values near Number.MAX_VALUE', () => {
  const lower = Number.MAX_VALUE / 2
  const upper = Number.MAX_VALUE
  const { yAxis } = createLossChartModel(
    [
      { step: 1, train: lower, eval: null },
      { step: 2, train: upper, eval: null },
    ],
    axisTestLabels
  )

  expectProvidedAxisIsSafe(yAxis)
  expect([yAxis.min, yAxis.max, yAxis.interval].every(Number.isFinite)).toBe(true)
  expect(yAxis.min!).toBeLessThan(yAxis.max!)
  expect(yAxis.interval).toBeGreaterThan(0)
  expect(yAxis.min).toBeLessThanOrEqual(lower)
  expect(yAxis.max).toBeGreaterThanOrEqual(upper)
  expect(yAxis.formatLabel(lower)).not.toBe(yAxis.formatLabel(upper))
  expect(yAxis.formatLabel(lower)).toMatch(/e/i)
  expect(yAxis.formatLabel(upper)).toMatch(/e/i)
})

test('loss chart axis keeps adjacent high-offset tick labels distinct', () => {
  const lower = 1_000_000
  const upper = 1_000_000.000001
  const { yAxis } = createLossChartModel(
    [
      { step: 1, train: lower, eval: null },
      { step: 2, train: upper, eval: null },
    ],
    axisTestLabels
  )

  expectProvidedAxisIsSafe(yAxis)
  expect(yAxis.interval).toBeDefined()
  const nextTick = lower + yAxis.interval!
  expect(nextTick).toBeGreaterThan(lower)
  expect(yAxis.formatLabel(lower)).not.toBe(yAxis.formatLabel(nextTick))
  expect(yAxis.formatLabel(lower)).not.toBe(yAxis.formatLabel(upper))
})

test('loss chart axis advances and labels nextUp values around one distinctly', () => {
  const lower = 1
  const upper = 1 + Number.EPSILON
  expect(nextUp(lower)).toBe(upper)
  const { yAxis } = createLossChartModel(
    [
      { step: 1, train: lower, eval: null },
      { step: 2, train: upper, eval: null },
    ],
    axisTestLabels
  )

  expectProvidedAxisIsSafe(yAxis)
  expect(yAxis.formatLabel(lower)).not.toBe(yAxis.formatLabel(upper))
})

test('loss chart axis advances and labels nextUp values around one million distinctly', () => {
  const lower = 1_000_000
  const upper = nextUp(lower)
  const adjacentDifference = upper - lower
  expect(adjacentDifference).toBeGreaterThan(0)
  const { yAxis } = createLossChartModel(
    [
      { step: 1, train: lower, eval: null },
      { step: 2, train: upper, eval: null },
    ],
    axisTestLabels
  )

  expectProvidedAxisIsSafe(yAxis)
  expect(yAxis.interval!).toBeGreaterThanOrEqual(adjacentDifference)
  expect(yAxis.formatLabel(lower)).not.toBe(yAxis.formatLabel(upper))
})

test('loss chart model rejects an unsafe signed y span with safe fallback labels', () => {
  const model = createLossChartModel(
    [
      { step: 1, train: -Number.MAX_VALUE, eval: null },
      { step: 2, train: Number.MAX_VALUE, eval: null },
    ],
    axisTestLabels
  )
  const { yAxis } = model

  expect(model.renderable).toBe(false)
  expectProvidedAxisIsSafe(yAxis)
  expect(yAxis.min).toBeUndefined()
  expect(yAxis.max).toBeUndefined()
  expect(yAxis.interval).toBeUndefined()
  expect(yAxis.formatLabel(-Number.MAX_VALUE)).not.toBe(yAxis.formatLabel(Number.MAX_VALUE))
})

test('loss chart model rejects an unrepresentable signed x-axis span', () => {
  const model = createLossChartModel(
    [
      { step: -Number.MAX_VALUE, train: 1, eval: null },
      { step: Number.MAX_VALUE, train: 2, eval: null },
    ],
    axisTestLabels
  )

  expect(model.renderable).toBe(false)
})

test('loss chart model keeps same-sign near-MAX x and y spans renderable', () => {
  const model = createLossChartModel(
    [
      { step: Number.MAX_VALUE / 2, train: Number.MAX_VALUE / 2, eval: null },
      { step: Number.MAX_VALUE, train: Number.MAX_VALUE, eval: null },
    ],
    axisTestLabels
  )

  expect(model.renderable).toBe(true)
})

for (const locale of locales) {
  test(`${locale.language} loss curve localizes chart semantics and exposes one normalized table`, async ({
    page,
  }) => {
    expectCompleteChartModel(locale)
    const metricsGate = deferred()
    await preparePage(page, locale.language)
    await mockLossHistory(page, comprehensiveHistory, metricsGate.promise)

    await page.goto('/training/task-loss-curve')
    const card = lossCard(page, locale.title)
    await expect(card.locator('.ant-card-head-title')).toHaveText(locale.title)
    await expect(card.getByText(locale.loading, { exact: true })).toBeVisible()

    metricsGate.resolve()

    const chart = card.locator('div[role="img"]')
    await expect(chart).toHaveAttribute('aria-label', locale.ariaDescription)
    await expect(chart.locator('canvas')).toHaveCount(1)
    await expect(chart).toHaveAttribute('aria-label', new RegExp(locale.axisStep, 'i'))
    await expect(chart).toHaveAttribute('aria-label', new RegExp(locale.trainSeries, 'i'))
    await expect(chart).toHaveAttribute('aria-label', new RegExp(locale.evalSeries, 'i'))

    const details = card.locator('details')
    await expect(details.locator('summary')).toHaveText(locale.tableToggle)
    await details.locator('summary').click()

    const table = card.getByRole('table', { name: locale.tableCaption })
    await expect(table.getByRole('columnheader', { name: locale.axisStep })).toBeVisible()
    await expect(table.getByRole('columnheader', { name: locale.trainSeries })).toBeVisible()
    await expect(table.getByRole('columnheader', { name: locale.evalSeries })).toBeVisible()
    await expect(table.locator('thead th[scope="col"]')).toHaveCount(3)
    await expect(table.getByRole('rowheader')).toHaveCount(11)
    await expect
      .poll(() => readTable(table))
      .toEqual([
        ['1', '0.987654', '0.8765'],
        ['2', '0.5', locale.missing],
        ['3', locale.missing, '0.333333'],
        ['4', '0.25', locale.missing],
        ['5', locale.missing, '0.125'],
        ['7', '0.07', locale.missing],
        ['8', locale.missing, '0.0087654'],
        ['9', '0.7', '0.8'],
        ['10', locale.missing, '0.2'],
        ['11', '0.11', locale.missing],
        ['12', locale.missing, locale.missing],
      ])
  })

  test(`${locale.language} loss curve falls back to its table for an unsupported signed range`, async ({
    page,
  }) => {
    await preparePage(page, locale.language)
    await mockLossHistory(page, unsupportedSignedHistory)

    await page.goto('/training/task-loss-curve')

    const card = lossCard(page, locale.title)
    const details = card.locator('details')
    await expect(details).toBeVisible()
    expect.soft(await card.getByText(locale.rangeUnsupported, { exact: true }).count()).toBe(1)
    expect.soft(await card.locator('div[role="img"]').count()).toBe(0)
    expect.soft(await card.locator('canvas').count()).toBe(0)

    await details.locator('summary').click()
    const table = card.getByRole('table', { name: locale.tableCaption })
    await expect(table.getByRole('rowheader', { name: '1', exact: true })).toBeVisible()
    await expect(table.getByRole('rowheader', { name: '2', exact: true })).toBeVisible()
    await expect(table.getByRole('rowheader')).toHaveCount(2)
    await expect
      .poll(() => readTable(table))
      .toEqual([
        ['1', String(-Number.MAX_VALUE), locale.missing],
        ['2', String(Number.MAX_VALUE), locale.missing],
      ])
  })

  test(`${locale.language} loss curve localizes the empty history state`, async ({ page }) => {
    await preparePage(page, locale.language)
    await mockLossHistory(page, [])

    await page.goto('/training/task-loss-curve')

    const card = lossCard(page, locale.title)
    await expect(card.locator('.ant-card-head-title')).toHaveText(locale.title)
    await expect(card.getByText(locale.empty, { exact: true })).toBeVisible()
    await expect(card.locator('div[role="img"]')).toHaveCount(0)
    await expect(card.locator('details')).toHaveCount(0)
  })
}

test('ResizeObserver resizes the rendered canvas and navigation remounts cleanly', async ({
  page,
}) => {
  await preparePage(page, 'en')
  await mockLossHistory(page, comprehensiveHistory)
  await page.goto('/training/task-loss-curve')

  const card = lossCard(page, 'Loss Curve')
  const chart = card.getByRole('img', {
    name: locales[1].ariaDescription,
    exact: true,
  })
  const canvas = chart.locator('canvas')
  await expect(canvas).toHaveCount(1)

  const initialBox = await canvas.boundingBox()
  expect(initialBox).not.toBeNull()
  const targetWidth = Math.max(320, Math.floor(initialBox!.width - 180))
  expect(initialBox!.width - targetWidth).toBeGreaterThan(100)

  await chart.evaluate((element, width) => {
    ;(element as HTMLElement).style.width = `${width}px`
  }, targetWidth)

  await expect
    .poll(async () => (await canvas.boundingBox())?.width ?? 0)
    .toBeGreaterThan(targetWidth - 2)
  await expect
    .poll(async () => (await canvas.boundingBox())?.width ?? 0)
    .toBeLessThan(targetWidth + 2)

  await page.locator('.ant-menu-item').filter({ hasText: 'Datasets' }).click()
  await expect(page).toHaveURL(/\/datasets$/)
  await page.goBack()
  await expect(page).toHaveURL(/\/training\/task-loss-curve$/)
  await expect(lossCard(page, 'Loss Curve').locator('canvas')).toBeVisible()
  await waitForAnimationFrames(page)
})
