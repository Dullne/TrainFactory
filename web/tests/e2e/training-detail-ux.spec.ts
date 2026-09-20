import { expect, test, type Page } from '@playwright/test'
import { mockApi } from './helpers/mockApi'
import type { TrainingTaskEvent } from '../../src/types'

const task = {
  task_id: 'detail-ux',
  task_name: 'Detail UX task',
  model_type: 'embedding',
  training_method: 'sft',
  base_model_path: '',
  train_dataset_path: '',
  output_dir: '/output',
  status: 'succeeded',
  progress: 100,
  created_at: '2026-09-19T01:00:00Z',
  updated_at: '2026-09-19T01:05:00Z',
}

async function setup(page: Page, language = 'en', overrides: Record<string, unknown> = {}) {
  await page.addInitScript((lang) => localStorage.setItem('tf_language', lang), language)
  await mockApi(page)
  await page.route('**/api/train/detail-ux', (route) =>
    route.fulfill({ json: { ...task, ...overrides } })
  )
  await page.route('**/api/train/detail-ux/metrics**', (route) =>
    route.fulfill({
      json: {
        task_id: task.task_id,
        loss_history: [],
        summary: null,
        current_metrics: null,
        has_data: false,
      },
    })
  )
}

function events(count: number): TrainingTaskEvent[] {
  return Array.from({ length: count }, (_, index) => ({
    event_id: `event-${index}`,
    task_id: task.task_id,
    event_type: 'progress_updated',
    payload: { progress: index },
    created_at: '2026-09-19T01:00:00Z',
  }))
}

for (const language of ['zh', 'en']) {
  test(`${language}: events have readable transitions, safe unknown names and usable history`, async ({
    page,
  }) => {
    await setup(page, language)
    const history = events(12)
    history[0] = {
      ...history[0],
      event_type: 'status_changed',
      payload: { from: 'pending', to: 'running' },
    }
    history[1] = { ...history[1], event_type: '<img src=x onerror=alert(1)>' }
    await page.route('**/api/train/detail-ux/events**', (route) =>
      route.fulfill({ json: { events: history, total: 12 } })
    )
    await page.goto('/training/detail-ux')
    const card = page.locator('.ant-card').filter({
      has: page.locator('.ant-card-head-title', {
        hasText: language === 'zh' ? '任务事件' : 'Task Events',
      }),
    })
    await expect(
      card.getByText(language === 'zh' ? '状态已变更' : 'Status changed', { exact: true })
    ).toBeVisible()
    await expect(
      card.getByText(language === 'zh' ? '等待中 → 运行中' : 'Pending → Running', { exact: true })
    ).toBeVisible()
    await expect(
      card.getByText(
        language === 'zh'
          ? '其他事件：<img src=x onerror=alert(1)>'
          : 'Other event: <img src=x onerror=alert(1)>',
        { exact: true }
      )
    ).toBeVisible()
    await expect(card.locator('img')).toHaveCount(0)
    await expect(card.getByRole('listitem')).toHaveCount(10)
    await card
      .getByRole('button', { name: language === 'zh' ? '查看更多事件' : 'Show more events' })
      .click()
    await expect(card.getByRole('listitem')).toHaveCount(12)
    await expect(
      card.getByText(
        language === 'zh' ? '已显示 12 / 12 条已加载事件' : 'Showing 12 of 12 loaded events'
      )
    ).toBeVisible()
    await expect(
      card.getByRole('button', { name: /查看更多事件|Show more events|Load older events/ })
    ).toHaveCount(0)
  })
}

test('history grows the supported limit and retains records after an older-history failure', async ({
  page,
}) => {
  await setup(page)
  const requested: number[] = []
  let failOlder = true
  await page.route('**/api/train/detail-ux/events**', (route) => {
    const limit = Number(new URL(route.request().url()).searchParams.get('limit'))
    requested.push(limit)
    if (limit > 100 && failOlder)
      return route.fulfill({ status: 503, json: { detail: 'Unavailable' } })
    const history = events(Math.min(limit, 105))
    return route.fulfill({ json: { events: history, total: history.length } })
  })
  await page.goto('/training/detail-ux')
  const card = page
    .locator('.ant-card')
    .filter({ has: page.locator('.ant-card-head-title', { hasText: 'Task Events' }) })
  for (let i = 0; i < 9; i += 1)
    await card.getByRole('button', { name: 'Show more events' }).click()
  await expect(card.getByRole('listitem')).toHaveCount(100)
  await card.getByRole('button', { name: 'Load older events' }).click()
  await expect(
    card.getByText('Events could not be loaded. Previously loaded records are preserved.')
  ).toBeVisible()
  await expect(card.getByRole('listitem')).toHaveCount(100)
  failOlder = false
  await card.getByRole('button', { name: 'Load older events' }).click()
  await expect(card.getByRole('listitem')).toHaveCount(105)
  expect(requested[0]).toBe(100)
  // StrictMode may issue two initial loads; each explicit older-history attempt uses 200.
  expect(requested.filter((limit) => limit > 100)).toEqual([200, 200])
  await expect(card.getByRole('button', { name: 'Load older events' })).toHaveCount(0)
})

test('history stops at the supported 500-event boundary without claiming an all-time total', async ({
  page,
}) => {
  await setup(page)
  const requested: number[] = []
  await page.route('**/api/train/detail-ux/events**', (route) => {
    const limit = Number(new URL(route.request().url()).searchParams.get('limit'))
    requested.push(limit)
    return route.fulfill({ json: { events: events(limit), total: limit } })
  })
  await page.goto('/training/detail-ux')
  const card = page
    .locator('.ant-card')
    .filter({ has: page.locator('.ant-card-head-title', { hasText: 'Task Events' }) })
  for (let visible = 10; visible < 500; visible += 10) {
    await card
      .getByRole('button', { name: visible % 100 ? 'Show more events' : 'Load older events' })
      .click()
    await expect(card.getByRole('listitem')).toHaveCount(visible + 10)
  }
  await expect(card.getByText('Showing 500 of 500 loaded events')).toBeVisible()
  await expect(
    card.getByText('Only the latest 500 events are available here; older records are not included.')
  ).toBeVisible()
  await expect(card.getByRole('button', { name: /more events|older events/ })).toHaveCount(0)
  expect(requested.filter((limit) => limit > 100)).toEqual([200, 300, 400, 500])
})

test('a poll during an older-history request preserves the expanded history scope', async ({
  page,
}) => {
  await page.clock.install()
  await setup(page, 'en', { status: 'running' })
  const requested: number[] = []
  let release!: () => void
  const delayed = new Promise<void>((resolve) => {
    release = resolve
  })
  let olderRequests = 0
  await page.route('**/api/train/detail-ux/events**', async (route) => {
    const limit = Number(new URL(route.request().url()).searchParams.get('limit'))
    requested.push(limit)
    if (limit === 200 && ++olderRequests === 1) await delayed
    const history = events(Math.min(limit, 105))
    await route.fulfill({ json: { events: history, total: history.length } })
  })
  await page.goto('/training/detail-ux')
  const card = page
    .locator('.ant-card')
    .filter({ has: page.locator('.ant-card-head-title', { hasText: 'Task Events' }) })
  for (let i = 0; i < 9; i += 1)
    await card.getByRole('button', { name: 'Show more events' }).click()
  await card.getByRole('button', { name: 'Load older events' }).click()
  try {
    await expect.poll(() => olderRequests).toBe(1)
    await page.clock.fastForward(5000)
    await expect.poll(() => requested.slice(-2)).toEqual([200, 200])
  } finally {
    release()
  }
  await expect(card.getByRole('listitem')).toHaveCount(105)
})

test('initial event failure has a retry and does not claim that the task has no events', async ({
  page,
}) => {
  await setup(page)
  let fail = true
  await page.route('**/api/train/detail-ux/events**', (route) =>
    route.fulfill(
      fail
        ? { status: 503, json: { detail: 'Unavailable' } }
        : { json: { events: events(1), total: 1 } }
    )
  )
  await page.goto('/training/detail-ux')
  const card = page
    .locator('.ant-card')
    .filter({ has: page.locator('.ant-card-head-title', { hasText: 'Task Events' }) })
  await expect(card.getByRole('button', { name: 'Retry loading events' })).toBeVisible()
  await expect(card.getByText('No events yet')).toHaveCount(0)
  fail = false
  await card.getByRole('button', { name: 'Retry loading events' }).click()
  await expect(card.getByRole('listitem')).toHaveCount(1)
  await expect(card.getByRole('alert')).toHaveCount(0)
})

for (const width of [1440, 390]) {
  test(`${width}px: empty detail and event history remain readable and keyboard accessible`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 })
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await setup(page)
    await page.route('**/api/train/detail-ux/events**', (route) =>
      route.fulfill({ json: { events: events(12), total: 12 } })
    )
    await page.goto('/training/detail-ux')
    const more = page.getByRole('button', { name: 'Show more events' })
    await more.focus()
    await page.keyboard.press('Enter')
    await expect(page.getByText('Showing 12 of 12 loaded events')).toBeVisible()
    await page.getByRole('region', { name: 'Task Events' }).focus()
    await page.keyboard.press('End')
    await expect(page.getByRole('listitem').last()).toBeInViewport()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
    ).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`detail-${width}.png`), fullPage: true })
    expect(errors).toEqual([])
  })
}

for (const [status, hint] of [
  ['pending', 'The task has not started recording loss.'],
  ['running', 'No plottable loss records have arrived yet.'],
  ['succeeded', 'The task has ended without plottable loss records.'],
  ['failed', 'The task failed without plottable loss records.'],
]) {
  test(`empty loss explains the actual ${status} state`, async ({ page }) => {
    await setup(page, 'en', { status })
    await page.goto('/training/detail-ux')
    await expect(page.getByText(hint, { exact: false })).toBeVisible()
    const progress = page
      .locator('.ant-card')
      .filter({ has: page.locator('.ant-card-head-title', { hasText: 'Training Progress' }) })
    await expect(progress.locator('.ant-tag')).toHaveCount(0)
  })
}

test('failed metrics are not called empty and can be retried without navigating away', async ({
  page,
}) => {
  await setup(page)
  let failMetrics = true
  await page.route('**/api/train/detail-ux/metrics**', (route) =>
    route.fulfill(
      failMetrics
        ? { status: 503, json: { detail: 'Unavailable' } }
        : {
            json: {
              task_id: task.task_id,
              loss_history: [{ step: 1, train_loss: 0.5 }],
              summary: null,
              current_metrics: null,
              has_data: true,
            },
          }
    )
  )
  await page.goto('/training/detail-ux')
  await expect(
    page.getByText('Training metrics could not be loaded, so loss availability is unknown.')
  ).toBeVisible()
  failMetrics = false
  await page.getByRole('button', { name: 'Reload metrics' }).click()
  await expect(
    page.getByRole('img', {
      name: 'Line chart of training loss and evaluation loss by training step',
    })
  ).toBeVisible()
  await expect(
    page.getByText('Training metrics could not be loaded, so loss availability is unknown.')
  ).toHaveCount(0)
})

test('conditional dataset and evaluation tables preserve values and standard deviation disclosure', async ({
  page,
}) => {
  await setup(page, 'en', {
    dataset_configs: [
      { path: '/datasets/train.jsonl', split: 'train', num_rows: 24, max_samples: 12 },
    ],
    final_metrics: {
      test_before: { num_samples: 5, accuracy: 0.6, accuracy_std: 0.1 },
      test_after: { accuracy: 0.8, accuracy_std: 0.05 },
      test_delta: { accuracy: 0.2, accuracy_std: -0.05 },
    },
  })
  await page.goto('/training/detail-ux')
  await expect(page.getByRole('cell', { name: '/datasets/train.jsonl', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '0.6000', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '+0.2000', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: 'accuracy_std', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Show Std Dev (1)' }).click()
  await expect(page.getByRole('cell', { name: 'accuracy_std', exact: true })).toBeVisible()
  await expect(page.getByRole('cell', { name: '-0.0500', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Hide Std Dev' }).click()
  await expect(page.getByRole('cell', { name: 'accuracy_std', exact: true })).toHaveCount(0)
})
