import { expect, test } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

type GenerationTask = {
  task_id: string
  task_name: string
  status: string
  generation_mode: string
  progress: number
  total_docs: number
  processed_docs: number
  output_sample_count: number
  created_at: string
}

function buildTask(index: number, status = 'completed'): GenerationTask {
  return {
    task_id: `generation-${String(index).padStart(3, '0')}`,
    task_name: `生成任务 ${index}`,
    status,
    generation_mode: 'doc_to_training',
    progress: status === 'running' ? 50 : 100,
    total_docs: 10,
    processed_docs: status === 'running' ? 5 : 10,
    output_sample_count: index,
    created_at: '2026-08-09T10:00:00',
  }
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('background polling keeps the current page without loading or error toasts', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let failPolling = false
  let failedPollingRequests = 0
  let releasePolling!: () => void
  const pollingGate = new Promise<void>((resolve) => {
    releasePolling = resolve
  })

  await page.route('**/api/generation/tasks*', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    if (failPolling) {
      failedPollingRequests += 1
      await pollingGate
      return route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'temporary failure' }),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: [{ ...buildTask(1, 'running'), task_name: 'Stable generation task' }],
        total: 1,
        stats: { total: 1, pending: 0, running: 1, completed: 0, failed: 0, stopped: 0 },
        limit: 10,
        offset: 0,
      }),
    })
  })

  await page.goto('/datasets/generation')
  await expect(page.getByText('Stable generation task')).toBeVisible({ timeout: 15_000 })
  await expect(page.locator('.ant-spin-spinning')).toHaveCount(0)
  failPolling = true
  await expect.poll(() => failedPollingRequests, { timeout: 8_000 }).toBeGreaterThanOrEqual(1)
  const loadingCount = await page.locator('.ant-spin-spinning').count()
  releasePolling()

  expect(loadingCount).toBe(0)
  await expect(page.getByText('Stable generation task')).toBeVisible()
  await page.waitForTimeout(300)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
})

test('slow background polling does not supersede an in-flight response', async ({ page }) => {
  test.setTimeout(45_000)
  let slowPolling = false
  let slowRequests = 0
  let releasePolling!: () => void
  const pollingGate = new Promise<void>((resolve) => {
    releasePolling = resolve
  })

  await page.route('**/api/generation/tasks*', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()

    if (slowPolling) {
      slowRequests += 1
      await pollingGate
    }

    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: [
          {
            ...buildTask(1, 'running'),
            task_name: slowPolling ? 'Updated slow generation task' : 'Initial generation task',
          },
        ],
        total: 1,
        stats: { total: 1, pending: 0, running: 1, completed: 0, failed: 0, stopped: 0 },
        limit: 10,
        offset: 0,
      }),
    })
  })

  await page.goto('/datasets/generation')
  await expect(page.getByText('Initial generation task')).toBeVisible({ timeout: 15_000 })
  slowPolling = true
  await expect.poll(() => slowRequests, { timeout: 8_000 }).toBe(1)
  await page.waitForTimeout(5200)
  const requestsWhileHeld = slowRequests
  releasePolling()

  expect(requestsWhileHeld).toBe(1)
  await expect(page.getByText('Updated slow generation task')).toBeVisible({ timeout: 10_000 })
})

test('uses server pagination, global stats, and polls the current page', async ({ page }) => {
  const requests: Array<{ limit: number; offset: number }> = []

  await page.route('**/api/generation/tasks*', async (route) => {
    const request = route.request()
    if (request.method() !== 'GET') return route.fallback()

    const url = new URL(request.url())
    const limit = Number(url.searchParams.get('limit'))
    const offset = Number(url.searchParams.get('offset'))
    requests.push({ limit, offset })
    const tasks = Array.from({ length: Math.min(limit, 21 - offset) }, (_, itemIndex) => {
      const index = offset + itemIndex + 1
      return buildTask(index, offset === 10 && itemIndex === 0 ? 'running' : 'completed')
    })

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks,
        total: 21,
        stats: {
          total: 42,
          pending: 3,
          running: 7,
          completed: 30,
          failed: 2,
          stopped: 0,
        },
        limit,
        offset,
      }),
    })
  })

  await page.goto('/datasets/generation')
  await expect(page.getByRole('heading', { name: '数据生成' })).toBeVisible()
  await expect(page.getByText('42', { exact: true })).toBeVisible()
  await expect(
    page.locator('.ant-card').filter({ hasText: '运行中' }).getByText('7', { exact: true })
  ).toBeVisible()
  await page.locator('.ant-pagination-item-2').click()
  await expect(page.getByText('生成任务 11', { exact: true })).toBeVisible()
  await page.waitForTimeout(5200)

  expect(requests[0]).toEqual({ limit: 10, offset: 0 })
  expect(requests.filter((request) => request.offset === 10).length).toBeGreaterThanOrEqual(2)
  expect(requests.at(-1)).toEqual({ limit: 10, offset: 10 })
})

test('deleting the last row falls back to the previous server page', async ({ page }) => {
  let total = 21
  const listOffsets: number[] = []

  await page.route('**/api/generation/tasks*', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    if (request.method() === 'DELETE') {
      total -= 1
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      })
    }
    if (request.method() !== 'GET') return route.fallback()

    const limit = Number(url.searchParams.get('limit'))
    const offset = Number(url.searchParams.get('offset'))
    listOffsets.push(offset)
    const tasks = Array.from(
      { length: Math.max(0, Math.min(limit, total - offset)) },
      (_, itemIndex) => buildTask(offset + itemIndex + 1)
    )
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks,
        total,
        stats: {
          total,
          pending: 0,
          running: 0,
          completed: total,
          failed: 0,
          stopped: 0,
        },
        limit,
        offset,
      }),
    })
  })

  await page.goto('/datasets/generation')
  await page.locator('.ant-pagination-item-3').click()
  const lastRow = page.getByRole('row').filter({ hasText: '生成任务 21' })
  await expect(lastRow).toBeVisible()
  await lastRow.locator('button.ant-btn-dangerous').click()
  await page.locator('.ant-popconfirm-buttons .ant-btn-primary').click()

  await expect(page.locator('.ant-pagination-item-2')).toHaveClass(/ant-pagination-item-active/)
  await expect(page.getByText('生成任务 11', { exact: true })).toBeVisible()
  expect(listOffsets.at(-1)).toBe(10)
})

test('a delayed older page response cannot replace the newest page', async ({ page }) => {
  let markPageTwoRequested!: () => void
  let releasePageTwo!: () => void
  let pageTwoCompleted = false
  const pageTwoRequested = new Promise<void>((resolve) => {
    markPageTwoRequested = resolve
  })
  const pageTwoReleased = new Promise<void>((resolve) => {
    releasePageTwo = resolve
  })

  await page.route('**/api/generation/tasks*', async (route) => {
    const request = route.request()
    if (request.method() !== 'GET') return route.fallback()

    const url = new URL(request.url())
    const limit = Number(url.searchParams.get('limit'))
    const offset = Number(url.searchParams.get('offset'))
    if (offset === 10) {
      markPageTwoRequested()
      await pageTwoReleased
    }

    const tasks = Array.from({ length: Math.min(limit, 21 - offset) }, (_, itemIndex) => {
      const index = offset + itemIndex + 1
      return { ...buildTask(index), task_name: `Page task ${index}` }
    })
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks,
        total: 21,
        stats: {
          total: 21,
          pending: 0,
          running: 0,
          completed: 21,
          failed: 0,
          stopped: 0,
        },
        limit,
        offset,
      }),
    })
    if (offset === 10) pageTwoCompleted = true
  })

  await page.goto('/datasets/generation')
  await expect(page.getByText('Page task 1', { exact: true })).toBeVisible()

  await page.locator('.ant-pagination-item-2').click()
  await pageTwoRequested
  await page.locator('.ant-pagination-item-3').dispatchEvent('click')
  await expect(page.getByText('Page task 21', { exact: true })).toBeVisible()

  releasePageTwo()
  await expect.poll(() => pageTwoCompleted).toBe(true)
  await expect(page.locator('.ant-pagination-item-3')).toHaveClass(/ant-pagination-item-active/)
  await expect(page.getByText('Page task 21', { exact: true })).toBeVisible()
  await expect(page.getByText('Page task 11', { exact: true })).toHaveCount(0)
})

test('publishing remains polled and exposes no stop or delete action', async ({ page }) => {
  let listRequests = 0
  const publishingTask = {
    ...buildTask(1, 'publishing'),
    task_name: 'Publishing generation task',
    progress: 100,
    processed_docs: 10,
  }

  await page.route('**/api/generation/tasks*', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    listRequests += 1
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: [publishingTask],
        total: 1,
        stats: {
          total: 1,
          pending: 0,
          running: 0,
          publishing: 1,
          completed: 0,
          failed: 0,
          stopped: 0,
        },
        limit: 10,
        offset: 0,
      }),
    })
  })

  await page.goto('/datasets/generation')
  const row = page.getByRole('row').filter({ hasText: publishingTask.task_name })
  await expect(row).toBeVisible()
  await expect(row.getByText('发布中', { exact: true })).toBeVisible()
  await expect(row.locator('.anticon-stop')).toHaveCount(0)
  await expect(row.locator('.anticon-delete')).toHaveCount(0)
  await page.waitForTimeout(5200)
  expect(listRequests).toBeGreaterThanOrEqual(2)
})

test('pending remains polled, can be stopped, and cannot be deleted', async ({ page }) => {
  let listRequests = 0
  const pendingTask = {
    ...buildTask(1, 'pending'),
    task_name: 'Pending generation task',
    progress: 0,
    processed_docs: 0,
  }

  await page.route('**/api/generation/tasks*', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    listRequests += 1
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: [pendingTask],
        total: 1,
        stats: {
          total: 1,
          pending: 1,
          running: 0,
          publishing: 0,
          completed: 0,
          failed: 0,
          stopped: 0,
        },
        limit: 10,
        offset: 0,
      }),
    })
  })

  await page.goto('/datasets/generation')
  const row = page.getByRole('row').filter({ hasText: pendingTask.task_name })
  await expect(row).toBeVisible()
  await expect(row.locator('.anticon-stop')).toHaveCount(1)
  await expect(row.locator('.anticon-delete')).toHaveCount(0)
  await page.waitForTimeout(5200)
  expect(listRequests).toBeGreaterThanOrEqual(2)
})

test('stopping and recovering remain polled without mutation actions', async ({ page }) => {
  let listRequests = 0
  const transitionalTasks = [
    {
      ...buildTask(1, 'stopping'),
      task_name: 'Stopping generation task',
      progress: 50,
      processed_docs: 5,
    },
    {
      ...buildTask(2, 'recovering'),
      task_name: 'Recovering generation task',
      progress: 50,
      processed_docs: 5,
    },
    {
      ...buildTask(3, 'restarting'),
      task_name: 'Restarting generation task',
      progress: 50,
      processed_docs: 5,
    },
  ]

  await page.route('**/api/generation/tasks*', async (route) => {
    if (route.request().method() !== 'GET') return route.fallback()
    listRequests += 1
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        tasks: transitionalTasks,
        total: transitionalTasks.length,
        stats: {
          total: transitionalTasks.length,
          pending: 0,
          running: 0,
          stopping: 1,
          recovering: 1,
          restarting: 1,
          publishing: 0,
          completed: 0,
          failed: 0,
          stopped: 0,
        },
        limit: 10,
        offset: 0,
      }),
    })
  })

  await page.goto('/datasets/generation')
  const stoppingRow = page.getByRole('row').filter({ hasText: transitionalTasks[0].task_name })
  const recoveringRow = page.getByRole('row').filter({ hasText: transitionalTasks[1].task_name })
  const restartingRow = page.getByRole('row').filter({ hasText: transitionalTasks[2].task_name })

  for (const row of [stoppingRow, recoveringRow, restartingRow]) {
    await expect(row).toBeVisible()
    await expect(row.locator('.anticon-stop')).toHaveCount(0)
    await expect(row.locator('.anticon-play-circle')).toHaveCount(0)
    await expect(row.locator('.anticon-redo')).toHaveCount(0)
    await expect(row.locator('.anticon-delete')).toHaveCount(0)
  }
  await expect(stoppingRow.getByText('停止中', { exact: true })).toBeVisible()
  await expect(recoveringRow.getByText('恢复中', { exact: true })).toBeVisible()
  await expect(restartingRow.getByText('重启准备中', { exact: true })).toBeVisible()
  await page.waitForTimeout(5200)
  expect(listRequests).toBeGreaterThanOrEqual(2)
})

test('publishing detail keeps polling and is non-stoppable', async ({ page }) => {
  let detailRequests = 0
  const taskId = 'publishing-detail'
  await page.route(`**/api/generation/tasks/${taskId}*`, async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/artifacts')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ task_id: taskId, total_artifacts: 0, stages: [] }),
      })
    }
    if (route.request().method() !== 'GET') return route.fallback()
    detailRequests += 1
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ...buildTask(1, 'publishing'),
        task_id: taskId,
        task_name: 'Publishing detail task',
        input_path: '/managed/input.jsonl',
        input_format: 'jsonl',
        output_format: 'universal',
        auto_register_dataset: true,
        llm_config: {},
        worker_config: {},
        steps_config: {},
      }),
    })
  })

  await page.goto(`/datasets/generation/${taskId}`)
  await expect(page.getByRole('heading', { name: 'Publishing detail task' })).toBeVisible()
  await expect(page.getByText('发布中', { exact: true })).toBeVisible()
  await expect(page.locator('.anticon-stop')).toHaveCount(0)
  await expect(page.locator('.anticon-delete')).toHaveCount(0)
  await page.waitForTimeout(3200)
  expect(detailRequests).toBeGreaterThanOrEqual(2)
})

for (const status of ['stopping', 'recovering', 'restarting'] as const) {
  test(`${status} detail keeps polling without mutation actions`, async ({ page }) => {
    let detailRequests = 0
    const taskId = `${status}-detail`
    await page.route(`**/api/generation/tasks/${taskId}*`, async (route) => {
      const path = new URL(route.request().url()).pathname
      if (path.endsWith('/artifacts')) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ task_id: taskId, total_artifacts: 0, stages: [] }),
        })
      }
      if (route.request().method() !== 'GET') return route.fallback()
      detailRequests += 1
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...buildTask(1, status),
          task_id: taskId,
          task_name: `${status} detail task`,
          input_path: '/managed/input.jsonl',
          input_format: 'jsonl',
          output_format: 'universal',
          auto_register_dataset: true,
          llm_config: {},
          worker_config: {},
          steps_config: {},
        }),
      })
    })

    await page.goto(`/datasets/generation/${taskId}`)
    await expect(page.getByRole('heading', { name: `${status} detail task` })).toBeVisible()
    await expect(page.locator('.anticon-stop')).toHaveCount(0)
    await expect(page.locator('.anticon-play-circle')).toHaveCount(0)
    await expect(page.locator('.anticon-redo')).toHaveCount(0)
    await expect(page.locator('.anticon-delete')).toHaveCount(0)
    await page.waitForTimeout(3200)
    expect(detailRequests).toBeGreaterThanOrEqual(2)
  })
}
