import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('training detail polling keeps old data and does not emit error toasts', async ({ page }) => {
  test.setTimeout(45_000)
  let failPolling = false
  let failures = 0
  let releasePolling!: () => void
  const pollingGate = new Promise<void>((resolve) => {
    releasePolling = resolve
  })
  await page.route('**/api/train/task-polling**', async (route) => {
    if (failPolling) {
      failures += 1
      await pollingGate
      return route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'temporary failure' }),
      })
    }

    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/metrics')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ task_id: 'task-polling', loss_history: [], has_data: false }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-polling',
        task_name: 'Polling Training Task',
        model_type: 'embedding',
        training_method: 'sft',
        status: 'running',
        progress: 25,
        current_step: 25,
        total_steps: 100,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-polling')
  await expect(page.getByText('Polling Training Task')).toBeVisible({ timeout: 15_000 })
  failPolling = true

  await expect.poll(() => failures, { timeout: 8_000 }).toBeGreaterThanOrEqual(3)
  await expect(page.locator('.ant-spin-spinning')).toHaveCount(0)
  releasePolling()
  await expect(page.getByTestId('training-stale-state')).toBeVisible({ timeout: 8_000 })
  await expect(page.getByText('Polling Training Task')).toBeVisible()
  await expect.poll(() => failures, { timeout: 12_000 }).toBeGreaterThanOrEqual(6)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
})

test('training manual refresh keeps stale state until task metrics and events all recover', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let failMetrics = false
  let failedMetricsRequests = 0
  let taskStatus = 'running'

  await page.route('**/api/train/task-aggregate-refresh**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/metrics')) {
      if (failMetrics) {
        failedMetricsRequests += 1
        return route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'metrics unavailable' }),
        })
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          task_id: 'task-aggregate-refresh',
          loss_history: [],
          has_data: false,
        }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-aggregate-refresh',
        task_name: 'Aggregate refresh task',
        model_type: 'embedding',
        training_method: 'sft',
        status: taskStatus,
        progress: 25,
        created_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-aggregate-refresh')
  await expect(page.getByText('Aggregate refresh task')).toBeVisible({ timeout: 15_000 })
  failMetrics = true
  await expect.poll(() => failedMetricsRequests, { timeout: 8_000 }).toBeGreaterThanOrEqual(1)
  await expect(page.getByTestId('training-stale-state')).toBeVisible({ timeout: 8_000 })

  taskStatus = 'failed'
  const refreshButton = page.locator('button:has(.anticon-reload)').first()
  await expect(refreshButton).toBeVisible()
  await refreshButton.click()
  await expect.poll(() => failedMetricsRequests).toBeGreaterThanOrEqual(2)
  await expect(page.getByTestId('training-stale-state')).toBeVisible()

  failMetrics = false
  await refreshButton.click()
  await expect(page.getByTestId('training-stale-state')).toHaveCount(0)
})

test('an older training poll cannot clear a newer manual refresh failure', async ({ page }) => {
  test.setTimeout(45_000)
  let holdOldPoll = false
  let heldRequests = 0
  let releasedRequests = 0
  let failManualMetrics = false
  let manualMetricsFailures = 0
  let taskStatus = 'running'
  let releaseOldPoll!: () => void
  const oldPollGate = new Promise<void>((resolve) => {
    releaseOldPoll = resolve
  })

  await page.route('**/api/train/task-refresh-ordering**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const holdThisRequest = holdOldPoll
    const responseStatus = taskStatus
    if (holdThisRequest) {
      heldRequests += 1
      await oldPollGate
      releasedRequests += 1
    }
    if (path.endsWith('/metrics')) {
      if (!holdThisRequest && failManualMetrics) {
        manualMetricsFailures += 1
        return route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'newer metrics failure' }),
        })
      }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          task_id: 'task-refresh-ordering',
          loss_history: [],
          has_data: false,
        }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-refresh-ordering',
        task_name: 'Refresh ordering task',
        model_type: 'embedding',
        training_method: 'sft',
        status: responseStatus,
        progress: 25,
        created_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-refresh-ordering')
  await expect(page.getByText('Refresh ordering task')).toBeVisible({ timeout: 15_000 })
  holdOldPoll = true
  await expect.poll(() => heldRequests, { timeout: 8_000 }).toBeGreaterThanOrEqual(3)
  holdOldPoll = false
  failManualMetrics = true
  taskStatus = 'failed'

  const refreshButton = page.locator('button:has(.anticon-reload)').first()
  await refreshButton.click()
  await expect.poll(() => manualMetricsFailures).toBeGreaterThanOrEqual(1)
  await expect(page.getByTestId('training-stale-state')).toBeVisible()

  releaseOldPoll()
  await expect.poll(() => releasedRequests).toBeGreaterThanOrEqual(3)
  await page.waitForTimeout(250)
  await expect(page.getByTestId('training-stale-state')).toBeVisible()
})

test('training detail ignores a slow task response older than a manual refresh', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let holdOldTask = false
  let heldTaskRequests = 0
  let releasedTaskRequests = 0
  let useNewerSnapshot = false
  let releaseOldTask!: () => void
  const oldTaskGate = new Promise<void>((resolve) => {
    releaseOldTask = resolve
  })

  await page.route('**/api/train/task-ordering**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/metrics')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ task_id: 'task-ordering', loss_history: [], has_data: false }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }

    if (holdOldTask) {
      holdOldTask = false
      heldTaskRequests += 1
      await oldTaskGate
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          task_id: 'task-ordering',
          task_name: 'Older Training Snapshot',
          model_type: 'embedding',
          training_method: 'sft',
          status: 'running',
          progress: 20,
          created_at: new Date().toISOString(),
        }),
      })
      releasedTaskRequests += 1
      return
    }

    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: 'task-ordering',
        task_name: useNewerSnapshot ? 'Newer Training Snapshot' : 'Initial Training Snapshot',
        model_type: 'embedding',
        training_method: 'sft',
        status: 'running',
        progress: useNewerSnapshot ? 40 : 10,
        created_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-ordering')
  await expect(page.getByText('Initial Training Snapshot')).toBeVisible({ timeout: 15_000 })

  holdOldTask = true
  await expect.poll(() => heldTaskRequests, { timeout: 8_000 }).toBe(1)
  useNewerSnapshot = true
  await page.getByRole('button', { name: /Refresh|刷新/ }).click()
  await expect(page.getByText('Newer Training Snapshot')).toBeVisible({ timeout: 5_000 })

  releaseOldTask()
  await expect.poll(() => releasedTaskRequests, { timeout: 3_000 }).toBe(1)
  await page.waitForTimeout(150)

  expect(await page.getByText('Older Training Snapshot').count()).toBe(0)
  await expect(page.getByText('Newer Training Snapshot')).toBeVisible()
})

test('training route change hides the old task and keeps the new request loading', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let holdRefreshA = false
  let releaseRefreshA!: () => void
  let releaseTaskB!: () => void
  let refreshAStarted = 0
  let taskBStarted = 0
  let refreshAReleased = 0
  const refreshAGate = new Promise<void>((resolve) => {
    releaseRefreshA = resolve
  })
  const taskBGate = new Promise<void>((resolve) => {
    releaseTaskB = resolve
  })

  await page.route('**/api/train/task-route-**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const taskId = path.includes('task-route-b') ? 'task-route-b' : 'task-route-a'
    if (path.endsWith('/metrics')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ task_id: taskId, loss_history: [], has_data: false }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }
    if (taskId === 'task-route-a' && holdRefreshA) {
      holdRefreshA = false
      refreshAStarted += 1
      await refreshAGate
      refreshAReleased += 1
    }
    if (taskId === 'task-route-b') {
      taskBStarted += 1
      await taskBGate
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: taskId,
        task_name: taskId === 'task-route-b' ? 'Route Task B' : 'Route Task A',
        model_type: 'embedding',
        training_method: 'sft',
        status: 'running',
        progress: 25,
        created_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-route-a')
  await expect(page.getByText('Route Task A')).toBeVisible({ timeout: 15_000 })

  const refreshButton = page.locator('button:has(.anticon-reload)').first()
  await expect(refreshButton).toBeVisible()
  holdRefreshA = true
  await refreshButton.click()
  await expect.poll(() => refreshAStarted, { timeout: 5_000 }).toBe(1)
  await page.evaluate(() => {
    window.history.pushState({}, '', '/training/task-route-b')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect.poll(() => taskBStarted, { timeout: 5_000 }).toBe(1)

  await expect(page.getByText('Route Task A')).toHaveCount(0)
  await expect(page.locator('.ant-spin-spinning')).toBeVisible()

  releaseRefreshA()
  await expect.poll(() => refreshAReleased, { timeout: 3_000 }).toBe(1)
  await page.waitForTimeout(150)
  await expect(page.getByText('Task Not Found')).toHaveCount(0)
  await expect(page.locator('.ant-spin-spinning')).toBeVisible()

  releaseTaskB()
  await expect(page.getByText('Route Task B')).toBeVisible({ timeout: 5_000 })
})

test('training route change ignores model resolution from the previous task', async ({ page }) => {
  test.setTimeout(45_000)
  let modelListCalls = 0
  let oldModelResponseReleased = 0
  let releaseOldModels!: () => void
  let releaseNewModels!: () => void
  const oldModelsGate = new Promise<void>((resolve) => {
    releaseOldModels = resolve
  })
  const newModelsGate = new Promise<void>((resolve) => {
    releaseNewModels = resolve
  })

  await page.route('**/api/models**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (!path.endsWith('/api/models')) return route.fallback()
    modelListCalls += 1
    const isOldRequest = modelListCalls === 1
    await (isOldRequest ? oldModelsGate : newModelsGate)
    if (isOldRequest) oldModelResponseReleased += 1
    const suffix = isOldRequest ? 'a' : 'b'
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        models: [
          {
            model_id: `registered-model-${suffix}`,
            model_name: `Registered Model ${suffix.toUpperCase()}`,
            model_type: 'embedding',
            model_path: `/models/base-${suffix}`,
            status: 'available',
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      }),
    })
  })

  await page.route('**/api/train/task-model-**', async (route) => {
    const path = new URL(route.request().url()).pathname
    const taskId = path.includes('task-model-b') ? 'task-model-b' : 'task-model-a'
    if (path.endsWith('/metrics')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ task_id: taskId, loss_history: [], has_data: false }),
      })
    }
    if (path.endsWith('/events')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ events: [], total: 0 }),
      })
    }
    const suffix = taskId === 'task-model-b' ? 'b' : 'a'
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        task_id: taskId,
        task_name: `Model Task ${suffix.toUpperCase()}`,
        model_type: 'embedding',
        training_method: 'sft',
        status: 'completed',
        progress: 100,
        base_model_path: `/models/base-${suffix}`,
        created_at: new Date().toISOString(),
      }),
    })
  })

  await page.goto('/training/task-model-a')
  await expect(page.getByText('Model Task A')).toBeVisible({ timeout: 15_000 })
  await expect.poll(() => modelListCalls, { timeout: 5_000 }).toBe(1)

  await page.evaluate(() => {
    window.history.pushState({}, '', '/training/task-model-b')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect(page.getByText('Model Task B')).toBeVisible({ timeout: 5_000 })
  await expect.poll(() => modelListCalls, { timeout: 5_000 }).toBe(2)

  releaseOldModels()
  await expect.poll(() => oldModelResponseReleased, { timeout: 3_000 }).toBe(1)
  await page.waitForTimeout(150)
  await expect(page.getByText('Registered Model A')).toHaveCount(0)
  await expect(page.getByText('Model Task B')).toBeVisible()

  releaseNewModels()
  await expect(page.getByText('Registered Model B')).toBeVisible({ timeout: 5_000 })
})

test('evaluation list polling is silent and retains its running row', async ({ page }) => {
  test.setTimeout(45_000)
  let failPolling = false
  let failures = 0
  await page.route('**/api/evaluations/tasks**', async (route) => {
    if (failPolling) {
      failures += 1
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
        items: [
          {
            task_id: 'eval-polling',
            task_name: 'Polling Evaluation Task',
            eval_type: 'single_model',
            status: 'running',
            progress: 30,
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      }),
    })
  })

  await page.goto('/evaluations')
  await expect(page.getByText('Polling Evaluation Task')).toBeVisible({ timeout: 15_000 })
  failPolling = true

  await expect(page.getByTestId('evaluation-stale-state')).toBeVisible({ timeout: 6_000 })
  await expect(page.getByText('Polling Evaluation Task')).toBeVisible()
  await expect.poll(() => failures, { timeout: 8_000 }).toBeGreaterThanOrEqual(2)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
})

test('evaluation pagination ignores a delayed response from the previous page', async ({
  page,
}) => {
  test.setTimeout(45_000)
  let holdPageOne = false
  let heldPageOneRequests = 0
  let releasedPageOneRequests = 0
  let releasePageOne!: () => void
  const pageOneGate = new Promise<void>((resolve) => {
    releasePageOne = resolve
  })

  await page.route('**/api/evaluations/tasks**', async (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get('offset') || 0)
    const wasHeldPageOne = holdPageOne && offset === 0
    if (wasHeldPageOne) {
      holdPageOne = false
      heldPageOneRequests += 1
      await pageOneGate
    }
    const isPageTwo = offset >= 10
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [
          {
            task_id: isPageTwo ? 'eval-page-two' : 'eval-page-one',
            task_name: isPageTwo ? 'Evaluation Page Two' : 'Evaluation Page One',
            eval_type: 'single_model',
            status: 'completed',
            progress: 100,
            created_at: new Date().toISOString(),
          },
        ],
        total: 20,
      }),
    })
    if (wasHeldPageOne) releasedPageOneRequests += 1
  })

  await page.goto('/evaluations')
  await expect(page.getByText('Evaluation Page One')).toBeVisible({ timeout: 15_000 })

  holdPageOne = true
  await page.getByRole('button', { name: /Refresh|刷新/ }).click()
  await expect.poll(() => heldPageOneRequests, { timeout: 5_000 }).toBe(1)
  await page.locator('.ant-pagination-item-2').click()
  await expect(page.getByText('Evaluation Page Two')).toBeVisible({ timeout: 5_000 })

  releasePageOne()
  await expect.poll(() => releasedPageOneRequests, { timeout: 3_000 }).toBe(1)
  await page.waitForTimeout(150)

  expect(await page.getByText('Evaluation Page One').count()).toBe(0)
  await expect(page.getByText('Evaluation Page Two')).toBeVisible()
})

test('deep evaluation polling is silent and retains its running row', async ({ page }) => {
  test.setTimeout(45_000)
  let failPolling = false
  let failures = 0
  await page.route('**/api/deep-evaluation/datasets/eval-ready', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route('**/api/deep-evaluation/tasks**', async (route) => {
    if (failPolling) {
      failures += 1
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
        items: [
          {
            task_id: 'deep-polling',
            task_name: 'Polling Deep Evaluation Task',
            source_type: 'dataset',
            status: 'running',
            progress: 20,
            selected_metrics: [],
            model_groups: [],
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('tab').nth(1).click()
  await expect(page.getByText('Polling Deep Evaluation Task')).toBeVisible({ timeout: 15_000 })
  failPolling = true

  await expect(page.getByTestId('deep-evaluation-stale-state')).toBeVisible({ timeout: 6_000 })
  await expect(page.getByText('Polling Deep Evaluation Task')).toBeVisible()
  await expect.poll(() => failures, { timeout: 8_000 }).toBeGreaterThanOrEqual(2)
  await expect(page.locator('.ant-message-notice')).toHaveCount(0)
})

test('deep evaluation groups staggered initial failures into one local message', async ({
  page,
}) => {
  test.setTimeout(45_000)
  await page.route('**/api/deep-evaluation/datasets/eval-ready', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route('**/api/deep-evaluation/tasks**', (route) =>
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'tasks unavailable' }),
    })
  )
  await page.route('**/api/datasets**', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 5_000))
    return route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'datasets unavailable' }),
    })
  })
  await page.route('**/api/deep-evaluation/metrics**', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 10_000))
    return route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'metrics unavailable' }),
    })
  })

  await page.goto('/evaluations')
  await page.evaluate(() => {
    const testWindow = window as typeof window & {
      __toastEpochs?: number
      __toastTimer?: number
    }
    let wasVisible = false
    testWindow.__toastEpochs = 0
    testWindow.__toastTimer = window.setInterval(() => {
      const isVisible = document.querySelectorAll('.ant-message-notice').length > 0
      if (isVisible && !wasVisible) testWindow.__toastEpochs = (testWindow.__toastEpochs || 0) + 1
      wasVisible = isVisible
    }, 50)
  })

  await page.getByRole('tab').nth(1).click()
  await page.waitForTimeout(11_000)

  const toastEpochs = await page.evaluate(() => {
    const testWindow = window as typeof window & {
      __toastEpochs?: number
      __toastTimer?: number
    }
    if (testWindow.__toastTimer) window.clearInterval(testWindow.__toastTimer)
    return testWindow.__toastEpochs || 0
  })
  expect(toastEpochs).toBe(1)
})

test('deep evaluation ignores a slow poll older than a manual refresh', async ({ page }) => {
  test.setTimeout(45_000)
  let holdOldTasks = false
  let heldTaskRequests = 0
  let releasedTaskRequests = 0
  let useNewerSnapshot = false
  let releaseOldTasks!: () => void
  const oldTasksGate = new Promise<void>((resolve) => {
    releaseOldTasks = resolve
  })

  await page.route('**/api/deep-evaluation/datasets/eval-ready', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route('**/api/deep-evaluation/tasks**', async (route) => {
    if (holdOldTasks) {
      holdOldTasks = false
      heldTaskRequests += 1
      await oldTasksGate
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          items: [
            {
              task_id: 'deep-ordering',
              task_name: 'Older Deep Evaluation Snapshot',
              source_type: 'dataset',
              status: 'running',
              progress: 20,
              selected_metrics: [],
              model_groups: [],
              created_at: new Date().toISOString(),
            },
          ],
          total: 1,
        }),
      })
      releasedTaskRequests += 1
      return
    }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        items: [
          {
            task_id: 'deep-ordering',
            task_name: useNewerSnapshot
              ? 'Newer Deep Evaluation Snapshot'
              : 'Initial Deep Evaluation Snapshot',
            source_type: 'dataset',
            status: 'running',
            progress: useNewerSnapshot ? 40 : 10,
            selected_metrics: [],
            model_groups: [],
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      }),
    })
  })

  await page.goto('/evaluations')
  await page.getByRole('tab').nth(1).click()
  await expect(page.getByText('Initial Deep Evaluation Snapshot')).toBeVisible({
    timeout: 15_000,
  })

  holdOldTasks = true
  await expect.poll(() => heldTaskRequests, { timeout: 6_000 }).toBe(1)
  useNewerSnapshot = true
  await page.getByRole('button', { name: /Refresh|刷新/ }).click()
  await expect(page.getByText('Newer Deep Evaluation Snapshot')).toBeVisible({ timeout: 5_000 })

  releaseOldTasks()
  await expect.poll(() => releasedTaskRequests, { timeout: 3_000 }).toBe(1)
  await page.waitForTimeout(150)

  expect(await page.getByText('Older Deep Evaluation Snapshot').count()).toBe(0)
  await expect(page.getByText('Newer Deep Evaluation Snapshot')).toBeVisible()
})
