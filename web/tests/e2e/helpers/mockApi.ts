import type { Page, Route } from '@playwright/test'

type JsonValue = Record<string, unknown> | unknown[]

const AUTH_COOKIE = 'access_token=mock-access-token; Path=/; HttpOnly; SameSite=Lax'

function jsonResponse(
  route: Route,
  data: JsonValue,
  status = 200,
  headers?: Record<string, string>
) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
    headers,
  })
}

export interface MockAuthOptions {
  authenticated: boolean
  isAdmin?: boolean
  username?: string
  registrationEnabled?: boolean
  directStorageRegistrationEnabled?: boolean
  omitDirectStorageRegistrationEnabled?: boolean
  configStatus?: number
  meStatuses?: number[]
  loginStatus?: number
  registerStatus?: number
  changePasswordStatus?: number
  logoutStatus?: number
  logoutErrorDetail?: string
  adminUsers?: MockAuthUser[]
  adminListStatuses?: number[]
  adminListDelayMs?: number
  adminCreateStatus?: number
  adminCreateDelayMs?: number
  adminPatchStatus?: number
  adminResetStatus?: number
  adminResetDelayMs?: number
  adminActionError?: string
}

export interface MockAuthUser {
  user_id: string
  username: string
  email: string | null
  is_active: boolean
  is_admin: boolean
  created_at: string
  updated_at: string
}

export async function mockAuth(page: Page, options: MockAuthOptions) {
  let authenticated = options.authenticated
  let sessionRequiresCookie = false
  let meRequestCount = 0
  const user: MockAuthUser = {
    user_id: options.isAdmin ? 'admin-1' : 'user-1',
    username: options.username ?? (options.isAdmin ? 'admin' : 'runtime_verify'),
    email: null,
    is_active: true,
    is_admin: options.isAdmin ?? false,
    created_at: '2026-08-08T12:00:00',
    updated_at: '2026-08-08T12:00:00',
  }
  const defaultAdminUsers: MockAuthUser[] = [
    user,
    {
      user_id: 'user-2',
      username: 'operator',
      email: 'operator@example.com',
      is_active: true,
      is_admin: false,
      created_at: '2026-08-07T09:30:00',
      updated_at: '2026-08-07T09:30:00',
    },
    {
      user_id: 'admin-2',
      username: 'manager',
      email: 'manager@example.com',
      is_active: true,
      is_admin: true,
      created_at: '2026-08-06T08:15:00',
      updated_at: '2026-08-06T08:15:00',
    },
  ]
  const adminUsers = (options.adminUsers ?? defaultAdminUsers).map((account) => ({
    ...account,
  }))
  let adminListRequestCount = 0

  await page.route('**/api/auth/**', async (route) => {
    const request = route.request()
    const method = request.method()
    const path = new URL(request.url()).pathname

    if (method === 'GET' && path.endsWith('/auth/config')) {
      const status = options.configStatus ?? 200
      return jsonResponse(
        route,
        status >= 400
          ? { detail: 'Authentication API unavailable' }
          : {
              self_registration_enabled: options.registrationEnabled ?? false,
              ...(options.omitDirectStorageRegistrationEnabled
                ? {}
                : {
                    direct_storage_registration_enabled:
                      options.directStorageRegistrationEnabled ?? false,
                  }),
            },
        status
      )
    }

    if (method === 'GET' && path.endsWith('/auth/me')) {
      const configuredStatus = options.meStatuses?.[meRequestCount]
      meRequestCount += 1
      if (configuredStatus !== undefined) {
        return configuredStatus >= 400
          ? jsonResponse(route, { detail: 'Not authenticated' }, configuredStatus)
          : jsonResponse(route, user, configuredStatus)
      }
      const hasAuthCookie = request
        .headers()['cookie']
        ?.split(';')
        .some((cookie) => cookie.trim() === 'access_token=mock-access-token')
      return authenticated && (!sessionRequiresCookie || hasAuthCookie)
        ? jsonResponse(route, user)
        : jsonResponse(route, { detail: 'Not authenticated' }, 401)
    }

    if (method === 'POST' && path.endsWith('/auth/login')) {
      const status = options.loginStatus ?? 200
      if (status >= 400) {
        return jsonResponse(route, { detail: 'Invalid username or password' }, status)
      }
      authenticated = true
      sessionRequiresCookie = true
      return jsonResponse(
        route,
        { access_token: 'mock-access-token', token_type: 'bearer' },
        status,
        { 'set-cookie': AUTH_COOKIE }
      )
    }

    if (method === 'POST' && path.endsWith('/auth/register')) {
      const status = options.registerStatus ?? 200
      if (status >= 400) {
        return jsonResponse(route, { detail: 'Username or email already exists' }, status)
      }
      authenticated = true
      sessionRequiresCookie = true
      return jsonResponse(
        route,
        { access_token: 'mock-access-token', token_type: 'bearer' },
        status,
        { 'set-cookie': AUTH_COOKIE }
      )
    }

    if (method === 'POST' && path.endsWith('/auth/change-password')) {
      const status = options.changePasswordStatus ?? 200
      if (status >= 400) {
        return jsonResponse(route, { detail: 'Current password is incorrect' }, status)
      }
      authenticated = false
      return jsonResponse(
        route,
        { message: 'Password changed successfully. Please log in again.' },
        status
      )
    }

    if (method === 'POST' && path.endsWith('/auth/logout')) {
      const status = options.logoutStatus ?? 200
      if (status >= 400) {
        return jsonResponse(
          route,
          { detail: options.logoutErrorDetail ?? 'Unable to log out' },
          status
        )
      }
      authenticated = false
      return jsonResponse(route, { message: 'Logged out successfully' }, status)
    }

    if (path.endsWith('/auth/admin/users')) {
      if (method === 'GET') {
        if (options.adminListDelayMs) {
          await new Promise((resolve) => setTimeout(resolve, options.adminListDelayMs))
        }
        const status = options.adminListStatuses?.[adminListRequestCount] ?? 200
        adminListRequestCount += 1
        if (status >= 400) {
          return jsonResponse(route, { detail: 'Unable to load accounts' }, status)
        }
        const url = new URL(request.url())
        const limit = Number(url.searchParams.get('limit') ?? 20)
        const offset = Number(url.searchParams.get('offset') ?? 0)
        return jsonResponse(route, {
          users: adminUsers.slice(offset, offset + limit),
          total: adminUsers.length,
          limit,
          offset,
        })
      }
      if (method === 'POST') {
        if (options.adminCreateDelayMs) {
          await new Promise((resolve) => setTimeout(resolve, options.adminCreateDelayMs))
        }
        const status = options.adminCreateStatus ?? 201
        const body = request.postDataJSON() as {
          username: string
          password: string
          email?: string
          is_admin: boolean
        }
        const duplicate = adminUsers.some(
          (account) =>
            account.username === body.username || (!!body.email && account.email === body.email)
        )
        if (status >= 400 || duplicate) {
          return jsonResponse(
            route,
            { detail: options.adminActionError ?? 'Username or email already exists' },
            status >= 400 ? status : 409
          )
        }
        const created: MockAuthUser = {
          user_id: `user-${adminUsers.length + 1}`,
          username: body.username,
          email: body.email ?? null,
          is_active: true,
          is_admin: body.is_admin,
          created_at: '2026-08-08T13:00:00',
          updated_at: '2026-08-08T13:00:00',
        }
        adminUsers.unshift(created)
        return jsonResponse(route, created, 201)
      }
    }

    if (/\/auth\/admin\/users\/[^/]+$/.test(path) && method === 'PATCH') {
      const status = options.adminPatchStatus ?? 200
      if (status >= 400) {
        return jsonResponse(
          route,
          {
            detail: options.adminActionError ?? 'Account update conflicts with safety rules',
          },
          status
        )
      }
      const userId = decodeURIComponent(path.split('/').pop() ?? '')
      const account = adminUsers.find((candidate) => candidate.user_id === userId)
      if (!account) {
        return jsonResponse(route, { detail: 'User not found' }, 404)
      }
      const body = request.postDataJSON() as {
        is_active?: boolean
        is_admin?: boolean
      }
      if (body.is_active !== undefined) account.is_active = body.is_active
      if (body.is_admin !== undefined) account.is_admin = body.is_admin
      account.updated_at = '2026-08-08T13:05:00'
      if (account.user_id === user.user_id) Object.assign(user, account)
      return jsonResponse(route, account)
    }

    if (/\/auth\/admin\/users\/[^/]+\/reset-password$/.test(path) && method === 'POST') {
      if (options.adminResetDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.adminResetDelayMs))
      }
      const status = options.adminResetStatus ?? 200
      if (status >= 400) {
        return jsonResponse(
          route,
          { detail: options.adminActionError ?? 'Unable to reset password' },
          status
        )
      }
      const userId = decodeURIComponent(path.split('/').at(-2) ?? '')
      if (userId === user.user_id) authenticated = false
      return jsonResponse(route, { message: 'Password reset successfully' })
    }

    return jsonResponse(route, { detail: `Unhandled auth endpoint: ${method} ${path}` }, 404)
  })
}

export async function mockApi(
  page: Page,
  authOptions: Omit<MockAuthOptions, 'authenticated'> = {},
) {
  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const method = request.method()
    const url = new URL(request.url())
    const path = url.pathname

    if (method === 'GET' && path.endsWith('/api/train')) {
      return jsonResponse(route, {
        tasks: [
          {
            task_id: 'task-12345678',
            task_name: '示例训练任务',
            model_type: 'embedding',
            training_method: 'sft',
            base_model_path: 'BAAI/bge-base-zh-v1.5',
            train_dataset_path: '/data/datasets/example.jsonl',
            output_dir: '/data/output/task-12345678',
            status: 'running',
            progress: 45,
            current_step: 450,
            total_steps: 1000,
            current_epoch: 1,
            total_epochs: 3,
            train_loss: 0.2345,
            eval_loss: 0.2876,
            learning_rate: 2e-5,
            gpu_ids: [0],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        total: 1,
      })
    }

    if (method === 'GET' && /\/api\/train\/[^/]+$/.test(path)) {
      return jsonResponse(route, {
        task_id: 'task-12345678',
        task_name: '示例训练任务',
        model_type: 'embedding',
        training_method: 'sft',
        base_model_path: 'BAAI/bge-base-zh-v1.5',
        train_dataset_path: '/data/datasets/example.jsonl',
        output_dir: '/data/output/task-12345678',
        status: 'succeeded',
        progress: 100,
        current_step: 1000,
        total_steps: 1000,
        current_epoch: 3,
        total_epochs: 3,
        train_loss: 0.1234,
        eval_loss: 0.1567,
        learning_rate: 2e-5,
        gpu_ids: [0],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        started_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
      })
    }

    if (method === 'GET' && /\/api\/train\/[^/]+\/events$/.test(path)) {
      return jsonResponse(route, {
        events: [
          {
            event_id: 'evt-001',
            task_id: 'task-12345678',
            event_type: 'task_created',
            payload: { status: 'pending' },
            created_at: new Date().toISOString(),
          },
          {
            event_id: 'evt-002',
            task_id: 'task-12345678',
            event_type: 'status_changed',
            payload: { from: 'pending', to: 'running' },
            created_at: new Date().toISOString(),
          },
        ],
        total: 2,
      })
    }

    if (method === 'GET' && path.endsWith('/api/datasets/downloads')) {
      return jsonResponse(route, { downloads: [], total: 0 })
    }

    if (method === 'GET' && path.endsWith('/api/datasets')) {
      return jsonResponse(route, {
        datasets: [
          {
            dataset_id: 'ds-123',
            dataset_name: 'example-dataset',
            name: 'example-dataset',
            description: '测试数据集',
            dataset_type: 'embedding_pair',
            usage: 'train',
            model_type: ['embedding'],
            source_type: 'local',
            storage_path: '/data/datasets/example.jsonl',
            file_format: 'jsonl',
            file_size: 102400,
            num_rows: 1280,
            columns: [
              { name: 'query', type: 'string' },
              { name: 'doc', type: 'string' },
              { name: 'label', type: 'int' },
            ],
            status: 'ready',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          {
            dataset_id: 'ds-456',
            dataset_name: 'eval-rerank-dataset',
            name: 'eval-rerank-dataset',
            description: '评估用重排序数据集',
            dataset_type: 'rerank_pair',
            usage: 'eval',
            model_type: ['rerank'],
            source_type: 'huggingface',
            storage_path: '/data/datasets/eval-rerank.parquet',
            file_format: 'parquet',
            file_size: 2048000,
            num_rows: 5000,
            columns: [
              { name: 'query', type: 'string' },
              { name: 'passage', type: 'string' },
              { name: 'score', type: 'float' },
            ],
            status: 'ready',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          {
            dataset_id: 'ds-789',
            dataset_name: 'test-rerank-dataset',
            name: 'test-rerank-dataset',
            description: '测试用重排序数据集',
            dataset_type: 'rerank_listwise',
            usage: 'test',
            model_type: ['rerank'],
            source_type: 'local',
            storage_path: '/data/datasets/test-rerank.jsonl',
            file_format: 'jsonl',
            file_size: 512000,
            num_rows: 800,
            num_test: 800,
            columns: [
              { name: 'query', type: 'string' },
              { name: 'documents', type: 'array' },
              { name: 'labels', type: 'array' },
            ],
            status: 'ready',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        total: 3,
      })
    }

    if (method === 'GET' && /\/api\/datasets\/[^/]+\/preview$/.test(path)) {
      return jsonResponse(route, [
        { query: 'hello', doc: 'world', label: 1 },
        { query: 'foo', doc: 'bar', label: 0 },
      ])
    }

    if (method === 'GET' && /\/api\/datasets\/[^/]+$/.test(path)) {
      const datasetId = path.split('/').pop()
      if (!datasetId || ['types', 'downloads', 'download'].includes(datasetId)) {
        return route.continue()
      }
      return jsonResponse(route, {
        dataset_id: datasetId,
        dataset_name: 'eval-rerank-dataset',
        name: 'eval-rerank-dataset',
        description: '评估用重排序数据集',
        dataset_type: 'rerank_pair',
        usage: 'eval',
        model_type: ['rerank'],
        source_type: 'huggingface',
        storage_path: '/data/datasets/eval-rerank.parquet',
        file_format: 'parquet',
        file_size: 2048000,
        num_rows: 5000,
        columns: [
          { name: 'query', type: 'string' },
          { name: 'passage', type: 'string' },
          { name: 'answer', type: 'string' },
          { name: 'contexts', type: 'array' },
        ],
        sample_data: [
          { query: '问题', passage: '文档', answer: '答案', contexts: ['上下文1', '上下文2'] },
        ],
        status: 'ready',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
    }

    if (method === 'GET' && path.endsWith('/api/deep-evaluation/metrics')) {
      return jsonResponse(route, [
        {
          name: 'answer_relevancy',
          description: '答案与问题的相关性',
          category: 'relevancy',
          requires_llm: true,
          requires_expected_output: false,
          requires_actual_output: true,
          requires_retrieval_context: false,
        },
        {
          name: 'contextual_precision',
          description: '检索上下文的精准度',
          category: 'retrieval',
          requires_llm: true,
          requires_expected_output: false,
          requires_actual_output: false,
          requires_retrieval_context: true,
        },
      ])
    }

    if (method === 'POST' && path.endsWith('/api/deep-evaluation/tasks')) {
      return jsonResponse(route, {
        task_id: 'deep-eval-001',
        message: 'deep evaluation task created',
      })
    }

    if (method === 'GET' && path.endsWith('/api/deep-evaluation/tasks')) {
      return jsonResponse(route, {
        items: [
          {
            task_id: 'deep-eval-001',
            task_name: '示例深度评估任务',
            source_type: 'dataset',
            dataset_id: 'ds-456',
            sample_size: 200,
            split: 'eval',
            field_mapping: {
              input: 'query',
              retrieval_context: 'contexts',
              actual_output: 'answer',
            },
            metrics: ['answer_relevancy', 'contextual_precision'],
            llm_config: { endpoint: 'http://localhost:8000/v1', model: 'qwen-max', api_key: '***' },
            concurrency: 5,
            status: 'completed',
            progress: 100,
            total_samples: 200,
            processed_samples: 200,
            results_summary: {
              total_samples: 200,
              overall: { mean: 0.76, min: 0.31, max: 0.93 },
              metrics: {
                answer_relevancy: { mean: 0.78, min: 0.2, max: 0.95, count: 200 },
                contextual_precision: { mean: 0.74, min: 0.25, max: 0.92, count: 200 },
              },
            },
            results_path: '/app/output/deep_evaluations/deep-eval-001.json',
            created_at: new Date().toISOString(),
          },
          {
            task_id: 'deep-eval-002',
            task_name: '运行中的深度评估',
            source_type: 'dataset',
            dataset_id: 'ds-456',
            sample_size: 100,
            split: 'eval',
            field_mapping: { input: 'query', retrieval_context: 'contexts' },
            metrics: ['answer_relevancy'],
            llm_config: { endpoint: 'http://localhost:8000/v1', model: 'qwen-max', api_key: '***' },
            concurrency: 5,
            status: 'running',
            progress: 45,
            total_samples: 100,
            processed_samples: 45,
            created_at: new Date().toISOString(),
          },
        ],
        total: 2,
      })
    }

    if (method === 'POST' && /\/api\/deep-evaluation\/tasks\/[^/]+\/cancel$/.test(path)) {
      return jsonResponse(route, { message: 'Task cancelled', task_id: 'deep-eval-002' })
    }

    if (method === 'POST' && /\/api\/deep-evaluation\/tasks\/[^/]+\/resume$/.test(path)) {
      return jsonResponse(route, { message: 'Task resumed', task_id: 'deep-eval-001' })
    }

    if (method === 'DELETE' && /\/api\/deep-evaluation\/tasks\/[^/]+$/.test(path)) {
      return jsonResponse(route, { message: 'Task deleted', task_id: 'deep-eval-001' })
    }

    if (method === 'POST' && path.endsWith('/api/deep-evaluation/evaluate')) {
      return jsonResponse(route, {
        results: {
          answer_relevancy: { score: 0.82, reason: '回答与问题高度相关' },
        },
        overall_score: 0.82,
      })
    }

    if (method === 'POST' && path.endsWith('/api/deep-evaluation/batch')) {
      return jsonResponse(route, {
        results: [
          {
            sample: { input: '问题 1' },
            metric_results: {
              answer_relevancy: { score: 0.8, reason: '相关' },
            },
            overall_score: 0.8,
          },
        ],
        summary: {
          total_samples: 1,
          metrics: {
            answer_relevancy: { mean: 0.8, min: 0.8, max: 0.8, count: 1 },
          },
          overall: { mean: 0.8, min: 0.8, max: 0.8 },
        },
      })
    }

    if (method === 'GET' && path.endsWith('/api/models/downloads')) {
      return jsonResponse(route, { downloads: [], total: 0 })
    }

    if (method === 'GET' && path.endsWith('/api/models')) {
      return jsonResponse(route, {
        models: [
          {
            model_id: 'model-001',
            model_name: 'bge-base-zh',
            model_type: 'embedding',
            model_path: '/data/models/bge-base-zh',
            base_model_path: 'BAAI/bge-base-zh-v1.5',
            description: '示例模型',
            source_type: 'trained',
            extra_metadata: { trainer: 'train-factory' },
            status: 'available',
            metrics: { accuracy: 0.9123 },
            tags: ['demo'],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        total: 1,
      })
    }

    if (method === 'GET' && path.endsWith('/api/deployments')) {
      return jsonResponse(route, {
        deployments: [
          {
            deployment_id: 'dep-001',
            model_id: 'model-001',
            model_uid: 'bge-base-zh-v1',
            deployment_name: 'xinference-deployment',
            xinference_endpoint: 'http://localhost:9997',
            replica: 1,
            gpu_memory_utilization: 0.9,
            deploy_mode: 'container',
            container_name: 'xinference-dep-001',
            gpu_id: 0,
            port: 9997,
            inference_framework: 'xinference',
            enable_lora: false,
            max_loras: 4,
            max_lora_rank: 64,
            status: 'running',
            config: {
              docker_cmd:
                'docker run --gpus device=0 -p 9997:9997 xinference:latest xinference-local --host 0.0.0.0 --port 9997',
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          {
            deployment_id: 'dep-002',
            model_id: 'model-001',
            model_uid: 'bge-base-zh-v2',
            deployment_name: 'vllm-lora-deployment',
            xinference_endpoint: 'http://localhost:8000',
            replica: 1,
            gpu_memory_utilization: 0.85,
            deploy_mode: 'container',
            container_name: 'vllm-dep-002',
            gpu_id: 1,
            port: 8000,
            inference_framework: 'vllm',
            enable_lora: true,
            max_loras: 4,
            max_lora_rank: 64,
            replica_instances: [
              {
                replica_id: '11111111-1111-4111-8111-111111111111',
                deployment_id: 'dep-002',
                replica_index: 0,
                endpoint: 'http://127.0.0.1:8000',
                port: 8000,
                gpu_ids: [0, 1],
                status: 'running',
                health_status: 'HEALTHY',
              },
              {
                replica_id: '22222222-2222-4222-8222-222222222222',
                deployment_id: 'dep-002',
                replica_index: 1,
                endpoint: 'http://127.0.0.1:8002',
                port: 8002,
                gpu_ids: [2, 3],
                status: 'running',
                health_status: 'HEALTHY',
              },
            ],
            status: 'running',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          {
            deployment_id: 'dep-003',
            model_id: 'model-001',
            model_uid: 'bge-base-zh-v3',
            deployment_name: 'sglang-deployment',
            xinference_endpoint: 'http://localhost:8001',
            replica: 1,
            gpu_memory_utilization: 0.9,
            deploy_mode: 'container',
            container_name: 'sglang-dep-003',
            gpu_id: 2,
            port: 8001,
            inference_framework: 'sglang',
            enable_lora: true,
            max_loras: 8,
            max_lora_rank: 128,
            status: 'stopped',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        total: 3,
      })
    }

    if (method === 'GET' && path.endsWith('/api/deployments/discover-models')) {
      const endpoint = url.searchParams.get('endpoint') || 'http://localhost:9997'
      const framework = url.searchParams.get('framework') || 'xinference'
      return jsonResponse(route, {
        endpoint,
        framework,
        models: [
          {
            model_uid: 'bind-model-001',
            model_name: 'bge-base-zh-v1',
            model_type: 'embedding',
            model_path: '/models/bge-base-zh',
            status: 'running',
          },
        ],
      })
    }

    if (method === 'POST' && path.endsWith('/api/deployments/bind-existing')) {
      const body = request.postDataJSON() as Record<string, unknown>
      return jsonResponse(route, {
        deployment_id: 'dep-bind-001',
        model_id: 'model-bind-001',
        model_uid: body.model_uid || 'bind-model-001',
        deployment_name: body.deployment_name || 'bound-model',
        xinference_endpoint: body.endpoint || 'http://localhost:9997',
        replica: 1,
        gpu_memory_utilization: 0.0,
        deploy_mode: 'shared',
        container_name: body.container_name || 'xinference',
        inference_framework: body.inference_framework || 'xinference',
        enable_lora: false,
        max_loras: 4,
        max_lora_rank: 64,
        status: 'running',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
    }

    const replicaActionMatch = path.match(
      /\/api\/deployments\/([^/]+)\/replicas\/([^/]+)\/(start|stop|restart|recreate)$/
    )
    if (method === 'POST' && replicaActionMatch) {
      const [, deploymentId, replicaId, action] = replicaActionMatch
      return jsonResponse(route, {
        replica_id: replicaId,
        deployment_id: deploymentId,
        replica_index: replicaId.startsWith('2222') ? 1 : 0,
        endpoint: replicaId.startsWith('2222')
          ? 'http://127.0.0.1:8002'
          : 'http://127.0.0.1:8000',
        port: replicaId.startsWith('2222') ? 8002 : 8000,
        gpu_ids: replicaId.startsWith('2222') ? [2, 3] : [0, 1],
        status: action === 'stop' ? 'stopped' : 'running',
        health_status: action === 'stop' ? 'UNKNOWN' : 'HEALTHY',
      })
    }

    if (method === 'POST' && /\/api\/deployments\/[^/]+\/restart$/.test(path)) {
      const deploymentId = path.split('/')[3]
      return jsonResponse(route, {
        deployment_id: deploymentId,
        model_id: 'model-001',
        model_uid: 'bge-base-zh-v1',
        deployment_name: 'xinference-deployment',
        xinference_endpoint: 'http://localhost:9997',
        replica: 1,
        gpu_memory_utilization: 0.9,
        deploy_mode: 'container',
        container_name: 'xinference-dep-001',
        gpu_id: 0,
        port: 9997,
        inference_framework: 'xinference',
        enable_lora: false,
        max_loras: 4,
        max_lora_rank: 64,
        status: 'running',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
    }

    // Adapter APIs
    if (method === 'GET' && /\/api\/deployments\/[^/]+\/adapters$/.test(path)) {
      return jsonResponse(route, {
        adapters: [
          {
            adapter_id: 'adapter-001',
            deployment_id: 'dep-002',
            adapter_name: 'my-lora-adapter',
            adapter_path: '/data/output/task-001/checkpoint-100',
            source_task_id: 'task-001',
            status: 'loaded',
            loaded_at: new Date().toISOString(),
          },
        ],
      })
    }

    if (method === 'GET' && path.endsWith('/api/adapters/available')) {
      return jsonResponse(route, {
        adapters: [
          {
            source: 'training_task',
            source_id: 'task-001',
            name: 'task-task-00',
            path: '/data/output/task-001/checkpoint-100',
            base_model_id: 'model-001',
            created_at: new Date().toISOString(),
            description: 'From training task: LoRA 微调任务',
          },
          {
            source: 'model_registry',
            source_id: 'model-002',
            name: 'registered-lora',
            path: '/data/models/registered-lora',
            base_model_id: 'model-001',
            created_at: new Date().toISOString(),
            description: '注册的 LoRA adapter',
          },
        ],
      })
    }

    if (method === 'GET' && path.endsWith('/api/configs')) {
      return jsonResponse(route, {
        configs: [
          {
            config_id: 'cfg-001',
            name: 'openai-default',
            provider: 'openai',
            api_endpoint: 'https://api.openai.com/v1',
            api_key: 'sk-test',
            model_name: 'gpt-4o-mini',
            model_type: 'llm',
            source_type: 'external_api',
            is_active: true,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
          {
            config_id: 'cfg-002',
            name: 'local-vllm',
            provider: 'custom',
            api_endpoint: 'http://localhost:8080/v1',
            api_key: '',
            model_name: 'Qwen3-Reranker-0.6B',
            model_type: 'reranker',
            source_type: 'local_deployed',
            is_active: true,
            check_status: 'healthy',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        total: 2,
      })
    }

    if (method === 'GET' && path.endsWith('/api/resources/status')) {
      return jsonResponse(route, {
        gpu: {
          total_gpus: 1,
          allocated_gpus: 0,
          free_gpus: 1,
          cpu_summary: {
            logical_cores: 16,
            physical_cores: 8,
            cpu_usage_percent: 12.5,
          },
          system_memory: {
            total_gb: 64,
            used_gb: 12,
            free_gb: 52,
            usage_percent: 18.75,
          },
        },
        system: {
          cpu_percent: 12.5,
          memory_percent: 18.75,
          memory_used_mb: 12288,
          memory_total_mb: 65536,
          disk_usage_percent: 45.2,
          disk_used_gb: 120,
          disk_total_gb: 256,
          open_files: 120,
          thread_count: 64,
        },
      })
    }

    if (method === 'GET' && path.endsWith('/api/resources/gpus')) {
      return jsonResponse(route, {
        total: 1,
        gpus: [
          {
            id: 0,
            name: 'NVIDIA A100',
            memory_total_gb: 80,
            memory_used_gb: 12,
            memory_free_gb: 68,
            memory_usage_percent: 15,
            gpu_utilization: 10,
            temperature: 45,
            power_usage_w: 120,
            power_limit_w: 300,
            is_allocated: false,
            allocated_task: null,
          },
        ],
      })
    }

    if (method === 'GET' && path.endsWith('/api/resources/gpus/processes')) {
      return jsonResponse(route, {
        total: 1,
        gpus: [
          {
            gpu_index: 0,
            name: 'NVIDIA A100',
            uuid: 'GPU-test-0',
            total_memory_mb: 81920,
            processes: [
              {
                pid: 1234,
                name: 'python',
                cmdline: 'python train.py --config configs/reranker.yaml',
                user: 'root',
                gpu_memory_mb: 10240,
                container_id: 'abcdef123456',
                container_name: 'train-factory-api-dev',
                started_at: new Date().toISOString(),
              },
            ],
          },
        ],
      })
    }

    if (method === 'GET' && path.endsWith('/api/evaluations/tasks')) {
      return jsonResponse(route, {
        items: [
          {
            task_id: 'eval-001',
            task_name: '对比评估任务',
            eval_type: 'multi_model',
            model_configs: [
              {
                endpoint: 'http://localhost:9997/v1',
                model_name: 'Qwen3-Reranker',
                name: 'baseline',
                gpu_id: 0,
              },
            ],
            dataset_configs: [{ type: 'mteb', name: 'T2Reranking' }],
            status: 'succeeded',
            progress: 100,
            current_model: null,
            current_dataset: null,
            results: { baseline: { T2Reranking: { 'NDCG@10': 0.8765 } } },
            created_at: new Date().toISOString(),
            started_at: new Date().toISOString(),
            completed_at: new Date().toISOString(),
          },
          {
            task_id: 'eval-002',
            task_name: '失败的评估任务',
            eval_type: 'single',
            model_configs: [
              {
                endpoint: 'http://localhost:9998/v1',
                model_name: 'test-model',
                name: 'test',
                gpu_id: 0,
              },
            ],
            dataset_configs: [{ type: 'mteb', name: 'T2Reranking' }],
            status: 'failed',
            progress: 30,
            current_model: 'test',
            current_dataset: 'T2Reranking',
            results: null,
            error_message: 'Connection refused',
            created_at: new Date().toISOString(),
            started_at: new Date().toISOString(),
            completed_at: null,
          },
        ],
        total: 2,
      })
    }

    if (method === 'DELETE' && /\/api\/evaluations\/tasks\/[^/]+$/.test(path)) {
      return jsonResponse(route, { message: '任务已删除', task_id: path.split('/').pop() })
    }

    if (method === 'GET' && path.endsWith('/api/evaluations/datasets')) {
      return jsonResponse(route, {
        datasets: {
          T2Reranking: { split: 'dev', lang: 'zh', description: '中文通用重排序' },
          MMarcoReranking: { split: 'dev', lang: 'zh', description: '中文 MS MARCO' },
        },
        groups: {
          chinese: ['T2Reranking', 'MMarcoReranking'],
          all: ['T2Reranking', 'MMarcoReranking'],
        },
      })
    }

    // Config check connection
    if (method === 'POST' && /\/api\/configs\/[^/]+\/check$/.test(path)) {
      return jsonResponse(route, {
        success: true,
        latency_ms: 12.34,
        models: ['gpt-4o-mini', 'text-embedding-3-small'],
      })
    }

    // ── Sync: External API Configs ──────────────────────────
    if (method === 'GET' && path.endsWith('/api/sync/api-configs')) {
      return jsonResponse(route, {
        configs: [
          {
            config_id: 'apicfg-001',
            config_name: '业务系统A',
            user_id: '',
            api_url: 'http://business-system.example.com:9003/api/embedding_texts',
            auth_config: { token: '***' },
            description: '生产环境业务系统',
            status: 'active',
            created_at: '2026-02-10T08:00:00',
            updated_at: '2026-02-10T08:00:00',
          },
          {
            config_id: 'apicfg-002',
            config_name: '测试系统B',
            user_id: '',
            api_url: 'http://another-system.example.com:8080/api/texts',
            auth_config: { token: '***' },
            description: '',
            status: 'inactive',
            created_at: '2026-02-09T10:00:00',
            updated_at: '2026-02-09T10:00:00',
          },
        ],
        total: 2,
      })
    }

    if (method === 'POST' && path.endsWith('/api/sync/api-configs')) {
      const body = request.postDataJSON() as Record<string, unknown>
      return jsonResponse(
        route,
        {
          message: 'API config created',
          config: {
            config_id: 'apicfg-new',
            config_name: body.config_name || 'New Config',
            user_id: '',
            api_url: body.api_url || '',
            auth_config: { token: '***' },
            description: body.description || '',
            status: 'active',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        },
        201
      )
    }

    if (method === 'PATCH' && /\/api\/sync\/api-configs\/[^/]+$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      return jsonResponse(route, {
        message: 'API config updated',
        config: {
          config_id: path.split('/').pop(),
          config_name: body.config_name || '业务系统A',
          user_id: '',
          api_url: body.api_url || 'http://business-system.example.com:9003/api/embedding_texts',
          auth_config: { token: '***' },
          description: body.description || '',
          status: body.status || 'active',
          created_at: '2026-02-10T08:00:00',
          updated_at: new Date().toISOString(),
        },
      })
    }

    if (method === 'DELETE' && /\/api\/sync\/api-configs\/[^/]+$/.test(path)) {
      return jsonResponse(route, { message: 'API config deleted' })
    }

    if (method === 'POST' && path.endsWith('/api/sync/api-configs/test-connection')) {
      return jsonResponse(route, { success: true, message: '100', status_code: 200 })
    }

    if (method === 'POST' && /\/api\/sync\/api-configs\/[^/]+\/test-connection$/.test(path)) {
      return jsonResponse(route, { success: true, message: '100', status_code: 200 })
    }

    // ── Sync: Sync Tasks ──────────────────────────────────
    if (method === 'GET' && path.endsWith('/api/sync/tasks')) {
      return jsonResponse(route, {
        tasks: [
          {
            task_id: 'sync-001',
            task_name: '自动同步-业务A',
            user_id: '',
            external_api_config_id: 'apicfg-001',
            external_api_url: '',
            sync_interval_seconds: 300,
            last_sync_at: '2026-02-12T06:30:00',
            generation_threshold: 500,
            generation_mode: 'doc_to_training',
            generation_config: {},
            training_threshold: 1000,
            training_config: {},
            base_deployment_id: null,
            pending_record_count: 120,
            pending_training_samples: 50,
            total_record_count: 2500,
            total_training_samples: 800,
            total_trainings: 2,
            current_adapter_name: 'sync-adapter-r2',
            current_adapter_id: null,
            current_training_id: null,
            is_active: true,
            status: 'syncing',
            error_message: null,
            created_at: '2026-02-01T10:00:00',
            updated_at: '2026-02-12T06:30:00',
          },
          {
            task_id: 'sync-002',
            task_name: '手动同步-测试B',
            user_id: '',
            external_api_config_id: null,
            external_api_url: 'http://another-system.example.com:8080/api/texts',
            sync_interval_seconds: 600,
            last_sync_at: null,
            generation_threshold: 200,
            generation_mode: 'doc_to_training',
            generation_config: {},
            training_threshold: 500,
            training_config: {},
            base_deployment_id: null,
            pending_record_count: 0,
            pending_training_samples: 0,
            total_record_count: 0,
            total_training_samples: 0,
            total_trainings: 0,
            current_adapter_name: null,
            current_adapter_id: null,
            current_training_id: null,
            is_active: false,
            status: 'idle',
            error_message: null,
            created_at: '2026-02-05T14:00:00',
            updated_at: '2026-02-05T14:00:00',
          },
        ],
        total: 2,
      })
    }

    if (method === 'POST' && path.endsWith('/api/sync/tasks')) {
      const body = request.postDataJSON() as Record<string, unknown>
      return jsonResponse(
        route,
        {
          message: 'Sync task created',
          task: {
            task_id: 'sync-new',
            task_name: body.task_name || 'New Sync',
            user_id: '',
            external_api_config_id: body.external_api_config_id || null,
            external_api_url: (body.external_api_url as string) || '',
            sync_interval_seconds: body.sync_interval_seconds || 300,
            last_sync_at: null,
            generation_threshold: body.generation_threshold || 500,
            generation_mode: 'doc_to_training',
            generation_config: body.generation_config || {},
            training_threshold: body.training_threshold || 1000,
            training_config: body.training_config || {},
            base_deployment_id: null,
            pending_record_count: 0,
            pending_training_samples: 0,
            total_record_count: 0,
            total_training_samples: 0,
            total_trainings: 0,
            current_adapter_name: null,
            current_adapter_id: null,
            current_training_id: null,
            is_active: true,
            status: 'idle',
            error_message: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        },
        201
      )
    }

    if (method === 'GET' && /\/api\/sync\/tasks\/[^/]+\/batches$/.test(path)) {
      return jsonResponse(route, {
        batches: [
          {
            batch_id: 'batch-001',
            task_id: 'sync-001',
            user_id: '',
            record_count: 85,
            storage_path: '/app/data/sync/batch_001.jsonl',
            since_time: '2026-02-12T05:00:00',
            until_time: '2026-02-12T06:00:00',
            fetched_at: '2026-02-12T06:01:00',
            status: 'generation_done',
            generation_task_id: 'gen-task-001',
          },
          {
            batch_id: 'batch-002',
            task_id: 'sync-001',
            user_id: '',
            record_count: 35,
            storage_path: '/app/data/sync/batch_002.jsonl',
            since_time: '2026-02-12T06:00:00',
            until_time: '2026-02-12T06:30:00',
            fetched_at: '2026-02-12T06:31:00',
            status: 'fetched',
            generation_task_id: null,
          },
        ],
        total: 2,
      })
    }

    if (method === 'GET' && /\/api\/sync\/tasks\/[^/]+\/generations$/.test(path)) {
      return jsonResponse(route, {
        generations: [
          {
            task_id: 'sync-001',
            generation_task_id: 'gen-task-001',
            user_id: '',
            input_batch_ids: ['batch-001'],
            input_record_count: 85,
            output_dataset_id: 'ds-gen-001',
            output_sample_count: 420,
            status: 'completed',
            created_at: '2026-02-12T06:02:00',
            completed_at: '2026-02-12T06:15:00',
          },
        ],
        total: 1,
      })
    }

    if (method === 'GET' && /\/api\/sync\/tasks\/[^/]+\/trainings$/.test(path)) {
      return jsonResponse(route, {
        trainings: [
          {
            task_id: 'sync-001',
            training_task_id: 'train-task-001',
            user_id: '',
            input_dataset_ids: ['ds-gen-001'],
            total_samples: 420,
            training_round: 2,
            output_adapter_path: '/app/output/sync-adapter-r2',
            loaded_adapter_name: 'sync-adapter-r2',
            loaded_adapter_id: 'adapter-sync-r2',
            previous_training_task_id: 'train-task-000',
            status: 'adapter_loaded',
            created_at: '2026-02-12T06:20:00',
            completed_at: '2026-02-12T06:35:00',
          },
        ],
        total: 1,
      })
    }

    if (method === 'GET' && /\/api\/sync\/tasks\/[^/]+\/status$/.test(path)) {
      return jsonResponse(route, {
        task_id: 'sync-001',
        status: 'idle',
        is_active: true,
        worker_status: 'running',
        pending_record_count: 120,
        generation_threshold: 500,
        pending_training_samples: 50,
        training_threshold: 1000,
        total_record_count: 2500,
        total_training_samples: 800,
        total_trainings: 2,
        current_adapter_name: 'sync-adapter-r2',
        last_sync_at: '2026-02-12T06:30:00',
      })
    }

    if (method === 'GET' && /\/api\/sync\/tasks\/[^/]+$/.test(path)) {
      return jsonResponse(route, {
        task: {
          task_id: 'sync-001',
          task_name: '自动同步-业务A',
          user_id: '',
          external_api_config_id: 'apicfg-001',
          external_api_url: 'http://business-system.example.com:9003/api/embedding_texts',
          sync_interval_seconds: 300,
          last_sync_at: '2026-02-12T06:30:00',
          generation_threshold: 500,
          generation_mode: 'doc_to_training',
          generation_config: {},
          training_threshold: 1000,
          training_config: {},
          base_deployment_id: null,
          pending_record_count: 120,
          pending_training_samples: 50,
          total_record_count: 2500,
          total_training_samples: 800,
          total_trainings: 2,
          current_adapter_name: 'sync-adapter-r2',
          current_adapter_id: null,
          current_training_id: null,
          is_active: true,
          status: 'idle',
          error_message: null,
          created_at: '2026-02-01T10:00:00',
          updated_at: '2026-02-12T06:30:00',
        },
      })
    }

    if (
      method === 'POST' &&
      /\/api\/sync\/tasks\/[^/]+\/(start|stop|sync-now|trigger-generation|trigger-training)$/.test(
        path
      )
    ) {
      const action = path.split('/').pop()
      if (action === 'start')
        return jsonResponse(route, { message: 'Sync started', worker_status: 'running' })
      if (action === 'stop')
        return jsonResponse(route, { message: 'Sync stopped', worker_status: 'not_found' })
      return jsonResponse(route, { message: `${action} completed` })
    }

    if (method === 'PATCH' && /\/api\/sync\/tasks\/[^/]+$/.test(path)) {
      return jsonResponse(route, { message: 'Sync task updated', task: {} })
    }

    if (method === 'DELETE' && /\/api\/sync\/tasks\/[^/]+$/.test(path)) {
      return jsonResponse(route, { message: 'Sync task deleted' })
    }

    if (path.includes('/api/auth/')) {
      return jsonResponse(route, { detail: `Unhandled auth endpoint: ${method} ${path}` }, 404)
    }

    // ── Fallback ────────────────────────────────────────────
    if (method !== 'GET') {
      return jsonResponse(route, { success: true })
    }

    return jsonResponse(route, {})
  })

  await mockAuth(page, { ...authOptions, authenticated: true })
}
