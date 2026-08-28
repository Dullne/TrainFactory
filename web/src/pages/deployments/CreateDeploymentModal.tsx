import { useEffect, useMemo, useState } from 'react'
import {
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Button,
  Space,
  Switch,
  Collapse,
  Alert,
  Radio,
  Spin,
  message,
} from 'antd'
import { DeleteOutlined, PlusOutlined, SearchOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { GpuSelect } from '@/components/GpuSelect'
import { deploymentApi, DiscoveredModel, externalApiConfigApi } from '@/services/api'
import {
  isSglangRerankerLoraUnsupported,
  normalizeDeploymentLoraEnabled,
  supportsDeploymentLora,
} from './deploymentLoraPolicy'
import { isValidSglangExpertParallelSize } from './deploymentConfigPolicy'
import type {
  DeploymentLaunchConfig,
  ExternalApiConfig,
  RegisteredModel,
  CreateDeploymentRequest,
  ReplicaGpuOverride,
} from '@/types'

const VLLM_QUANTIZATION_OPTIONS = [
  'awq',
  'auto_awq',
  'fp8',
  'fbgemm_fp8',
  'fp_quant',
  'modelopt',
  'modelopt_fp4',
  'modelopt_mxfp8',
  'modelopt_mixed',
  'auto_gptq',
  'gptq',
  'gptq_marlin',
  'awq_marlin',
  'humming',
  'compressed-tensors',
  'bitsandbytes',
  'experts_int8',
  'quark',
  'moe_wna16',
  'torchao',
  'inc',
  'mxfp4',
  'gpt_oss_mxfp4',
  'deepseek_v4_fp8',
  'online',
  'fp8_per_tensor',
  'fp8_per_block',
  'fp8_per_channel',
  'int8_per_channel_weight_only',
  'nvfp4_per_token',
  'mxfp8',
] as const

const SGLANG_QUANTIZATION_OPTIONS = [
  'awq',
  'fp8',
  'mxfp8',
  'gptq',
  'marlin',
  'gptq_marlin',
  'awq_marlin',
  'bitsandbytes',
  'gguf',
  'modelopt',
  'modelopt_fp8',
  'modelopt_fp4',
  'nvfp4_online',
  'modelopt_mixed',
  'petit_nvfp4',
  'w8a8_int8',
  'w8a8_fp8',
  'moe_wna16',
  'w4afp8',
  'mxfp4',
  'auto-round',
  'auto-round-int8',
  'compressed-tensors',
  'modelslim',
  'mxfp_w4a8',
  'quark',
  'quark_int4fp8_moe',
  'quark_mxfp4',
  'mlx_q4',
  'mlx_q8',
  'unquant',
  'humming',
] as const

const DTYPE_OPTIONS = ['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'] as const

const VLLM_KV_CACHE_OPTIONS = [
  'auto',
  'float16',
  'bfloat16',
  'fp8',
  'fp8_e4m3',
  'fp8_e5m2',
  'fp8_inc',
  'fp8_ds_mla',
  'turboquant_k8v4',
  'turboquant_4bit_nc',
  'turboquant_k3v4_nc',
  'turboquant_3bit_nc',
  'int4_per_token_head',
  'int8_per_token_head',
  'fp8_per_token_head',
  'nvfp4',
] as const

const SGLANG_KV_CACHE_OPTIONS = [
  'auto',
  'fp8_e5m2',
  'fp8_e4m3',
  'mxfp8',
  'bf16',
  'bfloat16',
  'nvfp4',
  'fp4_mx_block16',
  'fp4_e2m1',
] as const

const SGLANG_ATTENTION_OPTIONS = [
  'triton',
  'torch_native',
  'flex_attention',
  'dsa',
  'nsa',
  'dsv4',
  'compressed',
  'cutlass_mla',
  'fa3',
  'fa4',
  'flashinfer',
  'flashmla',
  'trtllm_mla',
  'cutedsl_mla',
  'tokenspeed_mla',
  'trtllm_mha',
  'dual_chunk_flash_attn',
  'hpc_ops',
  'aiter',
  'wave',
  'intel_amx',
  'ascend',
  'intel_xpu',
] as const

interface CreateDeploymentModalProps {
  visible: boolean
  models: RegisteredModel[]
  creating: boolean
  onCancel: () => void
  onCreate: (values: CreateDeploymentRequest & { _useContainer?: boolean }) => void
  onBindExisting?: (values: {
    endpoint: string
    model_uid: string
    model_name?: string
    model_type: string
    deployment_name?: string
    inference_framework: string
    container_name?: string
    gpu_id?: number
    external_api_config_id?: string
  }) => void
}

export function CreateDeploymentModal({
  visible,
  models,
  creating,
  onCancel,
  onCreate,
  onBindExisting,
}: CreateDeploymentModalProps) {
  const { t } = useTranslation(['deployments', 'common'])
  const [form] = Form.useForm()
  const [framework, setFramework] = useState('xinference')
  const [enableLora, setEnableLora] = useState(false)
  const [deployMode, setDeployMode] = useState<'new' | 'bind'>('new')
  const [containerMode, setContainerMode] = useState<'shared' | 'container'>('shared')

  const [modelTypeFilter, setModelTypeFilter] = useState<string>()

  // Discover models state
  const [discovering, setDiscovering] = useState(false)
  const [discoveredModels, setDiscoveredModels] = useState<DiscoveredModel[]>([])
  const [selectedDiscoveredModel, setSelectedDiscoveredModel] = useState<DiscoveredModel | null>(null)
  const [apiConfigs, setApiConfigs] = useState<ExternalApiConfig[]>([])
  const [apiConfigsLoading, setApiConfigsLoading] = useState(false)
  const selectedModelId = Form.useWatch('model_id', form)
  const selectedModelType = useMemo(
    () => models.find((model) => model.model_id === selectedModelId)?.model_type,
    [models, selectedModelId],
  )
  const replicaCount = Form.useWatch('replica', form) ?? 1
  const tensorParallelSize = Form.useWatch('tensor_parallel_size', form) ?? 1
  const pipelineParallelSize = Form.useWatch('pipeline_parallel_size', form) ?? 1
  const dataParallelSize = Form.useWatch('data_parallel_size', form) ?? 1
  const watchedGpuPool = Form.useWatch('gpu_pool', form)
  const watchedReplicaGpuOverrides = Form.useWatch('replica_gpu_overrides', form)
  const gpuPool = useMemo(() => watchedGpuPool ?? [], [watchedGpuPool])
  const replicaGpuOverrides = useMemo(
    () => watchedReplicaGpuOverrides ?? [],
    [watchedReplicaGpuOverrides],
  )
  const allowGpuReuse = Form.useWatch('allow_gpu_reuse', form) ?? false

  const topologyPreview = useMemo(() => {
    const requiredPerReplica =
      tensorParallelSize * pipelineParallelSize * dataParallelSize
    const overrides = new Map<number, number[]>(
      (replicaGpuOverrides as ReplicaGpuOverride[])
        .filter(
          (item) =>
            Number.isInteger(item?.replica_index) && Array.isArray(item?.gpu_ids),
        )
        .map((item) => [item.replica_index, item.gpu_ids]),
    )
    const reserved = new Set([...overrides.values()].flat())
    const remaining = (gpuPool as number[]).filter((gpuId) => !reserved.has(gpuId))
    let cursor = 0
    const replicas = Array.from({ length: replicaCount }, (_, replicaIndex) => {
      const override = overrides.get(replicaIndex)
      if (override) {
        return { replicaIndex, gpuIds: override, manual: true }
      }
      const gpuIds = remaining.slice(cursor, cursor + requiredPerReplica)
      cursor += requiredPerReplica
      return { replicaIndex, gpuIds, manual: false }
    })
    const assignedGpuIds = replicas.flatMap((item) => item.gpuIds)
    return {
      requiredPerReplica,
      replicas,
      hasGpuReuse: new Set(assignedGpuIds).size !== assignedGpuIds.length,
    }
  }, [
    dataParallelSize,
    gpuPool,
    pipelineParallelSize,
    replicaCount,
    replicaGpuOverrides,
    tensorParallelSize,
  ])

  const MODEL_TYPE_OPTIONS = [
    { label: t('common:modelType.embedding'), value: 'embedding' },
    { label: t('common:modelType.reranker'), value: 'reranker' },
    { label: t('common:modelType.decoderReranker'), value: 'decoder_reranker' },
    { label: t('common:modelType.llm'), value: 'llm' },
  ]

  const filteredModels = modelTypeFilter
    ? models.filter((m) => m.model_type === modelTypeFilter)
    : models

  const INFERENCE_FRAMEWORKS = [
    { value: 'xinference', label: 'Xinference', description: t('create.frameworkXinference') },
    { value: 'vllm', label: 'vLLM', description: t('create.frameworkVllm') },
    { value: 'sglang', label: 'SGLang', description: t('create.frameworkSglang') },
  ]

  const handleFinish = (values: CreateDeploymentRequest & Record<string, unknown>) => {
    if (deployMode === 'bind' && selectedDiscoveredModel) {
      // Bind existing model
      onBindExisting?.({
        endpoint: values.bind_endpoint as string,
        model_uid: selectedDiscoveredModel.model_uid,
        model_name: selectedDiscoveredModel.model_name || selectedDiscoveredModel.model_uid,
        model_type: selectedDiscoveredModel.model_type || 'embedding',
        deployment_name: values.deployment_name as string,
        inference_framework: values.bind_framework as string,
        container_name: values.bind_container_name as string | undefined,
        gpu_id: values.bind_gpu_id as number | undefined,
        external_api_config_id: values.external_api_config_id as string | undefined,
      })
    } else {
      // Deploy new model
      const {
        dtype,
        enforce_eager,
        attention_backend,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
        max_context_length,
        max_concurrent_requests,
        quantization,
        kv_cache_dtype,
        gpu_pool,
        replica_gpu_overrides,
        allow_gpu_reuse,
        enable_expert_parallel,
        expert_parallel_size,
        ...rest
      } = values as Record<string, unknown>
      delete rest.bind_endpoint
      delete rest.bind_framework
      delete rest.bind_container_name
      delete rest.bind_gpu_id
      const useContainer = framework !== 'xinference' || containerMode === 'container'
      let launchConfig: DeploymentLaunchConfig | undefined
      if (framework === 'vllm' || framework === 'sglang') {
        const commonLaunchConfig = {
          tensor_parallel_size: (tensor_parallel_size as number | undefined) ?? 1,
          pipeline_parallel_size: (pipeline_parallel_size as number | undefined) ?? 1,
          data_parallel_size: (data_parallel_size as number | undefined) ?? 1,
          max_context_length: (max_context_length as number | undefined) ?? null,
          max_concurrent_requests: (max_concurrent_requests as number | undefined) ?? null,
          dtype: dtype as DeploymentLaunchConfig['dtype'],
          quantization: (quantization as string | undefined) ?? null,
          kv_cache_dtype: kv_cache_dtype as string,
          gpu_pool: (gpu_pool as number[] | undefined) ?? [],
          replica_gpu_overrides:
            (replica_gpu_overrides as ReplicaGpuOverride[] | undefined) ?? [],
          allow_gpu_reuse: Boolean(allow_gpu_reuse),
        }
        launchConfig =
          framework === 'vllm'
            ? {
                ...commonLaunchConfig,
                framework: 'vllm',
                enable_expert_parallel: Boolean(enable_expert_parallel),
                enforce_eager: Boolean(enforce_eager),
              }
            : {
                ...commonLaunchConfig,
                framework: 'sglang',
                expert_parallel_size: (expert_parallel_size as number | undefined) ?? 1,
                attention_backend: (attention_backend as string | undefined) ?? null,
              }
      }
      const request = {
        ...rest,
        enable_lora: normalizeDeploymentLoraEnabled(
          Boolean(rest.enable_lora),
          framework,
          selectedModelType,
        ),
        ...(launchConfig ? { launch_config: launchConfig } : {}),
        _useContainer: useContainer,
      } as CreateDeploymentRequest & { _useContainer?: boolean }
      onCreate(request)
    }
    handleReset()
  }

  const handleReset = () => {
    form.resetFields()
    setFramework('xinference')
    setEnableLora(false)
    setDeployMode('new')
    setContainerMode('shared')
    setModelTypeFilter(undefined)
    setDiscoveredModels([])
    setSelectedDiscoveredModel(null)
  }

  const handleCancel = () => {
    handleReset()
    onCancel()
  }

  const handleFrameworkChange = (value: string) => {
    setFramework(value)
    const quantization = form.getFieldValue('quantization') as string | undefined
    const kvCacheDtype = form.getFieldValue('kv_cache_dtype') as string | undefined
    // Reset LoRA settings when switching to Xinference
    if (value === 'xinference') {
      setEnableLora(false)
      form.setFieldsValue({
        enable_lora: false,
        quantization: undefined,
        kv_cache_dtype: 'auto',
        enable_expert_parallel: false,
        enforce_eager: false,
        expert_parallel_size: 1,
        attention_backend: undefined,
        gpu_pool: [],
        replica_gpu_overrides: [],
        allow_gpu_reuse: false,
      })
      setContainerMode('shared')
    } else {
      setContainerMode('container')
      form.setFieldValue('gpu_id', undefined)
      if (value === 'vllm') {
        form.setFieldsValue({
          quantization: VLLM_QUANTIZATION_OPTIONS.includes(
            quantization as (typeof VLLM_QUANTIZATION_OPTIONS)[number],
          )
            ? quantization
            : undefined,
          kv_cache_dtype: VLLM_KV_CACHE_OPTIONS.includes(
            kvCacheDtype as (typeof VLLM_KV_CACHE_OPTIONS)[number],
          )
            ? kvCacheDtype
            : 'auto',
          enable_expert_parallel: false,
          enforce_eager: false,
          expert_parallel_size: 1,
          attention_backend: undefined,
        })
      } else {
        form.setFieldsValue({
          quantization: SGLANG_QUANTIZATION_OPTIONS.includes(
            quantization as (typeof SGLANG_QUANTIZATION_OPTIONS)[number],
          )
            ? quantization
            : undefined,
          kv_cache_dtype: SGLANG_KV_CACHE_OPTIONS.includes(
            kvCacheDtype as (typeof SGLANG_KV_CACHE_OPTIONS)[number],
          )
            ? kvCacheDtype
            : 'auto',
          enable_expert_parallel: false,
          enforce_eager: false,
          expert_parallel_size: 1,
          attention_backend: undefined,
        })
      }
    }
    // Clear discovered models when framework changes
    setDiscoveredModels([])
    setSelectedDiscoveredModel(null)
  }

  const handleDiscoverModels = async () => {
    const endpoint = form.getFieldValue('bind_endpoint')
    const bindFramework = form.getFieldValue('bind_framework') || 'xinference'

    if (!endpoint) {
      message.warning(t('create.endpointRequired'))
      return
    }

    setDiscovering(true)
    setDiscoveredModels([])
    setSelectedDiscoveredModel(null)

    try {
      const result = await deploymentApi.discoverModels(endpoint, bindFramework)
      if (result.models.length === 0) {
        message.info(t('create.discoverNoModels'))
      } else {
        message.success(t('create.discoverSuccess', { count: result.models.length }))
        setDiscoveredModels(result.models)
      }
    } catch (error) {
      message.error(t('create.discoverFailed', { error: (error as Error).message }))
    } finally {
      setDiscovering(false)
    }
  }

  const handleSelectDiscoveredModel = (modelUid: string | string[] | undefined) => {
    const uid = Array.isArray(modelUid) ? modelUid[0] : modelUid
    if (!uid) {
      setSelectedDiscoveredModel(null)
      return
    }
    const model = discoveredModels.find(m => m.model_uid === uid)
    setSelectedDiscoveredModel(model || null)
  }

  // 父组件在创建成功后直接置 visible=false（不经 onCancel），
  // 这里在每次打开时重置表单与本地状态，避免残留上一次的选择
  useEffect(() => {
    if (visible) {
      handleReset()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible])

  useEffect(() => {
    if (!visible) {
      return
    }
    let cancelled = false
    const loadApiConfigs = async () => {
      setApiConfigsLoading(true)
      try {
        const res = await externalApiConfigApi.list()
        if (!cancelled) {
          setApiConfigs(res.configs || [])
        }
      } catch {
        if (!cancelled) {
          setApiConfigs([])
        }
      } finally {
        if (!cancelled) {
          setApiConfigsLoading(false)
        }
      }
    }
    void loadApiConfigs()
    return () => {
      cancelled = true
    }
  }, [visible])

  const supportsManagedLaunch = framework === 'vllm' || framework === 'sglang'
  const sglangRerankerLoraUnsupported = isSglangRerankerLoraUnsupported(
    framework,
    selectedModelType,
  )
  const supportsLora = supportsDeploymentLora(framework, selectedModelType)

  useEffect(() => {
    if (enableLora && !supportsLora) {
      setEnableLora(false)
      form.setFieldValue('enable_lora', false)
    }
  }, [enableLora, form, supportsLora])

  return (
    <Modal
      title={t('create.title')}
      open={visible}
      onCancel={handleCancel}
      onOk={() => form.submit()}
      okText={deployMode === 'new' ? t('create.submitNew') : t('create.submitBind')}
      cancelText={t('common:action.cancel')}
      confirmLoading={creating}
      okButtonProps={{
        disabled: creating || (deployMode === 'bind' && !selectedDiscoveredModel),
      }}
      destroyOnHidden
      width={640}
      styles={{ body: { maxHeight: 'calc(100vh - 220px)', overflowY: 'auto' } }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        scrollToFirstError
        initialValues={{
          replica: 1,
          inference_framework: 'xinference',
          bind_framework: 'xinference',
          enable_lora: false,
          max_loras: 4,
          max_lora_rank: 64,
          tensor_parallel_size: 1,
          pipeline_parallel_size: 1,
          data_parallel_size: 1,
          dtype: 'auto',
          kv_cache_dtype: 'auto',
          gpu_pool: [],
          replica_gpu_overrides: [],
          allow_gpu_reuse: false,
          enable_expert_parallel: false,
          enforce_eager: false,
          expert_parallel_size: 1,
        }}
      >
        {/* Deploy Mode Selection */}
        <Form.Item label={t('create.deployMode')}>
          <Radio.Group
            value={deployMode}
            onChange={(e) => {
              setDeployMode(e.target.value)
              setDiscoveredModels([])
              setSelectedDiscoveredModel(null)
            }}
          >
            <Radio.Button value="new">{t('create.deployModeNew')}</Radio.Button>
            <Radio.Button value="bind">{t('create.deployModeBind')}</Radio.Button>
          </Radio.Group>
        </Form.Item>

        <Form.Item
          name="deployment_name"
          label={t('create.deploymentName')}
          rules={[{ required: true, message: t('create.deploymentNameRequired') }]}
        >
          <Input placeholder={t('create.deploymentNamePlaceholder')} />
        </Form.Item>

        <Form.Item
          name="external_api_config_id"
          label={t('create.externalApiConfig')}
          extra={t('create.externalApiConfigHint')}
        >
          <Select
            allowClear
            showSearch
            loading={apiConfigsLoading}
            placeholder={t('create.externalApiConfigPlaceholder')}
            optionFilterProp="label"
            options={apiConfigs.map((cfg) => ({
              label: `${cfg.config_name} (${cfg.config_id.slice(0, 8)})`,
              value: cfg.config_id,
            }))}
          />
        </Form.Item>

        {deployMode === 'new' ? (
          <>
            {/* Deploy New Model Form */}
            <Form.Item label={t('create.selectModel')} required style={{ marginBottom: 0 }}>
              <Space.Compact style={{ width: '100%', marginBottom: 16 }}>
                <Select
                  allowClear
                  placeholder={t('create.modelTypeFilter')}
                  style={{ width: 160 }}
                  options={MODEL_TYPE_OPTIONS}
                  value={modelTypeFilter}
                  onChange={(v) => {
                    setModelTypeFilter(v || undefined)
                    form.setFieldValue('model_id', undefined)
                  }}
                />
                <Form.Item
                  name="model_id"
                  noStyle
                  rules={[{ required: true, message: t('create.selectModelRequired') }]}
                >
                  <Select
                    showSearch
                    filterOption={(input, option) =>
                      (option?.label as string ?? '').toLowerCase().includes(input.toLowerCase())
                    }
                    placeholder={t('create.selectModelPlaceholder')}
                    style={{ flex: 1 }}
                    options={filteredModels.map((m) => ({
                      label: `${m.model_name} (${m.model_type})`,
                      value: m.model_id,
                    }))}
                  />
                </Form.Item>
              </Space.Compact>
            </Form.Item>

            <Form.Item
              name="inference_framework"
              label={t('create.inferenceFramework')}
              extra={t('create.inferenceFrameworkHint')}
            >
              <Select
                options={INFERENCE_FRAMEWORKS.map((f) => ({
                  label: `${f.label} - ${f.description}`,
                  value: f.value,
                }))}
                onChange={handleFrameworkChange}
              />
            </Form.Item>

            {framework === 'xinference' && (
              <>
                <Form.Item label={t('create.xinferenceMode')}>
                  <Radio.Group
                    value={containerMode}
                    onChange={(e) => setContainerMode(e.target.value)}
                  >
                    <Radio.Button value="shared">{t('create.xinferenceModeShared')}</Radio.Button>
                    <Radio.Button value="container">{t('create.xinferenceModeContainer')}</Radio.Button>
                  </Radio.Group>
                </Form.Item>

                {containerMode === 'shared' && (
                  <Form.Item
                    name="xinference_endpoint"
                    label={t('create.xinferenceEndpoint')}
                    extra={t('create.xinferenceEndpointHint')}
                  >
                    <Input placeholder="http://xinference:9997" />
                  </Form.Item>
                )}
              </>
            )}

            {sglangRerankerLoraUnsupported && (
              <Alert
                type="warning"
                showIcon
                message={t('create.sglangRerankerLoraUnsupported')}
                style={{ marginBottom: 16 }}
              />
            )}

            {supportsLora && (
              <>
                <Form.Item
                  name="enable_lora"
                  label={t('create.enableLora')}
                  valuePropName="checked"
                >
                  <Switch onChange={setEnableLora} />
                </Form.Item>

                {enableLora && (
                  <Alert
                    type="info"
                    showIcon
                    message={t('create.loraEnabledAlert')}
                    description={t('create.loraEnabledDesc')}
                    style={{ marginBottom: 16 }}
                  />
                )}

                <Collapse
                  ghost
                  items={[
                    {
                      key: 'lora_settings',
                      label: t('create.loraAdvanced'),
                      children: (
                        <>
                          <Form.Item
                            name="max_loras"
                            label={t('create.maxLoras')}
                            extra={t('create.maxLorasHint')}
                          >
                            <InputNumber min={1} max={16} style={{ width: '100%' }} />
                          </Form.Item>

                          <Form.Item
                            name="max_lora_rank"
                            label={t('create.maxLoraRank')}
                            extra={t('create.maxLoraRankHint')}
                          >
                            <InputNumber min={8} max={256} style={{ width: '100%' }} />
                          </Form.Item>
                        </>
                      ),
                    },
                  ]}
                />
              </>
            )}

            <Form.Item name="replica" label={t('create.replica')}>
              <InputNumber min={1} max={8} style={{ width: '100%' }} />
            </Form.Item>

            {framework === 'xinference' ? (
              <Form.Item
                name="gpu_id"
                label={t('create.gpuSelect')}
                extra={t('create.gpuSelectHint')}
              >
                <GpuSelect mode="single" />
              </Form.Item>
            ) : null}

            <Form.Item name="gpu_memory_utilization" label={t('create.gpuMemoryUtilization')} extra={t('create.gpuMemoryUtilizationHint')}>
              <InputNumber
                min={0.05}
                max={1.0}
                step={0.05}
                style={{ width: '100%' }}
                placeholder={t('create.gpuMemoryUtilizationPlaceholder')}
              />
            </Form.Item>

            {(framework === 'vllm' || framework === 'sglang') && (
              <Collapse
                defaultActiveKey={['parallel_settings', 'gpu_settings']}
                items={[
                  {
                    key: 'parallel_settings',
                    label: t('create.parallelSettings'),
                    children: (
                      <>
                        <div
                          data-testid="parallel-size-grid"
                          style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                            gap: 12,
                            width: '100%',
                          }}
                        >
                          <Form.Item
                            name="tensor_parallel_size"
                            label={t('create.tensorParallelSize')}
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber min={1} max={64} precision={0} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                          <Form.Item
                            name="pipeline_parallel_size"
                            label={t('create.pipelineParallelSize')}
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber min={1} max={64} precision={0} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                          <Form.Item
                            name="data_parallel_size"
                            label={t('create.dataParallelSize')}
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber min={1} max={64} precision={0} step={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </div>

                        <Form.Item
                          name="max_context_length"
                          label={t('create.maxContextLength')}
                          extra={t('create.maxContextLengthHint')}
                        >
                          <InputNumber min={1} max={4194304} style={{ width: '100%' }} />
                        </Form.Item>

                        <Form.Item
                          name="max_concurrent_requests"
                          label={t('create.maxConcurrentRequests')}
                          extra={t('create.maxConcurrentRequestsHint')}
                        >
                          <InputNumber min={1} max={4096} style={{ width: '100%' }} />
                        </Form.Item>

                        <Form.Item
                          name="dtype"
                          label={t('create.dtype')}
                          extra={t('create.dtypeHint')}
                        >
                          <Select
                            options={DTYPE_OPTIONS.map((value) => ({ label: value, value }))}
                          />
                        </Form.Item>

                        <Form.Item name="quantization" label={t('create.quantization')}>
                          <Select
                            allowClear
                            showSearch
                            placeholder={t('create.quantizationAuto')}
                            options={(framework === 'vllm'
                              ? VLLM_QUANTIZATION_OPTIONS
                              : SGLANG_QUANTIZATION_OPTIONS
                            ).map((value) => ({ label: value, value }))}
                          />
                        </Form.Item>

                        <Form.Item name="kv_cache_dtype" label={t('create.kvCacheDtype')}>
                          <Select
                            options={(framework === 'vllm'
                              ? VLLM_KV_CACHE_OPTIONS
                              : SGLANG_KV_CACHE_OPTIONS
                            ).map((value) => ({ label: value, value }))}
                          />
                        </Form.Item>

                        {framework === 'vllm' && (
                          <>
                            <Form.Item
                              name="enable_expert_parallel"
                              label={t('create.enableExpertParallel')}
                              valuePropName="checked"
                            >
                              <Switch />
                            </Form.Item>
                            <Form.Item
                              name="enforce_eager"
                              label={t('create.enforceEager')}
                              valuePropName="checked"
                              extra={t('create.enforceEagerHint')}
                            >
                              <Switch />
                            </Form.Item>
                          </>
                        )}

                        {framework === 'sglang' && (
                          <>
                            <Form.Item
                              name="expert_parallel_size"
                              label={t('create.expertParallelSize')}
                              extra={t('create.expertParallelSizeHint')}
                              dependencies={['tensor_parallel_size']}
                              rules={[
                                ({ getFieldValue }) => ({
                                  validator(_, value: number | undefined) {
                                    const expertSize = value ?? 1
                                    const tensorSize =
                                      (getFieldValue('tensor_parallel_size') as number | undefined) ?? 1
                                    return isValidSglangExpertParallelSize(
                                      expertSize,
                                      tensorSize,
                                    )
                                      ? Promise.resolve()
                                      : Promise.reject(
                                          new Error(t('create.expertParallelSizeInvalid')),
                                        )
                                  },
                                }),
                              ]}
                            >
                              <InputNumber min={1} max={64} precision={0} step={1} style={{ width: '100%' }} />
                            </Form.Item>
                            <Form.Item
                              name="attention_backend"
                              label={t('create.attentionBackend')}
                              extra={t('create.attentionBackendHint')}
                            >
                              <Select
                                allowClear
                                showSearch
                                placeholder={t('create.attentionBackendPlaceholder')}
                                options={SGLANG_ATTENTION_OPTIONS.map((value) => ({
                                  label: value,
                                  value,
                                }))}
                              />
                            </Form.Item>
                          </>
                        )}
                      </>
                    ),
                  },
                  {
                    key: 'gpu_settings',
                    label: t('create.gpuAllocation'),
                    children: (
                      <>
                        <Form.Item
                          name="gpu_pool"
                          label={t('create.gpuPool')}
                          extra={t('create.gpuPoolHint')}
                        >
                          <GpuSelect mode="multiple" allowClear />
                        </Form.Item>

                        <Form.Item
                          name="allow_gpu_reuse"
                          label={t('create.allowGpuReuse')}
                          valuePropName="checked"
                          extra={t('create.allowGpuReuseHint')}
                        >
                          <Switch />
                        </Form.Item>

                        {allowGpuReuse ? (
                          <Alert
                            type="warning"
                            showIcon
                            message={t('create.gpuReuseWarning')}
                            style={{ marginBottom: 16 }}
                          />
                        ) : null}

                        <Form.List name="replica_gpu_overrides">
                          {(fields, { add, remove }) => (
                            <Space direction="vertical" style={{ width: '100%' }}>
                              {fields.map(({ key, name, ...restField }) => (
                                <div
                                  key={key}
                                  data-testid={`replica-gpu-override-${name}`}
                                  style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                                    gap: 12,
                                    position: 'relative',
                                    width: '100%',
                                    paddingRight: 40,
                                  }}
                                >
                                  <Form.Item
                                    {...restField}
                                    name={[name, 'replica_index']}
                                    label={t('create.replicaIndex')}
                                    rules={[{ required: true }]}
                                    style={{ marginBottom: 0 }}
                                  >
                                    <InputNumber
                                      min={0}
                                      max={Math.max(0, replicaCount - 1)}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                  <Form.Item
                                    {...restField}
                                    name={[name, 'gpu_ids']}
                                    label={t('create.replicaGpuIds')}
                                    rules={[{ required: true }]}
                                    style={{ minWidth: 0, marginBottom: 0 }}
                                  >
                                    <GpuSelect mode="multiple" />
                                  </Form.Item>
                                  <Button
                                    danger
                                    type="text"
                                    icon={<DeleteOutlined />}
                                    aria-label={t('create.removeReplicaOverride')}
                                    onClick={() => remove(name)}
                                    style={{ position: 'absolute', right: 0, top: 30 }}
                                  />
                                </div>
                              ))}
                              <Button
                                type="dashed"
                                icon={<PlusOutlined />}
                                onClick={() => add()}
                                block
                              >
                                {t('create.addReplicaOverride')}
                              </Button>
                            </Space>
                          )}
                        </Form.List>
                      </>
                    ),
                  },
                ]}
              />
            )}

            {supportsManagedLaunch ? (
              <Alert
                type={topologyPreview.hasGpuReuse && !allowGpuReuse ? 'error' : 'info'}
                showIcon
                message={t('create.topologyPreview', {
                  replicas: replicaCount,
                  gpus: topologyPreview.requiredPerReplica,
                })}
                description={
                  <Space direction="vertical" size={2}>
                    {topologyPreview.replicas.map((item) => (
                      <span key={item.replicaIndex}>
                        {t('create.topologyReplica', {
                          index: item.replicaIndex,
                          gpus:
                            item.gpuIds.length > 0
                              ? item.gpuIds.join(', ')
                              : t('create.topologyAutomatic'),
                          mode: item.manual
                            ? t('create.topologyManual')
                            : t('create.topologyPool'),
                        })}
                      </span>
                    ))}
                  </Space>
                }
                style={{ marginTop: 16, marginBottom: 16 }}
              />
            ) : null}
          </>
        ) : (
          <>
            {/* Bind Existing Model Form */}
            <Alert
              type="info"
              showIcon
              message={t('create.bindAlert')}
              description={t('create.bindAlertDesc')}
              style={{ marginBottom: 16 }}
            />

            <Form.Item
              name="bind_framework"
              label={t('create.bindFramework')}
              rules={[{ required: deployMode === 'bind', message: t('create.bindFrameworkRequired') }]}
            >
              <Select
                options={INFERENCE_FRAMEWORKS.map((f) => ({
                  label: f.label,
                  value: f.value,
                }))}
              />
            </Form.Item>

            <Form.Item
              name="bind_endpoint"
              label={t('create.bindEndpoint')}
              rules={[{ required: deployMode === 'bind', message: t('create.bindEndpointRequired') }]}
              extra={t('create.bindEndpointHint')}
            >
              <Input placeholder="http://localhost:9997" />
            </Form.Item>

            <Form.Item
              name="bind_container_name"
              label={t('create.bindContainerName')}
              extra={t('create.bindContainerNameHint')}
            >
              <Input placeholder={t('create.bindContainerNamePlaceholder')} />
            </Form.Item>

            <Form.Item
              name="bind_gpu_id"
              label={t('create.bindGpu')}
              extra={t('create.bindGpuHint')}
            >
              <GpuSelect mode="single" allowClear />
            </Form.Item>

            <Form.Item label={t('create.discoverModelsLabel')}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button
                  icon={<SearchOutlined />}
                  onClick={handleDiscoverModels}
                  loading={discovering}
                >
                  {t('create.discoverModels')}
                </Button>

                {discovering && (
                  <div style={{ textAlign: 'center', padding: '16px 0' }}>
                    <Spin tip={t('create.discovering')} />
                  </div>
                )}

                {discoveredModels.length > 0 && (
                  <Collapse
                    accordion
                    activeKey={selectedDiscoveredModel?.model_uid}
                    onChange={handleSelectDiscoveredModel}
                    items={discoveredModels.map((m) => ({
                      key: m.model_uid,
                      label: (
                        <span>
                          <strong>{m.model_name || m.model_uid}</strong>
                          {m.model_type && <span style={{ color: '#888', marginLeft: 8 }}>({m.model_type})</span>}
                        </span>
                      ),
                      children: (
                        <div>
                          <div>UID: {m.model_uid}</div>
                          {m.model_type && <div>{t('create.discoveredType', { type: m.model_type })}</div>}
                          {m.model_path && <div>{t('create.discoveredPath', { path: m.model_path })}</div>}
                        </div>
                      ),
                    }))}
                  />
                )}

                {selectedDiscoveredModel && (
                  <Alert
                    type="success"
                    message={t('create.selectedModel', { name: selectedDiscoveredModel.model_name || selectedDiscoveredModel.model_uid })}
                    description={
                      <div>
                        <div>UID: {selectedDiscoveredModel.model_uid}</div>
                        {selectedDiscoveredModel.model_type && <div>{t('create.discoveredType', { type: selectedDiscoveredModel.model_type })}</div>}
                        {selectedDiscoveredModel.model_path && <div>{t('create.discoveredPath', { path: selectedDiscoveredModel.model_path })}</div>}
                      </div>
                    }
                  />
                )}
              </Space>
            </Form.Item>
          </>
        )}

      </Form>
    </Modal>
  )
}
