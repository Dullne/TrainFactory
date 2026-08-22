import { useEffect, useState } from 'react'
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
import { SearchOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { GpuSelect } from '@/components/GpuSelect'
import { deploymentApi, DiscoveredModel, externalApiConfigApi } from '@/services/api'
import type { RegisteredModel, CreateDeploymentRequest, ExternalApiConfig } from '@/types'

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
      const { dtype, enforce_eager, attention_backend, ...rest } = values as Record<string, unknown>
      delete rest.bind_endpoint
      delete rest.bind_framework
      delete rest.bind_container_name
      delete rest.bind_gpu_id
      const config: Record<string, unknown> = {}
      if (dtype) config.dtype = dtype
      if (enforce_eager) config.enforce_eager = true
      if (attention_backend) config.attention_backend = attention_backend
      const useContainer = framework !== 'xinference' || containerMode === 'container'
      const request = {
        ...rest,
        ...(Object.keys(config).length > 0 ? { config } : {}),
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
    // Reset LoRA settings when switching to Xinference
    if (value === 'xinference') {
      setEnableLora(false)
      form.setFieldsValue({ enable_lora: false })
      setContainerMode('shared')
    } else {
      setContainerMode('container')
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

  const supportsLora = framework === 'vllm' || framework === 'sglang'

  return (
    <Modal
      title={t('create.title')}
      open={visible}
      onCancel={handleCancel}
      footer={null}
      destroyOnHidden
      width={640}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        initialValues={{
          replica: 1,
          inference_framework: 'xinference',
          bind_framework: 'xinference',
          enable_lora: false,
          max_loras: 4,
          max_lora_rank: 64,
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

            <Form.Item name="gpu_id" label={t('create.gpuSelect')} extra={t('create.gpuSelectHint')}>
              <GpuSelect mode="single" />
            </Form.Item>

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
                ghost
                items={[
                  {
                    key: 'advanced_settings',
                    label: t('create.inferenceAdvanced'),
                    children: (
                      <>
                        <Form.Item
                          name="dtype"
                          label={t('create.dtype')}
                          extra={t('create.dtypeHint')}
                        >
                          <Select
                            allowClear
                            placeholder={t('create.dtypePlaceholder')}
                            options={[
                              { label: t('create.dtypeBfloat16'), value: 'bfloat16' },
                              { label: 'float16', value: 'float16' },
                              { label: 'float32', value: 'float32' },
                              { label: 'auto', value: 'auto' },
                            ]}
                          />
                        </Form.Item>

                        {framework === 'vllm' && (
                          <Form.Item
                            name="enforce_eager"
                            label={t('create.enforceEager')}
                            valuePropName="checked"
                            extra={t('create.enforceEagerHint')}
                          >
                            <Switch />
                          </Form.Item>
                        )}

                        {framework === 'sglang' && (
                          <Form.Item
                            name="attention_backend"
                            label={t('create.attentionBackend')}
                            extra={t('create.attentionBackendHint')}
                          >
                            <Select
                              allowClear
                              placeholder={t('create.attentionBackendPlaceholder')}
                              options={[
                                { label: t('create.attentionFlashinfer'), value: 'flashinfer' },
                                { label: t('create.attentionTorchNative'), value: 'torch_native' },
                                { label: t('create.attentionTriton'), value: 'triton' },
                                { label: t('create.attentionFa3'), value: 'fa3' },
                              ]}
                            />
                          </Form.Item>
                        )}
                      </>
                    ),
                  },
                ]}
              />
            )}
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

        <Form.Item>
          <Space>
            <Button onClick={handleCancel}>{t('common:action.cancel')}</Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={creating}
              disabled={creating || (deployMode === 'bind' && !selectedDiscoveredModel)}
            >
              {deployMode === 'new' ? t('create.submitNew') : t('create.submitBind')}
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  )
}
