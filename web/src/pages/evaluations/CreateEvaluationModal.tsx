import { useState, useEffect } from 'react'
import {
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Button,
  Space,
  Collapse,
  Tag,
  message,
  Card,
  Typography,
} from 'antd'
import { PlusOutlined, DeleteOutlined, ApiOutlined, DatabaseOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { useAuth } from '@/auth/AuthContext'
import type { Deployment, CreateEvaluationRequest, AvailableDatasets, Dataset } from '@/types'
import { evaluationApi, deploymentApi, datasetApi } from '@/services/api'
import { BG_ELEVATED, BORDER_SECONDARY, TEXT_SECONDARY, STATUS_INFO } from '@/theme'

const { Text } = Typography

interface ModelConfigItem {
  key: string
  source: 'deployment' | 'custom'
  deployment_id?: string
  deployment_replica_id?: string
  endpoint: string
  model_name?: string
  name: string
  inference_framework?: string
}

interface DatasetConfigItem {
  key: string
  type: 'mteb' | 'local' | 'registered'
  name: string
  path?: string
  dataset_id?: string
}

interface CreateEvaluationModalProps {
  visible: boolean
  onCancel: () => void
  onSuccess: () => void
}

const getHealthyEvaluationReplicas = (deployment: Deployment) =>
  (deployment.replica_instances ?? []).filter(
    (replica) =>
      replica.deployment_id === deployment.deployment_id &&
      replica.status === 'running' &&
      replica.health_status === 'HEALTHY'
  )

const hasReplicaLifecycleMarker = (deployment: Deployment) => {
  const version = deployment.config?.replica_schema_version
  return (
    (deployment.inference_framework === 'vllm' || deployment.inference_framework === 'sglang') &&
    typeof version === 'number' &&
    Number.isInteger(version) &&
    version === 1
  )
}

const usesReplicaLifecycle = (deployment: Deployment) =>
  hasReplicaLifecycleMarker(deployment) || (deployment.replica_instances?.length ?? 0) > 0

const isEvaluationDeploymentEligible = (deployment: Deployment) => {
  if (deployment.status !== 'running' && deployment.status !== 'degraded') return false
  if (!usesReplicaLifecycle(deployment)) return deployment.status === 'running'
  return getHealthyEvaluationReplicas(deployment).length > 0
}

export function CreateEvaluationModal({
  visible,
  onCancel,
  onSuccess,
}: CreateEvaluationModalProps) {
  const { t } = useTranslation(['evaluations', 'common'])
  const { config: authConfig } = useAuth()
  const [form] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [availableDatasets, setAvailableDatasets] = useState<AvailableDatasets | null>(null)
  const [registeredDatasets, setRegisteredDatasets] = useState<Dataset[]>([])
  const [modelConfigs, setModelConfigs] = useState<ModelConfigItem[]>([])
  const [datasetConfigs, setDatasetConfigs] = useState<DatasetConfigItem[]>([])
  const [loading, setLoading] = useState(false)
  const [deploymentListLoaded, setDeploymentListLoaded] = useState(false)

  // Fetch deployments, available datasets, and registered eval datasets
  useEffect(() => {
    setDeployments([])
    setDeploymentListLoaded(false)
    if (!visible) {
      setLoading(false)
      return
    }

    let ignore = false
    setLoading(true)
    Promise.all([
      deploymentApi.list({ page_size: 100 }),
      evaluationApi.getAvailableDatasets(),
      datasetApi.list({ usage: 'eval', page_size: 100 }),
    ])
      .then(([deployRes, datasetsRes, registeredRes]) => {
        if (ignore) return
        setDeployments(deployRes.items.filter(isEvaluationDeploymentEligible))
        setDeploymentListLoaded(true)
        setAvailableDatasets(datasetsRes)
        setRegisteredDatasets(registeredRes.items || [])
      })
      .catch(() => {
        if (!ignore) message.error(t('create.loadDataFailed'))
      })
      .finally(() => {
        if (!ignore) setLoading(false)
      })

    return () => {
      ignore = true
    }
  }, [visible, t])

  const handleAddModel = () => {
    setModelConfigs([
      ...modelConfigs,
      {
        key: Date.now().toString(),
        source: 'deployment',
        endpoint: '',
        name: '',
      },
    ])
  }

  const handleRemoveModel = (key: string) => {
    setModelConfigs(modelConfigs.filter((m) => m.key !== key))
  }

  const handleModelChange = (key: string, field: keyof ModelConfigItem, value: unknown) => {
    setModelConfigs((current) =>
      current.map((m) => {
        if (m.key === key) {
          if (field === 'source') {
            return {
              key: m.key,
              source: value as ModelConfigItem['source'],
              endpoint: '',
              name: '',
            }
          }
          const updated = { ...m, [field]: value }
          // Auto-fill from deployment
          if (field === 'deployment_id') {
            updated.deployment_replica_id = undefined
            updated.endpoint = ''
            updated.model_name = undefined
            updated.inference_framework = undefined
            const deployment = deployments.find((d) => d.deployment_id === value)
            if (deployment) {
              const healthyReplicas = getHealthyEvaluationReplicas(deployment)
              const selectedReplica = healthyReplicas.length === 1 ? healthyReplicas[0] : undefined
              updated.deployment_replica_id = selectedReplica?.replica_id
              updated.endpoint = selectedReplica?.endpoint ?? (
                !usesReplicaLifecycle(deployment) ? deployment.xinference_endpoint : ''
              )
              updated.name = deployment.deployment_name || deployment.model_uid || ''
              updated.model_name = deployment.model_uid || 'reranker'
              updated.inference_framework = deployment.inference_framework
            }
          }
          if (field === 'deployment_replica_id') {
            const deployment = deployments.find(
              (item) => item.deployment_id === updated.deployment_id,
            )
            const replica = deployment && getHealthyEvaluationReplicas(deployment).find(
              (item) => item.replica_id === value,
            )
            updated.deployment_replica_id = replica?.replica_id
            if (replica) {
              updated.endpoint = replica.endpoint
            } else if (deployment && usesReplicaLifecycle(deployment)) {
              updated.endpoint = ''
            }
          }
          return updated
        }
        return m
      })
    )
  }

  const handleAddDataset = (type: 'mteb' | 'local' | 'registered') => {
    setDatasetConfigs([
      ...datasetConfigs,
      {
        key: Date.now().toString(),
        type,
        name: '',
        path: type === 'local' ? '' : undefined,
      },
    ])
  }

  // Handle selecting a registered dataset
  const handleSelectRegisteredDataset = (key: string, datasetId: string) => {
    const dataset = registeredDatasets.find((d) => d.dataset_id === datasetId)
    if (dataset) {
      setDatasetConfigs(
        datasetConfigs.map((dc) =>
          dc.key === key
            ? {
                ...dc,
                name: dataset.dataset_name || dataset.name || dataset.display_name || '',
                dataset_id: dataset.dataset_id,
                path: undefined,
              }
            : dc
        )
      )
    }
  }

  const handleRemoveDataset = (key: string) => {
    setDatasetConfigs(datasetConfigs.filter((d) => d.key !== key))
  }

  const handleDatasetChange = (key: string, field: keyof DatasetConfigItem, value: string) => {
    setDatasetConfigs(
      datasetConfigs.map((d) => {
        if (d.key === key) {
          return { ...d, [field]: value }
        }
        return d
      })
    )
  }

  const handleQuickSelectGroup = (group: string) => {
    if (!availableDatasets?.groups[group]) return
    const existingNames = new Set(datasetConfigs.map((d) => d.name))
    const newDatasets = availableDatasets.groups[group]
      .filter((name) => !existingNames.has(name))
      .map((name) => ({
        key: `${Date.now()}-${name}`,
        type: 'mteb' as const,
        name,
      }))
    setDatasetConfigs([...datasetConfigs, ...newDatasets])
  }

  const handleFinish = async (values: {
    task_name?: string
    max_samples?: number
    batch_size?: number
    workers?: number
    model_workers?: number
  }) => {
    if (modelConfigs.length === 0) {
      message.error(t('create.atLeastOneModel'))
      return
    }
    if (datasetConfigs.length === 0) {
      message.error(t('create.atLeastOneDataset'))
      return
    }

    // Validate model configs and rebuild deployment bindings from trusted list data.
    const trustedModelConfigs: CreateEvaluationRequest['model_configs'] = []
    for (const mc of modelConfigs) {
      if (mc.source === 'custom') {
        if (!mc.endpoint || !mc.name) {
          message.error(t('create.completeModelConfig'))
          return
        }
        trustedModelConfigs.push({ endpoint: mc.endpoint, name: mc.name })
        continue
      }

      if (!deploymentListLoaded) {
        message.error(t('create.completeModelConfig'))
        return
      }

      const deployment = deployments.find(
        (item) => item.deployment_id === mc.deployment_id
      )
      if (!deployment) {
        message.error(t('create.completeModelConfig'))
        return
      }

      const replica = getHealthyEvaluationReplicas(deployment).find(
        (item) => item.replica_id === mc.deployment_replica_id
      )
      if (usesReplicaLifecycle(deployment) && !replica) {
        message.error(t('create.completeModelConfig'))
        return
      }

      const endpoint = replica?.endpoint ?? deployment.xinference_endpoint
      const name = deployment.deployment_name || deployment.model_uid || ''
      if (!endpoint || !name) {
        message.error(t('create.completeModelConfig'))
        return
      }
      trustedModelConfigs.push({
        endpoint,
        name,
        deployment_id: deployment.deployment_id,
        ...(replica ? { deployment_replica_id: replica.replica_id } : {}),
        model_name: deployment.model_uid || 'reranker',
        inference_framework: deployment.inference_framework,
      })
    }

    // Check for duplicate model names
    const modelNames = trustedModelConfigs.map((mc) => mc.name)
    const uniqueNames = new Set(modelNames)
    if (uniqueNames.size < modelNames.length) {
      message.error(t('create.duplicateModelName'))
      return
    }

    // Validate dataset configs
    for (const dc of datasetConfigs) {
      if (
        !dc.name ||
        (dc.type === 'local' && !dc.path) ||
        (dc.type === 'registered' && !dc.dataset_id)
      ) {
        message.error(t('create.completeDatasetConfig'))
        return
      }
    }

    const request: CreateEvaluationRequest = {
      task_name: values.task_name,
      model_configs: trustedModelConfigs,
      dataset_configs: datasetConfigs.map((d) => ({
        type: d.type,
        name: d.name,
        ...(d.path ? { path: d.path } : {}),
        ...(d.dataset_id ? { dataset_id: d.dataset_id } : {}),
      })),
      max_samples: values.max_samples,
      batch_size: values.batch_size,
      workers: values.workers,
      model_workers: values.model_workers,
    }

    setCreating(true)
    try {
      await evaluationApi.createTask(request)
      message.success(t('create.taskCreated'))
      handleCancel()
      onSuccess()
    } catch (err) {
      message.error(t('create.createTaskFailed', { error: (err as Error).message }))
    } finally {
      setCreating(false)
    }
  }

  const handleCancel = () => {
    form.resetFields()
    setModelConfigs([])
    setDatasetConfigs([])
    onCancel()
  }

  const mtebDatasetOptions = availableDatasets?.datasets
    ? Object.entries(availableDatasets.datasets).map(([name, info]) => ({
        label: `${name} (${info.lang}) - ${info.description}`,
        value: name,
      }))
    : []

  return (
    <Modal
      title={t('create.title')}
      open={visible}
      onCancel={handleCancel}
      onOk={() => form.submit()}
      okText={t('create.createEvalTask')}
      cancelText={t('common:action.cancel')}
      confirmLoading={creating}
      okButtonProps={{ disabled: creating }}
      destroyOnClose
      width={800}
      styles={{ body: { maxHeight: 'calc(100vh - 220px)', overflowY: 'auto' } }}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        scrollToFirstError
        initialValues={{
          batch_size: 50,
          workers: 8,
          model_workers: 2,
        }}
      >
        <Form.Item name="task_name" label={t('create.taskName')}>
          <Input placeholder={t('create.taskNamePlaceholder')} />
        </Form.Item>

        {/* Model Configuration Section */}
        <Card
          size="small"
          title={
            <Space>
              <ApiOutlined />
              <span>{t('create.modelConfig')}</span>
              <Tag color={STATUS_INFO}>{modelConfigs.length}</Tag>
            </Space>
          }
          extra={
            <Button type="dashed" size="small" icon={<PlusOutlined />} onClick={handleAddModel}>
              {t('create.addModel')}
            </Button>
          }
          style={{ marginBottom: 16, background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}
        >
          {modelConfigs.length === 0 ? (
            <div style={{ textAlign: 'center', color: TEXT_SECONDARY, padding: 20 }}>
              {t('create.addModelHint')}
            </div>
          ) : (
            <Space direction="vertical" style={{ width: '100%' }}>
              {modelConfigs.map((mc) => (
                <Card
                  key={mc.key}
                  size="small"
                  style={{ background: 'transparent' }}
                  extra={
                    <Button
                      type="text"
                      danger
                      size="small"
                      icon={<DeleteOutlined />}
                      onClick={() => handleRemoveModel(mc.key)}
                    />
                  }
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Space>
                      <Text>{t('create.source')}</Text>
                      <Select
                        size="small"
                        value={mc.source}
                        style={{ width: 150 }}
                        onChange={(v) => handleModelChange(mc.key, 'source', v)}
                        options={[
                          { label: t('create.sourceDeployment'), value: 'deployment' },
                          { label: t('create.sourceCustom'), value: 'custom' },
                        ]}
                      />
                    </Space>

                    {mc.source === 'deployment' ? (
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Select
                          placeholder={t('create.selectDeployment')}
                          value={mc.deployment_id}
                          onChange={(v) => handleModelChange(mc.key, 'deployment_id', v)}
                          style={{ width: '100%' }}
                          loading={loading}
                          disabled={!deploymentListLoaded}
                          options={deployments.map((d) => ({
                            label: `${d.deployment_name || d.model_uid} (${d.replica_instances?.length || 1} ${t('create.replicas')})`,
                            value: d.deployment_id,
                          }))}
                        />
                        {(() => {
                          const deployment = deployments.find(
                            (item) => item.deployment_id === mc.deployment_id,
                          )
                          const replicas = deployment?.replica_instances ?? []
                          if (replicas.length === 0) return null
                          const healthyReplicaIds = new Set(
                            deployment
                              ? getHealthyEvaluationReplicas(deployment).map(
                                  (replica) => replica.replica_id
                                )
                              : []
                          )
                          return (
                            <Select
                              placeholder={t('create.selectReplica')}
                              value={mc.deployment_replica_id}
                              onChange={(value) =>
                                handleModelChange(
                                  mc.key,
                                  'deployment_replica_id',
                                  value,
                                )
                              }
                              options={replicas.map((replica) => ({
                                label: `#${replica.replica_index} · ${replica.endpoint} · GPU ${replica.gpu_ids.join(', ')}`,
                                value: replica.replica_id,
                                disabled: !healthyReplicaIds.has(replica.replica_id),
                              }))}
                            />
                          )
                        })()}
                      </Space>
                    ) : (
                      <Space.Compact style={{ width: '100%' }}>
                        <Input
                          placeholder={t('create.endpointPlaceholder')}
                          value={mc.endpoint}
                          onChange={(e) => handleModelChange(mc.key, 'endpoint', e.target.value)}
                          style={{ flex: 1 }}
                        />
                        <Input
                          placeholder={t('create.displayNamePlaceholder')}
                          value={mc.name}
                          onChange={(e) => handleModelChange(mc.key, 'name', e.target.value)}
                          style={{ width: 150 }}
                        />
                      </Space.Compact>
                    )}
                  </Space>
                </Card>
              ))}
            </Space>
          )}
        </Card>

        {/* Dataset Configuration Section */}
        <Card
          size="small"
          title={
            <Space>
              <DatabaseOutlined />
              <span>{t('create.datasetConfig')}</span>
              <Tag color={STATUS_INFO}>{datasetConfigs.length}</Tag>
            </Space>
          }
          extra={
            <Space>
              <Button
                type="dashed"
                size="small"
                icon={<PlusOutlined />}
                onClick={() => handleAddDataset('registered')}
              >
                {t('create.registered')}
              </Button>
              <Button
                type="dashed"
                size="small"
                icon={<PlusOutlined />}
                onClick={() => handleAddDataset('mteb')}
              >
                MTEB
              </Button>
              {authConfig.direct_storage_registration_enabled ? (
                <Button
                  type="dashed"
                  size="small"
                  icon={<PlusOutlined />}
                  onClick={() => handleAddDataset('local')}
                >
                  {t('create.local')}
                </Button>
              ) : null}
            </Space>
          }
          style={{ marginBottom: 16, background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}
        >
          {/* Quick Select Groups */}
          {availableDatasets?.groups && (
            <div style={{ marginBottom: 12 }}>
              <Text type="secondary">{t('create.quickSelect')}</Text>
              <Space size={4}>
                {Object.keys(availableDatasets.groups).map((group) => (
                  <Tag
                    key={group}
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleQuickSelectGroup(group)}
                  >
                    {group}
                  </Tag>
                ))}
              </Space>
            </div>
          )}

          {datasetConfigs.length === 0 ? (
            <div style={{ textAlign: 'center', color: TEXT_SECONDARY, padding: 20 }}>
              {t('create.addDatasetHint')}
            </div>
          ) : (
            <Space direction="vertical" style={{ width: '100%' }}>
              {datasetConfigs.map((dc) => (
                <Space key={dc.key} style={{ width: '100%' }}>
                  <Tag
                    color={
                      dc.type === 'mteb' ? 'blue' : dc.type === 'registered' ? 'purple' : 'green'
                    }
                  >
                    {dc.type === 'registered' ? t('create.registered') : dc.type.toUpperCase()}
                  </Tag>
                  {dc.type === 'mteb' ? (
                    <Select
                      placeholder={t('create.selectMtebDataset')}
                      value={dc.name || undefined}
                      onChange={(v) => handleDatasetChange(dc.key, 'name', v)}
                      style={{ width: 400 }}
                      showSearch
                      optionFilterProp="label"
                      options={mtebDatasetOptions}
                    />
                  ) : dc.type === 'registered' ? (
                    <Select
                      placeholder={t('create.selectRegisteredDataset')}
                      value={dc.dataset_id}
                      onChange={(v) => handleSelectRegisteredDataset(dc.key, v)}
                      style={{ width: 400 }}
                      showSearch
                      optionFilterProp="label"
                      loading={loading}
                      options={registeredDatasets.map((d) => ({
                        label: `${d.dataset_name || d.name || d.display_name}${d.model_type?.length ? ` (${d.model_type.join(', ')})` : ''}`,
                        value: d.dataset_id,
                      }))}
                      notFoundContent={
                        registeredDatasets.length === 0
                          ? t('create.noRegisteredDatasets')
                          : undefined
                      }
                    />
                  ) : (
                    <Space.Compact style={{ width: 400 }}>
                      <Input
                        placeholder={t('create.datasetNamePlaceholder')}
                        value={dc.name}
                        onChange={(e) => handleDatasetChange(dc.key, 'name', e.target.value)}
                        style={{ width: 150 }}
                      />
                      <Input
                        placeholder={t('create.datasetPathPlaceholder')}
                        value={dc.path}
                        onChange={(e) => handleDatasetChange(dc.key, 'path', e.target.value)}
                        style={{ flex: 1 }}
                      />
                    </Space.Compact>
                  )}
                  <Button
                    type="text"
                    danger
                    size="small"
                    icon={<DeleteOutlined />}
                    onClick={() => handleRemoveDataset(dc.key)}
                  />
                </Space>
              ))}
            </Space>
          )}
        </Card>

        {/* Evaluation Parameters */}
        <Collapse
          ghost
          items={[
            {
              key: 'params',
              label: t('create.evalParams'),
              children: (
                <>
                  <Form.Item
                    name="max_samples"
                    label={t('create.maxSamples')}
                    extra={t('create.maxSamplesExtra')}
                  >
                    <InputNumber
                      min={1}
                      style={{ width: '100%' }}
                      placeholder={t('create.allSamples')}
                    />
                  </Form.Item>

                  <Form.Item
                    name="batch_size"
                    label={t('create.batchSize')}
                    extra={t('create.batchSizeExtra')}
                  >
                    <InputNumber min={1} max={200} style={{ width: '100%' }} />
                  </Form.Item>

                  <Form.Item
                    name="workers"
                    label={t('create.apiWorkers')}
                    extra={t('create.apiWorkersExtra')}
                  >
                    <InputNumber min={1} max={32} style={{ width: '100%' }} />
                  </Form.Item>

                  <Form.Item
                    name="model_workers"
                    label={t('create.modelWorkers')}
                    extra={t('create.modelWorkersExtra')}
                  >
                    <InputNumber min={1} max={8} style={{ width: '100%' }} />
                  </Form.Item>
                </>
              ),
            },
          ]}
        />

      </Form>
    </Modal>
  )
}
