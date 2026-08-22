import { useState, useEffect, useCallback } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Select,
  Input,
  message,
  Popconfirm,
  Empty,
  Alert,
  Spin,
} from 'antd'
import {
  PlusOutlined,
  DeleteOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { adapterApi } from '@/services/api'
import type { LoadedAdapter, AvailableAdapter, Deployment } from '@/types'

interface AdapterManagerProps {
  deployment: Deployment
  onRefresh?: () => void
}

export function AdapterManager({ deployment, onRefresh }: AdapterManagerProps) {
  const { t } = useTranslation(['deployments', 'common'])
  const [adapters, setAdapters] = useState<LoadedAdapter[]>([])
  const [availableAdapters, setAvailableAdapters] = useState<AvailableAdapter[]>([])
  const [loading, setLoading] = useState(false)
  const [loadModalVisible, setLoadModalVisible] = useState(false)
  const [loadingAdapter, setLoadingAdapter] = useState(false)
  const [form] = Form.useForm()

  const supportsLora =
    deployment.inference_framework === 'vllm' ||
    deployment.inference_framework === 'sglang'

  const fetchAdapters = useCallback(async () => {
    if (!supportsLora || deployment.status !== 'running') return

    setLoading(true)
    try {
      const response = await adapterApi.listLoaded(deployment.deployment_id)
      setAdapters(response.adapters || [])
    } catch (error) {
      console.error('Failed to fetch adapters:', error)
    } finally {
      setLoading(false)
    }
  }, [deployment.deployment_id, deployment.status, supportsLora])

  const fetchAvailableAdapters = useCallback(async () => {
    try {
      const response = await adapterApi.listAvailable()
      setAvailableAdapters(response.adapters || [])
    } catch (error) {
      console.error('Failed to fetch available adapters:', error)
    }
  }, [])

  useEffect(() => {
    if (supportsLora && deployment.status === 'running') {
      Promise.all([fetchAdapters(), fetchAvailableAdapters()])
    }
  }, [deployment.deployment_id, deployment.status, supportsLora, fetchAdapters, fetchAvailableAdapters])

  const handleLoadAdapter = async (values: {
    adapter_source: string
    adapter_name?: string
  }) => {
    setLoadingAdapter(true)
    try {
      const selectedAdapter = availableAdapters.find(
        (a) => `${a.source}:${a.source_id}` === values.adapter_source
      )
      if (!selectedAdapter) {
        message.error(t('adapter.message.notFound'))
        return
      }

      await adapterApi.load(deployment.deployment_id, {
        adapter_name: values.adapter_name || selectedAdapter.name,
        adapter_path: selectedAdapter.path,
        source_task_id:
          selectedAdapter.source === 'training_task' ? selectedAdapter.source_id : undefined,
        source_model_id:
          selectedAdapter.source === 'model_registry' ? selectedAdapter.source_id : undefined,
      })

      message.success(t('adapter.message.loadSuccess'))
      setLoadModalVisible(false)
      form.resetFields()
      fetchAdapters()
      onRefresh?.()
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : t('adapter.message.loadFailed')
      message.error(errorMessage)
    } finally {
      setLoadingAdapter(false)
    }
  }

  const handleUnloadAdapter = async (adapterName: string) => {
    try {
      await adapterApi.unload(deployment.deployment_id, adapterName)
      message.success(t('adapter.message.unloadSuccess'))
      fetchAdapters()
      onRefresh?.()
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : t('adapter.message.unloadFailed')
      message.error(errorMessage)
    }
  }

  const handleSync = async () => {
    try {
      await adapterApi.sync(deployment.deployment_id)
      message.success(t('adapter.message.syncSuccess'))
      fetchAdapters()
    } catch (error) {
      message.error(t('adapter.message.syncFailed'))
    }
  }

  const getStatusTag = (status: string) => {
    const statusMap: Record<string, { color: string; key: string }> = {
      loading: { color: 'blue', key: 'adapter.status.loading' },
      loaded: { color: 'green', key: 'adapter.status.loaded' },
      unloading: { color: 'orange', key: 'adapter.status.unloading' },
      unloaded: { color: 'default', key: 'adapter.status.unloaded' },
      failed: { color: 'red', key: 'adapter.status.failed' },
    }
    const config = statusMap[status] || { color: 'default', key: '' }
    return <Tag color={config.color}>{config.key ? t(config.key) : status}</Tag>
  }

  const columns = [
    {
      title: t('adapter.columns.adapterName'),
      dataIndex: 'adapter_name',
      key: 'adapter_name',
    },
    {
      title: t('adapter.columns.status'),
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
    },
    {
      title: t('adapter.columns.source'),
      key: 'source',
      render: (_: unknown, record: LoadedAdapter) => {
        if (record.source_task_id) {
          return <Tag color="blue">{t('adapter.source.trainingTask')}</Tag>
        }
        if (record.source_model_id) {
          return <Tag color="purple">{t('adapter.source.modelRegistry')}</Tag>
        }
        return <Tag>{t('adapter.source.manual')}</Tag>
      },
    },
    {
      title: t('adapter.columns.loadedAt'),
      dataIndex: 'loaded_at',
      key: 'loaded_at',
      render: (time: string) => (time ? new Date(time).toLocaleString() : '-'),
    },
    {
      title: t('adapter.columns.actions'),
      key: 'actions',
      render: (_: unknown, record: LoadedAdapter) => (
        <Space>
          {record.status === 'loaded' && (
            <Popconfirm
              title={t('adapter.confirmUnload')}
              onConfirm={() => handleUnloadAdapter(record.adapter_name)}
            >
              <Button type="text" danger icon={<DeleteOutlined />}>
                {t('adapter.unload')}
              </Button>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ]

  // If deployment doesn't support LoRA
  if (!supportsLora) {
    return (
      <Card title={t('adapter.title')}>
        <Alert
          type="warning"
          showIcon
          icon={<ExclamationCircleOutlined />}
          message={t('adapter.notSupportedMessage')}
          description={t('adapter.notSupportedDesc', { framework: deployment.inference_framework || 'xinference' })}
        />
      </Card>
    )
  }

  // If deployment is not running
  if (deployment.status !== 'running') {
    return (
      <Card title={t('adapter.title')}>
        <Alert
          type="info"
          showIcon
          message={t('adapter.notRunningMessage')}
          description={t('adapter.notRunningDesc')}
        />
      </Card>
    )
  }

  // If LoRA is not enabled
  if (!deployment.enable_lora) {
    return (
      <Card title={t('adapter.title')}>
        <Alert
          type="info"
          showIcon
          message={t('adapter.loraDisabledMessage')}
          description={t('adapter.loraDisabledDesc')}
        />
      </Card>
    )
  }

  return (
    <Card
      title={t('adapter.title')}
      extra={
        <Space>
          <Button icon={<ReloadOutlined />} onClick={handleSync}>
            {t('adapter.syncStatus')}
          </Button>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setLoadModalVisible(true)}
            disabled={adapters.filter((a) => a.status === 'loaded').length >= (deployment.max_loras || 4)}
          >
            {t('adapter.loadAdapter')}
          </Button>
        </Space>
      }
    >
      <Spin spinning={loading}>
        {adapters.length === 0 ? (
          <Empty description={t('adapter.noAdapters')} />
        ) : (
          <Table
            dataSource={adapters}
            columns={columns}
            rowKey="adapter_id"
            pagination={false}
          />
        )}
      </Spin>

      <Modal
        title={t('adapter.loadModal.title')}
        open={loadModalVisible}
        onCancel={() => {
          setLoadModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        destroyOnHidden
      >
        <Form form={form} layout="vertical" onFinish={handleLoadAdapter}>
          <Form.Item
            name="adapter_source"
            label={t('adapter.loadModal.selectAdapter')}
            rules={[{ required: true, message: t('adapter.loadModal.selectAdapterRequired') }]}
          >
            <Select
              placeholder={t('adapter.loadModal.selectAdapterPlaceholder')}
              options={availableAdapters.map((a) => ({
                label: `${a.name} (${a.source === 'training_task' ? t('adapter.source.trainingTask') : t('adapter.source.modelRegistry')})`,
                value: `${a.source}:${a.source_id}`,
              }))}
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
            />
          </Form.Item>

          <Form.Item
            name="adapter_name"
            label={t('adapter.loadModal.adapterName')}
            extra={t('adapter.loadModal.adapterNameHint')}
          >
            <Input placeholder={t('adapter.loadModal.adapterNamePlaceholder')} />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                onClick={() => {
                  setLoadModalVisible(false)
                  form.resetFields()
                }}
              >
                {t('common:action.cancel')}
              </Button>
              <Button type="primary" htmlType="submit" loading={loadingAdapter} disabled={loadingAdapter}>
                {t('adapter.loadModal.load')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}
