import { useState, useEffect, useCallback } from 'react'
import {
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  message,
} from 'antd'
import { useTranslation } from 'react-i18next'
import { milvusApi } from '@/services/api'
import { configApi } from '@/services/api'
import type { ModelConfig } from '@/types'

interface Props {
  open: boolean
  onClose: () => void
  onSuccess: () => void
}

export default function CreateCollectionModal({ open, onClose, onSuccess }: Props) {
  const { t } = useTranslation(['vectordb', 'common'])
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [embeddingConfigs, setEmbeddingConfigs] = useState<ModelConfig[]>([])

  const metricOptions = [
    { value: 'COSINE', label: t('create.metricCosine') },
    { value: 'L2', label: t('create.metricL2') },
    { value: 'IP', label: t('create.metricIP') },
  ]

  useEffect(() => {
    if (open) {
      configApi.list().then((res) => {
        setEmbeddingConfigs((res.items || []).filter((c) => c.model_type === 'embedding'))
      }).catch(() => {})
    }
  }, [open])

  const handleEmbeddingChange = useCallback((configId: string) => {
    const config = embeddingConfigs.find((c) => c.config_id === configId)
    if (config) {
      // Auto-fill display name with model name info
      const currentName = form.getFieldValue('name')
      if (!currentName) {
        form.setFieldValue('display_name', config.model_name || config.config_name)
      }
    }
  }, [embeddingConfigs, form])

  const handleOk = useCallback(async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)
      const res = await milvusApi.createCollection({
        name: values.name,
        dim: values.dim,
        metric_type: values.metric_type,
        description: values.description || undefined,
        embedding_config_id: values.embedding_config_id || undefined,
        display_name: values.display_name || undefined,
        enable_hybrid: values.enable_hybrid || false,
      })
      message.success(res.message || t('create.createSuccess'))
      form.resetFields()
      onSuccess()
      onClose()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'errorFields' in err) return
      message.error(t('create.createFailed', { error: err instanceof Error ? err.message : String(err) }))
    } finally {
      setLoading(false)
    }
  }, [form, onClose, onSuccess, t])

  return (
    <Modal
      title={t('create.title')}
      open={open}
      onOk={handleOk}
      onCancel={onClose}
      confirmLoading={loading}
      destroyOnHidden
    >
      <Form form={form} layout="vertical" initialValues={{ dim: 1024, metric_type: 'COSINE' }}>
        <Form.Item
          name="name"
          label={t('create.collectionName')}
          rules={[{ required: true, message: t('create.collectionNameRequired') }]}
          extra={t('create.collectionNameExtra')}
        >
          <Input placeholder={t('create.collectionNamePlaceholder')} />
        </Form.Item>
        <Form.Item name="display_name" label={t('create.displayName')}>
          <Input placeholder={t('create.displayNamePlaceholder')} />
        </Form.Item>
        <Form.Item
          name="embedding_config_id"
          label={t('create.embeddingModel')}
          extra={t('create.embeddingModelExtra')}
        >
          <Select
            allowClear
            placeholder={t('create.embeddingModelPlaceholder')}
            onChange={handleEmbeddingChange}
            options={embeddingConfigs.map((c) => ({
              value: c.config_id,
              label: `${c.config_name} (${c.model_name})`,
            }))}
          />
        </Form.Item>
        <Form.Item
          name="dim"
          label={t('create.vectorDimension')}
          rules={[{ required: true, message: t('create.vectorDimensionRequired') }]}
        >
          <InputNumber min={1} max={4096} style={{ width: '100%' }} />
        </Form.Item>
        <Form.Item name="metric_type" label={t('create.distanceMetric')}>
          <Select options={metricOptions} />
        </Form.Item>
        <Form.Item name="description" label={t('create.description')}>
          <Input.TextArea rows={2} placeholder={t('create.optional')} />
        </Form.Item>
        <Form.Item
          name="enable_hybrid"
          label={t('create.enableHybrid')}
          valuePropName="checked"
          extra={t('create.enableHybridExtra')}
        >
          <Switch />
        </Form.Item>
      </Form>
    </Modal>
  )
}
