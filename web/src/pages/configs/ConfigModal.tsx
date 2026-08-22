import { useEffect } from 'react'
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  Space,
} from 'antd'
import { useTranslation } from 'react-i18next'
import type { ModelConfig, CreateModelConfigRequest } from '@/types'

const modelTypeOptions = [
  { label: 'Embedding', value: 'embedding' },
  { label: 'Reranker', value: 'reranker' },
  { label: 'Decoder Reranker', value: 'decoder_reranker' },
  { label: 'LLM', value: 'llm' },
]

interface ConfigModalProps {
  visible: boolean
  editingConfig: ModelConfig | null
  saving: boolean
  onCancel: () => void
  onSave: (values: CreateModelConfigRequest) => void
}

export function ConfigModal({
  visible,
  editingConfig,
  saving,
  onCancel,
  onSave,
}: ConfigModalProps) {
  const { t } = useTranslation(['configs', 'common'])
  const [form] = Form.useForm()

  const providerOptions = [
    { label: 'OpenAI', value: 'openai' },
    { label: 'Anthropic', value: 'anthropic' },
    { label: 'Azure OpenAI', value: 'azure' },
    { label: 'Hugging Face', value: 'huggingface' },
    { label: 'Xinference', value: 'xinference' },
    { label: 'Ollama', value: 'ollama' },
    { label: t('modal.providerLocal'), value: 'local' },
    { label: 'Custom', value: 'custom' },
  ]

  useEffect(() => {
    if (visible) {
      if (editingConfig) {
        // 编辑时不预填 api_key：后端 GET 返回的是脱敏串（sk-****abcd），
        // 灌入表单会在保存时把真实密钥覆盖成掩码占位符（数据不可逆）。
        // 留空表示“不修改密钥”。
        const rest = { ...editingConfig }
        delete rest.api_key
        form.setFieldsValue(rest)
        form.setFieldValue('api_key', '')
      } else {
        form.resetFields()
      }
    }
  }, [visible, editingConfig, form])

  const handleFinish = (values: CreateModelConfigRequest) => {
    const payload = { ...values }
    if (!payload.api_key) {
      // 编辑时未重新输入密钥：不提交 api_key（后端仅在非 None 时覆盖）
      delete payload.api_key
    }
    onSave(payload)
  }

  return (
    <Modal
      title={editingConfig ? t('modal.editTitle') : t('modal.addTitle')}
      open={visible}
      onCancel={onCancel}
      footer={null}
      destroyOnClose
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleFinish}
        initialValues={{ model_type: 'embedding' }}
      >
        <Form.Item
          name="config_name"
          label={t('modal.configName')}
          rules={[{ required: true, message: t('modal.configNameRequired') }]}
        >
          <Input placeholder={t('modal.configNamePlaceholder')} />
        </Form.Item>

        <Form.Item
          name="model_type"
          label={t('modal.modelType')}
          rules={[{ required: true, message: t('modal.modelTypeRequired') }]}
        >
          <Select options={modelTypeOptions} placeholder={t('modal.modelTypePlaceholder')} />
        </Form.Item>

        <Form.Item
          name="provider"
          label={t('modal.provider')}
          rules={[{ required: true, message: t('modal.providerRequired') }]}
        >
          <Select options={providerOptions} placeholder={t('modal.providerPlaceholder')} />
        </Form.Item>

        <Form.Item
          name="api_endpoint"
          label={t('modal.apiEndpoint')}
          rules={[{ required: true, message: t('modal.apiEndpointRequired') }]}
        >
          <Input placeholder={t('modal.apiEndpointPlaceholder')} />
        </Form.Item>

        <Form.Item
          name="api_key"
          label={t('modal.apiKey')}
          extra={t('modal.apiKeyExtra')}
        >
          <Input.Password placeholder={t('modal.apiKeyPlaceholder')} />
        </Form.Item>

        <Form.Item name="model_name" label={t('modal.modelName')}>
          <Input placeholder={t('modal.modelNamePlaceholder')} />
        </Form.Item>

        <Form.Item name="description" label={t('modal.description')}>
          <Input.TextArea placeholder={t('modal.descriptionPlaceholder')} rows={2} />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button onClick={onCancel}>{t('common:action.cancel')}</Button>
            <Button type="primary" htmlType="submit" loading={saving} disabled={saving}>
              {t('common:action.save')}
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </Modal>
  )
}

