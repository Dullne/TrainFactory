import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  message,
  Space,
} from 'antd'
import { FolderOpenOutlined } from '@ant-design/icons'
import { modelApi } from '@/services/api'

interface RegisterModelModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
}

interface RegisterFormValues {
  model_name: string
  model_path: string
  model_type: string
  version: string
  base_model_path?: string
  description?: string
}

export function RegisterModelModal({ open, onCancel, onSuccess }: RegisterModelModalProps) {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const { t } = useTranslation(['models', 'common'])

  const modelTypeOptions = [
    { label: t('register.embeddingLabel'), value: 'embedding' },
    { label: t('register.rerankerLabel'), value: 'reranker' },
    { label: t('register.decoderRerankerLabel'), value: 'decoder_reranker' },
    { label: t('register.llmLabel'), value: 'llm' },
  ]

  const handleSubmit = async (values: RegisterFormValues) => {
    setLoading(true)
    try {
      await modelApi.register({
        model_name: values.model_name,
        model_path: values.model_path,
        model_type: values.model_type,
        version: values.version || 'v1.0.0',
        base_model_path: values.base_model_path,
        description: values.description,
      })
      message.success(t('register.registerSuccess'))
      form.resetFields()
      onSuccess()
    } catch (err) {
      message.error(err instanceof Error ? err.message : t('register.registerFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    form.resetFields()
    onCancel()
  }

  return (
    <Modal
      title={
        <Space>
          <FolderOpenOutlined />
          {t('register.title')}
        </Space>
      }
      open={open}
      onCancel={handleClose}
      footer={
        <Space>
          <Button onClick={handleClose}>{t('common:action.cancel')}</Button>
          <Button type="primary" loading={loading} onClick={() => form.submit()}>
            {t('register.registerButton')}
          </Button>
        </Space>
      }
      width={520}
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          model_type: 'embedding',
          version: 'v1.0.0',
        }}
      >
        <Form.Item
          name="model_name"
          label={t('register.modelName')}
          rules={[{ required: true, message: t('register.modelNameRequired') }]}
        >
          <Input placeholder={t('register.modelNamePlaceholder')} />
        </Form.Item>

        <Form.Item
          name="model_path"
          label={t('register.modelPath')}
          rules={[{ required: true, message: t('register.modelPathRequired') }]}
          extra={t('register.modelPathExtra')}
        >
          <Input placeholder={t('register.modelPathPlaceholder')} />
        </Form.Item>

        <Form.Item
          name="model_type"
          label={t('register.modelType')}
          rules={[{ required: true, message: t('register.modelTypeRequired') }]}
        >
          <Select options={modelTypeOptions} />
        </Form.Item>

        <Form.Item name="version" label={t('register.version')}>
          <Input placeholder="v1.0.0" />
        </Form.Item>

        <Form.Item name="base_model_path" label={t('register.baseModel')}>
          <Input placeholder={t('register.baseModelPlaceholder')} />
        </Form.Item>

        <Form.Item name="description" label={t('register.description')}>
          <Input.TextArea rows={2} placeholder={t('register.optionalPlaceholder')} />
        </Form.Item>
      </Form>
    </Modal>
  )
}
