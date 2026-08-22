import { useState } from 'react'
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
import { useTranslation } from 'react-i18next'
import { datasetApi } from '@/services/api'

interface RegisterDatasetModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
}

const fileFormatOptions = [
  { label: 'Parquet', value: 'parquet' },
  { label: 'JSONL', value: 'jsonl' },
  { label: 'CSV', value: 'csv' },
  { label: 'Arrow', value: 'arrow' },
]

interface RegisterFormValues {
  dataset_name: string
  storage_path: string
  dataset_type: string
  usage: string
  file_format: string
  display_name?: string
  description?: string
  content_field?: string
}

export function RegisterDatasetModal({ open, onCancel, onSuccess }: RegisterDatasetModalProps) {
  const { t } = useTranslation(['datasets', 'common'])
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [usage, setUsage] = useState<string | undefined>(undefined)

  const datasetTypeOptions = [
    { label: t('options.downloadDatasetType.embeddingPair'), value: 'embedding_pair' },
    { label: t('options.downloadDatasetType.embeddingTriplet'), value: 'embedding_triplet' },
    { label: t('options.downloadDatasetType.rerankPair'), value: 'rerank_pair' },
    { label: t('options.downloadDatasetType.rerankListwise'), value: 'rerank_listwise' },
    { label: t('options.downloadDatasetType.sftInstruct'), value: 'sft_instruct' },
    { label: t('options.downloadDatasetType.dpoPreference'), value: 'dpo_preference' },
    { label: t('options.downloadDatasetType.rlReward'), value: 'rl_reward' },
    { label: t('options.downloadDatasetType.custom'), value: 'custom' },
  ]

  const usageOptions = [
    { label: t('options.usageDataset.raw'), value: 'raw' },
    { label: t('options.usageDataset.train'), value: 'train' },
    { label: t('options.usageDataset.eval'), value: 'eval' },
    { label: t('options.usageDataset.test'), value: 'test' },
  ]

  const handleSubmit = async (values: RegisterFormValues) => {
    setLoading(true)
    try {
      const resolvedDatasetType = values.dataset_type
      const resolvedUsage = values.usage

      const extra_metadata: Record<string, string> = {}
      if (resolvedUsage === 'raw' && values.content_field) {
        extra_metadata.content_field = values.content_field
      }

      await datasetApi.create({
        dataset_name: values.dataset_name,
        storage_path: values.storage_path,
        dataset_type: resolvedDatasetType,
        usage: resolvedUsage as 'raw' | 'train' | 'eval' | 'test' | undefined,
        file_format: values.file_format,
        display_name: values.display_name,
        description: values.description,
        source_type: 'local',
        extra_metadata: Object.keys(extra_metadata).length > 0 ? extra_metadata : undefined,
      })

      message.success(t('register.message.registerSuccess'))
      form.resetFields()
      setUsage(undefined)
      onSuccess()
    } catch (err) {
      message.error(err instanceof Error ? err.message : t('register.message.registerFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    form.resetFields()
    setUsage(undefined)
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
            {t('common:action.register')}
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
          dataset_type: 'custom',
          usage: 'train',
          file_format: 'jsonl',
        }}
      >
        <Form.Item
          name="dataset_name"
          label={t('register.form.datasetName.label')}
          rules={[{ required: true, message: t('register.form.datasetName.required') }]}
        >
          <Input placeholder={t('register.form.datasetName.placeholder')} />
        </Form.Item>

        <Form.Item
          name="storage_path"
          label={t('register.form.storagePath.label')}
          rules={[{ required: true, message: t('register.form.storagePath.required') }]}
          extra={t('register.form.storagePath.extra')}
        >
          <Input placeholder={t('register.form.storagePath.placeholder')} />
        </Form.Item>

        <Form.Item
          name="usage"
          label={t('register.form.usage.label')}
        >
          <Select options={usageOptions} allowClear placeholder={t('options.optional')} onChange={setUsage} />
        </Form.Item>

        <Form.Item
          name="dataset_type"
          label={t('register.form.datasetType.label')}
          rules={[{ required: true, message: t('register.form.datasetType.required') }]}
        >
          <Select options={datasetTypeOptions} />
        </Form.Item>

        {usage === 'raw' && (
          <Form.Item
            name="content_field"
            label={t('register.form.contentField.label')}
            extra={t('register.form.contentField.extra')}
          >
            <Input placeholder={t('register.form.contentField.placeholder')} />
          </Form.Item>
        )}

        <Form.Item
          name="file_format"
          label={t('register.form.fileFormat.label')}
          rules={[{ required: true, message: t('register.form.fileFormat.required') }]}
        >
          <Select options={fileFormatOptions} />
        </Form.Item>

        <Form.Item name="display_name" label={t('register.form.displayName.label')}>
          <Input placeholder={t('register.form.displayName.placeholder')} />
        </Form.Item>

        <Form.Item name="description" label={t('register.form.description.label')}>
          <Input.TextArea rows={2} placeholder={t('register.form.description.placeholder')} />
        </Form.Item>
      </Form>
    </Modal>
  )
}
