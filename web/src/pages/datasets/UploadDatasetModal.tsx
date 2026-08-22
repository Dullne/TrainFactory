import { useCallback, useEffect, useState } from 'react'
import {
  Modal,
  Form,
  Input,
  Select,
  Upload,
  Checkbox,
  Button,
  message,
  Progress,
  Alert,
  Space,
  Typography,
} from 'antd'
import {
  CloudUploadOutlined,
  InboxOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons'
import type { UploadFile } from 'antd'
import { useTranslation } from 'react-i18next'
import { datasetApi } from '@/services/api'
import type { DatasetModelType } from '@/types'

const { Dragger } = Upload
const { Text } = Typography

const MAX_FILE_SIZE = 500 * 1024 * 1024 // 500MB

interface UploadDatasetModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
}

export function UploadDatasetModal({ open, onCancel, onSuccess }: UploadDatasetModalProps) {
  const { t } = useTranslation(['datasets', 'common'])
  const [form] = Form.useForm()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadPercent, setUploadPercent] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [errorMsg, setErrorMsg] = useState<string>('')

  const resetUploadState = useCallback(() => {
    form.resetFields()
    setFileList([])
    setUploading(false)
    setUploadPercent(0)
    setUploadStatus('idle')
    setErrorMsg('')
  }, [form])

  useEffect(() => {
    if (!open) {
      resetUploadState()
    }
  }, [open, resetUploadState])

  const datasetTypeOptions = [
    { label: 'Custom', value: 'custom' },
    { label: 'Embedding Pair', value: 'embedding_pair' },
    { label: 'Embedding Triplet', value: 'embedding_triplet' },
    { label: 'Rerank Pair', value: 'rerank_pair' },
    { label: 'SFT Instruct', value: 'sft_instruct' },
    { label: 'DPO Preference', value: 'dpo_preference' },
  ]

  const usageOptions = [
    { label: t('options.usage.raw'), value: 'raw' },
    { label: t('options.usage.train'), value: 'train' },
    { label: t('options.usage.eval'), value: 'eval' },
    { label: t('options.usage.test'), value: 'test' },
  ]

  const modelTypeOptions = [
    { label: 'Embedding', value: 'embedding' },
    { label: 'Rerank', value: 'rerank' },
    { label: 'LLM', value: 'llm' },
  ]

  const handleSubmit = async (values: {
    dataset_name: string
    dataset_type: string
    usage?: string
    model_type?: DatasetModelType[]
    description?: string
  }) => {
    if (fileList.length === 0) return

    const file = fileList[0].originFileObj
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)
    formData.append('dataset_name', values.dataset_name)
    formData.append('dataset_type', values.dataset_type)
    if (values.usage) {
      formData.append('usage', values.usage)
    }
    if (values.model_type?.length) {
      formData.append('model_type', values.model_type.join(','))
    }
    if (values.description) {
      formData.append('description', values.description)
    }

    setUploading(true)
    setUploadStatus('uploading')
    setUploadPercent(0)
    setErrorMsg('')

    try {
      await datasetApi.upload(formData, (percent) => {
        setUploadPercent(percent)
      })
      setUploadStatus('success')
      message.success(t('upload.success'))
      resetUploadState()
      onSuccess()
    } catch (err: unknown) {
      setUploadStatus('error')
      const msg = err instanceof Error ? err.message : String(err)
      setErrorMsg(msg)
      message.error(t('upload.failed'))
    } finally {
      setUploading(false)
    }
  }

  const handleClose = () => {
    resetUploadState()
    onCancel()
  }

  const getStatusIcon = () => {
    switch (uploadStatus) {
      case 'uploading':
        return <LoadingOutlined spin style={{ color: '#1890ff' }} />
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return null
    }
  }

  return (
    <Modal
      title={
        <Space>
          <CloudUploadOutlined />
          {t('upload.title')}
        </Space>
      }
      open={open}
      onCancel={handleClose}
      footer={
        uploadStatus === 'success' ? (
          <Button onClick={handleClose}>{t('common:action.close')}</Button>
        ) : (
          <Space>
            <Button onClick={handleClose} disabled={uploading}>{t('common:action.cancel')}</Button>
            <Button
              type="primary"
              loading={uploading}
              onClick={() => form.submit()}
              disabled={fileList.length === 0}
            >
              {uploading ? t('upload.uploading') : t('upload.title')}
            </Button>
          </Space>
        )
      }
      width={520}
      maskClosable={!uploading}
      closable={!uploading}
      forceRender
    >
      {uploadStatus === 'idle' || uploadStatus === 'error' ? (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            dataset_type: 'custom',
            usage: 'train',
          }}
        >
          <Form.Item
            label={t('upload.dragText')}
            required
            validateStatus={fileList.length === 0 ? undefined : 'success'}
          >
            <Dragger
              fileList={fileList}
              maxCount={1}
              accept=".jsonl,.json,.csv,.parquet"
              beforeUpload={(file) => {
                if (file.size > MAX_FILE_SIZE) {
                  message.error(t('upload.fileTooLarge'))
                  return Upload.LIST_IGNORE
                }
                setFileList([{ ...file, uid: file.uid, name: file.name, originFileObj: file } as UploadFile])
                // Pre-fill dataset_name from filename (without extension)
                const nameWithoutExt = file.name.replace(/\.\w+$/, '')
                if (!form.getFieldValue('dataset_name')) {
                  form.setFieldsValue({ dataset_name: nameWithoutExt })
                }
                return false // prevent auto upload
              }}
              onRemove={() => {
                setFileList([])
              }}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">{t('upload.dragText')}</p>
              <p className="ant-upload-hint">{t('upload.dragHint')}</p>
            </Dragger>
          </Form.Item>

          <Form.Item
            name="dataset_name"
            label={t('download.form.datasetName.label')}
            rules={[{ required: true, message: t('download.form.datasetName.required') }]}
          >
            <Input placeholder={t('download.form.datasetName.placeholder')} />
          </Form.Item>

          <Form.Item
            name="dataset_type"
            label={t('download.form.datasetType.label')}
            rules={[{ required: true, message: t('download.form.datasetType.required') }]}
          >
            <Select options={datasetTypeOptions} />
          </Form.Item>

          <Form.Item name="usage" label={t('download.form.usage.label')}>
            <Select options={usageOptions} allowClear placeholder={t('options.optional')} />
          </Form.Item>

          <Form.Item name="model_type" label={t('download.form.modelType.label')}>
            <Checkbox.Group options={modelTypeOptions} />
          </Form.Item>

          <Form.Item name="description" label={t('download.form.description.label')}>
            <Input.TextArea rows={2} placeholder={t('download.form.description.placeholder')} />
          </Form.Item>

          {uploadStatus === 'error' && errorMsg && (
            <Alert message={t('upload.failed')} description={errorMsg} type="error" style={{ marginBottom: 16 }} />
          )}
        </Form>
      ) : (
        <div>
          <Alert
            message={uploading ? t('upload.uploading') : t('upload.success')}
            description={
              <Text>
                {fileList[0]?.name}
              </Text>
            }
            type={uploadStatus === 'success' ? 'success' : 'info'}
            showIcon
            icon={getStatusIcon()}
            style={{ marginBottom: 16 }}
          />
          <Progress
            percent={uploadPercent}
            status={uploadStatus === 'success' ? 'success' : 'active'}
          />
        </div>
      )}
    </Modal>
  )
}
