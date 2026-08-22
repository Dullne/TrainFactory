import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Modal,
  Form,
  Input,
  Select,
  Button,
  message,
  Progress,
  Alert,
  Space,
  Typography,
} from 'antd'
import {
  CloudDownloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons'
import { modelApi, DownloadModelRequest, DownloadProgress } from '@/services/api'

const { Text } = Typography

interface DownloadModelModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
}

export function DownloadModelModal({ open, onCancel, onSuccess }: DownloadModelModalProps) {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [registryId, setRegistryId] = useState<string | null>(null)
  const { t } = useTranslation(['models', 'common'])

  const modelTypeOptions = [
    { label: t('download.embeddingLabel'), value: 'embedding' },
    { label: t('download.rerankerLabel'), value: 'reranker' },
    { label: t('download.decoderRerankerLabel'), value: 'decoder_reranker' },
    { label: t('download.llmLabel'), value: 'llm' },
  ]

  // Poll download progress with request guard
  useEffect(() => {
    if (!registryId || !downloading) return

    let polling = true
    let fetching = false
    let errorCount = 0
    const MAX_ERRORS = 10

    const poll = async () => {
      if (!polling || fetching) return
      fetching = true
      try {
        const data = await modelApi.getDownloadProgress(registryId)
        errorCount = 0
        setProgress(data)

        if (data.status === 'available') {
          polling = false
          setDownloading(false)
          message.success(t('download.downloadComplete'))
          onSuccess()
        } else if (data.status === 'failed') {
          polling = false
          setDownloading(false)
          message.error(t('download.downloadFailed', { error: data.error || t('download.unknownError') }))
        }
      } catch {
        errorCount++
        if (errorCount >= MAX_ERRORS) {
          polling = false
          setDownloading(false)
          message.error(t('download.progressFetchFailed'))
        }
      } finally {
        fetching = false
      }
    }

    const interval = setInterval(poll, 3000)
    return () => { polling = false; clearInterval(interval) }
  }, [registryId, downloading, onSuccess, t])

  const handleSubmit = async (values: DownloadModelRequest) => {
    setLoading(true)
    try {
      const result = await modelApi.download(values)
      setRegistryId(result.registry_id)
      setDownloading(true)
      setProgress({
        registry_id: result.registry_id,
        model_name: result.model_name,
        display_name: result.display_name,
        status: 'downloading',
        progress: 0,
      })
      message.info(t('download.taskCreated'))
    } catch (err) {
      message.error(t('download.taskCreateFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleClose = () => {
    if (downloading) {
      message.info(t('download.backgroundContinue'))
    }
    form.resetFields()
    setProgress(null)
    setRegistryId(null)
    setDownloading(false)
    onCancel()
  }

  const getStatusIcon = () => {
    if (!progress) return null
    switch (progress.status) {
      case 'downloading':
        return <LoadingOutlined spin style={{ color: '#1890ff' }} />
      case 'available':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return null
    }
  }

  return (
    <Modal
      title={
        <Space>
          <CloudDownloadOutlined />
          {t('download.title')}
        </Space>
      }
      open={open}
      onCancel={handleClose}
      footer={
        downloading ? (
          <Button onClick={handleClose}>{t('download.closeBackground')}</Button>
        ) : (
          <Space>
            <Button onClick={handleClose}>{t('common:action.cancel')}</Button>
            <Button type="primary" loading={loading} onClick={() => form.submit()}>
              {t('download.startDownload')}
            </Button>
          </Space>
        )
      }
      width={520}
    >
      {!downloading ? (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            download_source: 'modelscope',
            model_type: 'embedding',
          }}
        >
          <Form.Item
            name="download_source"
            label={t('download.source')}
            rules={[{ required: true, message: t('download.sourceRequired') }]}
          >
            <Select
              options={[
                { label: t('download.modelscope'), value: 'modelscope' },
                { label: 'HuggingFace', value: 'huggingface' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="remote_repo"
            label={t('download.repoAddress')}
            rules={[{ required: true, message: t('download.repoRequired') }]}
            extra={t('download.repoExample')}
          >
            <Input placeholder={t('download.repoPlaceholder')} />
          </Form.Item>

          <Form.Item
            name="model_type"
            label={t('download.modelType')}
            rules={[{ required: true, message: t('download.modelTypeRequired') }]}
          >
            <Select options={modelTypeOptions} />
          </Form.Item>

          <Form.Item name="display_name" label={t('download.displayName')}>
            <Input placeholder={t('download.displayNamePlaceholder')} />
          </Form.Item>

          <Form.Item name="description" label={t('download.description')}>
            <Input.TextArea rows={2} placeholder={t('download.optionalPlaceholder')} />
          </Form.Item>
        </Form>
      ) : (
        <div>
          <Alert
            message={t('download.downloading')}
            description={
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>
                  {t('download.modelLabel')} <Text strong>{progress?.model_name}</Text>
                </Text>
                {progress?.remote_repo && (
                  <Text type="secondary">{t('download.sourceLabel')} {progress.remote_repo}</Text>
                )}
              </Space>
            }
            type="info"
            showIcon
            icon={getStatusIcon()}
            style={{ marginBottom: 16 }}
          />

          <Progress
            percent={progress?.progress || 0}
            status={
              progress?.status === 'failed'
                ? 'exception'
                : progress?.status === 'available'
                ? 'success'
                : 'active'
            }
          />

          {progress?.error && (
            <Alert
              message={t('download.downloadFailedTitle')}
              description={progress.error}
              type="error"
              style={{ marginTop: 16 }}
            />
          )}

          <Text type="secondary" style={{ display: 'block', marginTop: 16 }}>
            {t('download.downloadHint')}
          </Text>
        </div>
      )}
    </Modal>
  )
}
