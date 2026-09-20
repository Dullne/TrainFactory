import { useCallback, useEffect, useRef, useState } from 'react'
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
import {
  modelApi,
  DownloadModelRequest,
  DownloadProgress,
  SILENT_REQUEST_CONFIG,
} from '@/services/api'
import { useAuth } from '@/auth/AuthContext'
import { DownloadTaskHistory } from '@/components/DownloadTaskHistory'
import { useDownloadHistory } from '@/hooks/useDownloadHistory'

const { Text } = Typography
const getModelDownloadId = (task: DownloadProgress) => task.registry_id
const isModelDownloadActive = (task: DownloadProgress) =>
  task.status === 'pending' || task.status === 'downloading'
const loadModelDownloads = () => modelApi.listDownloads(SILENT_REQUEST_CONFIG)

interface DownloadModelModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
  onTasksChanged: () => void
}

export function DownloadModelModal({
  open,
  onCancel,
  onSuccess,
  onTasksChanged,
}: DownloadModelModalProps) {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [registryId, setRegistryId] = useState<string | null>(null)
  const { t } = useTranslation(['models', 'common'])
  const { user } = useAuth()
  const scopeKey = user?.user_id ?? ''
  const scopeRef = useRef(scopeKey)
  const operationGenerationRef = useRef(0)

  useEffect(() => {
    scopeRef.current = scopeKey
    operationGenerationRef.current += 1
    setLoading(false)
    setDownloading(false)
    setProgress(null)
    setRegistryId(null)
  }, [scopeKey])

  const handleTaskSettled = useCallback(
    (task: DownloadProgress) => {
      if (task.status === 'available') {
        message.success(t('download.downloadComplete'))
      } else if (task.status === 'failed') {
        message.error(
          t('download.downloadFailed', { error: task.error || t('download.unknownError') })
        )
      }

      if (task.registry_id === registryId) {
        setProgress(task)
        setDownloading(false)
        if (open && task.status === 'available') {
          onSuccess()
          return
        }
      }
      onTasksChanged()
    },
    [onSuccess, onTasksChanged, open, registryId, t]
  )

  const history = useDownloadHistory({
    scopeKey,
    getId: getModelDownloadId,
    isActive: isModelDownloadActive,
    load: loadModelDownloads,
    onTaskSettled: handleTaskSettled,
  })
  const refreshDownloads = history.refresh
  useEffect(() => {
    if (open) void refreshDownloads()
  }, [open, refreshDownloads])

  const modelTypeOptions = [
    { label: t('download.embeddingLabel'), value: 'embedding' },
    { label: t('download.rerankerLabel'), value: 'reranker' },
    { label: t('download.decoderRerankerLabel'), value: 'decoder_reranker' },
    { label: t('download.llmLabel'), value: 'llm' },
  ]

  useEffect(() => {
    if (!registryId) return
    const currentTask = history.tasks.find((task) => task.registry_id === registryId)
    if (currentTask) setProgress(currentTask)
  }, [history.tasks, registryId])

  const handleSubmit = async (values: DownloadModelRequest) => {
    const requestScope = scopeKey
    const operationGeneration = ++operationGenerationRef.current
    setLoading(true)
    try {
      const result = await modelApi.download(values)
      if (scopeRef.current !== requestScope) return
      const task: DownloadProgress = {
        registry_id: result.registry_id,
        model_name: result.model_name,
        display_name: result.display_name,
        status: 'downloading',
        progress: 0,
      }
      if (!history.upsert(task)) return
      message.info(t('download.taskCreated'))
      if (operationGenerationRef.current !== operationGeneration) return
      setRegistryId(result.registry_id)
      setDownloading(true)
      setProgress(task)
    } catch (err) {
      if (
        operationGenerationRef.current === operationGeneration &&
        scopeRef.current === requestScope
      ) {
        message.error(t('download.taskCreateFailed'))
      }
    } finally {
      if (
        operationGenerationRef.current === operationGeneration &&
        scopeRef.current === requestScope
      ) {
        setLoading(false)
      }
    }
  }

  const handleClose = () => {
    if (downloading) {
      message.info(t('download.backgroundContinue'))
    }
    operationGenerationRef.current += 1
    form.resetFields()
    setLoading(false)
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
                  <Text type="secondary">
                    {t('download.sourceLabel')} {progress.remote_repo}
                  </Text>
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
      <DownloadTaskHistory
        items={history.tasks.map((task) => ({
          id: task.registry_id,
          name: task.display_name || task.model_name || task.registry_id,
          source: task.remote_repo,
          status: task.status,
          progress: task.progress,
          error: task.error,
        }))}
        loading={history.loading}
        failed={history.failed}
        title={t('download.history.title')}
        emptyText={t('download.history.empty')}
        errorText={t('download.history.error')}
        retryText={t('download.history.retry')}
        onRetry={() => void history.refresh()}
      />
    </Modal>
  )
}
