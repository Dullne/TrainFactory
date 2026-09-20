import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Modal,
  Form,
  Input,
  Select,
  Checkbox,
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
import { useTranslation } from 'react-i18next'
import {
  datasetApi,
  DownloadDatasetRequest,
  DatasetDownloadProgress,
  SILENT_REQUEST_CONFIG,
} from '@/services/api'
import { useAuth } from '@/auth/AuthContext'
import { DownloadTaskHistory } from '@/components/DownloadTaskHistory'
import { useDownloadHistory } from '@/hooks/useDownloadHistory'

const { Text } = Typography
const getDatasetDownloadId = (task: DatasetDownloadProgress) => task.dataset_id
const isDatasetDownloadActive = (task: DatasetDownloadProgress) =>
  task.status === 'pending' || task.status === 'downloading'
const loadDatasetDownloads = () => datasetApi.listDownloads(SILENT_REQUEST_CONFIG)

interface DownloadDatasetModalProps {
  open: boolean
  onCancel: () => void
  onSuccess: () => void
  onTasksChanged: () => void
}

export function DownloadDatasetModal({
  open,
  onCancel,
  onSuccess,
  onTasksChanged,
}: DownloadDatasetModalProps) {
  const { t } = useTranslation(['datasets', 'common'])
  const { user } = useAuth()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [progress, setProgress] = useState<DatasetDownloadProgress | null>(null)
  const [datasetId, setDatasetId] = useState<string | null>(null)
  const scopeKey = user?.user_id ?? ''
  const scopeRef = useRef(scopeKey)
  const operationGenerationRef = useRef(0)

  useEffect(() => {
    scopeRef.current = scopeKey
    operationGenerationRef.current += 1
    setLoading(false)
    setDownloading(false)
    setProgress(null)
    setDatasetId(null)
  }, [scopeKey])

  const handleTaskSettled = useCallback(
    (task: DatasetDownloadProgress) => {
      if (task.status === 'ready') {
        message.success(t('download.message.downloadComplete'))
      } else if (task.status === 'failed') {
        message.error(
          t('download.message.downloadFailed', {
            error: task.error || t('download.message.unknownError'),
          })
        )
      }

      if (task.dataset_id === datasetId) {
        setProgress(task)
        setDownloading(false)
        if (open && task.status === 'ready') {
          onSuccess()
          return
        }
      }
      onTasksChanged()
    },
    [datasetId, onSuccess, onTasksChanged, open, t]
  )

  const history = useDownloadHistory({
    scopeKey,
    getId: getDatasetDownloadId,
    isActive: isDatasetDownloadActive,
    load: loadDatasetDownloads,
    onTaskSettled: handleTaskSettled,
  })
  const refreshDownloads = history.refresh
  useEffect(() => {
    if (open) void refreshDownloads()
  }, [open, refreshDownloads])

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

  useEffect(() => {
    if (!datasetId) return
    const currentTask = history.tasks.find((task) => task.dataset_id === datasetId)
    if (currentTask) setProgress(currentTask)
  }, [datasetId, history.tasks])

  const handleSubmit = async (values: DownloadDatasetRequest) => {
    const requestScope = scopeKey
    const operationGeneration = ++operationGenerationRef.current
    setLoading(true)
    try {
      const result = await datasetApi.download(values)
      if (scopeRef.current !== requestScope) return
      const task: DatasetDownloadProgress = {
        dataset_id: result.dataset_id,
        dataset_name: result.dataset_name,
        status: 'downloading',
        progress: 0,
      }
      if (!history.upsert(task)) return
      message.info(t('download.message.taskCreated'))
      if (operationGenerationRef.current !== operationGeneration) return
      setDatasetId(result.dataset_id)
      setDownloading(true)
      setProgress(task)
    } catch (err) {
      if (
        operationGenerationRef.current === operationGeneration &&
        scopeRef.current === requestScope
      ) {
        message.error(t('download.message.taskCreateFailed'))
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
      message.info(t('download.message.backgroundContinue'))
    }
    operationGenerationRef.current += 1
    form.resetFields()
    setLoading(false)
    setProgress(null)
    setDatasetId(null)
    setDownloading(false)
    onCancel()
  }

  const getStatusIcon = () => {
    if (!progress) return null
    switch (progress.status) {
      case 'downloading':
        return <LoadingOutlined spin style={{ color: '#1890ff' }} />
      case 'ready':
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
      forceRender
    >
      {!downloading ? (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            source_type: 'huggingface',
            dataset_type: 'custom',
            usage: 'train',
            output_format: 'jsonl',
          }}
        >
          <Form.Item
            name="source_type"
            label={t('download.form.sourceType.label')}
            rules={[{ required: true, message: t('download.form.sourceType.required') }]}
          >
            <Select
              options={[
                { label: 'HuggingFace', value: 'huggingface' },
                { label: t('options.sourceType.modelscope'), value: 'modelscope' },
              ]}
            />
          </Form.Item>

          <Form.Item
            name="remote_repo"
            label={t('download.form.remoteRepo.label')}
            rules={[{ required: true, message: t('download.form.remoteRepo.required') }]}
            extra={t('download.form.remoteRepo.extra')}
          >
            <Input placeholder={t('download.form.remoteRepo.placeholder')} />
          </Form.Item>

          <Form.Item
            name="dataset_name"
            label={t('download.form.datasetName.label')}
            rules={[{ required: true, message: t('download.form.datasetName.required') }]}
          >
            <Input placeholder={t('download.form.datasetName.placeholder')} />
          </Form.Item>

          <Form.Item name="hf_subset" label={t('download.form.hfSubset.label')}>
            <Input placeholder={t('download.form.hfSubset.placeholder')} />
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

          <Form.Item name="output_format" label={t('download.form.outputFormat.label')}>
            <Select
              options={[
                { label: t('options.outputFormat.jsonlRecommended'), value: 'jsonl' },
                { label: 'JSON', value: 'json' },
                { label: 'Parquet', value: 'parquet' },
                { label: 'Arrow', value: 'arrow' },
              ]}
            />
          </Form.Item>

          <Form.Item name="description" label={t('download.form.description.label')}>
            <Input.TextArea rows={2} placeholder={t('download.form.description.placeholder')} />
          </Form.Item>
        </Form>
      ) : (
        <div>
          <Alert
            message={t('download.progress.title')}
            description={
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>
                  {t('download.progress.dataset')}: <Text strong>{progress?.dataset_name}</Text>
                </Text>
                {progress?.remote_repo && (
                  <Text type="secondary">
                    {t('download.progress.source', { source: progress.remote_repo })}
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
                : progress?.status === 'ready'
                  ? 'success'
                  : 'active'
            }
          />

          {progress?.error && (
            <Alert
              message={t('download.downloadFailedAlert')}
              description={progress.error}
              type="error"
              style={{ marginTop: 16 }}
            />
          )}

          <Text type="secondary" style={{ display: 'block', marginTop: 16 }}>
            {t('download.progress.backgroundHint')}
          </Text>
        </div>
      )}
      <DownloadTaskHistory
        items={history.tasks.map((task) => ({
          id: task.dataset_id,
          name: task.dataset_name || task.dataset_id,
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
