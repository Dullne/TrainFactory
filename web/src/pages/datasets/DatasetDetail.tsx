import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import {
  Card,
  Button,
  Space,
  Tag,
  Typography,
  Descriptions,
  Table,
  Popconfirm,
  message,
  Spin,
  Row,
  Col,
  Tooltip,
  Tabs,
  Empty,
  Form,
  Input,
  Select,
  Checkbox,
  Statistic,
} from 'antd'
import {
  ArrowLeftOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EditOutlined,
  SaveOutlined,
  CloseOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  SettingOutlined,
  CloudOutlined,
  FolderOutlined,
  DownloadOutlined,
  InboxOutlined,
  UndoOutlined,
} from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { datasetApi } from '@/services/api'
import { formatDate, formatBytes } from '@/utils'
import type { Dataset, DatasetUsage, DatasetModelType, DatasetColumn } from '@/types'
import { StatusTag } from '@/components/StatusTag'
import {
  STATUS_SUCCESS,
  STATUS_WARNING,
  STATUS_INFO,
  BG_ELEVATED,
} from '@/theme'

const { Title, Text, Paragraph } = Typography

const typeColorMap: Record<string, string> = {
  embedding_universal: 'purple',
  embedding_pair: 'blue',
  embedding_triplet: 'cyan',
  embedding_multi_neg: 'geekblue',
  embedding_dynamic_neg: 'volcano',
  embedding_cosine: 'gold',
  embedding_margin: 'lime',
  embedding_margin_multi: 'green',
  embedding_scored: 'orange',
  embedding_score_triplet: 'magenta',
  rerank_pair: 'processing',
  rerank_triplet: 'warning',
  rerank_listwise: 'error',
  sft_instruct: 'success',
  dpo_preference: 'red',
  rl_reward: 'pink',
  custom: 'default',
}

export default function DatasetDetail() {
  const { t } = useTranslation(['datasets', 'common'])
  const { datasetId } = useParams<{ datasetId: string }>()
  const navigate = useNavigate()
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [loading, setLoading] = useState(true)
  const [previewData, setPreviewData] = useState<{ columns: DatasetColumn[]; rows: Record<string, unknown>[]; total_rows: number }>({
    columns: [],
    rows: [],
    total_rows: 0,
  })
  const [previewLoading, setPreviewLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [editing, setEditing] = useState(false)
  const [form] = Form.useForm()

  const usageOptions: { label: string; value: DatasetUsage }[] = [
    { label: t('options.usage.raw'), value: 'raw' },
    { label: t('options.usage.train'), value: 'train' },
    { label: t('options.usage.eval'), value: 'eval' },
    { label: t('options.usage.test'), value: 'test' },
  ]

  const modelTypeOptions: { label: string; value: DatasetModelType }[] = [
    { label: 'Embedding', value: 'embedding' },
    { label: 'Rerank', value: 'rerank' },
    { label: 'LLM', value: 'llm' },
  ]

  const fetchDataset = useCallback(async () => {
    if (!datasetId) return
    try {
      const data = await datasetApi.get(datasetId)
      setDataset(data)
    } catch (err) {
      message.error(t('detail.message.fetchFailed'))
    } finally {
      setLoading(false)
    }
  }, [datasetId, t])

  const fetchPreview = useCallback(async () => {
    if (!datasetId) return
    setPreviewLoading(true)
    try {
      const data = await datasetApi.preview(datasetId, 20)
      // Handle different response formats
      if (Array.isArray(data)) {
        setPreviewData({ columns: [], rows: data, total_rows: data.length })
      } else if (data && typeof data === 'object') {
        setPreviewData({
          columns: (data as { columns?: DatasetColumn[] }).columns || [],
          rows: (data as { rows?: Record<string, unknown>[] }).rows || [],
          total_rows: (data as { total_rows?: number }).total_rows || 0,
        })
      }
    } catch (err) {
      console.error('Failed to fetch preview:', err)
    } finally {
      setPreviewLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    fetchDataset()
  }, [fetchDataset])

  useEffect(() => {
    if (dataset) {
      fetchPreview()
    }
  }, [dataset, fetchPreview])

  const handleDelete = async () => {
    if (!datasetId) return
    try {
      await datasetApi.delete(datasetId)
      message.success(t('detail.message.deleteSuccess'))
      navigate('/datasets')
    } catch (err) {
      message.error(t('detail.message.deleteFailed'))
    }
  }

  const handleToggleArchive = async () => {
    if (!datasetId || !dataset) return
    const newStatus = dataset.status === 'archived' ? 'ready' : 'archived'
    try {
      await datasetApi.update(datasetId, { status: newStatus })
      message.success(t('detail.message.updateSuccess'))
      fetchDataset()
    } catch {
      message.error(t('detail.message.updateFailed'))
    }
  }

  const handleEdit = () => {
    if (!dataset) return
    form.setFieldsValue({
      display_name: dataset.display_name,
      description: dataset.description,
      usage: dataset.usage,
      model_type: dataset.model_type || [],
      tags: dataset.tags?.join(', ') || '',
    })
    setEditing(true)
  }

  const handleSave = async () => {
    if (!datasetId) return
    try {
      const values = await form.validateFields()
      await datasetApi.update(datasetId, {
        display_name: values.display_name || undefined,
        description: values.description || undefined,
        usage: values.usage,
        model_type: values.model_type,
        tags: values.tags ? values.tags.split(',').map((t: string) => t.trim()).filter(Boolean) : undefined,
      })
      message.success(t('detail.message.updateSuccess'))
      setEditing(false)
      fetchDataset()
    } catch (err) {
      message.error(t('detail.message.updateFailed'))
    }
  }

  const handleCancelEdit = () => {
    setEditing(false)
    form.resetFields()
  }

  const handleExportJsonl = async () => {
    if (!datasetId) return
    setExporting(true)
    try {
      const result = await datasetApi.export(datasetId, { format: 'jsonl' })
      const downloadUrl = result.proxy_download_url || result.download_url
      if (!downloadUrl) {
        throw new Error('No download url returned')
      }
      window.open(downloadUrl, '_blank', 'noopener,noreferrer')
      message.success(`导出链接已生成${result.row_count ? `（${result.row_count} 条）` : ''}`)
    } catch (err) {
      message.error(`导出失败: ${(err as Error).message}`)
    } finally {
      setExporting(false)
    }
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    )
  }

  if (!dataset) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Empty description={t('detail.notFound')} />
        <Button type="primary" onClick={() => navigate('/datasets')} style={{ marginTop: 16 }}>
          {t('detail.backToList')}
        </Button>
      </div>
    )
  }

  const datasetStatusText = t(`options.status.${dataset.status}`, { defaultValue: dataset.status })

  // Generate preview table columns dynamically
  const previewColumns = previewData.rows.length > 0
    ? Object.keys(previewData.rows[0]).map((key) => ({
        title: key,
        dataIndex: key,
        key,
        ellipsis: true,
        width: 200,
        render: (val: unknown) => {
          const strVal = typeof val === 'object' ? JSON.stringify(val) : String(val ?? '-')
          return (
            <Tooltip title={strVal} placement="topLeft">
              <span style={{ maxWidth: 180, display: 'inline-block' }}>{strVal}</span>
            </Tooltip>
          )
        },
      }))
    : []

  return (
    <div style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets')}>
            {t('common:action.back')}
          </Button>
          <Title level={4} style={{ margin: 0 }}>
            {dataset.display_name || dataset.dataset_name}
          </Title>
          <StatusTag status={dataset.status} text={datasetStatusText} />
          <Tag color={typeColorMap[dataset.dataset_type] || 'default'}>{dataset.dataset_type}</Tag>
        </Space>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchDataset}>
            {t('common:action.refresh')}
          </Button>
          <Button icon={<DownloadOutlined />} loading={exporting} onClick={handleExportJsonl}>
            {t('common:action.export')} JSONL
          </Button>
          {!editing ? (
            <Button icon={<EditOutlined />} onClick={handleEdit}>
              {t('common:action.edit')}
            </Button>
          ) : (
            <>
              <Button icon={<CloseOutlined />} onClick={handleCancelEdit}>
                {t('common:action.cancel')}
              </Button>
              <Button type="primary" icon={<SaveOutlined />} onClick={handleSave}>
                {t('common:action.save')}
              </Button>
            </>
          )}
          {dataset.status === 'ready' && (
            <Popconfirm title={t('detail.confirmArchive')} onConfirm={handleToggleArchive}>
              <Button icon={<InboxOutlined />}>
                {t('detail.archive')}
              </Button>
            </Popconfirm>
          )}
          {dataset.status === 'archived' && (
            <Button icon={<UndoOutlined />} onClick={handleToggleArchive}>
              {t('detail.unarchive')}
            </Button>
          )}
          <Popconfirm title={t('detail.confirmDelete')} onConfirm={handleDelete}>
            <Button danger icon={<DeleteOutlined />}>
              {t('common:action.delete')}
            </Button>
          </Popconfirm>
        </Space>
      </div>

      {/* Statistics Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title={t('detail.stats.totalSamples')}
              value={dataset.num_rows || dataset.num_samples || 0}
              valueStyle={{ color: STATUS_INFO }}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title={t('detail.stats.trainSet')}
              value={dataset.num_train || '-'}
              valueStyle={{ color: STATUS_SUCCESS }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title={t('detail.stats.evalSet')}
              value={dataset.num_eval || '-'}
              valueStyle={{ color: STATUS_WARNING }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title={t('detail.stats.fileSize')}
              value={formatBytes(dataset.file_size || dataset.size_bytes || 0)}
              valueStyle={{ color: STATUS_INFO }}
              prefix={<FolderOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Tabs */}
      <Tabs
        defaultActiveKey="info"
        items={[
          {
            key: 'info',
            label: (
              <span>
                <FileTextOutlined />
                {t('detail.tabs.basicInfo')}
              </span>
            ),
            children: (
              <Form form={form} layout="vertical" component={false}>
                {editing ? (
                  <Card>
                    <div style={{ maxWidth: 600 }}>
                      <Form.Item name="display_name" label={t('detail.info.displayName')}>
                        <Input placeholder={t('detail.info.displayNamePlaceholder')} />
                      </Form.Item>
                      <Form.Item name="description" label={t('detail.info.description')}>
                        <Input.TextArea rows={3} placeholder={t('detail.info.descriptionPlaceholder')} />
                      </Form.Item>
                      <Form.Item name="usage" label={t('detail.info.usage')}>
                        <Select options={usageOptions} />
                      </Form.Item>
                      <Form.Item name="model_type" label={t('detail.info.modelType')}>
                        <Checkbox.Group options={modelTypeOptions} />
                      </Form.Item>
                      <Form.Item name="tags" label={t('detail.info.tags')} extra={t('detail.info.tagsExtra')}>
                        <Input placeholder="tag1, tag2, tag3" />
                      </Form.Item>
                    </div>
                  </Card>
                ) : (
                  <Card>
                    <Descriptions column={2} bordered size="small">
                      <Descriptions.Item label={t('detail.info.datasetId')}>{dataset.dataset_id}</Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.datasetName')}>{dataset.dataset_name}</Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.displayName')}>{dataset.display_name || '-'}</Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.datasetType')}>
                        <Tag color={typeColorMap[dataset.dataset_type] || 'default'}>{dataset.dataset_type}</Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.usage')}>
                        {dataset.usage === 'raw' ? (
                          <Tag color="purple">{t('options.usage.raw')}</Tag>
                        ) : dataset.usage === 'train' ? (
                          <Tag color="blue">{t('options.usage.train')}</Tag>
                        ) : dataset.usage === 'eval' ? (
                          <Tag color="green">{t('options.usage.eval')}</Tag>
                        ) : dataset.usage === 'test' ? (
                          <Tag color="orange">{t('options.usage.test')}</Tag>
                        ) : (
                          '-'
                        )}
                      </Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.modelType')}>
                        {dataset.model_type?.length ? (
                          <Space>
                            {dataset.model_type.map((mt) => (
                              <Tag key={mt} color={mt === 'embedding' ? 'purple' : mt === 'rerank' ? 'orange' : 'cyan'}>
                                {mt.toUpperCase()}
                              </Tag>
                            ))}
                          </Space>
                        ) : (
                          '-'
                        )}
                      </Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.fileFormat')}>
                        <Tag>{dataset.file_format}</Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.status')}>
                        <StatusTag status={dataset.status} text={datasetStatusText} />
                      </Descriptions.Item>
                      {dataset.storage_path && (
                      <Descriptions.Item label={t('detail.info.storagePath')} span={2}>
                        <Text code copyable style={{ fontSize: 12 }}>
                          {dataset.storage_path}
                        </Text>
                      </Descriptions.Item>
                      )}
                      {dataset.storage_uri && (
                      <Descriptions.Item label="Storage URI" span={2}>
                        <Text code copyable style={{ fontSize: 12 }}>
                          {dataset.storage_uri}
                        </Text>
                      </Descriptions.Item>
                      )}
                      {dataset.description && (
                        <Descriptions.Item label={t('detail.info.description')} span={2}>
                          <Paragraph style={{ margin: 0 }}>{dataset.description}</Paragraph>
                        </Descriptions.Item>
                      )}
                      {dataset.tags?.length ? (
                        <Descriptions.Item label={t('detail.info.tags')} span={2}>
                          <Space>
                            {dataset.tags.map((tag) => (
                              <Tag key={tag}>{tag}</Tag>
                            ))}
                          </Space>
                        </Descriptions.Item>
                      ) : null}
                      <Descriptions.Item label={t('detail.info.createdAt')}>{formatDate(dataset.created_at)}</Descriptions.Item>
                      <Descriptions.Item label={t('detail.info.updatedAt')}>{formatDate(dataset.updated_at)}</Descriptions.Item>
                    </Descriptions>
                  </Card>
                )}
              </Form>
            ),
          },
          {
            key: 'source',
            label: (
              <span>
                <CloudOutlined />
                {t('detail.tabs.sourceInfo')}
              </span>
            ),
            children: (
              <Card>
                <Descriptions column={2} bordered size="small">
                  <Descriptions.Item label={t('detail.source.sourceType')}>
                    <Tag color={dataset.source_type === 'huggingface' ? 'orange' : dataset.source_type === 'modelscope' ? 'blue' : 'default'}>
                      {dataset.source_type}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label={t('detail.source.sourcePath')}>{dataset.source_path || '-'}</Descriptions.Item>
                  {dataset.source_dataset_id && (
                    <Descriptions.Item label={t('detail.source.sourceDataset')}>
                      <Link to={`/datasets/${dataset.source_dataset_id}`}>{dataset.source_dataset_id}</Link>
                    </Descriptions.Item>
                  )}
                  {dataset.remote_repo && (
                    <Descriptions.Item label={t('detail.source.remoteRepo')} span={2}>
                      <Text code copyable>{dataset.remote_repo}</Text>
                    </Descriptions.Item>
                  )}
                  {dataset.hf_subset && (
                    <Descriptions.Item label="HuggingFace Subset">{dataset.hf_subset}</Descriptions.Item>
                  )}
                  {dataset.extra_metadata && Object.keys(dataset.extra_metadata).length > 0 && (
                    <Descriptions.Item label={t('detail.source.extraMetadata')} span={2}>
                      <pre style={{ margin: 0, fontSize: 12, background: BG_ELEVATED, padding: 8, borderRadius: 4 }}>
                        {JSON.stringify(dataset.extra_metadata, null, 2)}
                      </pre>
                    </Descriptions.Item>
                  )}
                </Descriptions>
              </Card>
            ),
          },
          {
            key: 'schema',
            label: (
              <span>
                <SettingOutlined />
                {t('detail.tabs.schema')}
              </span>
            ),
            children: (
              <Card>
                {dataset.columns && dataset.columns.length > 0 ? (
                  <Table
                    dataSource={dataset.columns.map((col, idx) => ({
                      key: idx,
                      name: typeof col === 'string' ? col : col.name,
                      type: typeof col === 'string' ? '-' : col.type,
                    }))}
                    columns={[
                      { title: t('detail.schema.fieldName'), dataIndex: 'name', key: 'name' },
                      { title: t('detail.schema.fieldType'), dataIndex: 'type', key: 'type' },
                    ]}
                    pagination={false}
                    size="small"
                  />
                ) : (
                  <Empty description={t('detail.schema.noFields')} />
                )}
              </Card>
            ),
          },
          {
            key: 'preview',
            label: (
              <span>
                <DatabaseOutlined />
                {t('detail.tabs.preview')}
              </span>
            ),
            children: (
              <Card>
                {previewData.rows.length > 0 ? (
                  <Table
                    columns={previewColumns}
                    dataSource={previewData.rows.map((row, idx) => ({ ...row, _key: idx }))}
                    rowKey="_key"
                    loading={previewLoading}
                    scroll={{ x: 'max-content' }}
                    pagination={{
                      pageSize: 10,
                      showTotal: (total) => t('detail.preview.totalPreview', { total }),
                    }}
                    size="small"
                  />
                ) : (
                  <Empty description={previewLoading ? t('detail.preview.loading') : t('detail.preview.noData')} />
                )}
                <div style={{ marginTop: 16 }}>
                  <Text type="secondary">
                    {t('detail.preview.showFirst', { total: dataset.num_rows?.toLocaleString() || '-' })}
                  </Text>
                </div>
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}
