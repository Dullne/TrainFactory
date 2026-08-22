import { useCallback, useMemo } from 'react'
import {
  Modal,
  Table,
  Typography,
  Space,
  Tag,
  Descriptions,
  Divider,
  Card,
  Empty,
  Progress,
} from 'antd'
import {
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  ApiOutlined,
  DatabaseOutlined,
  LoadingOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import type { EvaluationDatasetConfig, EvaluationTask } from '@/types'
import { formatDate } from '@/utils'
import {
  BG_ELEVATED,
  BORDER_SECONDARY,
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  STATUS_SUCCESS,
  STATUS_ERROR,
  STATUS_WARNING,
} from '@/theme'

const { Text } = Typography

interface EvaluationDetailModalProps {
  visible: boolean
  task: EvaluationTask | null
  onClose: () => void
}

interface ResultRow {
  key: string
  model: string
  dataset: string
  datasetSource?: DatasetSource
  mrr?: number
  ap?: number
  [key: string]: string | number | undefined
}

type DatasetSource = EvaluationDatasetConfig['type']

interface DatasetDisplay {
  label: string
  source?: DatasetSource
}

const datasetSourceColors: Record<DatasetSource, string> = {
  mteb: 'blue',
  registered: 'green',
  local: 'default',
}

function normalizedDatasetSource(value: string): DatasetSource | undefined {
  const source = value.trim().toLowerCase()
  return source === 'mteb' || source === 'registered' || source === 'local' ? source : undefined
}

function datasetDisplayFromConfig(config: EvaluationDatasetConfig): DatasetDisplay {
  return {
    label: config.name,
    source: normalizedDatasetSource(config.type),
  }
}

function canonicalDatasetKey(config: EvaluationDatasetConfig): string | undefined {
  if (config.result_key) return config.result_key
  if (normalizedDatasetSource(config.type) === 'mteb') return `mteb:${config.name}`
  if (config.dataset_id) return `dataset:${config.dataset_id}`
  if (config.path) return `local:${config.path}`
  return undefined
}

function buildDatasetDisplayMap(
  configs: EvaluationDatasetConfig[] | undefined,
  identityDatasets: NonNullable<EvaluationTask['identity_map']>['datasets'] | undefined
): Map<string, DatasetDisplay> {
  const displays = new Map<string, DatasetDisplay>()
  const legacyAliases = new Map<string, DatasetDisplay[]>()

  Object.entries(identityDatasets || {}).forEach(([resultKey, descriptor]) => {
    displays.set(resultKey, {
      label: descriptor.name,
      source: normalizedDatasetSource(descriptor.type),
    })
  })

  configs?.forEach((config) => {
    const display = datasetDisplayFromConfig(config)
    const canonicalKey = canonicalDatasetKey(config)
    if (canonicalKey) displays.set(canonicalKey, display)
    const aliases = legacyAliases.get(config.name) || []
    aliases.push(display)
    legacyAliases.set(config.name, aliases)
  })
  legacyAliases.forEach((candidates, alias) => {
    if (candidates.length === 1 && !displays.has(alias)) {
      displays.set(alias, candidates[0])
    }
  })
  return displays
}

function resolveDatasetDisplay(
  resultKey: string,
  displays: Map<string, DatasetDisplay>
): DatasetDisplay {
  const configured = displays.get(resultKey)
  if (configured) return configured
  if (resultKey.startsWith('mteb:')) {
    return { label: resultKey.slice('mteb:'.length), source: 'mteb' }
  }
  if (resultKey.startsWith('local:')) {
    return { label: 'Local dataset', source: 'local' }
  }
  return { label: resultKey }
}

export function EvaluationDetailModal({ visible, task, onClose }: EvaluationDetailModalProps) {
  const { t } = useTranslation(['evaluations', 'common'])
  const datasetDisplays = useMemo(
    () => buildDatasetDisplayMap(task?.dataset_configs, task?.identity_map?.datasets),
    [task?.dataset_configs, task?.identity_map?.datasets]
  )
  const renderDatasetDisplay = useCallback(
    (display: DatasetDisplay) => {
      const sourceLabel =
        display.source === 'mteb'
          ? 'MTEB'
          : display.source === 'registered'
            ? t('create.registered')
            : display.source === 'local'
              ? t('create.local')
              : undefined
      return (
        <Space size={4} wrap>
          <Tag color="blue">{display.label}</Tag>
          {display.source && sourceLabel && (
            <Tag color={datasetSourceColors[display.source]}>{sourceLabel}</Tag>
          )}
        </Space>
      )
    },
    [t]
  )

  // Parse results into table rows
  const { rows, metricKeys } = useMemo(() => {
    if (!task?.results) {
      return { rows: [], metricKeys: [] }
    }

    const allKeys = new Set<string>()
    const tableRows: ResultRow[] = []

    // Results structure: { model_name: { dataset_name: { metric: value } } }
    Object.entries(task.results).forEach(([modelName, datasets]) => {
      if (typeof datasets !== 'object' || datasets === null) return

      Object.entries(datasets as Record<string, Record<string, number>>).forEach(
        ([datasetName, metrics]) => {
          if (typeof metrics !== 'object' || metrics === null) return
          if (datasetName === '_error') return

          const datasetDisplay = resolveDatasetDisplay(datasetName, datasetDisplays)

          const row: ResultRow = {
            key: `${modelName}-${datasetName}`,
            model: modelName,
            dataset: datasetDisplay.label,
            datasetSource: datasetDisplay.source,
          }

          Object.entries(metrics).forEach(([metricKey, value]) => {
            if (metricKey !== 'error') {
              allKeys.add(metricKey)
              row[metricKey] = value
            }
          })

          tableRows.push(row)
        }
      )
    })

    return {
      rows: tableRows,
      metricKeys: Array.from(allKeys).sort(),
    }
  }, [datasetDisplays, task?.results])

  // Generate columns dynamically
  const columns: ColumnsType<ResultRow> = useMemo(() => {
    const baseColumns: ColumnsType<ResultRow> = [
      {
        title: t('detail.model'),
        dataIndex: 'model',
        key: 'model',
        fixed: 'left',
        width: 150,
        render: (name: string) => (
          <Text strong style={{ color: TEXT_PRIMARY }}>
            {name}
          </Text>
        ),
      },
      {
        title: t('detail.dataset'),
        dataIndex: 'dataset',
        key: 'dataset',
        fixed: 'left',
        width: 180,
        render: (name: string, row) =>
          renderDatasetDisplay({
            label: name,
            source: row.datasetSource,
          }),
      },
    ]

    // Add metric columns
    const metricColumns: ColumnsType<ResultRow> = metricKeys.map((key) => ({
      title: key.toUpperCase(),
      dataIndex: key,
      key,
      width: 100,
      render: (value: number | string | undefined) => {
        if (value === undefined || value === null) return '-'
        const num = typeof value === 'number' ? value : Number(value)
        if (!Number.isFinite(num)) return '-'
        return <Text style={{ color: STATUS_SUCCESS, fontWeight: 500 }}>{num.toFixed(4)}</Text>
      },
      sorter: (a: ResultRow, b: ResultRow) => {
        const aVal = Number(a[key]) || 0
        const bVal = Number(b[key]) || 0
        return aVal - bVal
      },
    }))

    return [...baseColumns, ...metricColumns]
  }, [metricKeys, renderDatasetDisplay, t])

  // Group results by model for summary
  const modelSummaries = useMemo(() => {
    if (!task?.results) return []

    return Object.entries(task.results).map(([modelName, datasets]) => {
      const datasetCount =
        typeof datasets === 'object' && datasets !== null ? Object.keys(datasets).length : 0

      // Calculate average MRR if available
      let avgMrr = 0
      let mrrCount = 0
      if (typeof datasets === 'object' && datasets !== null) {
        Object.values(datasets as Record<string, Record<string, number>>).forEach((metrics) => {
          if (metrics?.MRR !== undefined) {
            avgMrr += metrics.MRR
            mrrCount++
          }
        })
      }

      return {
        name: modelName,
        datasets: datasetCount,
        avgMrr: mrrCount > 0 ? avgMrr / mrrCount : null,
      }
    })
  }, [task?.results])

  if (!task) return null

  const isFailed = task.status === 'failed'
  const isRunning = task.status === 'running'
  const isPending = task.status === 'pending'
  const isSucceeded = task.status === 'succeeded'

  const getModalTitle = () => {
    if (isFailed) return t('detail.evalFailed')
    if (isRunning) return t('detail.evalRunning')
    if (isPending) return t('detail.waitingExec')
    return t('detail.evalResults')
  }

  const getTitleIcon = () => {
    if (isFailed) return <CloseCircleOutlined style={{ color: STATUS_ERROR }} />
    if (isRunning) return <LoadingOutlined spin style={{ color: STATUS_WARNING }} />
    if (isPending) return <ClockCircleOutlined style={{ color: TEXT_SECONDARY }} />
    return <CheckCircleOutlined style={{ color: STATUS_SUCCESS }} />
  }

  const getStatusTag = () => {
    if (isFailed)
      return (
        <Tag color="error" icon={<CloseCircleOutlined />}>
          {t('common:status.failed')}
        </Tag>
      )
    if (isRunning)
      return (
        <Tag color="processing" icon={<PlayCircleOutlined spin />}>
          {t('common:status.running')}
        </Tag>
      )
    if (isPending)
      return (
        <Tag color="default" icon={<ClockCircleOutlined />}>
          {t('common:status.pending')}
        </Tag>
      )
    if (isSucceeded)
      return (
        <Tag color="success" icon={<CheckCircleOutlined />}>
          {t('common:status.succeeded')}
        </Tag>
      )
    return <Tag>{task.status}</Tag>
  }

  return (
    <Modal
      title={
        <Space>
          {getTitleIcon()}
          <span>{getModalTitle()}</span>
        </Space>
      }
      open={visible}
      onCancel={onClose}
      footer={null}
      width={1000}
      styles={{ body: { maxHeight: '70vh', overflowY: 'auto' } }}
    >
      {/* Task Info */}
      <Descriptions size="small" column={3} style={{ marginBottom: 16 }}>
        <Descriptions.Item label={t('detail.taskName')}>
          {task.task_name || task.task_id.substring(0, 8)}
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.taskId')}>
          {task.task_id.substring(0, 8)}...
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.type')}>
          <Tag>{task.eval_type}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.createdAt')}>
          {formatDate(task.created_at)}
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.completedAt')}>
          {formatDate(task.completed_at) || '-'}
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.status')}>{getStatusTag()}</Descriptions.Item>
      </Descriptions>

      {/* Progress for running tasks */}
      {isRunning && (
        <Card
          size="small"
          style={{
            marginBottom: 16,
            background: BG_ELEVATED,
            borderColor: BORDER_SECONDARY,
          }}
        >
          <Space direction="vertical" style={{ width: '100%' }} size={8}>
            <Space>
              <LoadingOutlined spin style={{ color: STATUS_WARNING }} />
              <Text strong>{t('detail.overallProgress')}</Text>
            </Space>
            <Progress percent={Math.round(task.progress || 0)} status="active" />
          </Space>
        </Card>
      )}

      {/* Model Progress Cards */}
      {task.model_progress && Object.keys(task.model_progress).length > 0 && (
        <>
          <Divider orientation="left" style={{ marginTop: 0 }}>
            <Space>
              <ApiOutlined />
              {t('detail.evalProgress')}
            </Space>
          </Divider>
          <Space direction="vertical" size={12} style={{ width: '100%' }}>
            {Object.entries(task.model_progress).map(([modelName, datasets]) => {
              // Calculate model overall progress
              const datasetEntries = Object.entries(datasets)
              const modelProgress =
                datasetEntries.length > 0
                  ? datasetEntries.reduce((sum, [, d]) => sum + (d.progress || 0), 0) /
                    datasetEntries.length
                  : 0
              const completedCount = datasetEntries.filter(
                ([, d]) => d.status === 'completed'
              ).length
              const runningCount = datasetEntries.filter(([, d]) => d.status === 'running').length

              return (
                <Card
                  key={modelName}
                  size="small"
                  title={
                    <Space>
                      <Text strong style={{ color: TEXT_PRIMARY }}>
                        {modelName}
                      </Text>
                      <Tag color="blue">
                        {completedCount}/{datasetEntries.length}
                      </Tag>
                      {runningCount > 0 && (
                        <Tag color="processing" icon={<LoadingOutlined spin />}>
                          {t('common:status.running')}
                        </Tag>
                      )}
                    </Space>
                  }
                  extra={
                    <Progress
                      type="circle"
                      percent={Math.round(modelProgress)}
                      size={40}
                      status={completedCount === datasetEntries.length ? 'success' : 'active'}
                    />
                  }
                  style={{ background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}
                >
                  <Space direction="vertical" size={8} style={{ width: '100%' }}>
                    {datasetEntries.map(([datasetName, data]) => {
                      const { progress, status } = data
                      const datasetDisplay = resolveDatasetDisplay(datasetName, datasetDisplays)
                      const color =
                        status === 'completed'
                          ? 'success'
                          : status === 'running'
                            ? 'processing'
                            : status === 'failed'
                              ? 'error'
                              : 'default'
                      const icon =
                        status === 'completed' ? (
                          <CheckCircleOutlined />
                        ) : status === 'running' ? (
                          <LoadingOutlined spin />
                        ) : status === 'failed' ? (
                          <CloseCircleOutlined />
                        ) : (
                          <ClockCircleOutlined />
                        )
                      const statusText =
                        status === 'completed'
                          ? t('detail.progressStatus.completed')
                          : status === 'running'
                            ? t('detail.progressStatus.running')
                            : status === 'failed'
                              ? t('detail.progressStatus.failed')
                              : t('detail.progressStatus.waiting')

                      return (
                        <div
                          key={datasetName}
                          style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                        >
                          <Tag color={color} icon={icon} style={{ margin: 0, minWidth: 70 }}>
                            {statusText}
                          </Tag>
                          <div style={{ minWidth: 140 }}>
                            {renderDatasetDisplay(datasetDisplay)}
                          </div>
                          <Progress
                            percent={Math.round(progress)}
                            size="small"
                            status={
                              status === 'running'
                                ? 'active'
                                : status === 'failed'
                                  ? 'exception'
                                  : undefined
                            }
                            style={{ flex: 1, margin: 0 }}
                          />
                        </div>
                      )
                    })}
                  </Space>
                </Card>
              )
            })}
          </Space>
        </>
      )}

      {/* Error Message for failed tasks */}
      {isFailed && task.error_message && (
        <Card
          size="small"
          style={{
            marginBottom: 16,
            background: '#2a1f1f',
            borderColor: STATUS_ERROR,
          }}
        >
          <Space direction="vertical" size={4}>
            <Text strong style={{ color: STATUS_ERROR }}>
              <CloseCircleOutlined /> {t('detail.errorInfo')}
            </Text>
            <Text style={{ color: TEXT_SECONDARY, whiteSpace: 'pre-wrap' }}>
              {task.error_message}
            </Text>
          </Space>
        </Card>
      )}

      {/* Model Summaries */}
      {modelSummaries.length > 0 && (
        <>
          <Divider orientation="left" style={{ marginTop: 0 }}>
            <Space>
              <ApiOutlined />
              {t('detail.modelSummary')}
            </Space>
          </Divider>
          <Space wrap style={{ marginBottom: 16 }}>
            {modelSummaries.map((model) => (
              <Card
                key={model.name}
                size="small"
                style={{ background: BG_ELEVATED, borderColor: BORDER_SECONDARY }}
              >
                <Space direction="vertical" size={4}>
                  <Text strong>{model.name}</Text>
                  <Space>
                    <Tag color="blue">
                      <DatabaseOutlined /> {t('detail.datasetsTag', { count: model.datasets })}
                    </Tag>
                    {typeof model.avgMrr === 'number' && Number.isFinite(model.avgMrr) && (
                      <Tag color="green">
                        {t('detail.avgMrr', { value: model.avgMrr.toFixed(4) })}
                      </Tag>
                    )}
                  </Space>
                </Space>
              </Card>
            ))}
          </Space>
        </>
      )}

      {/* Results Table - only show for completed tasks */}
      {(isSucceeded || isFailed) && (
        <>
          <Divider orientation="left">
            <Space>
              <DatabaseOutlined />
              {t('detail.detailedResults')}
            </Space>
          </Divider>

          {rows.length > 0 ? (
            <Table
              columns={columns}
              dataSource={rows}
              size="small"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => t('detail.totalResults', { total }),
              }}
              scroll={{ x: 'max-content' }}
            />
          ) : (
            <Empty description={isFailed ? t('detail.noResultsFailed') : t('detail.noResults')} />
          )}
        </>
      )}

      {/* Configuration Info */}
      <Divider orientation="left">
        <Space>
          <ClockCircleOutlined />
          {t('detail.configInfo')}
        </Space>
      </Divider>

      <Descriptions size="small" column={2}>
        <Descriptions.Item label={t('detail.model')}>
          <Space wrap size={4}>
            {task.model_configs?.map((mc, i) => <Tag key={i}>{mc.name || mc.endpoint}</Tag>) || '-'}
          </Space>
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.dataset')}>
          <Space wrap size={4}>
            {task.dataset_configs?.map((dc, i) => (
              <span key={`${canonicalDatasetKey(dc) || dc.name}-${i}`}>
                {renderDatasetDisplay(datasetDisplayFromConfig(dc))}
              </span>
            )) || '-'}
          </Space>
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.maxSamples')}>
          {task.max_samples || t('detail.allSamples')}
        </Descriptions.Item>
        <Descriptions.Item label={t('detail.batchSize')}>{task.batch_size || 50}</Descriptions.Item>
        <Descriptions.Item label={t('detail.apiWorkers')}>{task.workers || 8}</Descriptions.Item>
        <Descriptions.Item label={t('detail.modelWorkers')}>
          {task.model_workers || 2}
        </Descriptions.Item>
      </Descriptions>
    </Modal>
  )
}
