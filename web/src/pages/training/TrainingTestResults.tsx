import { useState } from 'react'
import { Card, Space, Tag, Table, Typography, Row, Col, Button } from 'antd'
import { DownOutlined, UpOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import type { TrainingTask } from '@/types'

const { Text } = Typography

export default function TrainingTestResults({ task }: { task: TrainingTask }) {
  const { t } = useTranslation('training')
  const [showAllMetrics, setShowAllMetrics] = useState(false)
  const testBefore = task.final_metrics!.test_before as Record<string, number> | undefined
  const testAfter = task.final_metrics!.test_after as Record<string, number> | undefined
  const testDelta = task.final_metrics!.test_delta as Record<string, number> | undefined

  // Get all metrics, filter out num_samples
  const allMetricKeys = Array.from(
    new Set([...Object.keys(testBefore || {}), ...Object.keys(testAfter || {})])
  ).filter((k) => k !== 'num_samples')

  // Core metrics (without _std)
  const coreMetrics = allMetricKeys.filter((k) => !k.endsWith('_std'))
  // Standard deviation metrics
  const stdMetrics = allMetricKeys.filter((k) => k.endsWith('_std'))

  const displayMetrics = showAllMetrics ? allMetricKeys : coreMetrics

  // Split into two columns for display
  const midIndex = Math.ceil(displayMetrics.length / 2)
  const leftMetrics = displayMetrics.slice(0, midIndex)
  const rightMetrics = displayMetrics.slice(midIndex)

  const createDataSource = (keys: string[]) =>
    keys.map((key) => ({
      key,
      metric: key,
      before: testBefore?.[key],
      after: testAfter?.[key],
      delta: testDelta?.[key],
    }))

  const columns = [
    {
      title: t('detail.testResult.columns.metric'),
      dataIndex: 'metric',
      key: 'metric',
      width: 120,
      render: (text: string) => (
        <Text strong style={{ fontSize: 12 }}>
          {text}
        </Text>
      ),
    },
    {
      title: t('detail.testResult.columns.before'),
      dataIndex: 'before',
      key: 'before',
      width: 90,
      align: 'right' as const,
      render: (val: number | undefined) => (
        <span style={{ fontSize: 12 }}>{typeof val === 'number' ? val.toFixed(4) : '-'}</span>
      ),
    },
    {
      title: t('detail.testResult.columns.after'),
      dataIndex: 'after',
      key: 'after',
      width: 90,
      align: 'right' as const,
      render: (val: number | undefined) => (
        <span style={{ fontSize: 12 }}>{typeof val === 'number' ? val.toFixed(4) : '-'}</span>
      ),
    },
    {
      title: t('detail.testResult.columns.delta'),
      dataIndex: 'delta',
      key: 'delta',
      width: 90,
      align: 'right' as const,
      render: (val: number | undefined) => {
        if (typeof val !== 'number') return <span style={{ fontSize: 12 }}>-</span>
        const color =
          val > 0 ? 'var(--tf-status-success)' : val < 0 ? 'var(--tf-status-error)' : undefined
        const prefix = val > 0 ? '+' : ''
        return (
          <span style={{ color, fontSize: 12 }}>
            {prefix}
            {val.toFixed(4)}
          </span>
        )
      },
    },
  ]

  return (
    <Card
      title={
        <Space wrap>
          <span>
            {testBefore?.num_samples
              ? t('detail.testResult.titleWithSamples', { count: testBefore.num_samples })
              : t('detail.testResult.title')}
          </span>
          <Tag color="blue">
            {t('detail.testResult.metricCount', { count: coreMetrics.length })}
          </Tag>
        </Space>
      }
      size="small"
      style={{ marginTop: 16 }}
      extra={
        stdMetrics.length > 0 && (
          <Button
            type="link"
            size="small"
            icon={showAllMetrics ? <UpOutlined /> : <DownOutlined />}
            onClick={() => setShowAllMetrics(!showAllMetrics)}
          >
            {showAllMetrics
              ? t('detail.testResult.hideStd')
              : t('detail.testResult.showStd', { count: stdMetrics.length })}
          </Button>
        )
      }
    >
      <Row gutter={[24, 16]}>
        <Col xs={24} lg={12}>
          <Table
            dataSource={createDataSource(leftMetrics)}
            scroll={{ x: 400 }}
            columns={columns}
            pagination={false}
            size="small"
            showHeader={true}
          />
        </Col>
        <Col xs={24} lg={12}>
          <Table
            dataSource={createDataSource(rightMetrics)}
            scroll={{ x: 400 }}
            columns={columns}
            pagination={false}
            size="small"
            showHeader={true}
          />
        </Col>
      </Row>
    </Card>
  )
}
