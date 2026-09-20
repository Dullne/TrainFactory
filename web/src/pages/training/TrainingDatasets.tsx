import { Card, Space, Tag, Table, Typography } from 'antd'
import { DatabaseOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import type { TrainingTask } from '@/types'

const { Text } = Typography

export default function TrainingDatasets({ task }: { task: TrainingTask }) {
  const { t } = useTranslation('training')
  const datasets =
    task.dataset_configs && task.dataset_configs.length > 0
      ? task.dataset_configs
      : task.train_dataset_path
        ? [{ path: task.train_dataset_path, split: 'train', max_samples: null }]
        : []

  if (datasets.length === 0) return null

  const splitColorMap: Record<string, string> = { train: 'green', eval: 'blue', test: 'orange' }
  const splitLabelMap: Record<string, string> = {
    train: t('detail.datasets.splitTrain'),
    eval: t('detail.datasets.splitEval'),
    test: t('detail.datasets.splitTest'),
  }

  const hasNumRows = datasets.some((ds: Record<string, unknown>) => ds.num_rows != null)

  const dsColumns = [
    {
      title: t('detail.datasets.colSplit'),
      dataIndex: 'split',
      key: 'split',
      width: 90,
      render: (split: string) => (
        <Tag color={splitColorMap[split] || 'default'}>{splitLabelMap[split] || split}</Tag>
      ),
    },
    {
      title: t('detail.datasets.colPath'),
      dataIndex: 'path',
      key: 'path',
      ellipsis: true,
      render: (path: string) => (
        <Text code style={{ fontSize: 12 }}>
          {path}
        </Text>
      ),
    },
    ...(hasNumRows
      ? [
          {
            title: t('detail.datasets.colNumRows'),
            dataIndex: 'num_rows',
            key: 'num_rows',
            width: 100,
            align: 'right' as const,
            render: (val: number | undefined) => (
              <span style={{ fontSize: 12 }}>{val != null ? val.toLocaleString() : '-'}</span>
            ),
          },
        ]
      : []),
    {
      title: t('detail.datasets.colMaxSamples'),
      dataIndex: 'max_samples',
      key: 'max_samples',
      width: 100,
      align: 'right' as const,
      render: (val: number | null | undefined) => (
        <span style={{ fontSize: 12 }}>
          {val != null ? val.toLocaleString() : t('detail.datasets.allData')}
        </span>
      ),
    },
  ]

  return (
    <Card
      title={
        <Space>
          <DatabaseOutlined />
          <span>{t('detail.datasets.title')}</span>
          <Tag color="blue">{datasets.length}</Tag>
        </Space>
      }
      size="small"
      style={{ marginBottom: 16 }}
    >
      <Table
        dataSource={datasets.map((ds: Record<string, unknown>, i: number) => ({ ...ds, key: i }))}
        columns={dsColumns}
        scroll={{ x: 480 }}
        pagination={false}
        size="small"
        showHeader={true}
      />
    </Card>
  )
}
