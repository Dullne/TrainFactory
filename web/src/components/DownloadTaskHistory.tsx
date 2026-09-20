import { Alert, Button, Empty, List, Progress, Space, Typography } from 'antd'
import { StatusTag } from '@/components/StatusTag'

const { Text, Title } = Typography

export interface DownloadTaskHistoryItem {
  id: string
  name: string
  source?: string
  status: string
  progress: number
  error?: string
}

interface DownloadTaskHistoryProps {
  items: DownloadTaskHistoryItem[]
  loading: boolean
  failed: boolean
  title: string
  emptyText: string
  errorText: string
  retryText: string
  onRetry: () => void
}

export function DownloadTaskHistory({
  items,
  loading,
  failed,
  title,
  emptyText,
  errorText,
  retryText,
  onRetry,
}: DownloadTaskHistoryProps) {
  return (
    <section data-testid="download-task-history" aria-label={title} style={{ marginTop: 20 }}>
      <Title level={5} style={{ marginBottom: 12 }}>
        {title}
      </Title>
      {failed ? (
        <Alert
          type="error"
          showIcon
          message={errorText}
          action={
            <Button size="small" onClick={onRetry}>
              {retryText}
            </Button>
          }
          style={{ marginBottom: 12 }}
        />
      ) : null}
      <List
        size="small"
        loading={loading}
        dataSource={items}
        locale={{
          emptyText: <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={emptyText} />,
        }}
        style={{ maxHeight: 240, overflowY: 'auto' }}
        renderItem={(item) => (
          <List.Item data-testid={`download-task-${item.id}`}>
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Space wrap>
                <Text strong>{item.name}</Text>
                <StatusTag status={item.status} showTooltip={false} />
              </Space>
              {item.source ? <Text type="secondary">{item.source}</Text> : null}
              <Progress
                percent={Math.max(0, Math.min(100, item.progress))}
                size="small"
                status={
                  item.status === 'failed'
                    ? 'exception'
                    : item.status === 'available' || item.status === 'ready'
                      ? 'success'
                      : 'active'
                }
              />
              {item.error ? <Text type="danger">{item.error}</Text> : null}
            </Space>
          </List.Item>
        )}
      />
    </section>
  )
}
