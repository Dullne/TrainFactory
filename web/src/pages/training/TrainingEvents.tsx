import { useState } from 'react'
import { Alert, Button, Card, Empty, Spin, Typography } from 'antd'
import { useTranslation } from 'react-i18next'
import type { TrainingTaskEvent } from '@/types'
import { formatDate, getStatusText } from '@/utils'

const EVENT_TYPES = new Set([
  'task_created',
  'status_changed',
  'progress_updated',
  'result_updated',
  'model_registered',
  'task_completed',
  'task_deleted',
  'metrics_updated',
  'output_dir_updated',
])

interface TrainingEventsProps {
  events: TrainingTaskEvent[]
  loading: boolean
  failed: boolean
  limit: number
  onLoadOlder: () => Promise<boolean>
  onRetry: () => Promise<boolean>
}

export default function TrainingEvents({
  events,
  loading,
  failed,
  limit,
  onLoadOlder,
  onRetry,
}: TrainingEventsProps) {
  const { t } = useTranslation(['trainingDetailExtras', 'training'])
  const [visibleCount, setVisibleCount] = useState(10)
  const hasLoadedMore = visibleCount < events.length
  const canLoadOlder = events.length >= limit && limit < 500
  const visibleEvents = events.slice(0, visibleCount)

  const showMore = async () => {
    if (hasLoadedMore || (await onLoadOlder())) setVisibleCount((count) => count + 10)
  }

  return (
    <Card title={t('training:detail.taskEvents')} size="small" style={{ marginBottom: 16 }}>
      {failed && <Alert type="warning" showIcon message={t('events.failed')} />}
      <div
        className="training-events-scroll"
        aria-busy={loading}
        role="region"
        aria-label={t('training:detail.taskEvents')}
        tabIndex={0}
      >
        {events.length ? (
          <ol className="training-events-list">
            {visibleEvents.map((event) => {
              const from = event.payload?.from
              const to = event.payload?.to
              return (
                <li key={event.event_id}>
                  <Typography.Text strong>
                    {EVENT_TYPES.has(event.event_type)
                      ? t(`events.${event.event_type}`)
                      : t('events.unknown', { type: event.event_type })}
                  </Typography.Text>
                  {event.event_type === 'status_changed' &&
                    typeof from === 'string' &&
                    typeof to === 'string' && (
                      <Typography.Text>
                        {t('events.transition', {
                          from: getStatusText(from),
                          to: getStatusText(to),
                        })}
                      </Typography.Text>
                    )}
                  <time dateTime={event.created_at}>
                    {event.created_at ? formatDate(event.created_at) : '-'}
                  </time>
                </li>
              )
            })}
          </ol>
        ) : loading ? (
          <Spin size="small" />
        ) : !failed ? (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={t('training:detail.noEvents')} />
        ) : null}
      </div>
      {events.length > 0 && (
        <Typography.Text className="training-events-count" type="secondary" aria-live="polite">
          {t('events.shown', { visible: visibleEvents.length, loaded: events.length })}
        </Typography.Text>
      )}
      {hasLoadedMore || canLoadOlder ? (
        <Button block onClick={() => void showMore()} loading={loading}>
          {t(hasLoadedMore ? 'events.showMore' : 'events.loadOlder')}
        </Button>
      ) : failed ? (
        <Button block onClick={() => void onRetry()} loading={loading}>
          {t('events.retry')}
        </Button>
      ) : null}
      {events.length >= 500 && (
        <Typography.Paragraph type="secondary" className="training-events-limit">
          {t('events.limit')}
        </Typography.Paragraph>
      )}
    </Card>
  )
}
