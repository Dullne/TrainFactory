import { Card, Statistic, Tooltip } from 'antd'
import type { CSSProperties, ReactNode } from 'react'
import { cardStyle, flexWithGap, coloredIconContainer } from '@/styles'

interface StatCardProps {
  title: string
  value: number | string
  icon?: ReactNode
  color?: string
  suffix?: string
  prefix?: ReactNode
  loading?: boolean
  tooltip?: string
}

const semanticColors: Record<string, string> = {
  '#58a6ff': 'var(--tf-status-info)',
  '#1677ff': 'var(--tf-status-info)',
  '#3fb950': 'var(--tf-status-success)',
  '#52c41a': 'var(--tf-status-success)',
  '#d29922': 'var(--tf-status-warning)',
  '#faad14': 'var(--tf-status-warning)',
  '#f85149': 'var(--tf-status-error)',
  '#ff4d4f': 'var(--tf-status-error)',
  '#8b949e': 'var(--tf-status-default)',
}

export function StatCard({
  title,
  value,
  icon,
  color = '#8b949e',
  suffix,
  prefix,
  loading = false,
  tooltip,
}: StatCardProps) {
  const accent = semanticColors[color.toLowerCase()] ?? color
  const card = (
    <Card
      className="stat-card"
      size="small"
      style={
        {
          ...cardStyle,
          '--tf-stat-accent': accent,
          backgroundColor: 'var(--tf-stat-bg)',
          borderRadius: 'var(--tf-radius)',
        } as CSSProperties
      }
      styles={{
        body: {
          padding: 'calc(var(--tf-card-padding) - 4px) var(--tf-card-padding)',
        },
      }}
    >
      <div className="stat-card-row" style={flexWithGap(12)}>
        {icon && (
          <div
            className="stat-card-icon"
            style={{ ...coloredIconContainer(accent), flexShrink: 0, fontSize: 18 }}
          >
            {icon}
          </div>
        )}
        <div className="stat-card-content">
          <div style={{ color: 'var(--tf-text-secondary)', fontSize: 12, marginBottom: 4 }}>
            {title}
          </div>
          <Statistic
            value={value}
            suffix={suffix}
            prefix={prefix}
            loading={loading}
            valueStyle={{
              fontSize: 24,
              fontWeight: 600,
              color: accent,
              lineHeight: 1,
            }}
          />
        </div>
      </div>
    </Card>
  )

  return tooltip ? <Tooltip title={tooltip}>{card}</Tooltip> : card
}

export default StatCard
