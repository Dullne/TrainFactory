import { Card, Statistic, Tooltip } from 'antd'
import type { ReactNode } from 'react'
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
  const card = (
    <Card
      size="small"
      style={{ ...cardStyle, borderRadius: 8 }}
      styles={{
        body: {
          padding: '16px 20px',
        }
      }}
    >
      <div style={flexWithGap(12)}>
        {icon && (
          <div style={{ ...coloredIconContainer(color), fontSize: 18 }}>
            {icon}
          </div>
        )}
        <div>
          <div style={{ color: '#8b949e', fontSize: 12, marginBottom: 4 }}>
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
              color: color,
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
