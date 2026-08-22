import { Tag, Tooltip } from 'antd'
import { useTranslation } from 'react-i18next'
import { getStatusMeta } from '@/utils'

interface StatusTagProps {
  status: string
  text?: string
  description?: string
  showTooltip?: boolean
}

export function StatusTag({
  status,
  text,
  description,
  showTooltip = true,
}: StatusTagProps) {
  const { t } = useTranslation('common')
  const meta = getStatusMeta(status)
  const color = meta.color
  const displayText = text || t(`status.${status}`, { defaultValue: status })
  const tooltipDescription = description ?? meta.description

  const tag = (
    <Tag
      style={{
        backgroundColor: `${color}20`,
        color: color,
        border: `1px solid ${color}40`,
        borderRadius: 4,
      }}
    >
      <span
        style={{
          display: 'inline-block',
          width: 6,
          height: 6,
          borderRadius: '50%',
          backgroundColor: color,
          marginRight: 6,
          animation: meta.isAnimated
            ? 'pulse 2s infinite'
            : 'none',
        }}
      />
      {displayText}
    </Tag>
  )

  if (!showTooltip || !tooltipDescription) {
    return tag
  }

  return (
    <Tooltip
      title={
        <div style={{ maxWidth: 280 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{displayText}</div>
          <div>{tooltipDescription}</div>
        </div>
      }
    >
      {tag}
    </Tooltip>
  )
}

export default StatusTag
