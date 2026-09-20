import { Tag, Tooltip } from 'antd'
import { useTranslation } from 'react-i18next'
import { getStatusMeta } from '@/utils'

interface StatusTagProps {
  status: string
  text?: string
  description?: string
  showTooltip?: boolean
}

export function StatusTag({ status, text, description, showTooltip = true }: StatusTagProps) {
  const { t } = useTranslation('common')
  const meta = getStatusMeta(status)
  const tone = meta.antdColor === 'processing' ? 'info' : meta.antdColor
  const displayText = text || t(`status.${status}`, { defaultValue: status })
  const tooltipDescription = description ?? meta.description

  const tag = (
    <Tag className="status-tag" style={{ color: `var(--tf-status-${tone})` }}>
      <span
        className="status-tag-dot"
        style={{ animation: meta.isAnimated ? 'pulse 2s infinite' : 'none' }}
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
