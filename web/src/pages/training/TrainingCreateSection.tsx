import { useId, type ReactNode } from 'react'
import { Card } from 'antd'
import { DownOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'

interface TrainingCreateSectionProps {
  sectionId: string
  title: string
  summary?: string
  collapsed: boolean
  onToggle: () => void
  children: ReactNode
  extra?: ReactNode
  className?: string
  hidden?: boolean
}

export function TrainingCreateSection({
  sectionId,
  title,
  summary,
  collapsed,
  onToggle,
  children,
  extra,
  className = '',
  hidden,
}: TrainingCreateSectionProps) {
  const contentId = useId()
  const { t } = useTranslation('training')

  return (
    <Card
      className={`training-create-section ${collapsed ? 'is-collapsed' : ''} ${className}`}
      data-training-section={sectionId}
      hidden={hidden}
      extra={extra}
      title={
        <h2>
          <button
            type="button"
            className="training-create-section-toggle"
            aria-label={title}
            aria-expanded={!collapsed}
            aria-controls={contentId}
            aria-describedby={collapsed && summary ? `${contentId}-summary` : undefined}
            onClick={onToggle}
          >
            <DownOutlined className="training-create-section-arrow" aria-hidden="true" />
            <span className="training-create-section-label">
              <span className="training-create-section-title">{title}</span>
              {collapsed && summary && (
                <span id={`${contentId}-summary`} className="training-create-section-summary">
                  {summary}
                </span>
              )}
            </span>
            <span className="training-create-section-hint" aria-hidden="true">
              {t(collapsed ? 'create.sections.expand' : 'create.sections.collapse')}
            </span>
          </button>
        </h2>
      }
    >
      {/* Keep fields mounted so folding never resets values or unregisters validation. */}
      <div id={contentId} className="training-create-section-content">
        {children}
      </div>
    </Card>
  )
}
