import { Card, Button, Space, Tag, Popconfirm, Tooltip, Typography, message } from 'antd'
import {
  EditOutlined,
  DeleteOutlined,
  ApiOutlined,
  CopyOutlined,
  CodeOutlined,
} from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import type { ModelConfig } from '@/types'
import { copyToClipboard } from '@/utils'
import {
  BG_ELEVATED,
  BORDER_SECONDARY,
  TEXT_PRIMARY,
  TEXT_SECONDARY,
  STATUS_SUCCESS,
  STATUS_ERROR,
  STATUS_DEFAULT,
} from '@/theme'

const { Text } = Typography

interface ConfigCardProps {
  config: ModelConfig
  connectionStatus: boolean | null
  checking: boolean
  grouped?: boolean // 在端点折叠组内时隐藏重复的端点和框架信息
  onEdit: (config: ModelConfig) => void
  onDelete: (configId: string) => void
  onCheckConnection: (configId: string) => void
  onApiTest: (config: ModelConfig) => void
}

export function ConfigCard({
  config,
  connectionStatus,
  checking,
  grouped,
  onEdit,
  onDelete,
  onCheckConnection,
  onApiTest,
}: ConfigCardProps) {
  const { t } = useTranslation(['configs', 'common'])

  const handleCopy = (text: string) => {
    copyToClipboard(text)
    message.success(t('card.copiedToClipboard'))
  }

  return (
    <Card
      size="small"
      style={{
        marginBottom: 12,
        background: BG_ELEVATED,
        borderColor: BORDER_SECONDARY,
        borderRadius: 'var(--tf-radius)',
      }}
      styles={{
        body: { padding: 'calc(var(--tf-card-padding-sm) * 0.75) var(--tf-card-padding-sm)' },
      }}
    >
      <div className="config-card-row">
        {/* 左侧：状态指示器 + 配置名称 + 端点 */}
        <div className="config-card-info">
          {/* 状态指示器 - 优先使用实时检测结果，否则使用 last_check_status */}
          <Tooltip
            title={
              connectionStatus === true
                ? t('card.connectionOk')
                : connectionStatus === false
                  ? t('card.connectionFailed')
                  : config.last_check_status === 'healthy'
                    ? t('card.connectionOk')
                    : config.last_check_status === 'error'
                      ? t('card.connectionFailedWithError', {
                          error: config.last_check_error || t('card.unknownError'),
                        })
                      : t('card.notTested')
            }
          >
            <div
              style={{
                width: 8,
                height: 8,
                flexShrink: 0,
                borderRadius: '50%',
                backgroundColor:
                  connectionStatus === true
                    ? STATUS_SUCCESS
                    : connectionStatus === false
                      ? STATUS_ERROR
                      : config.last_check_status === 'healthy'
                        ? STATUS_SUCCESS
                        : config.last_check_status === 'error'
                          ? STATUS_ERROR
                          : STATUS_DEFAULT,
              }}
            />
          </Tooltip>

          {/* 配置名称 + 模型类型 + 框架 */}
          <div className="config-card-name">
            <Text strong style={{ color: TEXT_PRIMARY, fontSize: 14 }}>
              {config.config_name}
            </Text>
            <Tag color="blue" style={{ fontSize: 11, margin: 0 }}>
              {config.model_type}
            </Tag>
            {config.inference_framework && !grouped && (
              <Tag
                color={
                  config.inference_framework === 'sglang'
                    ? 'purple'
                    : config.inference_framework === 'vllm'
                      ? 'orange'
                      : 'cyan'
                }
                style={{ fontSize: 11, margin: 0 }}
              >
                {config.inference_framework.toUpperCase()}
              </Tag>
            )}
          </div>
          <Tooltip title={config.config_id}>
            <Text
              copyable={{
                text: config.config_id,
                tooltips: false,
                onCopy: () => message.success(t('card.copiedToClipboard')),
              }}
              style={{ color: TEXT_SECONDARY, fontSize: 11 }}
            >
              {config.config_id.slice(0, 8)}
            </Text>
          </Tooltip>

          {/* 端点信息 - 紧跟在名称后面 */}
          {!grouped && (
            <div className="config-card-endpoint">
              <Tooltip title={config.api_endpoint}>
                <Text
                  style={{
                    color: TEXT_SECONDARY,
                    fontSize: 12,
                    maxWidth: 300,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {config.api_endpoint}
                </Text>
              </Tooltip>
              <Button
                type="text"
                size="small"
                icon={<CopyOutlined />}
                aria-label={t('common:action.copy')}
                onClick={() => handleCopy(config.api_endpoint)}
                style={{ marginLeft: 4 }}
              />
            </div>
          )}
        </div>

        {/* 右侧：状态 + 操作按钮 */}
        <div className="config-card-actions">
          {/* 状态标签 */}
          <Tag
            color={
              config.status === 'active'
                ? 'success'
                : config.status === 'error'
                  ? 'error'
                  : 'default'
            }
            style={{ margin: 0 }}
          >
            {config.status === 'active'
              ? t('card.statusActive')
              : config.status === 'error'
                ? t('card.statusError')
                : t('card.statusUnknown')}
          </Tag>

          {/* 操作按钮 */}
          <Space size={4}>
            <Tooltip title={t('card.apiTest')}>
              <Button
                type="text"
                size="small"
                icon={<CodeOutlined />}
                aria-label={t('card.apiTest')}
                onClick={() => onApiTest(config)}
              />
            </Tooltip>
            <Tooltip title={t('card.testConnection')}>
              <Button
                type="text"
                size="small"
                icon={<ApiOutlined />}
                aria-label={t('card.testConnection')}
                loading={checking}
                onClick={() => onCheckConnection(config.config_id)}
              />
            </Tooltip>
            <Tooltip title={t('common:action.edit')}>
              <Button
                type="text"
                size="small"
                icon={<EditOutlined />}
                aria-label={t('common:action.edit')}
                onClick={() => onEdit(config)}
              />
            </Tooltip>
            <Popconfirm
              title={t('card.deleteConfirm')}
              onConfirm={() => onDelete(config.config_id)}
            >
              <Tooltip title={t('common:action.delete')}>
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                  aria-label={t('common:action.delete')}
                />
              </Tooltip>
            </Popconfirm>
          </Space>
        </div>
      </div>
    </Card>
  )
}
