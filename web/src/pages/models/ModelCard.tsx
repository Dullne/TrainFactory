import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { Card, Button, Popconfirm, Tooltip, Typography, Progress, Col, message } from 'antd'
import {
  DeleteOutlined,
  RocketOutlined,
  AppstoreOutlined,
  NodeIndexOutlined,
  BranchesOutlined,
  CodeOutlined,
  RobotOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import type { RegisteredModel } from '@/types'
import { StatusTag } from '@/components/StatusTag'
import { formatDate, copyToClipboard } from '@/utils'
import {
  MODEL_TYPE_COLORS,
  BG_ELEVATED,
  BORDER_SECONDARY,
  TEXT_PRIMARY,
  TEXT_SECONDARY,
} from '@/theme'

const { Text } = Typography

// 模型类型图标映射
const modelTypeIcons: Record<string, React.ReactNode> = {
  embedding: <NodeIndexOutlined />,
  reranker: <BranchesOutlined />,
  decoder_reranker: <CodeOutlined />,
  llm: <RobotOutlined />,
}

// 模型类型 → i18n key 映射
const modelTypeI18nKeys: Record<string, string> = {
  embedding: 'common:modelType.embedding',
  reranker: 'common:modelType.reranker',
  decoder_reranker: 'common:modelType.decoderReranker',
  llm: 'common:modelType.llm',
}

interface ModelCardProps {
  model: RegisteredModel
  onDelete: (modelId: string) => void
  onDetail?: (model: RegisteredModel) => void
  baseModelMap?: Record<string, RegisteredModel>
}

export function ModelCard({ model, onDelete, onDetail, baseModelMap }: ModelCardProps) {
  const navigate = useNavigate()
  const { t } = useTranslation(['models', 'common'])
  const typeColor = MODEL_TYPE_COLORS[model.model_type] || '#8b949e'
  const metrics = model.metrics || {}
  const primaryMetric = Object.entries(metrics)[0]
  const displayName = model.model_name || model.display_name || model.model_id

  const pathSuffix = (p: string) => p.split('/').filter(Boolean).pop() || p
  const resolvedBase =
    model.base_model_path && baseModelMap
      ? baseModelMap[model.base_model_path] || baseModelMap[pathSuffix(model.base_model_path)]
      : undefined

  const sourceTypeLabels: Record<string, { label: string; color: string }> = {
    trained: { label: t('source.trained'), color: '#1677ff' },
    downloaded: { label: t('source.downloaded'), color: '#52c41a' },
    uploaded: { label: t('source.uploaded'), color: '#2f54eb' },
    external_bind: { label: t('source.externalBind'), color: '#fa8c16' },
  }

  return (
    <Col xs={24} sm={12} lg={8} xl={6}>
      <Card
        hoverable
        className="card-hover"
        onClick={() => onDetail?.(model)}
        style={{
          background: BG_ELEVATED,
          borderColor: BORDER_SECONDARY,
          borderRadius: 'var(--tf-radius-lg)',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          cursor: 'pointer',
        }}
        styles={{
          body: {
            padding: 'var(--tf-card-padding-sm)',
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
          },
        }}
      >
        {/* 头部：类型图标 + 名称 */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 12 }}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 'calc((var(--tf-radius) + var(--tf-radius-lg)) / 2)',
              backgroundColor: `${typeColor}20`,
              color: typeColor,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 22,
              flexShrink: 0,
            }}
          >
            {modelTypeIcons[model.model_type] || <AppstoreOutlined />}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <Tooltip title={displayName}>
                <Text
                  strong
                  style={{
                    color: TEXT_PRIMARY,
                    fontSize: 15,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: 160,
                  }}
                >
                  {displayName}
                </Text>
              </Tooltip>
              <Tooltip title={t('card.copyModelName')}>
                <CopyOutlined
                  style={{ color: TEXT_SECONDARY, cursor: 'pointer', fontSize: 12, flexShrink: 0 }}
                  onClick={(e) => {
                    e.stopPropagation()
                    copyToClipboard(displayName)
                    message.success(t('card.copiedModelName'))
                  }}
                />
              </Tooltip>
            </div>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12, display: 'block' }}>
              {modelTypeI18nKeys[model.model_type]
                ? t(modelTypeI18nKeys[model.model_type])
                : model.model_type}
            </Text>
            <Tooltip title={model.model_id}>
              <Text
                copyable={{ text: model.model_id, tooltips: false }}
                style={{ color: TEXT_SECONDARY, fontSize: 11, display: 'block' }}
              >
                {model.model_id.slice(0, 8)}
              </Text>
            </Tooltip>
          </div>
          <StatusTag status={model.status} />
        </div>

        {/* 基础模型 */}
        {model.base_model_path && (
          <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12, flexShrink: 0 }}>
              {t('card.baseModel')}
            </Text>
            {resolvedBase ? (
              <Tooltip title={model.base_model_path}>
                <Text
                  style={{
                    color: typeColor,
                    fontSize: 12,
                    maxWidth: 150,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    display: 'inline-block',
                    verticalAlign: 'bottom',
                    cursor: 'pointer',
                  }}
                  onClick={(e) => {
                    e.stopPropagation()
                    onDetail?.(resolvedBase)
                  }}
                >
                  {resolvedBase.model_name || resolvedBase.display_name || resolvedBase.model_id}
                </Text>
              </Tooltip>
            ) : (
              <Tooltip title={model.base_model_path}>
                <Text
                  style={{
                    color: TEXT_PRIMARY,
                    fontSize: 12,
                    maxWidth: 150,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    display: 'inline-block',
                    verticalAlign: 'bottom',
                  }}
                >
                  {pathSuffix(model.base_model_path)}
                </Text>
              </Tooltip>
            )}
          </div>
        )}

        {/* 来源与绑定信息 */}
        {model.source_type && (
          <div style={{ marginBottom: 8 }}>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{t('card.source')}</Text>
            <span style={{ marginLeft: 6 }}>
              <Tooltip
                title={
                  model.extra_metadata ? JSON.stringify(model.extra_metadata, null, 2) : undefined
                }
              >
                <span>
                  <Text
                    style={{
                      color: sourceTypeLabels[model.source_type]?.color ?? TEXT_PRIMARY,
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    {sourceTypeLabels[model.source_type]?.label ?? model.source_type}
                  </Text>
                </span>
              </Tooltip>
            </span>
          </div>
        )}

        {Boolean(model.extra_metadata?.bind_endpoint) && (
          <div style={{ marginBottom: 8 }}>
            <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{t('card.bindEndpoint')}</Text>
            <Tooltip title={String(model.extra_metadata?.bind_endpoint)}>
              <Text
                style={{
                  color: TEXT_PRIMARY,
                  fontSize: 12,
                  marginLeft: 4,
                  maxWidth: 180,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  display: 'inline-block',
                  verticalAlign: 'bottom',
                }}
              >
                {String(model.extra_metadata?.bind_endpoint)}
              </Text>
            </Tooltip>
          </div>
        )}

        {/* 指标展示 */}
        {primaryMetric && typeof primaryMetric[1] === 'number' && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <Text style={{ color: TEXT_SECONDARY, fontSize: 12 }}>{primaryMetric[0]}</Text>
              <Text style={{ color: typeColor, fontSize: 12, fontWeight: 600 }}>
                {primaryMetric[1].toFixed(4)}
              </Text>
            </div>
            <Progress
              percent={Math.min(primaryMetric[1] * 100, 100)}
              showInfo={false}
              strokeColor={typeColor}
              trailColor={`${typeColor}20`}
              size="small"
            />
          </div>
        )}

        {/* 创建时间 */}
        <div style={{ marginTop: 'auto', marginBottom: 12 }}>
          <Text style={{ color: TEXT_SECONDARY, fontSize: 11 }}>
            {t('card.createdAt', {
              date: formatDate(model.created_at, {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
              }),
            })}
          </Text>
        </div>

        {/* 操作按钮 */}
        <div style={{ display: 'flex', gap: 8 }}>
          {model.status === 'available' && (
            <Button
              type="primary"
              size="small"
              icon={<RocketOutlined />}
              onClick={() => navigate(`/deployments?model_id=${model.model_id}`)}
              style={{ flex: 1 }}
            >
              {t('list.deploy')}
            </Button>
          )}
          <Popconfirm title={t('list.confirmDelete')} onConfirm={() => onDelete(model.model_id)}>
            <Tooltip title={t('common:action.delete')}>
              <Button size="small" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </div>
      </Card>
    </Col>
  )
}

export { modelTypeIcons, modelTypeI18nKeys }
