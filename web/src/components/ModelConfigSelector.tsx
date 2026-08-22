import { useState, useEffect, useCallback } from 'react'
import { Select, Space, Tag, Spin, Typography } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined, CloudOutlined, DesktopOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { configApi } from '@/services/api'
import type { ModelConfig } from '@/types'

const { Text } = Typography

interface ModelConfigSelectorBaseProps {
  /** 模型类型: llm, embedding, rerank。不传则显示所有类型 */
  modelType?: 'llm' | 'embedding' | 'rerank'
  /** 是否允许清空 */
  allowClear?: boolean
  /** 占位文本 */
  placeholder?: string
  /** 是否禁用 */
  disabled?: boolean
  /** 样式 */
  style?: React.CSSProperties
}

interface ModelConfigSelectorSingleProps extends ModelConfigSelectorBaseProps {
  mode: 'single'
  value?: string
  onChange?: (configId: string | null, config: ModelConfig | null) => void
}

interface ModelConfigSelectorMultipleProps extends ModelConfigSelectorBaseProps {
  mode: 'multiple'
  value?: string[]
  onChange?: (configIds: string[], configs: ModelConfig[]) => void
}

type ModelConfigSelectorProps =
  | ModelConfigSelectorSingleProps
  | ModelConfigSelectorMultipleProps

const modelTypeI18nMap: Record<string, string> = {
  llm: 'modelType.llm',
  embedding: 'modelType.embedding',
  rerank: 'modelType.reranker',
}

export default function ModelConfigSelector({
  modelType,
  value,
  mode,
  onChange,
  allowClear = false,
  placeholder,
  disabled = false,
  style,
}: ModelConfigSelectorProps) {
  const { t } = useTranslation('common')
  const [configs, setConfigs] = useState<ModelConfig[]>([])
  const [loading, setLoading] = useState(false)
  const [configMap, setConfigMap] = useState<Record<string, ModelConfig>>({})

  const fetchConfigs = useCallback(async () => {
    setLoading(true)
    try {
      // 获取所有配置，前端按 model_type 过滤
      const res = await configApi.list({ page: 1, page_size: 200 })
      const filtered = (res.items || []).filter((c) => {
        if (c.status !== 'active') return false
        if (!modelType) return true
        // Handle rerank/reranker alias: frontend uses 'rerank', backend may have 'reranker' (legacy)
        if (modelType === 'rerank') {
          return c.model_type === 'rerank' || c.model_type === 'reranker'
        }
        return c.model_type === modelType
      })
      setConfigs(filtered)
      // 构建 id -> config 映射
      const map: Record<string, ModelConfig> = {}
      filtered.forEach((c) => {
        map[c.config_id] = c
      })
      setConfigMap(map)
    } catch (error) {
      console.error('Failed to load model configs:', error)
    } finally {
      setLoading(false)
    }
  }, [modelType])

  useEffect(() => {
    fetchConfigs()
  }, [fetchConfigs])

  const handleChange = (selected?: string | string[]) => {
    if (mode === 'multiple') {
      const selectedIds = Array.isArray(selected) ? selected : []
      const configs = selectedIds.map((id) => configMap[id]).filter(Boolean)
      onChange?.(selectedIds, configs)
      return
    }
    const selectedId = Array.isArray(selected) ? selected[0] : selected
    if (!selectedId) {
      onChange?.(null, null)
      return
    }
    const config = configMap[selectedId]
    if (config) {
      onChange?.(selectedId, config)
    }
  }

  const renderOption = (config: ModelConfig) => {
    const isHealthy = config.last_check_status === 'healthy'
    const isLocalDeployed = config.source_type === 'local_deployed'

    return (
      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
        <Space>
          {isLocalDeployed ? (
            <DesktopOutlined style={{ color: '#1890ff' }} />
          ) : (
            <CloudOutlined style={{ color: '#52c41a' }} />
          )}
          <span>{config.config_name}</span>
          {config.model_name && (
            <Text type="secondary" style={{ fontSize: 12 }}>
              ({config.model_name})
            </Text>
          )}
        </Space>
        <Space>
          {config.last_check_status && (
            <Tag
              color={isHealthy ? 'success' : 'error'}
              icon={isHealthy ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
            >
              {isHealthy ? t('modelConfig.healthy') : t('modelConfig.unhealthy')}
            </Tag>
          )}
        </Space>
      </Space>
    )
  }

  return (
    <Select
      value={value}
      onChange={handleChange}
      allowClear={allowClear}
      mode={mode === 'multiple' ? 'multiple' : undefined}
      placeholder={placeholder || t('modelConfig.selectModel', { type: modelType ? ` ${t(modelTypeI18nMap[modelType])}` : '' })}
      disabled={disabled}
      loading={loading}
      style={{ width: '100%', ...style }}
      notFoundContent={loading ? <Spin size="small" /> : t('modelConfig.noAvailableConfig', { type: modelType ? ` ${t(modelTypeI18nMap[modelType])}` : '' })}
      optionLabelProp="label"
    >
      {configs.map((config) => (
        <Select.Option
          key={config.config_id}
          value={config.config_id}
          label={config.config_name}
        >
          {renderOption(config)}
        </Select.Option>
      ))}
    </Select>
  )
}
