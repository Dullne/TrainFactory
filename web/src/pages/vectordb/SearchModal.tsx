import { useState, useCallback, useEffect } from 'react'
import {
  Modal,
  Input,
  InputNumber,
  Button,
  Table,
  Space,
  Typography,
  Tooltip,
  Radio,
  Select,
  Slider,
  message,
} from 'antd'
import { SearchOutlined, CopyOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { milvusApi } from '@/services/api'
import ModelConfigSelector from '@/components/ModelConfigSelector'
import type { MilvusSearchResult, MilvusSearchMode, MilvusRankerType, LinkedDatasetInfo } from '@/types'

const { Text } = Typography

interface Props {
  open: boolean
  collectionName: string
  hybridEnabled?: boolean
  metadataEnabled?: boolean
  linkedDatasets?: LinkedDatasetInfo[]
  onClose: () => void
}

export default function SearchModal({ open, collectionName, hybridEnabled = false, metadataEnabled = false, linkedDatasets = [], onClose }: Props) {
  const { t } = useTranslation(['vectordb', 'common'])
  const [queryText, setQueryText] = useState('')
  const [embeddingConfigId, setEmbeddingConfigId] = useState<string>('')
  const [topK, setTopK] = useState(10)
  const [searchMode, setSearchMode] = useState<MilvusSearchMode>('dense')
  const [ranker, setRanker] = useState<MilvusRankerType>('rrf')
  const [denseWeight, setDenseWeight] = useState(0.7)
  const [filterDatasetId, setFilterDatasetId] = useState<string | undefined>(undefined)
  const [filterExpr, setFilterExpr] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<MilvusSearchResult[]>([])
  const [searched, setSearched] = useState(false)

  const needsEmbedding = searchMode !== 'sparse'

  useEffect(() => {
    if (open) {
      setSearchMode('dense')
      setFilterDatasetId(undefined)
      setFilterExpr('')
    }
  }, [open, collectionName])

  useEffect(() => {
    if (!hybridEnabled && searchMode !== 'dense') {
      setSearchMode('dense')
    }
  }, [hybridEnabled, searchMode])

  const fallbackCopy = useCallback((text: string) => {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.style.position = 'fixed'
    ta.style.opacity = '0'
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
    message.success(t('common:message.copied'))
  }, [t])

  const copyToClipboard = useCallback((text: string) => {
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(text).then(
        () => message.success(t('common:message.copied')),
        () => fallbackCopy(text),
      )
    } else {
      fallbackCopy(text)
    }
  }, [fallbackCopy, t])

  const buildFilterExpr = useCallback((): string | undefined => {
    const parts: string[] = []
    if (filterDatasetId) {
      parts.push(`metadata["dataset_id"] == "${filterDatasetId}"`)
    }
    if (filterExpr.trim()) {
      parts.push(filterExpr.trim())
    }
    return parts.length > 0 ? parts.join(' && ') : undefined
  }, [filterDatasetId, filterExpr])

  const handleSearch = useCallback(async () => {
    if (!queryText.trim()) {
      message.warning(t('search.queryRequired'))
      return
    }
    if (needsEmbedding && !embeddingConfigId) {
      message.warning(t('search.embeddingRequired'))
      return
    }
    setLoading(true)
    try {
      const res = await milvusApi.search(collectionName, {
        query_text: queryText,
        embedding_config_id: needsEmbedding ? embeddingConfigId : undefined,
        top_k: topK,
        search_mode: searchMode,
        ranker: searchMode === 'hybrid' ? ranker : undefined,
        dense_weight: searchMode === 'hybrid' && ranker === 'weighted' ? denseWeight : undefined,
        sparse_weight: searchMode === 'hybrid' && ranker === 'weighted' ? (1 - denseWeight) : undefined,
        filter_expr: buildFilterExpr(),
      })
      setResults(res.results)
      setSearched(true)
    } catch (err: unknown) {
      message.error(t('search.searchFailed', { error: err instanceof Error ? err.message : String(err) }))
    } finally {
      setLoading(false)
    }
  }, [collectionName, queryText, embeddingConfigId, topK, searchMode, ranker, denseWeight, needsEmbedding, buildFilterExpr, t])

  const columns: ColumnsType<MilvusSearchResult> = [
    {
      title: '#',
      key: 'rank',
      width: 50,
      render: (_: unknown, __: unknown, index: number) => index + 1,
    },
    {
      title: 'chunk_id',
      dataIndex: 'chunk_id',
      width: 200,
      ellipsis: true,
      render: (v: string) => (
        <Text copyable={{ text: v }} style={{ fontSize: 12 }}>{v}</Text>
      ),
    },
    {
      title: t('search.contentColumn'),
      dataIndex: 'chunk_content',
      ellipsis: true,
      render: (v: string) => (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 4 }}>
          <Tooltip title={t('search.copyFullText')}>
            <Button
              type="text"
              size="small"
              icon={<CopyOutlined />}
              style={{ flexShrink: 0, marginTop: -2 }}
              onClick={() => copyToClipboard(v || '')}
            />
          </Tooltip>
          <Tooltip title={v?.slice(0, 500)} overlayStyle={{ maxWidth: 500 }}>
            <span style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {v?.slice(0, 100)}{v?.length > 100 ? '...' : ''}
            </span>
          </Tooltip>
        </div>
      ),
    },
    {
      title: t('search.similarity'),
      dataIndex: 'score',
      width: 100,
      render: (v: number) => (
        <Text strong style={{ color: v > 0.8 ? '#3fb950' : v > 0.5 ? '#d29922' : '#8b949e' }}>
          {(v * 100).toFixed(1)}%
        </Text>
      ),
    },
  ]

  return (
    <Modal
      title={t('search.title', { name: collectionName })}
      open={open}
      onCancel={() => { onClose(); setResults([]); setSearched(false) }}
      footer={null}
      width={900}
      destroyOnHidden
    >
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        <Input.TextArea
          rows={3}
          placeholder={t('search.queryPlaceholder')}
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
        />
        <div>
          <Space size={4} style={{ marginBottom: 8 }}>
            <span style={{ fontSize: 13 }}>{t('search.searchMode')}:</span>
            <Radio.Group value={searchMode} onChange={(e) => setSearchMode(e.target.value)} size="small">
              <Radio.Button value="dense">{t('search.modeDense')}</Radio.Button>
              <Radio.Button value="sparse" disabled={!hybridEnabled}>
                <Tooltip title={!hybridEnabled ? t('search.hybridNotSupported') : undefined}>
                  {t('search.modeSparse')}
                </Tooltip>
              </Radio.Button>
              <Radio.Button value="hybrid" disabled={!hybridEnabled}>
                <Tooltip title={!hybridEnabled ? t('search.hybridNotSupported') : undefined}>
                  {t('search.modeHybrid')}
                </Tooltip>
              </Radio.Button>
            </Radio.Group>
          </Space>
        </div>
        <Space wrap>
          {needsEmbedding && (
            <div style={{ width: 300 }}>
              <ModelConfigSelector
                modelType="embedding"
                mode="single"
                value={embeddingConfigId}
                onChange={(id) => setEmbeddingConfigId(id as string)}
              />
            </div>
          )}
          <InputNumber
            min={1}
            max={100}
            value={topK}
            onChange={(v) => setTopK(v || 10)}
            addonBefore="Top-K"
            style={{ width: 140 }}
          />
          <Button
            type="primary"
            icon={<SearchOutlined />}
            loading={loading}
            onClick={handleSearch}
          >
            {t('common:action.search')}
          </Button>
        </Space>
        {metadataEnabled && (
          <Space wrap align="center">
            <span style={{ fontSize: 13 }}>{t('search.filterByDataset')}:</span>
            <Select
              allowClear
              placeholder={t('search.filterByDatasetPlaceholder')}
              value={filterDatasetId}
              onChange={setFilterDatasetId}
              size="small"
              style={{ width: 200 }}
              options={linkedDatasets.map((ds) => ({
                value: ds.dataset_id,
                label: ds.dataset_name || ds.dataset_id.slice(0, 12),
              }))}
            />
            <span style={{ fontSize: 13 }}>{t('search.filterExpression')}:</span>
            <Input
              placeholder={t('search.filterExprPlaceholder')}
              value={filterExpr}
              onChange={(e) => setFilterExpr(e.target.value)}
              size="small"
              style={{ width: 280 }}
              allowClear
            />
          </Space>
        )}
        {searchMode === 'hybrid' && (
          <Space wrap align="center">
            <span style={{ fontSize: 13 }}>{t('search.ranker')}:</span>
            <Select
              value={ranker}
              onChange={setRanker}
              size="small"
              style={{ width: 140 }}
              options={[
                { value: 'rrf', label: t('search.rankerRrf') },
                { value: 'weighted', label: t('search.rankerWeighted') },
              ]}
            />
            {ranker === 'weighted' && (
              <>
                <span style={{ fontSize: 13 }}>{t('search.denseWeight')}:</span>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={denseWeight}
                  onChange={setDenseWeight}
                  style={{ width: 120 }}
                  tooltip={{ formatter: (v) => `${((v ?? 0) * 100).toFixed(0)}%` }}
                />
                <span style={{ fontSize: 12, color: '#888' }}>
                  {(denseWeight * 100).toFixed(0)}% / {((1 - denseWeight) * 100).toFixed(0)}%
                </span>
              </>
            )}
          </Space>
        )}
        {searched && (
          <Table
            dataSource={results}
            columns={columns}
            rowKey="chunk_id"
            size="small"
            pagination={false}
            locale={{ emptyText: t('search.noResults') }}
          />
        )}
      </Space>
    </Modal>
  )
}
