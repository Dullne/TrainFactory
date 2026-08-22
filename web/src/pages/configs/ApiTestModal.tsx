import { useState, useEffect, useMemo, useRef } from 'react'
import {
  Modal,
  Tabs,
  Input,
  InputNumber,
  Button,
  Space,
  Typography,
  Alert,
  Spin,
  message,
  Radio,
  Select,
} from 'antd'
import { PlayCircleOutlined, CopyOutlined, CodeOutlined, WarningOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import type { ModelConfig, LoadedAdapter } from '@/types'
import { configApi, adapterApi } from '@/services/api'
import { copyToClipboard as copyText } from '@/utils'

const { TextArea } = Input
const { Text, Title } = Typography
const MAX_LLM_TOKENS = 131072

interface ApiTestModalProps {
  visible: boolean
  config: ModelConfig | null
  onClose: () => void
  onSuccess?: () => void | Promise<void>
}

interface DefaultTestDataOptions {
  modelName?: string
  includeModelInRequest?: boolean
}

const supportsRuntimeModelDiscovery = (config: ModelConfig | null) => {
  if (!config) return false
  const framework = config.inference_framework?.toLowerCase()
  return (
    config.provider === 'xinference' ||
    framework === 'xinference' ||
    framework === 'vllm' ||
    framework === 'sglang'
  )
}

// Default test data for different model types
// Xinference 示例请求体显式包含 model，其他类型仍在发送前自动补全
// cURL examples include model field for copy-paste convenience
const getDefaultTestData = (config: ModelConfig, options: DefaultTestDataOptions = {}) => {
  const modelType = config.model_type
  const framework = config.inference_framework?.toLowerCase()
  const isVllm = framework === 'vllm'
  const isXinference = config.provider === 'xinference' || framework === 'xinference'
  const modelName = options.modelName || config.model_name || 'model-name'
  const includeModelInRequest = options.includeModelInRequest ?? isXinference

  switch (modelType) {
    case 'embedding': {
      const requestObj: Record<string, unknown> = {
        input: ['Hello, how are you?', 'This is a test document.'],
      }
      if (includeModelInRequest) {
        requestObj.model = modelName
      }
      const curlModel = isVllm || isXinference ? `\n    "model": "${modelName}",` : ''
      return {
        request: JSON.stringify(requestObj, null, 2),
        curlExample: (endpoint: string) => `curl -X POST "${endpoint}/v1/embeddings" \\
  -H "Content-Type: application/json" \\
  -d '{${curlModel}
    "input": ["Hello, how are you?", "This is a test document."]
  }'`,
      }
    }
    case 'reranker':
    case 'rerank': {
      const requestObj: Record<string, unknown> = {
        query: 'What is machine learning?',
        documents: [
          'Machine learning is a subset of artificial intelligence.',
          'Python is a popular programming language.',
          'Deep learning uses neural networks.',
        ],
        top_n: 3,
      }
      if (includeModelInRequest) {
        requestObj.model = modelName
      }
      const curlModel = isVllm || isXinference ? `\n    "model": "${modelName}",` : ''
      return {
        request: JSON.stringify(requestObj, null, 2),
        curlExample: (endpoint: string) => `curl -X POST "${endpoint}/v1/rerank" \\
  -H "Content-Type: application/json" \\
  -d '{${curlModel}
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of artificial intelligence.",
      "Python is a popular programming language.",
      "Deep learning uses neural networks."
    ],
    "top_n": 3
  }'`,
      }
    }
    case 'llm': {
      // LLM always needs model in request body (OpenAI compatible)
      const requestObj: Record<string, unknown> = {
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Hello!' },
        ],
        max_tokens: 100,
      }
      if (includeModelInRequest) {
        requestObj.model = modelName
      }
      return {
        request: JSON.stringify(requestObj, null, 2),
        curlExample: (endpoint: string) => `curl -X POST "${endpoint}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'`,
      }
    }
    default:
      return {
        request: '{}',
        curlExample: () => '# Unknown model type',
      }
  }
}

const getApiPath = (modelType: string) => {
  switch (modelType) {
    case 'embedding':
      return '/v1/embeddings'
    case 'reranker':
    case 'rerank':
      return '/v1/rerank'
    case 'llm':
      return '/v1/chat/completions'
    default:
      return '/v1/inference'
  }
}

export function ApiTestModal({ visible, config, onClose, onSuccess }: ApiTestModalProps) {
  const { t } = useTranslation(['configs', 'common'])
  const configId = config?.config_id
  const configModelName = config?.model_name
  const configProvider = config?.provider
  const configFramework = config?.inference_framework
  const normalizedConfigFramework = configFramework?.toLowerCase()
  const runtimeModelDiscoverySupported =
    configProvider === 'xinference' ||
    normalizedConfigFramework === 'xinference' ||
    normalizedConfigFramework === 'vllm' ||
    normalizedConfigFramework === 'sglang'
  const [requestBody, setRequestBody] = useState('')
  const [response, setResponse] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('test')
  const [temperature, setTemperature] = useState<number | null>(null)
  const [topP, setTopP] = useState<number | null>(null)
  const [maxTokens, setMaxTokens] = useState<number | null>(null)
  const [presencePenalty, setPresencePenalty] = useState<number | null>(null)
  const [frequencyPenalty, setFrequencyPenalty] = useState<number | null>(null)
  const [n, setN] = useState<number | null>(null)
  const [topN, setTopN] = useState<number | null>(null)
  const [embeddingMode, setEmbeddingMode] = useState<'vector' | 'similarity' | 'recall'>('vector')
  const [sentence1Text, setSentence1Text] = useState('你好\n请帮我总结一下这段话')
  const [sentence2Text, setSentence2Text] = useState('今天天气不错\n这段话的核心观点是什么')
  const [loadedAdapters, setLoadedAdapters] = useState<LoadedAdapter[]>([])
  const [selectedAdapter, setSelectedAdapter] = useState<string | null>(null)
  // 召回模式
  const [recallQueries, setRecallQueries] = useState('什么是机器学习？\n如何训练模型？')
  const [recallTopK, setRecallTopK] = useState(10)
  const [collections, setCollections] = useState<
    Array<{
      name: string
      match_type: 'exact' | 'model_match' | 'mismatch'
      embedding_model: string | null
      embedding_config_id: string | null
      dim: number | null
      status: string | null
    }>
  >([])
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [collectionsLoading, setCollectionsLoading] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedRuntimeModel, setSelectedRuntimeModel] = useState<string | null>(null)
  const [fetchingModels, setFetchingModels] = useState(false)
  const wasVisibleRef = useRef(false)

  // Fetch loaded adapters when config has a deployment_id
  useEffect(() => {
    if (!visible || !config?.deployment_id) {
      setLoadedAdapters([])
      return
    }
    adapterApi
      .listLoaded(config.deployment_id)
      .then((res) => {
        setLoadedAdapters((res.adapters || []).filter((a: LoadedAdapter) => a.status === 'loaded'))
      })
      .catch(() => {
        setLoadedAdapters([])
      })
  }, [visible, config?.deployment_id])

  // 加载匹配的向量库
  useEffect(() => {
    if (!visible || !config?.config_id || embeddingMode !== 'recall') return
    setCollectionsLoading(true)
    configApi
      .listCollections(config.config_id)
      .then((res) => {
        const colls = res.collections || []
        setCollections(colls)
        // 自动选中第一个精确匹配的
        const exactMatch = colls.find((c) => c.match_type === 'exact')
        if (exactMatch) {
          setSelectedCollection(exactMatch.name)
        } else if (colls.length > 0) {
          setSelectedCollection(colls[0].name)
        }
      })
      .catch(() => {
        setCollections([])
        setSelectedCollection(null)
      })
      .finally(() => {
        setCollectionsLoading(false)
      })
  }, [visible, config?.config_id, embeddingMode])

  const modelType = config?.model_type || 'embedding'
  const defaultConfig: ModelConfig = useMemo(
    () =>
      config || {
        config_id: '',
        config_name: '',
        model_type: 'embedding',
        provider: '',
        api_endpoint: '',
        status: '',
        created_at: '',
        updated_at: '',
      },
    [config]
  )
  const effectiveModelName =
    selectedAdapter || selectedRuntimeModel || defaultConfig.model_name || 'model-name'
  const defaultData = useMemo(
    () =>
      getDefaultTestData(defaultConfig, {
        modelName: effectiveModelName,
        includeModelInRequest:
          defaultConfig.provider === 'xinference' ||
          defaultConfig.inference_framework?.toLowerCase() === 'xinference',
      }),
    [defaultConfig, effectiveModelName]
  )

  useEffect(() => {
    if (!visible || !configId || !runtimeModelDiscoverySupported) {
      setAvailableModels([])
      setSelectedRuntimeModel(null)
      setFetchingModels(false)
      return
    }

    const currentConfigId = configId
    const currentModelName = configModelName
    let cancelled = false
    setFetchingModels(true)
    configApi
      .checkConnection(currentConfigId)
      .then((res) => {
        if (cancelled) return
        const models = Array.from(
          new Set(
            (res.models || []).filter(
              (model): model is string => typeof model === 'string' && Boolean(model)
            )
          )
        )
        setAvailableModels(models)

        if (!models.length) {
          setSelectedRuntimeModel(null)
          return
        }

        if (currentModelName && models.includes(currentModelName)) {
          setSelectedRuntimeModel(currentModelName)
          return
        }

        setSelectedRuntimeModel(models[0])
      })
      .catch(() => {
        if (cancelled) return
        setAvailableModels([])
        setSelectedRuntimeModel(null)
      })
      .finally(() => {
        if (!cancelled) {
          setFetchingModels(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [visible, configId, configModelName, runtimeModelDiscoverySupported])

  useEffect(() => {
    if (!visible) return
    if (modelType === 'embedding' && (embeddingMode === 'similarity' || embeddingMode === 'recall'))
      return
    setRequestBody(defaultData.request)
  }, [visible, defaultData.request, modelType, embeddingMode])

  const handleTest = async () => {
    if (!config) return

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      let body: Record<string, unknown>
      if (modelType === 'embedding' && embeddingMode === 'similarity') {
        const sentence1 = sentence1Text
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean)
        const sentence2 = sentence2Text
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean)

        if (!sentence1.length || !sentence2.length) {
          setError(t('apiTest.sentenceEmpty'))
          return
        }
        body = {
          mode: 'embedding_similarity',
          sentence1,
          sentence2,
        }
      } else if (modelType === 'embedding' && embeddingMode === 'recall') {
        const queries = recallQueries
          .split('\n')
          .map((s) => s.trim())
          .filter(Boolean)
        if (!queries.length) {
          setError(t('apiTest.recallQueriesEmpty'))
          return
        }
        if (!selectedCollection) {
          setError(t('apiTest.recallNoCollections'))
          return
        }
        body = {
          mode: 'embedding_recall',
          queries,
          collection_name: selectedCollection,
          top_k: recallTopK,
        }
      } else {
        body = JSON.parse(requestBody || defaultData.request)
      }
      const apiPath = getApiPath(modelType)

      if (modelType === 'llm') {
        if (temperature !== null && temperature !== undefined) {
          body.temperature = temperature
        }
        if (topP !== null && topP !== undefined) {
          body.top_p = topP
        }
        if (maxTokens !== null && maxTokens !== undefined) {
          body.max_tokens = Math.min(maxTokens, MAX_LLM_TOKENS)
        }
        if (body.max_tokens && typeof body.max_tokens === 'number') {
          body.max_tokens = Math.min(body.max_tokens, MAX_LLM_TOKENS)
        }
        if (body.max_completion_tokens && typeof body.max_completion_tokens === 'number') {
          body.max_completion_tokens = Math.min(body.max_completion_tokens, MAX_LLM_TOKENS)
        }
        if (presencePenalty !== null && presencePenalty !== undefined) {
          body.presence_penalty = presencePenalty
        }
        if (frequencyPenalty !== null && frequencyPenalty !== undefined) {
          body.frequency_penalty = frequencyPenalty
        }
        if (n !== null && n !== undefined) {
          body.n = n
        }
      }

      if (
        (modelType === 'reranker' || modelType === 'rerank') &&
        topN !== null &&
        topN !== undefined
      ) {
        body.top_n = topN
      }

      // 优先使用 adapter；否则使用运行时已发现的模型；最后退回配置模型名
      if (selectedAdapter) {
        body.model = selectedAdapter
      } else if (!body.model) {
        body.model = selectedRuntimeModel || config.model_name
      }

      // 使用后端代理 API（自动处理 vLLM reranker 预格式化）
      const result = await configApi.test(config.config_id, apiPath, body)

      if (!result.success) {
        setError(
          `HTTP ${result.status_code}\n\n${result.error || t('apiTest.requestFailedGeneric')}`
        )
      } else {
        setResponse(JSON.stringify(result.data, null, 2))
        message.success(t('apiTest.requestSuccess', { ms: Math.round(result.latency_ms) }))
        await onSuccess?.()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t('apiTest.requestFailedGeneric'))
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = (text: string) => {
    copyText(text)
    message.success(t('apiTest.copiedToClipboard'))
  }

  useEffect(() => {
    const justOpened = visible && !wasVisibleRef.current
    wasVisibleRef.current = visible
    if (!justOpened) return

    setResponse(null)
    setError(null)
    setActiveTab('test')
    setRequestBody(defaultData.request)
    setTemperature(null)
    setTopP(null)
    setMaxTokens(null)
    setPresencePenalty(null)
    setFrequencyPenalty(null)
    setN(null)
    setTopN(null)
    setEmbeddingMode('vector')
    setSentence1Text('你好\n请帮我总结一下这段话')
    setSentence2Text('今天天气不错\n这段话的核心观点是什么')
    setSelectedAdapter(null)
    setRecallQueries('什么是机器学习？\n如何训练模型？')
    setRecallTopK(10)
    setCollections([])
    setSelectedCollection(null)
    setAvailableModels([])
    setSelectedRuntimeModel(null)
  }, [defaultData.request, visible])

  if (!config) return null

  const curlCommand = defaultData.curlExample(config.api_endpoint)
  const displayModelName = selectedRuntimeModel || config.model_name || '-'
  const showRuntimeModelSelector = supportsRuntimeModelDiscovery(config)

  const tabItems = [
    {
      key: 'test',
      label: t('apiTest.tabTest'),
      children: (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {showRuntimeModelSelector ? (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.modelName')}</Text>
              </div>
              {availableModels.length > 1 ? (
                <Select
                  value={selectedRuntimeModel}
                  onChange={setSelectedRuntimeModel}
                  loading={fetchingModels}
                  style={{ width: '100%' }}
                  options={availableModels.map((model) => ({
                    label: model,
                    value: model,
                  }))}
                />
              ) : (
                <Input
                  value={fetchingModels ? '加载中...' : displayModelName}
                  readOnly
                  style={{
                    fontFamily: 'monospace',
                    fontSize: 12,
                    background: 'rgba(255, 255, 255, 0.04)',
                  }}
                  suffix={
                    <Button
                      type="text"
                      size="small"
                      icon={<CopyOutlined />}
                      onClick={() => copyToClipboard(displayModelName)}
                    />
                  }
                />
              )}
            </div>
          ) : (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.modelName')}</Text>
              </div>
              <Input
                value={displayModelName}
                readOnly
                style={{
                  fontFamily: 'monospace',
                  fontSize: 12,
                  background: 'rgba(255, 255, 255, 0.04)',
                }}
                suffix={
                  <Button
                    type="text"
                    size="small"
                    icon={<CopyOutlined />}
                    onClick={() => copyToClipboard(displayModelName)}
                  />
                }
              />
            </div>
          )}

          {loadedAdapters.length > 0 && (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.adapterOverride')}</Text>
                <Text type="secondary" style={{ fontSize: 12, marginLeft: 8 }}>
                  {t('apiTest.adapterOverrideHint')}
                </Text>
              </div>
              <Select
                allowClear
                value={selectedAdapter}
                onChange={setSelectedAdapter}
                placeholder={t('apiTest.noAdapter')}
                style={{ width: '100%' }}
                options={loadedAdapters.map((a) => ({
                  label: a.adapter_name,
                  value: a.adapter_name,
                }))}
              />
            </div>
          )}

          {modelType === 'llm' && (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.optionalParams')}</Text>
              </div>
              <div
                style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px 16px' }}
              >
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    temperature
                  </div>
                  <InputNumber
                    value={temperature}
                    onChange={setTemperature}
                    min={0}
                    max={2}
                    step={0.1}
                    placeholder={t('apiTest.defaultPlaceholder')}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    top_p
                  </div>
                  <InputNumber
                    value={topP}
                    onChange={setTopP}
                    min={0}
                    max={1}
                    step={0.05}
                    placeholder={t('apiTest.defaultPlaceholder')}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    max_tokens
                  </div>
                  <InputNumber
                    value={maxTokens}
                    onChange={setMaxTokens}
                    min={1}
                    max={MAX_LLM_TOKENS}
                    step={1}
                    placeholder={`≤${MAX_LLM_TOKENS}`}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    presence_penalty
                  </div>
                  <InputNumber
                    value={presencePenalty}
                    onChange={setPresencePenalty}
                    min={-2}
                    max={2}
                    step={0.1}
                    placeholder={t('apiTest.defaultPlaceholder')}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    frequency_penalty
                  </div>
                  <InputNumber
                    value={frequencyPenalty}
                    onChange={setFrequencyPenalty}
                    min={-2}
                    max={2}
                    step={0.1}
                    placeholder={t('apiTest.defaultPlaceholder')}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    {t('apiTest.nLabel')}
                  </div>
                  <InputNumber
                    value={n}
                    onChange={setN}
                    min={1}
                    step={1}
                    placeholder={t('apiTest.defaultPlaceholder')}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>
          )}

          {(modelType === 'reranker' || modelType === 'rerank') && (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.optionalParams')}</Text>
              </div>
              <Space wrap>
                <div>
                  <div
                    style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.65)', marginBottom: 4 }}
                  >
                    top_n
                  </div>
                  <InputNumber
                    value={topN}
                    onChange={setTopN}
                    min={1}
                    step={1}
                    placeholder={t('apiTest.defaultPlaceholder')}
                  />
                </div>
              </Space>
            </div>
          )}

          {modelType === 'embedding' && (
            <div>
              <div style={{ marginBottom: 8 }}>
                <Text strong>{t('apiTest.testMode')}</Text>
              </div>
              <Radio.Group
                value={embeddingMode}
                onChange={(e) => setEmbeddingMode(e.target.value)}
                optionType="button"
                buttonStyle="solid"
              >
                <Radio.Button value="vector">{t('apiTest.vectorMode')}</Radio.Button>
                <Radio.Button value="similarity">{t('apiTest.similarityMode')}</Radio.Button>
                <Radio.Button value="recall">{t('apiTest.recallMode')}</Radio.Button>
              </Radio.Group>
            </div>
          )}

          {modelType === 'embedding' && embeddingMode === 'similarity' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>{t('apiTest.sentence1Label')}</Text>
                </div>
                <TextArea
                  value={sentence1Text}
                  onChange={(e) => setSentence1Text(e.target.value)}
                  rows={4}
                  style={{ fontFamily: 'monospace', fontSize: 12 }}
                  placeholder={t('apiTest.sentence1Placeholder')}
                />
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>{t('apiTest.sentence2Label')}</Text>
                </div>
                <TextArea
                  value={sentence2Text}
                  onChange={(e) => setSentence2Text(e.target.value)}
                  rows={4}
                  style={{ fontFamily: 'monospace', fontSize: 12 }}
                  placeholder={t('apiTest.sentence2Placeholder')}
                />
              </div>
            </div>
          ) : modelType === 'embedding' && embeddingMode === 'recall' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>{t('apiTest.recallCollection')}</Text>
                </div>
                <Select
                  value={selectedCollection}
                  onChange={setSelectedCollection}
                  placeholder={t('apiTest.recallCollectionPlaceholder')}
                  loading={collectionsLoading}
                  style={{ width: '100%' }}
                  notFoundContent={
                    collectionsLoading
                      ? t('apiTest.recallCollectionLoading')
                      : t('apiTest.recallNoCollections')
                  }
                  options={collections.map((c) => ({
                    label: `${c.name}${c.match_type === 'exact' ? ` (${t('apiTest.recallCollectionMatchExact')})` : c.match_type === 'model_match' ? ` (${t('apiTest.recallCollectionMatchModel')})` : ''}`,
                    value: c.name,
                  }))}
                />
                {selectedCollection &&
                  collections.find((c) => c.name === selectedCollection)?.match_type ===
                    'mismatch' && (
                    <Alert
                      type="warning"
                      showIcon
                      icon={<WarningOutlined />}
                      message={t('apiTest.recallCollectionWarning')}
                      style={{ marginTop: 8 }}
                    />
                  )}
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>{t('apiTest.recallQueriesLabel')}</Text>
                </div>
                <TextArea
                  value={recallQueries}
                  onChange={(e) => setRecallQueries(e.target.value)}
                  rows={4}
                  style={{ fontFamily: 'monospace', fontSize: 12 }}
                  placeholder={t('apiTest.recallQueriesPlaceholder')}
                />
              </div>
              <div>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>{t('apiTest.recallTopK')}</Text>
                </div>
                <InputNumber
                  value={recallTopK}
                  onChange={(v) => setRecallTopK(v || 10)}
                  min={1}
                  max={100}
                  step={1}
                  style={{ width: 120 }}
                />
              </div>
            </div>
          ) : (
            <div>
              <div
                style={{
                  marginBottom: 8,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Text strong>{t('apiTest.requestBody')}</Text>
                <Button size="small" onClick={() => setRequestBody(defaultData.request)}>
                  {t('apiTest.resetToExample')}
                </Button>
              </div>
              <TextArea
                value={requestBody || defaultData.request}
                onChange={(e) => setRequestBody(e.target.value)}
                rows={8}
                style={{ fontFamily: 'monospace', fontSize: 12 }}
                placeholder={t('apiTest.requestBodyPlaceholder')}
              />
            </div>
          )}

          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleTest}
            loading={loading}
          >
            {t('apiTest.sendRequest')}
          </Button>

          {error && (
            <div>
              <div
                style={{
                  marginBottom: 8,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Text strong style={{ color: '#ff4d4f' }}>
                  {t('apiTest.requestFailed')}
                </Text>
                <Button size="small" icon={<CopyOutlined />} onClick={() => copyToClipboard(error)}>
                  {t('common:action.copy')}
                </Button>
              </div>
              <pre
                style={{
                  margin: 0,
                  padding: 12,
                  background: '#1a1a2e',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: 6,
                  maxHeight: 200,
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                  fontSize: 12,
                  fontFamily: 'monospace',
                  color: '#ff7875',
                }}
              >
                {error}
              </pre>
            </div>
          )}

          {response && (
            <div>
              <div
                style={{
                  marginBottom: 8,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Text strong>{t('apiTest.responseResult')}</Text>
                <Button
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={() => copyToClipboard(response)}
                >
                  {t('common:action.copy')}
                </Button>
              </div>
              <pre
                style={{
                  background: '#1a1a2e',
                  padding: 12,
                  borderRadius: 6,
                  maxHeight: 300,
                  overflow: 'auto',
                  fontSize: 12,
                  margin: 0,
                }}
              >
                {response}
              </pre>
            </div>
          )}
        </div>
      ),
    },
    {
      key: 'curl',
      label: t('apiTest.tabCurl'),
      children: (
        <div>
          <div
            style={{
              marginBottom: 8,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Text strong>{t('apiTest.curlCommand')}</Text>
            <Button
              size="small"
              icon={<CopyOutlined />}
              onClick={() => copyToClipboard(curlCommand)}
            >
              {t('common:action.copy')}
            </Button>
          </div>
          <pre
            style={{
              background: '#1a1a2e',
              padding: 12,
              borderRadius: 6,
              maxHeight: 400,
              overflow: 'auto',
              fontSize: 12,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              margin: 0,
            }}
          >
            {curlCommand}
          </pre>
        </div>
      ),
    },
    {
      key: 'docs',
      label: t('apiTest.tabDocs'),
      children: (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Alert
            type="info"
            message={t('apiTest.apiFormat', { type: modelType.toUpperCase() })}
            description={t('apiTest.endpointInfo', {
              endpoint: `${config.api_endpoint}${getApiPath(modelType)}`,
            })}
          />

          {modelType === 'embedding' && (
            <>
              <div>
                <Title level={5}>{t('apiTest.requestParams')}</Title>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  <li>
                    <code>input</code> {t('apiTest.embedding.inputDesc')}
                  </li>
                  <li>
                    <code>model</code> {t('apiTest.embedding.modelDesc')}
                  </li>
                  <li>
                    <code>encoding_format</code> {t('apiTest.embedding.encodingFormatDesc')}
                  </li>
                </ul>
              </div>
              <div>
                <Title level={5}>{t('apiTest.responseFormat')}</Title>
                <pre
                  style={{
                    background: '#1a1a2e',
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 12,
                    margin: 0,
                  }}
                >
                  {`{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ...]  ${t('apiTest.embedding.vectorDimComment')}
    }
  ],
  "model": "model-name",
  "usage": { "prompt_tokens": 10, "total_tokens": 10 }
}`}
                </pre>
              </div>
            </>
          )}

          {(modelType === 'reranker' || modelType === 'rerank') && (
            <>
              <div>
                <Title level={5}>{t('apiTest.requestParams')}</Title>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  <li>
                    <code>query</code> {t('apiTest.reranker.queryDesc')}
                  </li>
                  <li>
                    <code>documents</code> {t('apiTest.reranker.documentsDesc')}
                  </li>
                  <li>
                    <code>top_n</code> {t('apiTest.reranker.topNDesc')}
                  </li>
                  <li>
                    <code>return_documents</code> {t('apiTest.reranker.returnDocumentsDesc')}
                  </li>
                </ul>
              </div>
              <div>
                <Title level={5}>{t('apiTest.responseFormat')}</Title>
                <pre
                  style={{
                    background: '#1a1a2e',
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 12,
                    margin: 0,
                  }}
                >
                  {`{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "${t('apiTest.reranker.originalDocContent')}"
    },
    ...
  ],
  "model": "model-name"
}`}
                </pre>
              </div>
            </>
          )}

          {modelType === 'llm' && (
            <>
              <div>
                <Title level={5}>{t('apiTest.requestParams')}</Title>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  <li>
                    <code>model</code> {t('apiTest.llm.modelDesc')}
                  </li>
                  <li>
                    <code>messages</code> {t('apiTest.llm.messagesDesc')}
                  </li>
                  <li>
                    <code>max_tokens</code> {t('apiTest.llm.maxTokensDesc')}
                  </li>
                  <li>
                    <code>temperature</code> {t('apiTest.llm.temperatureDesc')}
                  </li>
                  <li>
                    <code>stream</code> {t('apiTest.llm.streamDesc')}
                  </li>
                </ul>
              </div>
              <div>
                <Title level={5}>{t('apiTest.responseFormat')}</Title>
                <pre
                  style={{
                    background: '#1a1a2e',
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 12,
                    margin: 0,
                  }}
                >
                  {`{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "${t('apiTest.llm.replyContent')}" },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30 }
}`}
                </pre>
              </div>
            </>
          )}
        </div>
      ),
    },
  ]

  return (
    <Modal
      title={
        <Space>
          <CodeOutlined />
          {t('apiTest.title', { name: config.config_name })}
        </Space>
      }
      open={visible}
      onCancel={onClose}
      footer={null}
      width={700}
      destroyOnClose
    >
      <Spin spinning={loading}>
        <Tabs activeKey={activeTab} onChange={setActiveTab} items={tabItems} />
      </Spin>
    </Modal>
  )
}
