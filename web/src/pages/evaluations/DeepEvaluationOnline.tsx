import { useState, useEffect } from 'react'
import {
  Form,
  Input,
  Button,
  Card,
  Space,
  Select,
  Typography,
  message,
  Divider,
  Row,
  Col,
  Table,
  Tag,
  Tabs,
  List,
} from 'antd'
import { PlayCircleOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { deepEvaluationApi as deepEvalApi } from '@/services/api'
import ModelConfigSelector from '@/components/ModelConfigSelector'
import type { ModelConfig } from '@/types'

const { Text, Paragraph } = Typography
const { TextArea } = Input

interface MetricInfo {
  name: string
  description: string
  category: string
  requires_llm: boolean
  requires_expected_output: boolean
  requires_actual_output: boolean
  requires_retrieval_context: boolean
}

interface EvaluationResult {
  score: number
  reason: string
  details?: Record<string, unknown>
}

interface SingleFormValues {
  input: string
  expected_output?: string
  actual_output?: string
  retrieval_context?: string
}

interface Sample {
  input: string
  expected_output?: string
  actual_output?: string
  retrieval_context: string[]
}

interface BatchMetricStat {
  mean: number
  min: number
  max: number
  count: number
}

interface BatchResultItem {
  sample: Record<string, unknown>
  metric_results: Record<string, EvaluationResult>
  overall_score: number
}

interface BatchResultSummary {
  total_samples: number
  metrics: Record<string, BatchMetricStat>
  overall: {
    mean: number
    min: number
    max: number
  }
}

interface BatchEvaluateResponse {
  results: BatchResultItem[]
  summary: BatchResultSummary
}

const formatScore = (value: unknown) => {
  const num = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(num)) return '-'
  return `${Math.round(num * 100)}/100`
}

export default function DeepEvaluationOnline() {
  const { t } = useTranslation(['evaluations', 'common'])

  // Metrics are fixed for online deep evaluation (LLM-as-Judge)
  const metrics: MetricInfo[] = [
    { name: 'answer_relevancy', description: t('online.metricDescription.answer_relevancy'), category: 'relevancy', requires_llm: true, requires_expected_output: false, requires_actual_output: true, requires_retrieval_context: false },
    { name: 'faithfulness', description: t('online.metricDescription.faithfulness'), category: 'faithfulness', requires_llm: true, requires_expected_output: false, requires_actual_output: true, requires_retrieval_context: true },
    { name: 'contextual_precision', description: t('online.metricDescription.contextual_precision'), category: 'retrieval', requires_llm: true, requires_expected_output: true, requires_actual_output: false, requires_retrieval_context: true },
    { name: 'contextual_recall', description: t('online.metricDescription.contextual_recall'), category: 'retrieval', requires_llm: true, requires_expected_output: true, requires_actual_output: false, requires_retrieval_context: true },
    { name: 'contextual_relevancy', description: t('online.metricDescription.contextual_relevancy'), category: 'retrieval', requires_llm: true, requires_expected_output: false, requires_actual_output: false, requires_retrieval_context: true },
  ]
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['answer_relevancy'])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<Record<string, EvaluationResult> | null>(null)
  const [overallScore, setOverallScore] = useState<number | null>(null)

  // Single sample form
  const [singleForm] = Form.useForm<SingleFormValues>()

  // Batch samples
  const [batchSamples, setBatchSamples] = useState<Sample[]>([
    { input: '', expected_output: '', actual_output: '', retrieval_context: [] },
  ])
  const [batchResults, setBatchResults] = useState<BatchEvaluateResponse | null>(null)

  // LLM Config (from model config selector)
  const [selectedLLMConfig, setSelectedLLMConfig] = useState<ModelConfig | null>(null)

  useEffect(() => {
    // Metrics are now hardcoded for online deep evaluation
  }, [])

  const handleSingleEvaluate = async (values: SingleFormValues) => {
    if (!selectedLLMConfig) {
      message.error(t('online.selectLLMFirst'))
      return
    }

    setLoading(true)
    setResults(null)
    try {
      const response = await deepEvalApi.evaluate({
        input: values.input,
        expected_output: values.expected_output,
        actual_output: values.actual_output,
        retrieval_context: values.retrieval_context?.split('\n').filter((s: string) => s.trim()) || [],
        metrics: selectedMetrics,
        llm_config: {
          config_id: selectedLLMConfig.config_id,
        },
      })
      setResults(response.results)
      setOverallScore(response.overall_score)
      message.success(t('online.evalComplete'))
    } catch {
      message.error(t('online.evalFailed'))
    } finally {
      setLoading(false)
    }
  }

  const handleBatchEvaluate = async () => {
    if (!selectedLLMConfig) {
      message.error(t('online.selectLLMFirst'))
      return
    }

    const validSamples = batchSamples.filter((s) => s.input.trim())
    if (validSamples.length === 0) {
      message.error(t('online.atLeastOneSample'))
      return
    }

    setLoading(true)
    setBatchResults(null)
    try {
      const results: BatchResultItem[] = []
      for (const sample of validSamples) {
        const response = await deepEvalApi.evaluate({
          input: sample.input,
          expected_output: sample.expected_output,
          actual_output: sample.actual_output,
          retrieval_context: sample.retrieval_context,
          metrics: selectedMetrics,
          llm_config: {
            config_id: selectedLLMConfig.config_id,
          },
        })
        results.push({
          sample: sample as unknown as Record<string, unknown>,
          metric_results: response.results,
          overall_score: response.overall_score,
        })
      }

      // Compute summary
      const metricStats: Record<string, BatchMetricStat> = {}
      const overallScores = results.map(r => r.overall_score)
      for (const metric of selectedMetrics) {
        const scores = results.map(r => r.metric_results[metric]?.score).filter(s => s !== undefined) as number[]
        if (scores.length > 0) {
          metricStats[metric] = {
            mean: scores.reduce((a, b) => a + b, 0) / scores.length,
            min: Math.min(...scores),
            max: Math.max(...scores),
            count: scores.length,
          }
        }
      }

      setBatchResults({
        results,
        summary: {
          total_samples: results.length,
          metrics: metricStats,
          overall: {
            mean: overallScores.reduce((a, b) => a + b, 0) / overallScores.length,
            min: Math.min(...overallScores),
            max: Math.max(...overallScores),
          },
        },
      })
      message.success(t('online.batchEvalComplete', { count: results.length }))
    } catch {
      message.error(t('online.batchEvalFailed'))
    } finally {
      setLoading(false)
    }
  }

  const addBatchSample = () => {
    setBatchSamples([...batchSamples, { input: '', expected_output: '', actual_output: '', retrieval_context: [] }])
  }

  const removeBatchSample = (index: number) => {
    setBatchSamples(batchSamples.filter((_, i) => i !== index))
  }

  const updateBatchSample = (index: number, field: keyof Sample, value: string) => {
    const newSamples = [...batchSamples]
    if (field === 'retrieval_context') {
      newSamples[index][field] = value.split('\n').filter((s: string) => s.trim())
    } else {
      newSamples[index][field] = value
    }
    setBatchSamples(newSamples)
  }

  const categoryColorMap: Record<string, string> = {
    retrieval: 'blue',
    generation: 'green',
    faithfulness: 'orange',
    relevancy: 'purple',
  }

  return (
    <div>
      <Paragraph type="secondary">
        {t('online.description')}
      </Paragraph>

      <Card title={t('online.llmConfig')} style={{ marginBottom: 24 }}>
        <Row gutter={24}>
          <Col span={12}>
            <Form.Item label={t('online.selectLLMModel')} required>
              <ModelConfigSelector
                mode="single"
                modelType="llm"
                onChange={(_, config) => setSelectedLLMConfig(config)}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            {selectedLLMConfig && (
              <div style={{ padding: '8px 12px', background: '#f5f5f5', borderRadius: 4, marginTop: 30 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {t('online.endpoint')}: {selectedLLMConfig.api_endpoint} | {t('online.modelLabel')}: {selectedLLMConfig.model_name || '-'}
                </Text>
              </div>
            )}
          </Col>
        </Row>
      </Card>

      <Card title={t('online.evalMetrics')} style={{ marginBottom: 24 }}>
        <Select
          mode="multiple"
          style={{ width: '100%' }}
          placeholder={t('online.selectMetrics')}
          value={selectedMetrics}
          onChange={setSelectedMetrics}
        >
          {metrics.map((m) => (
            <Select.Option key={m.name} value={m.name}>
              <Space>
                <Tag color={categoryColorMap[m.category] || 'default'}>{m.category}</Tag>
                <span>{m.name}</span>
                <Text type="secondary" style={{ fontSize: 12 }}>{m.description}</Text>
              </Space>
            </Select.Option>
          ))}
        </Select>

        <div style={{ marginTop: 16 }}>
          <Text type="secondary">{t('online.selectedRequirements')}</Text>
          <div style={{ marginTop: 8 }}>
            {selectedMetrics.map((name) => {
              const m = metrics.find((x) => x.name === name)
              if (!m) return null
              return (
                <Tag key={name} style={{ marginBottom: 4 }}>
                  {name}
                  {m.requires_expected_output && t('online.requiresExpectedOutput')}
                  {m.requires_actual_output && t('online.requiresActualOutput')}
                  {m.requires_retrieval_context && t('online.requiresRetrievalContext')}
                </Tag>
              )
            })}
          </div>
        </div>
      </Card>

      <Tabs
        defaultActiveKey="single"
        items={[
          {
            key: 'single',
            label: t('online.singleEval'),
            children: (
              <Card>
                <Form form={singleForm} layout="vertical" onFinish={handleSingleEvaluate}>
                  <Form.Item
                    name="input"
                    label={t('online.userQuery')}
                    rules={[{ required: true, message: t('online.userQueryRequired') }]}
                  >
                    <TextArea rows={2} placeholder={t('online.userQueryPlaceholder')} />
                  </Form.Item>

                  <Row gutter={24}>
                    <Col span={12}>
                      <Form.Item name="expected_output" label={t('online.expectedOutput')}>
                        <TextArea rows={4} placeholder={t('online.expectedOutputPlaceholder')} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item name="actual_output" label={t('online.actualOutput')}>
                        <TextArea rows={4} placeholder={t('online.actualOutputPlaceholder')} />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item name="retrieval_context" label={t('online.retrievalContext')}>
                    <TextArea rows={6} placeholder={t('online.retrievalContextPlaceholder')} />
                  </Form.Item>

                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={loading} disabled={loading} icon={<PlayCircleOutlined />}>
                      {t('online.startEval')}
                    </Button>
                  </Form.Item>
                </Form>

                {results && (
                  <div style={{ marginTop: 24 }}>
                    <Divider>{t('online.evalResultsTitle')}</Divider>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>{t('online.overallScore')}</Text>
                      <Text style={{ marginLeft: 8 }}>
                        {formatScore(overallScore)}
                      </Text>
                    </div>

                    <Table
                      dataSource={Object.entries(results).map(([name, result]) => ({
                        key: name,
                        name,
                        ...result,
                      }))}
                      columns={[
                        { title: t('online.metricColumn'), dataIndex: 'name', key: 'name', width: 200 },
                        {
                          title: t('online.scoreColumn'),
                          dataIndex: 'score',
                          key: 'score',
                          width: 150,
                          render: (score: number) => {
                            const num = typeof score === 'number' ? score : Number(score)
                            if (!Number.isFinite(num)) return '-'
                            return <Text>{Math.round(num * 100)}/100</Text>
                          },
                        },
                        { title: t('online.reasonColumn'), dataIndex: 'reason', key: 'reason' },
                      ]}
                      pagination={false}
                    />
                  </div>
                )}
              </Card>
            ),
          },
          {
            key: 'batch',
            label: t('online.batchEval'),
            children: (
              <Card>
                <List
                  dataSource={batchSamples}
                  renderItem={(sample, index) => (
                    <List.Item
                      actions={[
                        batchSamples.length > 1 && (
                          <Button
                            type="text"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => removeBatchSample(index)}
                          />
                        ),
                      ]}
                    >
                      <Card size="small" style={{ width: '100%' }} title={t('online.sample', { index: index + 1 })}>
                        <Row gutter={16}>
                          <Col span={24}>
                            <Form.Item label={t('online.userQuestion')}>
                              <Input
                                value={sample.input}
                                onChange={(e) => updateBatchSample(index, 'input', e.target.value)}
                                placeholder={t('online.userQuestion')}
                              />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={12}>
                            <Form.Item label={t('online.expectedAnswer')}>
                              <TextArea
                                rows={2}
                                value={sample.expected_output}
                                onChange={(e) => updateBatchSample(index, 'expected_output', e.target.value)}
                              />
                            </Form.Item>
                          </Col>
                          <Col span={12}>
                            <Form.Item label={t('online.actualOutputLabel')}>
                              <TextArea
                                rows={2}
                                value={sample.actual_output}
                                onChange={(e) => updateBatchSample(index, 'actual_output', e.target.value)}
                              />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Form.Item label={t('online.retrievalContextLabel')}>
                          <TextArea
                            rows={2}
                            value={sample.retrieval_context.join('\n')}
                            onChange={(e) => updateBatchSample(index, 'retrieval_context', e.target.value)}
                          />
                        </Form.Item>
                      </Card>
                    </List.Item>
                  )}
                />

                <Space style={{ marginTop: 16 }}>
                  <Button icon={<PlusOutlined />} onClick={addBatchSample}>
                    {t('online.addSample')}
                  </Button>
                  <Button type="primary" loading={loading} icon={<PlayCircleOutlined />} onClick={handleBatchEvaluate}>
                    {t('online.startBatchEval')}
                  </Button>
                </Space>

                {batchResults && (
                  <div style={{ marginTop: 24 }}>
                    <Divider>{t('online.batchResultsTitle')}</Divider>

                    <Card size="small" title={t('online.summaryStats')} style={{ marginBottom: 16 }}>
                      <Row gutter={16}>
                        <Col span={6}>
                          <Text>{t('online.totalSamples', { count: batchResults.summary.total_samples })}</Text>
                        </Col>
                        <Col span={6}>
                          <Text>{t('online.avgScore', { score: formatScore(batchResults.summary.overall?.mean) })}</Text>
                        </Col>
                        <Col span={6}>
                          <Text>{t('online.maxScore', { score: formatScore(batchResults.summary.overall?.max) })}</Text>
                        </Col>
                        <Col span={6}>
                          <Text>{t('online.minScore', { score: formatScore(batchResults.summary.overall?.min) })}</Text>
                        </Col>
                      </Row>
                    </Card>

                    <Table
                      dataSource={batchResults.results.map((r, i) => ({
                        key: i,
                        index: i + 1,
                        input: r.sample.input,
                        overall_score: r.overall_score,
                        metrics: r.metric_results,
                      }))}
                      columns={[
                        { title: '#', dataIndex: 'index', key: 'index', width: 50 },
                        {
                          title: t('online.questionColumn'),
                          dataIndex: 'input',
                          key: 'input',
                          ellipsis: true,
                        },
                        {
                          title: t('online.overallScoreColumn'),
                          dataIndex: 'overall_score',
                          key: 'overall_score',
                          width: 150,
                          render: (score: number) => formatScore(score),
                        },
                        ...selectedMetrics.map((metric) => ({
                          title: metric,
                          key: metric,
                          width: 120,
                          render: (_: unknown, record: { metrics: Record<string, EvaluationResult> }) => {
                            const result = record.metrics[metric]
                            if (!result) return '-'
                            const num = typeof result.score === 'number' ? result.score : Number(result.score)
                            if (!Number.isFinite(num)) return '-'
                            return `${Math.round(num * 100)}/100`
                          },
                        })),
                      ]}
                      pagination={{ pageSize: 10 }}
                    />
                  </div>
                )}
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}
