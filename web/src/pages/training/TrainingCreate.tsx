import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Form,
  Input,
  Select,
  InputNumber,
  Button,
  Card,
  Row,
  Col,
  Space,
  message,
  Typography,
  Divider,
  Switch,
} from 'antd'
import { ArrowLeftOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { trainingApi, datasetApi, modelApi } from '@/services/api'
import { GpuSelect } from '@/components/GpuSelect'
import type { CreateTrainingRequest, ModelType, TrainingMethod, TunerType, Dataset, RegisteredModel } from '@/types'

const { Title } = Typography

// Dataset item for dynamic form
interface DatasetItem {
  key: string
  path: string
  max_samples?: number
  split: 'train' | 'eval' | 'test'
}

const modelTypeOptions = [
  { label: 'Embedding', value: 'embedding' },
  { label: 'Reranker (CrossEncoder)', value: 'reranker' },
  { label: 'Decoder Reranker', value: 'decoder_reranker' },
  { label: 'LLM', value: 'llm' },
]

const lambdaMetricOptions = [
  { label: 'NDCG', value: 'ndcg' },
  { label: 'MAP', value: 'map' },
  { label: 'MRR', value: 'mrr' },
]

const rlRewardKeys = [
  { key: 'create.rl.rewardRankBased', value: 'rank_based' },
  { key: 'create.rl.rewardScoreBased', value: 'score_based' },
  { key: 'create.rl.rewardNdcgBased', value: 'ndcg_based' },
  { key: 'create.rl.rewardRecallBased', value: 'recall_based' },
]

const embeddingScaleLosses = [
  'MultipleNegativesRankingLoss',
  'CachedMultipleNegativesRankingLoss',
  'CoSENTLoss',
  'AnglELoss',
  'DynamicExplicitNegativesRankingLoss',
]
const embeddingMarginLosses = [
  'ContrastiveLoss',
  'OnlineContrastiveLoss',
  'BatchHardTripletLoss',
  'BatchSemiHardTripletLoss',
  'BatchAllTripletLoss',
]
const embeddingTripletLosses = ['TripletLoss']
const embeddingMegaBatchLosses = ['MegaBatchMarginLoss']
const embeddingSoftmaxLosses = ['SoftmaxLoss']
const embeddingGistLosses = ['GISTEmbedLoss']
const embeddingDynamicNegLosses = ['DynamicExplicitNegativesRankingLoss']
const embeddingMatryoshkaLosses = ['MatryoshkaLoss', 'Matryoshka2dLoss']

const rerankerScaleLosses = ['MultipleNegativesRankingLoss', 'CachedMultipleNegativesRankingLoss']
const rerankerMiniBatchLosses = ['CachedMultipleNegativesRankingLoss', 'RankNetLoss', 'LambdaLoss', 'ListMLELoss', 'ListNetLoss', 'PListMLELoss']
const rerankerNegativesLosses = ['MultipleNegativesRankingLoss', 'CachedMultipleNegativesRankingLoss']
const rerankerTopKLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerBetaLosses: string[] = []
const rerankerDeltaLosses: string[] = []
const rerankerTemperatureLosses: string[] = []
const rerankerSigmaLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerRespectInputOrderLosses = ['ListMLELoss', 'PListMLELoss']

const trainingMethodOptions: Record<ModelType, { label: string; value: TrainingMethod }[]> = {
  embedding: [{ label: 'SFT', value: 'sft' }],
  reranker: [{ label: 'SFT', value: 'sft' }],
  decoder_reranker: [
    { label: 'SFT', value: 'sft' },
    { label: 'GRPO', value: 'grpo' },
    { label: 'DAPO', value: 'dapo' },
    { label: 'Dr. GRPO', value: 'dr_grpo' },
    { label: 'DPO', value: 'dpo' },
  ],
  llm: [
    { label: 'SFT', value: 'sft' },
    { label: 'DPO', value: 'dpo' },
    { label: 'ORPO', value: 'orpo' },
  ],
}

export default function TrainingCreate() {
  const navigate = useNavigate()
  const { t } = useTranslation(['training', 'common'])
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<RegisteredModel[]>([])
  const [modelType, setModelType] = useState<ModelType>('embedding')
  const [datasetItems, setDatasetItems] = useState<DatasetItem[]>([
    { key: Date.now().toString(), path: '', max_samples: undefined, split: 'train' }
  ])

  const tunerTypeOptions: { label: string; value: TunerType }[] = [
    // Unsupported QLoRA/freeze modes stay out of the selectable capability list.
    { label: 'LoRA', value: 'lora' },
    { label: t('create.tuner.fullFinetune'), value: 'full' },
  ]

  const embeddingLossOptions = [
    { label: t('create.lossAuto'), value: 'auto' },
    { label: 'MultipleNegativesRankingLoss', value: 'MultipleNegativesRankingLoss' },
    { label: 'CachedMultipleNegativesRankingLoss', value: 'CachedMultipleNegativesRankingLoss' },
    { label: 'ContrastiveLoss', value: 'ContrastiveLoss' },
    { label: 'OnlineContrastiveLoss', value: 'OnlineContrastiveLoss' },
    { label: 'MegaBatchMarginLoss', value: 'MegaBatchMarginLoss' },
    { label: 'CosineSimilarityLoss', value: 'CosineSimilarityLoss' },
    { label: 'CoSENTLoss', value: 'CoSENTLoss' },
    { label: 'AnglELoss', value: 'AnglELoss' },
    { label: 'TripletLoss', value: 'TripletLoss' },
    { label: 'BatchHardTripletLoss', value: 'BatchHardTripletLoss' },
    { label: 'BatchSemiHardTripletLoss', value: 'BatchSemiHardTripletLoss' },
    { label: 'BatchAllTripletLoss', value: 'BatchAllTripletLoss' },
    { label: 'SoftmaxLoss', value: 'SoftmaxLoss' },
    { label: 'DistillKLDivLoss', value: 'DistillKLDivLoss' },
    { label: 'MSELoss', value: 'MSELoss' },
    { label: 'GISTEmbedLoss', value: 'GISTEmbedLoss' },
    { label: 'DynamicExplicitNegativesRankingLoss', value: 'DynamicExplicitNegativesRankingLoss' },
    { label: 'MatryoshkaLoss', value: 'MatryoshkaLoss' },
    { label: 'Matryoshka2dLoss', value: 'Matryoshka2dLoss' },
  ]

  const rerankerLossOptions = [
    { label: t('create.lossAuto'), value: 'auto' },
    { label: 'CrossEntropyLoss', value: 'CrossEntropyLoss' },
    { label: 'BinaryCrossEntropyLoss (BCEWithLogits)', value: 'BCEWithLogitsLoss' },
    { label: 'MSELoss', value: 'MSELoss' },
    { label: 'MarginMSELoss', value: 'MarginMSELoss' },
    { label: 'MultipleNegativesRankingLoss', value: 'MultipleNegativesRankingLoss' },
    { label: 'CachedMultipleNegativesRankingLoss', value: 'CachedMultipleNegativesRankingLoss' },
    { label: 'RankNetLoss', value: 'RankNetLoss' },
    { label: 'LambdaLoss', value: 'LambdaLoss' },
    { label: 'ListMLELoss', value: 'ListMLELoss' },
    { label: 'ListNetLoss', value: 'ListNetLoss' },
    { label: 'PListMLELoss', value: 'PListMLELoss' },
  ]

  useEffect(() => {
    fetchDatasets()
    fetchModels()
  }, [])

  const fetchDatasets = async () => {
    try {
      const res = await datasetApi.list({ page_size: 100 })
      setDatasets(res.items)
    } catch {
      // Ignore error, datasets can be entered manually
    }
  }

  const fetchModels = async () => {
    try {
      const res = await modelApi.list({ page_size: 100, status: 'available' })
      setModels(res.items)
    } catch {
      // Ignore error, model path can be entered manually
    }
  }

  const handleModelTypeChange = (value: ModelType) => {
    setModelType(value)
    // Reset training method when model type changes
    const methods = trainingMethodOptions[value]
    form.setFieldsValue({
      training_method: methods[0]?.value,
      ...(value === 'llm'
        ? {
            per_device_train_batch_size: 1,
            gradient_accumulation_steps: 16,
            max_length: 1024,
            gradient_checkpointing: true,
            mixed_precision: 'bf16',
          }
        : {
            per_device_train_batch_size: 16,
            gradient_accumulation_steps: 1,
            max_length: undefined,
            gradient_checkpointing: false,
          }),
    })
  }

  // Dataset item handlers
  const handleAddDataset = () => {
    setDatasetItems([...datasetItems, { key: Date.now().toString(), path: '', max_samples: undefined, split: 'train' }])
  }

  const handleRemoveDataset = (key: string) => {
    if (datasetItems.length > 1) {
      setDatasetItems(datasetItems.filter(item => item.key !== key))
    }
  }

  const handleDatasetChange = (key: string, field: 'path' | 'max_samples' | 'split', value: string | number | undefined) => {
    setDatasetItems(datasetItems.map(item =>
      item.key === key ? { ...item, [field]: value } : item
    ))
  }

  const handleSubmit = async (values: CreateTrainingRequest) => {
    // Build datasets array from datasetItems
    const validDatasets = datasetItems
      .filter(item => item.path)
      .map(item => ({
        path: item.path,
        max_samples: item.max_samples || undefined,
        split: item.split,
      }))

    // Check at least one training dataset
    const trainDatasets = validDatasets.filter(d => d.split === 'train')
    if (trainDatasets.length === 0) {
      message.error(t('create.dataset.requireTrain'))
      return
    }

    // Handle base_model_path from Select tags mode
    let modelPath = values.base_model_path
    if (Array.isArray(modelPath)) {
      modelPath = (modelPath as string[])[0] || ''
    }

    // Convert mixed_precision select to bf16/fp16 booleans
    const { mixed_precision, ...restValues } = values as CreateTrainingRequest & { mixed_precision?: string }

    const submitData: CreateTrainingRequest = {
      ...restValues,
      base_model_path: modelPath,
      bf16: mixed_precision === 'bf16',
      fp16: mixed_precision === 'fp16',
      datasets: validDatasets,
    }

    const lossConfig = submitData.loss_config
    if (lossConfig?.matryoshka_dims) {
      const rawDims = lossConfig.matryoshka_dims as Array<number | string> | string
      const dimsList = Array.isArray(rawDims)
        ? rawDims
        : typeof rawDims === 'string'
          ? rawDims.split(/[,，\s]+/)
          : []
      const parsedDims = dimsList
        .map((value: number | string) => (typeof value === 'number' ? value : Number(value)))
        .filter((value: number) => Number.isFinite(value) && value > 0)
      if (parsedDims.length > 0) {
        lossConfig.matryoshka_dims = parsedDims
      } else {
        delete lossConfig.matryoshka_dims
      }
    }

    setLoading(true)
    try {
      const task = await trainingApi.create(submitData)
      message.success(t('create.message.createSuccess'))
      navigate(`/training/${task.task_id}`)
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('create.message.createFailed'))
    } finally {
      setLoading(false)
    }
  }

  // Build model options: registered models + manual input
  const modelOptions = models.map(m => ({
    label: `${m.model_name} (${m.model_type})`,
    value: m.model_path || m.model_name,
  }))

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/training')}>
          {t('common:action.back')}
        </Button>
        <Title level={4} style={{ margin: 0 }}>{t('create.title')}</Title>
      </Space>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          model_type: 'embedding',
          training_method: 'sft',
          tuner_type: 'lora',
          num_train_epochs: 3,
          per_device_train_batch_size: 16,
          learning_rate: 2e-5,
          warmup_ratio: 0.1,
          gradient_accumulation_steps: 1,
          gradient_checkpointing: false,
          logging_steps: 1,
          eval_strategy: 'no',
          save_strategy: 'epoch',
          mixed_precision: 'bf16',
          lora_r: 16,
          lora_alpha: 32,
        }}
      >
        <Row gutter={24}>
          <Col span={12}>
            <Card title={t('create.basicConfig')} size="small">
              <Form.Item
                name="task_name"
                label={t('create.taskName.label')}
              >
                <Input placeholder={t('create.taskName.placeholder')} />
              </Form.Item>

              <Form.Item
                name="model_type"
                label={t('create.modelType.label')}
                rules={[{ required: true, message: t('create.modelType.required') }]}
              >
                <Select
                  options={modelTypeOptions}
                  onChange={handleModelTypeChange}
                />
              </Form.Item>

              <Form.Item
                name="training_method"
                label={t('create.trainingMethod.label')}
                rules={[{ required: true, message: t('create.trainingMethod.required') }]}
              >
                <Select options={trainingMethodOptions[modelType]} />
              </Form.Item>

              <Form.Item
                name="base_model_path"
                label={t('create.baseModel.label')}
                rules={[{ required: true, message: t('create.baseModel.required') }]}
                extra={t('create.baseModel.extra')}
              >
                <Select
                  showSearch
                  allowClear
                  placeholder={t('create.baseModel.placeholder')}
                  options={modelOptions}
                  mode="tags"
                  maxTagCount={1}
                  filterOption={(input, option) =>
                    (option?.label as string)?.toLowerCase().includes(input.toLowerCase()) ?? false
                  }
                />
              </Form.Item>

              {/* Dataset configuration - supports multiple datasets */}
              <div style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontWeight: 500 }}>{t('create.dataset.title')}</span>
                  <Button
                    type="link"
                    icon={<PlusOutlined />}
                    onClick={handleAddDataset}
                    size="small"
                  >
                    {t('create.dataset.add')}
                  </Button>
                </div>
                {datasetItems.map((item) => (
                  <Card
                    key={item.key}
                    size="small"
                    style={{ marginBottom: 8 }}
                    bodyStyle={{ padding: '8px 12px' }}
                  >
                    <Row gutter={8} align="middle">
                      <Col flex="80px">
                        <Select
                          value={item.split}
                          onChange={(value) => handleDatasetChange(item.key, 'split', value)}
                          options={[
                            { label: t('create.dataset.train'), value: 'train' },
                            { label: t('create.dataset.eval'), value: 'eval' },
                            { label: t('create.dataset.test'), value: 'test' },
                          ]}
                          style={{ width: '100%' }}
                        />
                      </Col>
                      <Col flex="1">
                        <Select
                          showSearch
                          allowClear
                          placeholder={t('create.dataset.placeholder')}
                          options={datasets.filter((d) => d.storage_path).map((d) => ({
                            label: `${d.name} (${d.dataset_type})`,
                            value: d.storage_path!,
                          }))}
                          mode="tags"
                          maxTagCount={1}
                          value={item.path ? [item.path] : []}
                          onChange={(values) => handleDatasetChange(item.key, 'path', values[values.length - 1] || '')}
                          style={{ width: '100%' }}
                        />
                      </Col>
                      <Col flex="160px">
                        <InputNumber
                          addonBefore={t('create.dataset.samplePrefix')}
                          placeholder={t('create.dataset.samplePlaceholder')}
                          min={1}
                          value={item.max_samples}
                          onChange={(value) => handleDatasetChange(item.key, 'max_samples', value ?? undefined)}
                          style={{ width: '100%' }}
                        />
                      </Col>
                      <Col flex="32px">
                        <Button
                          type="text"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={() => handleRemoveDataset(item.key)}
                          disabled={datasetItems.length === 1}
                        />
                      </Col>
                    </Row>
                  </Card>
                ))}
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {t('create.dataset.sampleHint')}
                </Typography.Text>
              </div>

              <Form.Item name="output_dir" label={t('create.outputDir.label')}>
                <Input placeholder={t('create.outputDir.placeholder')} />
              </Form.Item>

              {/* Model type specific parameters */}
              <Form.Item
                noStyle
                shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}
              >
                {({ getFieldValue }) => {
                  const type = getFieldValue('model_type')
                  if (type === 'embedding') {
                    return (
                      <>
                        <Divider>{t('create.embedding.divider')}</Divider>
                        <Form.Item name="embedding_loss_name" label={t('create.embedding.lossFunction')} initialValue="auto">
                          <Select options={embeddingLossOptions} />
                        </Form.Item>
                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, cur) => prev.embedding_loss_name !== cur.embedding_loss_name}
                        >
                          {({ getFieldValue }) => {
                            const lossName = getFieldValue('embedding_loss_name')
                            if (!lossName || lossName === 'auto') return null

                            const showScale = embeddingScaleLosses.includes(lossName)
                            const showMiniBatch = lossName === 'CachedMultipleNegativesRankingLoss'
                            const showMargin = embeddingMarginLosses.includes(lossName)
                            const showTriplet = embeddingTripletLosses.includes(lossName)
                            const showMegaBatch = embeddingMegaBatchLosses.includes(lossName)
                            const showSoftmax = embeddingSoftmaxLosses.includes(lossName)
                            const showGist = embeddingGistLosses.includes(lossName)
                            const showDynamicNeg = embeddingDynamicNegLosses.includes(lossName)
                            const showMatryoshka = embeddingMatryoshkaLosses.includes(lossName)
                            const marginDefault = ['BatchHardTripletLoss', 'BatchSemiHardTripletLoss', 'BatchAllTripletLoss'].includes(lossName)
                              ? 5.0
                              : 0.5

                            return (
                              <>
                                <Row gutter={16}>
                                  {showScale && (
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'scale']}
                                        label="Scale"
                                        initialValue={20.0}
                                        preserve={false}
                                      >
                                        <InputNumber min={0.1} max={200} step={0.1} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  )}
                                  {showMiniBatch && (
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'mini_batch_size']}
                                        label="Mini Batch Size"
                                        initialValue={32}
                                        preserve={false}
                                      >
                                        <InputNumber min={1} max={4096} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  )}
                                  {showMargin && (
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'margin']}
                                        label="Margin"
                                        initialValue={marginDefault}
                                        preserve={false}
                                      >
                                        <InputNumber min={0} max={100} step={0.1} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  )}
                                  {showTriplet && (
                                    <>
                                      <Col span={12}>
                                        <Form.Item
                                          name={['loss_config', 'distance_metric']}
                                          label={t('create.embedding.distanceMetric')}
                                          initialValue="cosine"
                                          preserve={false}
                                        >
                                          <Select options={[
                                            { label: 'Cosine', value: 'cosine' },
                                            { label: 'Euclidean', value: 'euclidean' },
                                          ]} />
                                        </Form.Item>
                                      </Col>
                                      <Col span={12}>
                                        <Form.Item
                                          name={['loss_config', 'triplet_margin']}
                                          label="Triplet Margin"
                                          initialValue={5.0}
                                          preserve={false}
                                        >
                                          <InputNumber min={0} max={100} step={0.1} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </>
                                  )}
                                  {showMegaBatch && (
                                    <>
                                      <Col span={12}>
                                        <Form.Item
                                          name={['loss_config', 'positive_margin']}
                                          label="Positive Margin"
                                          initialValue={0.8}
                                          preserve={false}
                                        >
                                          <InputNumber min={0} max={10} step={0.1} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                      <Col span={12}>
                                        <Form.Item
                                          name={['loss_config', 'negative_margin']}
                                          label="Negative Margin"
                                          initialValue={0.3}
                                          preserve={false}
                                        >
                                          <InputNumber min={0} max={10} step={0.1} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    </>
                                  )}
                                  {showSoftmax && (
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'num_labels']}
                                        label={t('create.embedding.numLabels')}
                                        initialValue={2}
                                        preserve={false}
                                      >
                                        <InputNumber min={2} max={1000} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  )}
                                </Row>
                                {showGist && (
                                  <Row gutter={16}>
                                    <Col span={24}>
                                      <Form.Item
                                        name={['loss_config', 'guide_model']}
                                        label={t('create.embedding.guideModel.label')}
                                        preserve={false}
                                      >
                                        <Input placeholder={t('create.embedding.guideModel.placeholder')} />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )}
                                {showDynamicNeg && (
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'normalize_by_logc']}
                                        label="normalize_by_logc"
                                        valuePropName="checked"
                                        initialValue={false}
                                        preserve={false}
                                      >
                                        <Switch />
                                      </Form.Item>
                                    </Col>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'ignore_empty']}
                                        label="ignore_empty"
                                        valuePropName="checked"
                                        initialValue={true}
                                        preserve={false}
                                      >
                                        <Switch />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )}
                                {showMatryoshka && (
                                  <Row gutter={16}>
                                    <Col span={24}>
                                      <Form.Item
                                        name={['loss_config', 'matryoshka_dims']}
                                        label={t('create.embedding.matryoshkaDims.label')}
                                        initialValue={['768', '512', '256', '128', '64']}
                                        preserve={false}
                                        extra={t('create.embedding.matryoshkaDims.extra')}
                                      >
                                        <Select
                                          mode="tags"
                                          tokenSeparators={[',', '\uff0c', ' ']}
                                          placeholder="768,512,256,128,64"
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )}
                              </>
                            )
                          }}
                        </Form.Item>
                      </>
                    )
                  }
                  if (type === 'reranker') {
                    return (
                      <>
                        <Divider>{t('create.reranker.divider')}</Divider>
                        <Form.Item name="reranker_loss_name" label={t('create.reranker.lossFunction')} initialValue="auto">
                          <Select options={rerankerLossOptions} />
                        </Form.Item>
                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, cur) => prev.reranker_loss_name !== cur.reranker_loss_name}
                        >
                          {({ getFieldValue }) => {
                            const lossName = getFieldValue('reranker_loss_name')
                            if (!lossName || lossName === 'auto') return null

                            const showScale = rerankerScaleLosses.includes(lossName)
                            const showMiniBatch = rerankerMiniBatchLosses.includes(lossName)
                            const showNegatives = rerankerNegativesLosses.includes(lossName)
                            const showTopK = rerankerTopKLosses.includes(lossName)
                            const showBeta = rerankerBetaLosses.includes(lossName)
                            const showDelta = rerankerDeltaLosses.includes(lossName)
                            const showTemperature = rerankerTemperatureLosses.includes(lossName)
                            const showSigma = rerankerSigmaLosses.includes(lossName)
                            const showRespectInputOrder = rerankerRespectInputOrderLosses.includes(lossName)

                            return (
                              <Row gutter={16}>
                                {showScale && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'scale']}
                                      label="Scale"
                                      initialValue={10.0}
                                      preserve={false}
                                    >
                                      <InputNumber min={0} max={100} step={0.1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showMiniBatch && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'mini_batch_size']}
                                      label="Mini Batch Size"
                                      initialValue={32}
                                      preserve={false}
                                    >
                                      <InputNumber min={1} max={512} step={1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showNegatives && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'num_negatives']}
                                      label="Num Negatives"
                                      initialValue={4}
                                      preserve={false}
                                    >
                                      <InputNumber min={1} max={128} step={1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showTopK && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'k']}
                                      label="Top K"
                                      preserve={false}
                                    >
                                      <InputNumber min={1} max={1024} step={1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showBeta && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'beta']}
                                      label="Beta"
                                      initialValue={1.0}
                                      preserve={false}
                                    >
                                      <InputNumber min={0} max={100} step={0.1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showDelta && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'delta']}
                                      label="Delta"
                                      initialValue={1.0}
                                      preserve={false}
                                    >
                                      <InputNumber min={0} max={100} step={0.1} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showTemperature && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'temperature']}
                                      label="Temperature"
                                      initialValue={1.0}
                                      preserve={false}
                                    >
                                      <InputNumber min={0.01} max={10} step={0.01} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showSigma && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'sigma']}
                                      label="Sigma"
                                      initialValue={1.0}
                                      preserve={false}
                                    >
                                      <InputNumber min={0.01} max={10} step={0.01} style={{ width: '100%' }} />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showRespectInputOrder && (
                                  <Col span={12}>
                                    <Form.Item
                                      name={['loss_config', 'respect_input_order']}
                                      label="Respect Input Order"
                                      initialValue={true}
                                      preserve={false}
                                      valuePropName="checked"
                                    >
                                      <Switch />
                                    </Form.Item>
                                  </Col>
                                )}
                              </Row>
                            )
                          }}
                        </Form.Item>
                      </>
                    )
                  }
                  return null
                }}
              </Form.Item>

              <Form.Item
                noStyle
                shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type || prev.training_method !== cur.training_method}
              >
                {({ getFieldValue }) => {
                  const type = getFieldValue('model_type')
                  const method = getFieldValue('training_method')
                  if (type !== 'decoder_reranker') return null
                  const isSft = method === 'sft'
                  return (
                    <>
                      <Divider>{t('create.decoderReranker.divider')}</Divider>
                      {isSft ? (
                        <>
                          <Row gutter={16}>
                            <Col span={12}>
                              <Form.Item name={['loss_config', 'n_docs']} label={t('create.decoderReranker.docsPerGroup')} initialValue={8} preserve={false}>
                                <InputNumber min={2} max={32} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                            <Col span={12}>
                              <Form.Item name={['loss_config', 'n_pos']} label={t('create.decoderReranker.positiveCount')} initialValue={1} extra={t('create.decoderReranker.positiveCountExtra')} preserve={false}>
                                <InputNumber min={0} max={8} style={{ width: '100%' }} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Row gutter={16}>
                            <Col span={12}>
                              <Form.Item
                                name={['loss_config', 'name']}
                                label={t('create.decoderReranker.lossFunction')}
                                initialValue="infonce"
                                extra={t('create.decoderReranker.lossFunctionExtra')}
                                preserve={false}
                              >
                                <Select options={[
                                  { label: 'InfoNCE', value: 'infonce' },
                                  { label: 'BCE', value: 'bce' },
                                  { label: 'LambdaLoss', value: 'lambda_loss' },
                                  { label: 'ListMLE', value: 'list_mle' },
                                  { label: 'RankNet', value: 'ranknet' },
                                ]} />
                              </Form.Item>
                            </Col>
                          </Row>
                          <Form.Item noStyle shouldUpdate={(prev, cur) => prev.loss_config?.name !== cur.loss_config?.name}>
                            {({ getFieldValue }) => {
                              const lossName = getFieldValue(['loss_config', 'name'])
                              if (lossName === 'infonce') {
                                return (
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'temperature']}
                                        label={t('create.decoderReranker.temperature')}
                                        initialValue={0.05}
                                        preserve={false}
                                      >
                                        <InputNumber min={0.01} max={1} step={0.01} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'infonce_mode']}
                                        label={t('create.decoderReranker.infonceMode')}
                                        initialValue="single"
                                        preserve={false}
                                      >
                                        <Select options={[
                                          { label: t('create.decoderReranker.infonceSingle'), value: 'single' },
                                          { label: t('create.decoderReranker.infoncePosset'), value: 'posset' },
                                          { label: t('create.decoderReranker.infonceAvgpos'), value: 'avgpos' },
                                        ]} />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )
                              }
                              if (lossName === 'ranknet') {
                                return (
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['loss_config', 'ranknet_max_pairs_per_batch']}
                                        label={t('create.decoderReranker.maxPairsPerBatch')}
                                        initialValue={2000000}
                                        preserve={false}
                                      >
                                        <InputNumber min={1000} max={10000000} step={1000} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )
                              }
                              return null
                            }}
                          </Form.Item>
                          <Form.Item
                            noStyle
                            shouldUpdate={(prev, cur) => prev.loss_config?.name !== cur.loss_config?.name}
                          >
                            {({ getFieldValue }) => {
                              const lossName = getFieldValue(['loss_config', 'name'])
                              if (lossName !== 'lambda_loss') return null
                              return (
                                <Row gutter={16}>
                                  <Col span={12}>
                                    <Form.Item name={['loss_config', 'metric']} label={t('create.decoderReranker.targetMetric')} initialValue="ndcg" preserve={false}>
                                      <Select options={lambdaMetricOptions} />
                                    </Form.Item>
                                  </Col>
                                </Row>
                              )
                            }}
                          </Form.Item>
                          <Form.Item name={['loss_config', 'chunk_size']} label={t('create.decoderReranker.chunkSize')} initialValue={0} extra={t('create.decoderReranker.chunkSizeExtra')} preserve={false}>
                            <InputNumber min={0} max={64} style={{ width: '100%' }} />
                          </Form.Item>
                        </>
                      ) : (
                        <Row gutter={16}>
                          <Col span={12}>
                            <Form.Item name={['rl_config', 'n_docs']} label={t('create.decoderReranker.docsPerGroup')} initialValue={8} preserve={false}>
                              <InputNumber min={2} max={32} style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                        </Row>
                      )}

                      {/* RL training specific parameters */}
                      {method && method !== 'sft' && (
                        <>
                          <Divider>{t('create.rl.divider')}</Divider>
                          <Form.Item
                            name="sft_checkpoint_path"
                            label={t('create.rl.sftModelPath.label')}
                            rules={[{ required: true, message: t('create.rl.sftModelPath.required') }]}
                            extra={t('create.rl.sftModelPath.extra')}
                          >
                            <Input placeholder={t('create.rl.sftModelPath.placeholder')} />
                          </Form.Item>
                          <Form.Item
                            noStyle
                            shouldUpdate={(prev, cur) => (
                              prev.model_type !== cur.model_type
                              || prev.training_method !== cur.training_method
                              || prev.rl_config?.reward_type !== cur.rl_config?.reward_type
                            )}
                          >
                            {({ getFieldValue }) => {
                              const type = getFieldValue('model_type')
                              const currentMethod = getFieldValue('training_method')
                              const isDpo = currentMethod === 'dpo'
                              const rewardType = getFieldValue(['rl_config', 'reward_type']) || 'rank_based'
                              const showRewardK = rewardType === 'ndcg_based' || rewardType === 'recall_based'
                              const showDecoderDpoBeta = type === 'decoder_reranker'

                              if (isDpo) {
                                return (
                                  <Row gutter={16}>
                                    {showDecoderDpoBeta ? (
                                      <Col span={12}>
                                        <Form.Item
                                          name={['rl_config', 'beta']}
                                          label="DPO Beta"
                                          initialValue={0.1}
                                          preserve={false}
                                        >
                                          <InputNumber min={0.01} max={10} step={0.01} style={{ width: '100%' }} />
                                        </Form.Item>
                                      </Col>
                                    ) : null}
                                    <Col span={12}>
                                      <Form.Item
                                        name={['rl_config', 'reference_free']}
                                        label="Reference Free"
                                        valuePropName="checked"
                                        initialValue={false}
                                        preserve={false}
                                      >
                                        <Switch />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                )
                              }

                              return (
                                <>
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item name={['rl_config', 'kl_coef']} label={t('create.rl.klCoef')} initialValue={0.1} preserve={false}>
                                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                    <Col span={12}>
                                      <Form.Item name={['rl_config', 'clip_range']} label="Clip Range" initialValue={0.2} preserve={false}>
                                        <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item name={['rl_config', 'reward_type']} label={t('create.rl.rewardType')} initialValue="rank_based" preserve={false}>
                                        <Select options={rlRewardKeys.map(r => ({ label: t(r.key), value: r.value }))} />
                                      </Form.Item>
                                    </Col>
                                    <Col span={12}>
                                      {showRewardK ? (
                                        <Form.Item name={['rl_config', 'reward_k']} label="Reward@K" initialValue={10} preserve={false}>
                                          <InputNumber min={1} max={100} style={{ width: '100%' }} />
                                        </Form.Item>
                                      ) : null}
                                    </Col>
                                  </Row>
                                  <Row gutter={16}>
                                    <Col span={12}>
                                      <Form.Item
                                        name={['rl_config', 'scale_rewards']}
                                        label="Scale Rewards"
                                        valuePropName="checked"
                                        initialValue={false}
                                        preserve={false}
                                      >
                                        <Switch />
                                      </Form.Item>
                                    </Col>
                                    <Col span={12}>
                                      <Form.Item name={['rl_config', 'num_iterations']} label={t('create.rl.updatePerBatch')} initialValue={1} preserve={false}>
                                        <InputNumber min={1} max={16} style={{ width: '100%' }} />
                                      </Form.Item>
                                    </Col>
                                  </Row>
                                </>
                              )
                            }}
                          </Form.Item>
                          <Form.Item
                            name={['rl_config', 'chunk_size']}
                            label={t('create.rl.chunkSize')}
                            initialValue={0}
                            extra={t('create.rl.chunkSizeExtra')}
                            preserve={false}
                          >
                            <InputNumber min={0} max={64} style={{ width: '100%' }} />
                          </Form.Item>
                        </>
                      )}
                    </>
                  )
                }}
              </Form.Item>

              <Form.Item
                noStyle
                shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type || prev.training_method !== cur.training_method}
              >
                {({ getFieldValue }) => {
                  const type = getFieldValue('model_type')
                  const method = getFieldValue('training_method')
                  if (type !== 'llm' || (method !== 'dpo' && method !== 'orpo')) return null

                  return (
                    <>
                      <Divider>{t('create.preference.divider')}</Divider>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item
                            name={['rl_config', 'beta']}
                            label={method === 'dpo' ? 'DPO Beta' : 'ORPO Beta'}
                            initialValue={0.1}
                            preserve={false}
                          >
                            <InputNumber min={0.01} max={10} step={0.01} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item
                            name={['rl_config', 'rankings_direction']}
                            label={t('create.preference.rankingsDirection')}
                            initialValue="auto"
                            extra={t('create.preference.rankingsDirectionExtra')}
                            preserve={false}
                          >
                            <Select
                              options={[
                                { label: t('create.preference.rankingsDirectionAuto'), value: 'auto' },
                                { label: t('create.preference.rankingsDirectionHigher'), value: 'higher_is_better' },
                                { label: t('create.preference.rankingsDirectionLower'), value: 'lower_is_better' },
                              ]}
                            />
                          </Form.Item>
                        </Col>
                      </Row>
                    </>
                  )
                }}
              </Form.Item>
            </Card>
          </Col>

          <Col span={12}>
            <Card title={t('create.trainingParams')} size="small">
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="num_train_epochs"
                    label={t('create.params.epochs')}
                    rules={[{ required: true }]}
                  >
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="per_device_train_batch_size"
                    label={t('create.params.batchSize')}
                    rules={[{ required: true }]}
                  >
                    <InputNumber min={1} max={256} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="learning_rate"
                    label={t('create.params.learningRate')}
                    rules={[{ required: true }]}
                  >
                    <InputNumber
                      min={0}
                      max={1}
                      step={0.00001}
                      style={{ width: '100%' }}
                      formatter={(value) => value ? `${value}` : ''}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="warmup_ratio" label={t('create.params.warmupRatio')}>
                    <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item name="gradient_accumulation_steps" label={t('create.params.gradientAccumulation')}>
                    <InputNumber min={1} max={128} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="logging_steps" label={t('create.params.loggingSteps')} extra={t('create.params.loggingStepsExtra')}>
                    <InputNumber min={1} max={1000} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item name="mixed_precision" label={t('create.params.mixedPrecision')}>
                    <Select
                      allowClear
                      placeholder={t('create.params.mixedPrecisionPlaceholder')}
                      options={[
                        { label: 'BF16', value: 'bf16' },
                        { label: 'FP16', value: 'fp16' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item name="max_length" label={t('create.params.maxLength')}>
                    <InputNumber min={32} max={32768} style={{ width: '100%' }} placeholder={t('create.params.maxLengthPlaceholder')} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                noStyle
                shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}
              >
                {({ getFieldValue }) => getFieldValue('model_type') === 'llm' ? (
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        name="gradient_checkpointing"
                        label={t('create.params.gradientCheckpointing')}
                        valuePropName="checked"
                      >
                        <Switch />
                      </Form.Item>
                    </Col>
                  </Row>
                ) : null}
              </Form.Item>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="deepspeed"
                    label={t('create.params.deepspeed')}
                    extra={t('create.params.deepspeedExtra')}
                  >
                    <Select
                      allowClear
                      placeholder={t('create.params.deepspeedPlaceholder')}
                      options={[
                        { label: 'ZeRO-2', value: 'zero2' },
                        { label: 'ZeRO-3', value: 'zero3' },
                        { label: 'ZeRO-2 + CPU Offload', value: 'zero2_offload' },
                        { label: 'ZeRO-3 + CPU Offload', value: 'zero3_offload' },
                      ]}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Divider>{t('create.evalAndSave.divider')}</Divider>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="eval_strategy"
                    label={t('create.evalAndSave.evalStrategy')}
                    extra={t('create.evalAndSave.evalStrategyExtra')}
                  >
                    <Select options={[
                      { label: t('create.evalAndSave.evalNoEval'), value: 'no' },
                      { label: t('create.evalAndSave.evalPerEpoch'), value: 'epoch' },
                      { label: t('create.evalAndSave.evalPerSteps'), value: 'steps' },
                    ]} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    noStyle
                    shouldUpdate={(prev, cur) => prev.eval_strategy !== cur.eval_strategy}
                  >
                    {({ getFieldValue }) =>
                      getFieldValue('eval_strategy') === 'steps' ? (
                        <Form.Item name="eval_steps" label={t('create.evalAndSave.evalSteps')}>
                          <InputNumber min={1} max={10000} style={{ width: '100%' }} placeholder={t('create.evalAndSave.evalStepsPlaceholder')} />
                        </Form.Item>
                      ) : null
                    }
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item name="save_strategy" label={t('create.evalAndSave.saveStrategy')}>
                    <Select options={[
                      { label: t('create.evalAndSave.saveNoSave'), value: 'no' },
                      { label: t('create.evalAndSave.savePerEpoch'), value: 'epoch' },
                      { label: t('create.evalAndSave.savePerSteps'), value: 'steps' },
                    ]} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    noStyle
                    shouldUpdate={(prev, cur) => prev.save_strategy !== cur.save_strategy}
                  >
                    {({ getFieldValue }) =>
                      getFieldValue('save_strategy') === 'steps' ? (
                        <Form.Item name="save_steps" label={t('create.evalAndSave.saveSteps')}>
                          <InputNumber min={1} max={10000} style={{ width: '100%' }} placeholder={t('create.evalAndSave.saveStepsPlaceholder')} />
                        </Form.Item>
                      ) : null
                    }
                  </Form.Item>
                </Col>
              </Row>

              <Divider>{t('create.tuner.divider')}</Divider>

              <Form.Item
                name="tuner_type"
                label={t('create.tuner.type')}
              >
                <Select options={tunerTypeOptions} />
              </Form.Item>

              <Form.Item
                noStyle
                shouldUpdate={(prev, cur) => prev.tuner_type !== cur.tuner_type}
              >
                {({ getFieldValue }) => {
                  const tuner = getFieldValue('tuner_type')
                  if (tuner !== 'lora') return null
                  return (
                    <Row gutter={16}>
                      <Col span={8}>
                        <Form.Item name="lora_r" label="LoRA Rank">
                          <InputNumber min={1} max={256} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item name="lora_alpha" label="LoRA Alpha">
                          <InputNumber min={1} max={256} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item name="lora_dropout" label="Dropout">
                          <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} placeholder="0.0" />
                        </Form.Item>
                      </Col>
                    </Row>
                  )
                }}
              </Form.Item>

              <Divider>{t('create.resource.divider')}</Divider>

              <Form.Item
                name="gpu_ids"
                label={t('create.resource.gpuSelect')}
                extra={t('create.resource.gpuExtra')}
              >
                <GpuSelect mode="multiple" />
              </Form.Item>
            </Card>
          </Col>
        </Row>

        <div style={{ marginTop: 24, textAlign: 'center' }}>
          <Space size="large">
            <Button onClick={() => navigate('/training')}>{t('common:action.cancel')}</Button>
            <Button type="primary" htmlType="submit" loading={loading} disabled={loading}>
              {t('create.submitButton')}
            </Button>
          </Space>
        </div>
      </Form>
    </div>
  )
}
