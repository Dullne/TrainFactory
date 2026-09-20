import { useState, useEffect, useLayoutEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  Form,
  Input,
  Select,
  InputNumber,
  Button,
  Row,
  Col,
  Space,
  message,
  Typography,
  Divider,
  Switch,
  Popconfirm,
  type FormProps,
} from 'antd'
import { ArrowLeftOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons'
import { trainingApi, datasetApi, modelApi } from '@/services/api'
import { GpuSelect } from '@/components/GpuSelect'
import { TrainingCreateSection } from './TrainingCreateSection'
import { useAuth } from '@/auth/AuthContext'
import { createDatasetItem, useTrainingDraft } from './useTrainingDraft'
import type {
  CreateTrainingRequest,
  ModelType,
  TrainingMethod,
  TunerType,
  Dataset,
  RegisteredModel,
} from '@/types'
import './TrainingCreate.css'

const { Title } = Typography
const sectionIds = ['basic', 'datasets', 'model', 'training', 'evaluation', 'tuner', 'resources']

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
const rerankerMiniBatchLosses = [
  'CachedMultipleNegativesRankingLoss',
  'RankNetLoss',
  'LambdaLoss',
  'ListMLELoss',
  'ListNetLoss',
  'PListMLELoss',
]
const rerankerNegativesLosses = [
  'MultipleNegativesRankingLoss',
  'CachedMultipleNegativesRankingLoss',
]
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
  const { user } = useAuth()
  const [revision, setRevision] = useState(0)
  if (!user) return null
  return (
    <TrainingCreateForm
      key={`${user.user_id}:${revision}`}
      accountId={user.user_id}
      onDiscard={() => setRevision((current) => current + 1)}
    />
  )
}

function TrainingCreateForm({
  accountId,
  onDiscard,
}: {
  accountId: string
  onDiscard: () => void
}) {
  const navigate = useNavigate()
  const { t, i18n } = useTranslation(['training', 'common'])
  const [form] = Form.useForm()
  const draft = useTrainingDraft(accountId, form)
  const { datasetItems, updateDatasets } = draft
  const trainingMethod = Form.useWatch('training_method', form)
  const summaryValues = Form.useWatch(
    (values) => ({
      epochs: values.num_train_epochs,
      batch: values.per_device_train_batch_size,
      rate: values.learning_rate,
      tuner: values.tuner_type,
      rank: values.lora_r,
      eval: values.eval_strategy,
      save: values.save_strategy,
      gpu: values.gpu_ids,
      embeddingLoss: values.embedding_loss_name,
      rerankerLoss: values.reranker_loss_name,
    }),
    form
  )
  const [loading, setLoading] = useState(false)
  const mountedRef = useRef(false)
  const requestGenerationRef = useRef(0)
  const pendingRequestRef = useRef(false)
  useLayoutEffect(() => {
    mountedRef.current = true
    const invalidate = () => {
      mountedRef.current = false
      requestGenerationRef.current += 1
    }
    // Invalidate synchronously when the session ends, including the interval
    // before auth causes this account-keyed form to unmount.
    window.addEventListener('tf:workspace-session-cleared', invalidate)
    return () => {
      invalidate()
      window.removeEventListener('tf:workspace-session-cleared', invalidate)
    }
  }, [accountId])
  const [collapsedSections, setCollapsedSections] = useState<string[]>([
    'model',
    'evaluation',
    'tuner',
    'resources',
  ])
  const [focusFieldId, setFocusFieldId] = useState<string | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [models, setModels] = useState<RegisteredModel[]>([])
  const [modelType, setModelType] = useState<ModelType>(
    (draft.initialValues.model_type as ModelType) ?? 'embedding'
  )

  const selectedLoss =
    modelType === 'embedding' ? summaryValues?.embeddingLoss : summaryValues?.rerankerLoss
  const modelSummary = ['embedding', 'reranker'].includes(modelType)
    ? selectedLoss && selectedLoss !== 'auto'
      ? selectedLoss
      : t('create.lossAuto')
    : (trainingMethod ?? 'sft').toUpperCase()
  const sectionSummaries: Record<string, string> = {
    basic: `${modelType} · ${(trainingMethod ?? 'sft').toUpperCase()}`,
    datasets: t('create.sections.datasetSummary', {
      count: datasetItems.filter((item) => item.path).length,
    }),
    model: `${modelType} · ${modelSummary}`,
    training: t('create.sections.trainingSummary', {
      epochs: summaryValues?.epochs ?? '—',
      batch: summaryValues?.batch ?? '—',
      rate: summaryValues?.rate ?? '—',
    }),
    evaluation: `${t('create.evalAndSave.evalStrategy')}: ${t(`create.sections.eval_${summaryValues?.eval ?? 'no'}`)} · ${t('create.evalAndSave.saveStrategy')}: ${t(`create.sections.save_${summaryValues?.save ?? 'epoch'}`)}`,
    tuner:
      summaryValues?.tuner === 'full'
        ? t('create.tuner.fullFinetune')
        : `LoRA · Rank ${summaryValues?.rank ?? 16}`,
    resources: summaryValues?.gpu?.length
      ? `GPU ${summaryValues.gpu.join(', ')}`
      : t('detail.gpuAutoAssign'),
  }

  useEffect(() => {
    if (!focusFieldId) return
    // Run after the section has expanded and React has committed its visible fields.
    const field = document.getElementById(focusFieldId)
    field?.scrollIntoView({ block: 'center' })
    field?.focus({ preventScroll: true })
    setFocusFieldId(null)
  }, [focusFieldId])

  const getSectionProps = (sectionId: string) => ({
    sectionId,
    summary: sectionSummaries[sectionId],
    collapsed: collapsedSections.includes(sectionId),
    onToggle: () =>
      setCollapsedSections((current) =>
        current.includes(sectionId)
          ? current.filter((id) => id !== sectionId)
          : [...current, sectionId]
      ),
  })

  const expandSection = (sectionId: string) => {
    setCollapsedSections((current) => current.filter((id) => id !== sectionId))
  }

  const handleValidationFailed: FormProps<CreateTrainingRequest>['onFinishFailed'] = ({
    errorFields,
  }) => {
    // Ant Design uses the name path joined by underscores for unnamed Form field IDs.
    const fieldIds = errorFields.map(({ name }) => name.join('_'))
    const invalidSections = new Set(
      fieldIds.map(
        (id) =>
          document.getElementById(id)?.closest<HTMLElement>('[data-training-section]')?.dataset
            .trainingSection
      )
    )
    setCollapsedSections((current) => current.filter((id) => !invalidSections.has(id)))
    setFocusFieldId(fieldIds[0] ?? null)
  }

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
    draft.save()
  }

  // Dataset item handlers
  const handleAddDataset = () => {
    expandSection('datasets')
    updateDatasets([...datasetItems, createDatasetItem()])
  }

  const handleRemoveDataset = (key: string) => {
    if (datasetItems.length > 1) {
      updateDatasets(datasetItems.filter((item) => item.key !== key))
    }
  }

  const handleDatasetChange = (
    key: string,
    field: 'path' | 'max_samples' | 'split',
    value: string | number | undefined
  ) => {
    updateDatasets(
      datasetItems.map((item) => (item.key === key ? { ...item, [field]: value } : item))
    )
  }

  const clearDraft = (expectedRevision?: number) => {
    const result = draft.clear(expectedRevision)
    if (result === false) message.warning(t('create.draft.clearFailed'))
    return result
  }

  const handleSubmit = async (values: CreateTrainingRequest) => {
    if (!mountedRef.current || pendingRequestRef.current) return
    // Build datasets array from datasetItems
    const validDatasets = datasetItems
      .filter((item) => item.path)
      .map((item) => ({
        path: item.path,
        max_samples: item.max_samples || undefined,
        split: item.split,
      }))

    // Check at least one training dataset
    const trainDatasets = validDatasets.filter((d) => d.split === 'train')
    if (trainDatasets.length === 0) {
      expandSection('datasets')
      setFocusFieldId(`dataset-path-${datasetItems[0].key}`)
      message.error(t('create.dataset.requireTrain'))
      return
    }

    // Handle base_model_path from Select tags mode
    let modelPath = values.base_model_path
    if (Array.isArray(modelPath)) {
      modelPath = (modelPath as string[])[0] || ''
    }

    // Convert mixed_precision select to bf16/fp16 booleans
    const { mixed_precision, ...restValues } = values as CreateTrainingRequest & {
      mixed_precision?: string
    }

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

    const submittedRevision = draft.getRevision()
    const generation = ++requestGenerationRef.current
    const isCurrentRequest = () => mountedRef.current && requestGenerationRef.current === generation
    pendingRequestRef.current = true
    setLoading(true)
    try {
      const task = await trainingApi.create(submitData)
      if (!isCurrentRequest()) return
      if (clearDraft(submittedRevision) === 'changed') {
        message.success(t('create.message.createdDraftKept', { taskId: task.task_id }))
        return
      }
      message.success(t('create.message.createSuccess'))
      navigate(`/training/${task.task_id}`)
    } catch (error) {
      if (isCurrentRequest()) {
        message.error(error instanceof Error ? error.message : t('create.message.createFailed'))
      }
    } finally {
      if (isCurrentRequest()) {
        pendingRequestRef.current = false
        setLoading(false)
      }
    }
  }

  // Build model options: registered models + manual input
  const modelOptions = models.map((m) => ({
    label: `${m.model_name} (${m.model_type})`,
    value: m.model_path || m.model_name,
  }))

  return (
    <div className="training-create">
      <div className="training-create-header">
        <Button
          className="training-create-back"
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/training')}
        >
          {t('common:action.back')}
        </Button>
        <div>
          <Title level={1}>{t('create.title')}</Title>
          <Typography.Paragraph type="secondary">{t('create.description')}</Typography.Paragraph>
        </div>
        <Space className="training-create-section-actions">
          <Button onClick={() => setCollapsedSections([])}>{t('create.sections.expandAll')}</Button>
          <Button onClick={() => setCollapsedSections(sectionIds)}>
            {t('create.sections.collapseAll')}
          </Button>
        </Space>
      </div>

      <Form
        form={form}
        layout="vertical"
        size="large"
        onFinish={handleSubmit}
        onFinishFailed={handleValidationFailed}
        onValuesChange={() => draft.save()}
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
          ...draft.initialValues,
        }}
      >
        <div className="training-create-columns">
          <TrainingCreateSection {...getSectionProps('basic')} title={t('create.basicConfig')}>
            <Form.Item name="task_name" label={t('create.taskName.label')}>
              <Input placeholder={t('create.taskName.placeholder')} />
            </Form.Item>

            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="model_type"
                  label={t('create.modelType.label')}
                  rules={[{ required: true, message: t('create.modelType.required') }]}
                >
                  <Select options={modelTypeOptions} onChange={handleModelTypeChange} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="training_method"
                  label={t('create.trainingMethod.label')}
                  rules={[{ required: true, message: t('create.trainingMethod.required') }]}
                >
                  <Select options={trainingMethodOptions[modelType]} />
                </Form.Item>
              </Col>
            </Row>

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
          </TrainingCreateSection>

          <TrainingCreateSection
            {...getSectionProps('datasets')}
            title={t('create.dataset.title')}
            extra={
              <Button
                type="default"
                icon={<PlusOutlined />}
                onClick={handleAddDataset}
                size="small"
              >
                {t('create.dataset.add')}
              </Button>
            }
          >
            {datasetItems.map((item) => (
              <div key={item.key} className="training-create-dataset-row">
                <div className="training-create-dataset-path">
                  <label htmlFor={`dataset-path-${item.key}`}>
                    {t('create.dataset.pathLabel')}
                  </label>
                  <Select
                    id={`dataset-path-${item.key}`}
                    aria-label={t('create.dataset.pathLabel')}
                    showSearch
                    allowClear
                    placeholder={t('create.dataset.placeholder')}
                    options={datasets
                      .filter((d) => d.storage_path)
                      .map((d) => ({
                        label: `${d.name} (${d.dataset_type})`,
                        value: d.storage_path!,
                      }))}
                    mode="tags"
                    maxTagCount={1}
                    value={item.path ? [item.path] : []}
                    onChange={(values) =>
                      handleDatasetChange(item.key, 'path', values[values.length - 1] || '')
                    }
                    style={{ width: '100%' }}
                  />
                </div>
                <div className="training-create-dataset-split">
                  <label htmlFor={`dataset-split-${item.key}`}>
                    {t('create.dataset.splitLabel')}
                  </label>
                  <Select
                    id={`dataset-split-${item.key}`}
                    aria-label={t('create.dataset.splitLabel')}
                    value={item.split}
                    onChange={(value) => handleDatasetChange(item.key, 'split', value)}
                    options={[
                      { label: t('create.dataset.train'), value: 'train' },
                      { label: t('create.dataset.eval'), value: 'eval' },
                      { label: t('create.dataset.test'), value: 'test' },
                    ]}
                    style={{ width: '100%' }}
                  />
                </div>
                <div className="training-create-dataset-samples">
                  <label htmlFor={`dataset-samples-${item.key}`}>
                    {t('create.dataset.samplesLabel')}
                  </label>
                  <InputNumber
                    id={`dataset-samples-${item.key}`}
                    aria-label={t('create.dataset.samplesLabel')}
                    placeholder={t('create.dataset.samplePlaceholder')}
                    min={1}
                    value={item.max_samples}
                    onChange={(value) =>
                      handleDatasetChange(item.key, 'max_samples', value ?? undefined)
                    }
                    style={{ width: '100%' }}
                  />
                </div>
                <Button
                  className="training-create-dataset-remove"
                  type="text"
                  danger
                  aria-label={t('create.dataset.remove')}
                  icon={<DeleteOutlined />}
                  onClick={() => handleRemoveDataset(item.key)}
                  disabled={datasetItems.length === 1}
                />
              </div>
            ))}
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
              {t('create.dataset.sampleHint')}
            </Typography.Text>
          </TrainingCreateSection>

          <TrainingCreateSection
            {...getSectionProps('training')}
            title={t('create.trainingParams')}
          >
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="num_train_epochs"
                  label={t('create.params.epochs')}
                  rules={[{ required: true }]}
                >
                  <InputNumber min={1} max={100} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
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
              <Col xs={24} sm={12}>
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
                    formatter={(value) => (value ? `${value}` : '')}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item name="warmup_ratio" label={t('create.params.warmupRatio')}>
                  <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="gradient_accumulation_steps"
                  label={t('create.params.gradientAccumulation')}
                >
                  <InputNumber min={1} max={128} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="logging_steps"
                  label={t('create.params.loggingSteps')}
                  extra={t('create.params.loggingStepsExtra')}
                >
                  <InputNumber min={1} max={1000} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col xs={24} sm={12}>
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
              <Col xs={24} sm={12}>
                <Form.Item name="max_length" label={t('create.params.maxLength')}>
                  <InputNumber
                    min={32}
                    max={32768}
                    style={{ width: '100%' }}
                    placeholder={t('create.params.maxLengthPlaceholder')}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item noStyle shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}>
              {({ getFieldValue }) =>
                getFieldValue('model_type') === 'llm' ? (
                  <Row gutter={16}>
                    <Col xs={24} sm={12}>
                      <Form.Item
                        name="gradient_checkpointing"
                        label={t('create.params.gradientCheckpointing')}
                        valuePropName="checked"
                      >
                        <Switch />
                      </Form.Item>
                    </Col>
                  </Row>
                ) : null
              }
            </Form.Item>

            <Row gutter={16}>
              <Col span={24}>
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
          </TrainingCreateSection>

          <TrainingCreateSection
            {...getSectionProps('model')}
            title={t('create.modelParams')}
            className="training-create-model-options"
            hidden={modelType === 'llm' && !['dpo', 'orpo'].includes(trainingMethod)}
          >
            {/* Model type specific parameters */}
            <Form.Item noStyle shouldUpdate={(prev, cur) => prev.model_type !== cur.model_type}>
              {({ getFieldValue }) => {
                const type = getFieldValue('model_type')
                if (type === 'embedding') {
                  return (
                    <>
                      <Form.Item
                        name="embedding_loss_name"
                        label={t('create.embedding.lossFunction')}
                        initialValue="auto"
                      >
                        <Select options={embeddingLossOptions} />
                      </Form.Item>
                      <Form.Item
                        noStyle
                        shouldUpdate={(prev, cur) =>
                          prev.embedding_loss_name !== cur.embedding_loss_name
                        }
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
                          const marginDefault = [
                            'BatchHardTripletLoss',
                            'BatchSemiHardTripletLoss',
                            'BatchAllTripletLoss',
                          ].includes(lossName)
                            ? 5.0
                            : 0.5

                          return (
                            <>
                              <Row gutter={16}>
                                {showScale && (
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['loss_config', 'scale']}
                                      label="Scale"
                                      initialValue={20.0}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={0.1}
                                        max={200}
                                        step={0.1}
                                        style={{ width: '100%' }}
                                      />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showMiniBatch && (
                                  <Col xs={24} sm={12}>
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
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['loss_config', 'margin']}
                                      label="Margin"
                                      initialValue={marginDefault}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={0}
                                        max={100}
                                        step={0.1}
                                        style={{ width: '100%' }}
                                      />
                                    </Form.Item>
                                  </Col>
                                )}
                                {showTriplet && (
                                  <>
                                    <Col xs={24} sm={12}>
                                      <Form.Item
                                        name={['loss_config', 'distance_metric']}
                                        label={t('create.embedding.distanceMetric')}
                                        initialValue="cosine"
                                        preserve={false}
                                      >
                                        <Select
                                          options={[
                                            { label: 'Cosine', value: 'cosine' },
                                            { label: 'Euclidean', value: 'euclidean' },
                                          ]}
                                        />
                                      </Form.Item>
                                    </Col>
                                    <Col xs={24} sm={12}>
                                      <Form.Item
                                        name={['loss_config', 'triplet_margin']}
                                        label="Triplet Margin"
                                        initialValue={5.0}
                                        preserve={false}
                                      >
                                        <InputNumber
                                          min={0}
                                          max={100}
                                          step={0.1}
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                  </>
                                )}
                                {showMegaBatch && (
                                  <>
                                    <Col xs={24} sm={12}>
                                      <Form.Item
                                        name={['loss_config', 'positive_margin']}
                                        label="Positive Margin"
                                        initialValue={0.8}
                                        preserve={false}
                                      >
                                        <InputNumber
                                          min={0}
                                          max={10}
                                          step={0.1}
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                    <Col xs={24} sm={12}>
                                      <Form.Item
                                        name={['loss_config', 'negative_margin']}
                                        label="Negative Margin"
                                        initialValue={0.3}
                                        preserve={false}
                                      >
                                        <InputNumber
                                          min={0}
                                          max={10}
                                          step={0.1}
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                  </>
                                )}
                                {showSoftmax && (
                                  <Col xs={24} sm={12}>
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
                                      <Input
                                        placeholder={t('create.embedding.guideModel.placeholder')}
                                      />
                                    </Form.Item>
                                  </Col>
                                </Row>
                              )}
                              {showDynamicNeg && (
                                <Row gutter={16}>
                                  <Col xs={24} sm={12}>
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
                                  <Col xs={24} sm={12}>
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
                      <Form.Item
                        name="reranker_loss_name"
                        label={t('create.reranker.lossFunction')}
                        initialValue="auto"
                      >
                        <Select options={rerankerLossOptions} />
                      </Form.Item>
                      <Form.Item
                        noStyle
                        shouldUpdate={(prev, cur) =>
                          prev.reranker_loss_name !== cur.reranker_loss_name
                        }
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
                          const showRespectInputOrder =
                            rerankerRespectInputOrderLosses.includes(lossName)

                          return (
                            <Row gutter={16}>
                              {showScale && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'scale']}
                                    label="Scale"
                                    initialValue={10.0}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={0}
                                      max={100}
                                      step={0.1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showMiniBatch && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'mini_batch_size']}
                                    label="Mini Batch Size"
                                    initialValue={32}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={1}
                                      max={512}
                                      step={1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showNegatives && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'num_negatives']}
                                    label="Num Negatives"
                                    initialValue={4}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={1}
                                      max={128}
                                      step={1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showTopK && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'k']}
                                    label="Top K"
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={1}
                                      max={1024}
                                      step={1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showBeta && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'beta']}
                                    label="Beta"
                                    initialValue={1.0}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={0}
                                      max={100}
                                      step={0.1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showDelta && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'delta']}
                                    label="Delta"
                                    initialValue={1.0}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={0}
                                      max={100}
                                      step={0.1}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showTemperature && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'temperature']}
                                    label="Temperature"
                                    initialValue={1.0}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={0.01}
                                      max={10}
                                      step={0.01}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showSigma && (
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'sigma']}
                                    label="Sigma"
                                    initialValue={1.0}
                                    preserve={false}
                                  >
                                    <InputNumber
                                      min={0.01}
                                      max={10}
                                      step={0.01}
                                      style={{ width: '100%' }}
                                    />
                                  </Form.Item>
                                </Col>
                              )}
                              {showRespectInputOrder && (
                                <Col xs={24} sm={12}>
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
              shouldUpdate={(prev, cur) =>
                prev.model_type !== cur.model_type || prev.training_method !== cur.training_method
              }
            >
              {({ getFieldValue }) => {
                const type = getFieldValue('model_type')
                const method = getFieldValue('training_method')
                if (type !== 'decoder_reranker') return null
                const isSft = method === 'sft'
                return (
                  <>
                    {isSft ? (
                      <>
                        <Row gutter={16}>
                          <Col xs={24} sm={12}>
                            <Form.Item
                              name={['loss_config', 'n_docs']}
                              label={t('create.decoderReranker.docsPerGroup')}
                              initialValue={8}
                              preserve={false}
                            >
                              <InputNumber min={2} max={32} style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                          <Col xs={24} sm={12}>
                            <Form.Item
                              name={['loss_config', 'n_pos']}
                              label={t('create.decoderReranker.positiveCount')}
                              initialValue={1}
                              extra={t('create.decoderReranker.positiveCountExtra')}
                              preserve={false}
                            >
                              <InputNumber min={0} max={8} style={{ width: '100%' }} />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Row gutter={16}>
                          <Col xs={24} sm={12}>
                            <Form.Item
                              name={['loss_config', 'name']}
                              label={t('create.decoderReranker.lossFunction')}
                              initialValue="infonce"
                              extra={t('create.decoderReranker.lossFunctionExtra')}
                              preserve={false}
                            >
                              <Select
                                options={[
                                  { label: 'InfoNCE', value: 'infonce' },
                                  { label: 'BCE', value: 'bce' },
                                  { label: 'LambdaLoss', value: 'lambda_loss' },
                                  { label: 'ListMLE', value: 'list_mle' },
                                  { label: 'RankNet', value: 'ranknet' },
                                ]}
                              />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, cur) =>
                            prev.loss_config?.name !== cur.loss_config?.name
                          }
                        >
                          {({ getFieldValue }) => {
                            const lossName = getFieldValue(['loss_config', 'name'])
                            if (lossName === 'infonce') {
                              return (
                                <Row gutter={16}>
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['loss_config', 'temperature']}
                                      label={t('create.decoderReranker.temperature')}
                                      initialValue={0.05}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={0.01}
                                        max={1}
                                        step={0.01}
                                        style={{ width: '100%' }}
                                      />
                                    </Form.Item>
                                  </Col>
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['loss_config', 'infonce_mode']}
                                      label={t('create.decoderReranker.infonceMode')}
                                      initialValue="single"
                                      preserve={false}
                                    >
                                      <Select
                                        options={[
                                          {
                                            label: t('create.decoderReranker.infonceSingle'),
                                            value: 'single',
                                          },
                                          {
                                            label: t('create.decoderReranker.infoncePosset'),
                                            value: 'posset',
                                          },
                                          {
                                            label: t('create.decoderReranker.infonceAvgpos'),
                                            value: 'avgpos',
                                          },
                                        ]}
                                      />
                                    </Form.Item>
                                  </Col>
                                </Row>
                              )
                            }
                            if (lossName === 'ranknet') {
                              return (
                                <Row gutter={16}>
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['loss_config', 'ranknet_max_pairs_per_batch']}
                                      label={t('create.decoderReranker.maxPairsPerBatch')}
                                      initialValue={2000000}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={1000}
                                        max={10000000}
                                        step={1000}
                                        style={{ width: '100%' }}
                                      />
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
                          shouldUpdate={(prev, cur) =>
                            prev.loss_config?.name !== cur.loss_config?.name
                          }
                        >
                          {({ getFieldValue }) => {
                            const lossName = getFieldValue(['loss_config', 'name'])
                            if (lossName !== 'lambda_loss') return null
                            return (
                              <Row gutter={16}>
                                <Col xs={24} sm={12}>
                                  <Form.Item
                                    name={['loss_config', 'metric']}
                                    label={t('create.decoderReranker.targetMetric')}
                                    initialValue="ndcg"
                                    preserve={false}
                                  >
                                    <Select options={lambdaMetricOptions} />
                                  </Form.Item>
                                </Col>
                              </Row>
                            )
                          }}
                        </Form.Item>
                        <Form.Item
                          name={['loss_config', 'chunk_size']}
                          label={t('create.decoderReranker.chunkSize')}
                          initialValue={0}
                          extra={t('create.decoderReranker.chunkSizeExtra')}
                          preserve={false}
                        >
                          <InputNumber min={0} max={64} style={{ width: '100%' }} />
                        </Form.Item>
                      </>
                    ) : (
                      <Row gutter={16}>
                        <Col xs={24} sm={12}>
                          <Form.Item
                            name={['rl_config', 'n_docs']}
                            label={t('create.decoderReranker.docsPerGroup')}
                            initialValue={8}
                            preserve={false}
                          >
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
                          rules={[
                            { required: true, message: t('create.rl.sftModelPath.required') },
                          ]}
                          extra={t('create.rl.sftModelPath.extra')}
                        >
                          <Input placeholder={t('create.rl.sftModelPath.placeholder')} />
                        </Form.Item>
                        <Form.Item
                          noStyle
                          shouldUpdate={(prev, cur) =>
                            prev.model_type !== cur.model_type ||
                            prev.training_method !== cur.training_method ||
                            prev.rl_config?.reward_type !== cur.rl_config?.reward_type
                          }
                        >
                          {({ getFieldValue }) => {
                            const type = getFieldValue('model_type')
                            const currentMethod = getFieldValue('training_method')
                            const isDpo = currentMethod === 'dpo'
                            const rewardType =
                              getFieldValue(['rl_config', 'reward_type']) || 'rank_based'
                            const showRewardK =
                              rewardType === 'ndcg_based' || rewardType === 'recall_based'
                            const showDecoderDpoBeta = type === 'decoder_reranker'

                            if (isDpo) {
                              return (
                                <Row gutter={16}>
                                  {showDecoderDpoBeta ? (
                                    <Col xs={24} sm={12}>
                                      <Form.Item
                                        name={['rl_config', 'beta']}
                                        label="DPO Beta"
                                        initialValue={0.1}
                                        preserve={false}
                                      >
                                        <InputNumber
                                          min={0.01}
                                          max={10}
                                          step={0.01}
                                          style={{ width: '100%' }}
                                        />
                                      </Form.Item>
                                    </Col>
                                  ) : null}
                                  <Col xs={24} sm={12}>
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
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['rl_config', 'kl_coef']}
                                      label={t('create.rl.klCoef')}
                                      initialValue={0.1}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        style={{ width: '100%' }}
                                      />
                                    </Form.Item>
                                  </Col>
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['rl_config', 'clip_range']}
                                      label="Clip Range"
                                      initialValue={0.2}
                                      preserve={false}
                                    >
                                      <InputNumber
                                        min={0}
                                        max={1}
                                        step={0.01}
                                        style={{ width: '100%' }}
                                      />
                                    </Form.Item>
                                  </Col>
                                </Row>
                                <Row gutter={16}>
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['rl_config', 'reward_type']}
                                      label={t('create.rl.rewardType')}
                                      initialValue="rank_based"
                                      preserve={false}
                                    >
                                      <Select
                                        options={rlRewardKeys.map((r) => ({
                                          label: t(r.key),
                                          value: r.value,
                                        }))}
                                      />
                                    </Form.Item>
                                  </Col>
                                  <Col xs={24} sm={12}>
                                    {showRewardK ? (
                                      <Form.Item
                                        name={['rl_config', 'reward_k']}
                                        label="Reward@K"
                                        initialValue={10}
                                        preserve={false}
                                      >
                                        <InputNumber min={1} max={100} style={{ width: '100%' }} />
                                      </Form.Item>
                                    ) : null}
                                  </Col>
                                </Row>
                                <Row gutter={16}>
                                  <Col xs={24} sm={12}>
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
                                  <Col xs={24} sm={12}>
                                    <Form.Item
                                      name={['rl_config', 'num_iterations']}
                                      label={t('create.rl.updatePerBatch')}
                                      initialValue={1}
                                      preserve={false}
                                    >
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
              shouldUpdate={(prev, cur) =>
                prev.model_type !== cur.model_type || prev.training_method !== cur.training_method
              }
            >
              {({ getFieldValue }) => {
                const type = getFieldValue('model_type')
                const method = getFieldValue('training_method')
                if (type !== 'llm' || (method !== 'dpo' && method !== 'orpo')) return null

                return (
                  <>
                    <Divider>{t('create.preference.divider')}</Divider>
                    <Row gutter={16}>
                      <Col xs={24} sm={12}>
                        <Form.Item
                          name={['rl_config', 'beta']}
                          label={method === 'dpo' ? 'DPO Beta' : 'ORPO Beta'}
                          initialValue={0.1}
                          preserve={false}
                        >
                          <InputNumber min={0.01} max={10} step={0.01} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col xs={24} sm={12}>
                        <Form.Item
                          name={['rl_config', 'rankings_direction']}
                          label={t('create.preference.rankingsDirection')}
                          initialValue="auto"
                          extra={t('create.preference.rankingsDirectionExtra')}
                          preserve={false}
                        >
                          <Select
                            options={[
                              {
                                label: t('create.preference.rankingsDirectionAuto'),
                                value: 'auto',
                              },
                              {
                                label: t('create.preference.rankingsDirectionHigher'),
                                value: 'higher_is_better',
                              },
                              {
                                label: t('create.preference.rankingsDirectionLower'),
                                value: 'lower_is_better',
                              },
                            ]}
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                  </>
                )
              }}
            </Form.Item>
          </TrainingCreateSection>

          <TrainingCreateSection
            {...getSectionProps('evaluation')}
            title={t('create.evalAndSave.divider')}
          >
            <Row gutter={16}>
              <Col xs={24} sm={12}>
                <Form.Item
                  name="eval_strategy"
                  label={t('create.evalAndSave.evalStrategy')}
                  extra={t('create.evalAndSave.evalStrategyExtra')}
                >
                  <Select
                    options={[
                      { label: t('create.evalAndSave.evalNoEval'), value: 'no' },
                      { label: t('create.evalAndSave.evalPerEpoch'), value: 'epoch' },
                      { label: t('create.evalAndSave.evalPerSteps'), value: 'steps' },
                    ]}
                  />
                </Form.Item>
                <Form.Item
                  noStyle
                  shouldUpdate={(prev, cur) => prev.eval_strategy !== cur.eval_strategy}
                >
                  {({ getFieldValue }) =>
                    getFieldValue('eval_strategy') === 'steps' ? (
                      <Form.Item name="eval_steps" label={t('create.evalAndSave.evalSteps')}>
                        <InputNumber
                          min={1}
                          max={10000}
                          style={{ width: '100%' }}
                          placeholder={t('create.evalAndSave.evalStepsPlaceholder')}
                        />
                      </Form.Item>
                    ) : null
                  }
                </Form.Item>
              </Col>
              <Col xs={24} sm={12}>
                <Form.Item name="save_strategy" label={t('create.evalAndSave.saveStrategy')}>
                  <Select
                    options={[
                      { label: t('create.evalAndSave.saveNoSave'), value: 'no' },
                      { label: t('create.evalAndSave.savePerEpoch'), value: 'epoch' },
                      { label: t('create.evalAndSave.savePerSteps'), value: 'steps' },
                    ]}
                  />
                </Form.Item>
                <Form.Item
                  noStyle
                  shouldUpdate={(prev, cur) => prev.save_strategy !== cur.save_strategy}
                >
                  {({ getFieldValue }) =>
                    getFieldValue('save_strategy') === 'steps' ? (
                      <Form.Item name="save_steps" label={t('create.evalAndSave.saveSteps')}>
                        <InputNumber
                          min={1}
                          max={10000}
                          style={{ width: '100%' }}
                          placeholder={t('create.evalAndSave.saveStepsPlaceholder')}
                        />
                      </Form.Item>
                    ) : null
                  }
                </Form.Item>
              </Col>
            </Row>
          </TrainingCreateSection>

          <TrainingCreateSection {...getSectionProps('tuner')} title={t('create.tuner.divider')}>
            <Form.Item name="tuner_type" label={t('create.tuner.type')}>
              <Select options={tunerTypeOptions} />
            </Form.Item>

            <Form.Item noStyle shouldUpdate={(prev, cur) => prev.tuner_type !== cur.tuner_type}>
              {({ getFieldValue }) => {
                const tuner = getFieldValue('tuner_type')
                if (tuner !== 'lora') return null
                return (
                  <Row gutter={16}>
                    <Col xs={24} sm={8}>
                      <Form.Item name="lora_r" label="LoRA Rank">
                        <InputNumber min={1} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={8}>
                      <Form.Item name="lora_alpha" label="LoRA Alpha">
                        <InputNumber min={1} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={8}>
                      <Form.Item name="lora_dropout" label="Dropout">
                        <InputNumber
                          min={0}
                          max={1}
                          step={0.05}
                          style={{ width: '100%' }}
                          placeholder="0.0"
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                )
              }}
            </Form.Item>
          </TrainingCreateSection>
        </div>

        <TrainingCreateSection
          {...getSectionProps('resources')}
          title={t('create.resource.divider')}
          className="training-create-resources"
        >
          <Row gutter={24}>
            <Col xs={24} md={12}>
              <Form.Item
                name="gpu_ids"
                label={t('create.resource.gpuSelect')}
                extra={t('create.resource.gpuExtra')}
              >
                <GpuSelect mode="multiple" />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item name="output_dir" label={t('create.outputDir.label')}>
                <Input placeholder={t('create.outputDir.placeholder')} />
              </Form.Item>
            </Col>
          </Row>
        </TrainingCreateSection>

        <div className="training-create-actions">
          <div
            className={`training-create-draft ${draft.status === 'unavailable' ? 'is-unsaved' : ''}`}
          >
            <span role="status" aria-live="polite">
              {t(`create.draft.${draft.status}`)}
              {draft.updatedAt && draft.status !== 'unavailable' && draft.status !== 'empty'
                ? ` · ${new Intl.DateTimeFormat(i18n.language, { hour: '2-digit', minute: '2-digit' }).format(draft.updatedAt)}`
                : ''}
            </span>
            {draft.hasDraft && (
              <Popconfirm
                title={t('create.draft.discardConfirm')}
                okText={t('create.draft.discardConfirmAction')}
                cancelText={t('common:action.cancel')}
                onConfirm={() => {
                  clearDraft()
                  onDiscard()
                }}
              >
                <Button type="link" size="small" disabled={loading}>
                  {t('create.draft.discard')}
                </Button>
              </Popconfirm>
            )}
          </div>
          <Space size="middle">
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
