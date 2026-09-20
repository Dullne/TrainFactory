import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import {
  Form,
  Input,
  Button,
  Card,
  Space,
  Select,
  InputNumber,
  Switch,
  Collapse,
  Typography,
  message,
  Divider,
  Row,
  Col,
  Radio,
  Tag,
  Tooltip,
} from 'antd'
import { ArrowLeftOutlined, DatabaseOutlined, FolderOutlined } from '@ant-design/icons'
import { generationApi, datasetApi, milvusApi } from '@/services/api'
import ModelConfigSelector from '@/components/ModelConfigSelector'
import type { ModelConfig, Dataset, MilvusCollectionSummary } from '@/types'
import './GenerationCreate.css'

const { Title, Text } = Typography
const { Panel } = Collapse
const MAX_LLM_TOKENS = 131072

interface FormValues {
  task_name: string
  input_path: string
  content_field?: string
  input_format: string
  output_format: string
  generation_mode: string
  // LLM Config
  llm_temperature: number
  llm_max_tokens: number
  // Worker Config
  llm_concurrency: number
  embedding_batch_size: number
  embedding_concurrency: number
  rerank_top_k: number
  rerank_batch_size: number
  rerank_concurrency: number
  timeout_per_doc: number
  // Steps
  doc_quality_enabled: boolean
  doc_quality_min_score: number
  keypoint_gen_enabled: boolean
  keypoint_gen_max: number
  qa_gen_enabled: boolean
  qa_gen_count: number
  qa_gen_use_role: boolean
  qa_gen_roles_per_doc: number
  pos_neg_enabled: boolean
  pos_neg_positive_count: number
  pos_neg_negative_count: number
  pos_neg_use_role: boolean
  pos_neg_roles_per_doc: number
  pos_neg_augment: boolean
  pos_neg_neg_detection_mode: string
  pos_neg_chunk_eval_mode: string
  pos_neg_supplement_positives: boolean
  pos_neg_confirm_positives: boolean
  pos_neg_confirm_negatives: boolean
  pos_neg_answer_rewrite: boolean
  pos_neg_rerank_score_classification: boolean
  pos_neg_evidence_removal: boolean
  pos_neg_evidence_pruning: boolean
  pos_neg_neg_chunk_scoring: boolean
  pos_neg_skip_easy_negatives: boolean
  pos_neg_skip_perfect_ap: boolean
  pos_neg_skip_zero_ap: boolean
  validation_enabled: boolean
  // Post Process
  dedup_enabled: boolean
  // Auto register
  auto_register_dataset: boolean
  // Training 模式配置（embedding）
  similarity_threshold: number
  retrieval_top_k: number
  // Training 模式配置（rerank）
  rerank_threshold: number
}

/** 正负例生成卡片（qa_to_training / doc_to_training 共享） */
function PosNegCard({ posNegMethod }: { posNegMethod: string }) {
  const { t } = useTranslation(['generation', 'common'])
  const skipEasy = Form.useWatch('pos_neg_skip_easy_negatives')
  const skipPerfect = Form.useWatch('pos_neg_skip_perfect_ap')
  const negMode = Form.useWatch('pos_neg_neg_detection_mode')
  const supplement = Form.useWatch('pos_neg_supplement_positives')

  return (
    <Card size="small" title={t('steps.posNeg.title')} style={{ marginBottom: 16 }}>
      <Form.Item name="pos_neg_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
      </Form.Item>
      <Row gutter={12}>
        <Col xs={24} md={12}>
          <Form.Item name="pos_neg_positive_count" label={t('steps.posNeg.positiveCount')}>
            <InputNumber min={1} max={20} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col xs={24} md={12}>
          <Form.Item name="pos_neg_negative_count" label={t('steps.posNeg.negativeCount')}>
            <InputNumber min={1} max={50} style={{ width: '100%' }} />
          </Form.Item>
        </Col>
      </Row>
      <Tooltip title={t('steps.posNeg.evalMode.tooltip')}>
        <Form.Item name="pos_neg_chunk_eval_mode" label={t('steps.posNeg.evalMode.label')} style={{ marginBottom: 8 }}>
          <Radio.Group size="small">
            <Radio.Button value="batch">Batch</Radio.Button>
            <Radio.Button value="individual">Individual</Radio.Button>
          </Radio.Group>
        </Form.Item>
      </Tooltip>
      <Collapse ghost size="small">
        <Panel header={<Text type="secondary" style={{ fontSize: 12 }}>{t('steps.posNeg.advanced')}</Text>} key="adv">
          {posNegMethod === 'llm' && (
            <>
              <Form.Item name="pos_neg_use_role" valuePropName="checked" style={{ marginBottom: 4 }}>
                <Switch size="small" checkedChildren={t('steps.posNeg.useRole')} unCheckedChildren={t('steps.posNeg.noRole')} />
              </Form.Item>
              <Form.Item noStyle shouldUpdate={(prev, cur) => prev.pos_neg_use_role !== cur.pos_neg_use_role}>
                {({ getFieldValue }) => getFieldValue('pos_neg_use_role') ? (
                  <Form.Item name="pos_neg_roles_per_doc" label={t('steps.posNeg.rolesPerDoc')} style={{ marginBottom: 4 }}>
                    <InputNumber min={1} max={10} style={{ width: '100%' }} />
                  </Form.Item>
                ) : null}
              </Form.Item>
            </>
          )}
          <Tooltip title={t('steps.posNeg.augment.tooltip')}>
            <Form.Item name="pos_neg_augment" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.augment.on')} unCheckedChildren={t('steps.posNeg.augment.off')} />
            </Form.Item>
          </Tooltip>
          <Form.Item name="pos_neg_neg_detection_mode" label={t('steps.posNeg.negMode.label')} style={{ marginBottom: 4 }}>
            <Radio.Group size="small">
              <Radio.Button value="chunk">{t('steps.posNeg.negMode.chunk')}</Radio.Button>
              <Radio.Button value="statement">{t('steps.posNeg.negMode.statement')}</Radio.Button>
            </Radio.Group>
          </Form.Item>
          <Tooltip title={t('steps.posNeg.supplementPositives.tooltip')}>
            <Form.Item name="pos_neg_supplement_positives" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.supplementPositives.on')} unCheckedChildren={t('steps.posNeg.supplementPositives.off')} />
            </Form.Item>
          </Tooltip>
          {negMode === 'statement' && supplement && (
            <Tooltip title={t('steps.posNeg.confirmPositives.tooltip')}>
              <Form.Item name="pos_neg_confirm_positives" valuePropName="checked" style={{ marginBottom: 4 }}>
                <Switch size="small" checkedChildren={t('steps.posNeg.confirmPositives.on')} unCheckedChildren={t('steps.posNeg.confirmPositives.off')} />
              </Form.Item>
            </Tooltip>
          )}
          {negMode === 'statement' && (
            <Tooltip title={t('steps.posNeg.confirmNegatives.tooltip')}>
              <Form.Item name="pos_neg_confirm_negatives" valuePropName="checked" style={{ marginBottom: 4 }}>
                <Switch size="small" checkedChildren={t('steps.posNeg.confirmNegatives.on')} unCheckedChildren={t('steps.posNeg.confirmNegatives.off')} />
              </Form.Item>
            </Tooltip>
          )}
          {negMode === 'statement' && (
            <Tooltip title={t('steps.posNeg.answerRewrite.tooltip')}>
              <Form.Item name="pos_neg_answer_rewrite" valuePropName="checked" style={{ marginBottom: 4 }}>
                <Switch size="small" checkedChildren={t('steps.posNeg.answerRewrite.on')} unCheckedChildren={t('steps.posNeg.answerRewrite.off')} />
              </Form.Item>
            </Tooltip>
          )}
          {negMode === 'statement' && (
            <Tooltip title={t('steps.posNeg.rerankScoreClassification.tooltip')}>
              <Form.Item name="pos_neg_rerank_score_classification" valuePropName="checked" style={{ marginBottom: 4 }}>
                <Switch size="small" checkedChildren={t('steps.posNeg.rerankScoreClassification.on')} unCheckedChildren={t('steps.posNeg.rerankScoreClassification.off')} />
              </Form.Item>
            </Tooltip>
          )}
          <Tooltip title={t('steps.posNeg.evidenceRemoval.tooltip')}>
            <Form.Item name="pos_neg_evidence_removal" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.evidenceRemoval.on')} unCheckedChildren={t('steps.posNeg.evidenceRemoval.off')} />
            </Form.Item>
          </Tooltip>
          <Tooltip title={t('steps.posNeg.evidencePruning.tooltip')}>
            <Form.Item name="pos_neg_evidence_pruning" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.evidencePruning.on')} unCheckedChildren={t('steps.posNeg.evidencePruning.off')} />
            </Form.Item>
          </Tooltip>
          <Tooltip title={negMode === 'statement'
            ? t('steps.posNeg.negChunkScoring.tooltipStatement')
            : t('steps.posNeg.negChunkScoring.tooltipChunk')}>
            <Form.Item name="pos_neg_neg_chunk_scoring" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.negChunkScoring.on')} unCheckedChildren={t('steps.posNeg.negChunkScoring.off')} />
            </Form.Item>
          </Tooltip>
          <Tooltip title={t('steps.posNeg.skipEasyNegatives.tooltip')}>
            <Form.Item name="pos_neg_skip_easy_negatives" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.skipEasyNegatives.on')} unCheckedChildren={t('steps.posNeg.skipEasyNegatives.off')} />
            </Form.Item>
          </Tooltip>
          <Tooltip title={t('steps.posNeg.skipPerfectAp.tooltip')}>
            <Form.Item name="pos_neg_skip_perfect_ap" valuePropName="checked" style={{ marginBottom: 4 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.skipPerfectAp.on')} unCheckedChildren={t('steps.posNeg.skipPerfectAp.off')} />
            </Form.Item>
          </Tooltip>
          {skipEasy && !skipPerfect && (
            <Text type="warning" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
              {t('steps.posNeg.skipEasyWarning')}
            </Text>
          )}
          <Tooltip title={t('steps.posNeg.skipZeroAp.tooltip')}>
            <Form.Item name="pos_neg_skip_zero_ap" valuePropName="checked" style={{ marginBottom: 0 }}>
              <Switch size="small" checkedChildren={t('steps.posNeg.skipZeroAp.on')} unCheckedChildren={t('steps.posNeg.skipZeroAp.off')} />
            </Form.Item>
          </Tooltip>
        </Panel>
      </Collapse>
    </Card>
  )
}

export default function GenerationCreate() {
  const { t } = useTranslation(['generation', 'common'])
  const navigate = useNavigate()
  const [form] = Form.useForm<FormValues>()
  const [loading, setLoading] = useState(false)
  const [selectedLLMConfig, setSelectedLLMConfig] = useState<ModelConfig | null>(null)
  const [selectedEvalLLMConfig, setSelectedEvalLLMConfig] = useState<ModelConfig | null>(null)
  const [selectedEmbeddingConfig, setSelectedEmbeddingConfig] = useState<ModelConfig | null>(null)
  const [selectedRerankConfig, setSelectedRerankConfig] = useState<ModelConfig | null>(null)
  const [formats, setFormats] = useState<{
    output_formats: Array<{ name: string; description: string }>
    source_types: string[]
    length_types: string[]
  } | null>(null)

  // 输入源相关状态
  const [inputSource, setInputSource] = useState<'dataset' | 'path'>('dataset')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [datasetsLoading, setDatasetsLoading] = useState(false)
  // QA 数据集（qa_to_training 用）
  const [qaDatasets, setQaDatasets] = useState<Dataset[]>([])
  const [selectedQaDataset, setSelectedQaDataset] = useState<Dataset | null>(null)
  const [qaLoadingState, setQaLoadingState] = useState(false)
  // 当前选择的生成模式
  const [generationMode, setGenerationMode] = useState<string>('doc_to_training')
  // 正负例生成方式
  const [posNegMethod, setPosNegMethod] = useState<string>('retrieval')
  // 集合选择
  const [useExistingCollection, setUseExistingCollection] = useState(false)
  const [collections, setCollections] = useState<MilvusCollectionSummary[]>([])
  const [selectedCollection, setSelectedCollection] = useState<MilvusCollectionSummary | null>(null)

  // 模式分类辅助变量
  const isTrainingMode = generationMode === 'doc_to_training' || generationMode === 'qa_to_training'
  const isEvalMode = generationMode === 'doc_to_eval' || generationMode === 'qa_to_eval'
  const needsDocInput = ['doc_to_training', 'doc_to_eval', 'qa_extraction'].includes(generationMode)
  const needsQaInput = ['qa_to_training', 'qa_to_eval'].includes(generationMode)
  useEffect(() => {
    generationApi.getFormats().then(setFormats).catch(console.error)
    // 加载数据集列表（筛选原始数据）
    setDatasetsLoading(true)
    datasetApi.list({ usage: 'raw', page_size: 100 })
      .then(res => setDatasets(res.items || []))
      .catch(console.error)
      .finally(() => setDatasetsLoading(false))
    // 加载 QA 数据集列表
    setQaLoadingState(true)
    datasetApi.list({ dataset_type: 'qa_pair', page_size: 100 })
      .then(res => setQaDatasets(res.items || []))
      .catch(console.error)
      .finally(() => setQaLoadingState(false))
    // 加载已有集合列表
    milvusApi.listCollections()
      .then(res => setCollections(res.collections || []))
      .catch(console.error)
  }, [])

  const handleSubmit = async (values: FormValues) => {
    // qa_to_eval 模式不需要 LLM
    if (!selectedLLMConfig && generationMode !== 'qa_to_eval') {
      message.warning(t('create.validation.selectLLM'))
      return
    }
    // 模式验证
    if (generationMode === 'qa_to_training') {
      if (!selectedQaDataset) {
        message.warning(t('create.validation.selectQaDataset'))
        return
      }
    } else if (generationMode === 'doc_to_training' || generationMode === 'qa_extraction') {
      if (inputSource === 'dataset' && !selectedDataset) {
        message.warning(t('create.validation.selectDataset'))
        return
      }
      if (inputSource === 'path' && !values.input_path) {
        message.warning(t('create.validation.inputPathRequired'))
        return
      }
    }

    // 评估模式验证：必须有 Embedding
    if (isEvalMode) {
      if (!selectedEmbeddingConfig) {
        message.warning(t('create.validation.evalEmbeddingRequired'))
        return
      }
      if (needsQaInput && !selectedQaDataset) {
        message.warning(t('create.validation.selectQaDataset'))
        return
      }
      if (needsDocInput) {
        if (inputSource === 'dataset' && !selectedDataset) {
          message.warning(t('create.validation.selectDataset'))
          return
        }
        if (inputSource === 'path' && !values.input_path) {
          message.warning(t('create.validation.inputPathRequired'))
          return
        }
      }
    }

    // retrieval 模式需要 embedding（使用已有集合时不需要手动配置）
    if (isTrainingMode && posNegMethod === 'retrieval' && !selectedEmbeddingConfig && !useExistingCollection) {
      message.warning(t('create.validation.retrievalEmbeddingRequired'))
      return
    }
    if (isTrainingMode && posNegMethod === 'retrieval' && useExistingCollection && !selectedCollection) {
      message.warning(t('create.validation.selectCollection'))
      return
    }

    setLoading(true)
    try {
      const payload: Parameters<typeof generationApi.createTask>[0] = {
        task_name: values.task_name,
        generation_mode: generationMode,
        pos_neg_method: isTrainingMode ? posNegMethod : undefined,
        llm_config: {
          config_id: selectedLLMConfig?.config_id ?? '',
          temperature: values.llm_temperature,
          max_tokens: values.llm_max_tokens,
          concurrency: values.llm_concurrency,
        },
        eval_llm_config: isTrainingMode && selectedEvalLLMConfig ? {
          config_id: selectedEvalLLMConfig.config_id,
        } : undefined,
        worker_config: {
          timeout_per_doc: values.timeout_per_doc,
        },
        auto_register_dataset: values.auto_register_dataset,
        input_format: values.input_format,
        output_format: isEvalMode ? undefined : values.output_format,
      }

      // qa_to_training 模式
      if (generationMode === 'qa_to_training') {
        payload.dataset_id = selectedQaDataset!.dataset_id

        // steps 配置
        payload.steps = {
          pos_neg_extraction: {
            enabled: values.pos_neg_enabled,
            num_positive: values.pos_neg_positive_count,
            num_negative: values.pos_neg_negative_count,
            use_role: values.pos_neg_use_role,
            roles_per_doc: values.pos_neg_roles_per_doc,
            augment: values.pos_neg_augment,
            neg_detection_mode: values.pos_neg_neg_detection_mode,
            chunk_eval_mode: values.pos_neg_chunk_eval_mode,
            supplement_positives: values.pos_neg_supplement_positives,
            confirm_positives: values.pos_neg_confirm_positives,
            confirm_negatives: values.pos_neg_confirm_negatives,
            answer_rewrite: values.pos_neg_answer_rewrite,
            rerank_score_classification: values.pos_neg_rerank_score_classification,

            evidence_removal: values.pos_neg_evidence_removal,
            evidence_pruning: values.pos_neg_evidence_pruning,
            neg_chunk_scoring: values.pos_neg_neg_chunk_scoring,
            skip_easy_negatives: values.pos_neg_skip_easy_negatives,
            skip_perfect_ap: values.pos_neg_skip_perfect_ap,
            skip_zero_ap: values.pos_neg_skip_zero_ap,
          },
          validation: {
            enabled: values.validation_enabled,
          },
        }
      } else if (generationMode === 'doc_to_training') {
        // doc_to_training 模式
        if (inputSource === 'dataset' && selectedDataset) {
          payload.dataset_id = selectedDataset.dataset_id
        } else {
          payload.input_path = values.input_path
          payload.content_field = values.content_field
        }

        // steps 配置（包含 QA 提取步骤）
        payload.steps = {
          doc_quality: {
            enabled: values.doc_quality_enabled,
            min_score: values.doc_quality_min_score,
          },
          keypoint_gen: {
            enabled: values.keypoint_gen_enabled,
            max_keypoints: values.keypoint_gen_max,
          },
          qa_gen: {
            enabled: values.qa_gen_enabled,
            num_qa_per_doc: values.qa_gen_count,
            use_role: values.qa_gen_use_role,
            roles_per_doc: values.qa_gen_roles_per_doc,
          },
          pos_neg_extraction: {
            enabled: values.pos_neg_enabled,
            num_positive: values.pos_neg_positive_count,
            num_negative: values.pos_neg_negative_count,
            use_role: values.pos_neg_use_role,
            roles_per_doc: values.pos_neg_roles_per_doc,
            augment: values.pos_neg_augment,
            neg_detection_mode: values.pos_neg_neg_detection_mode,
            chunk_eval_mode: values.pos_neg_chunk_eval_mode,
            supplement_positives: values.pos_neg_supplement_positives,
            confirm_positives: values.pos_neg_confirm_positives,
            confirm_negatives: values.pos_neg_confirm_negatives,
            answer_rewrite: values.pos_neg_answer_rewrite,
            rerank_score_classification: values.pos_neg_rerank_score_classification,

            evidence_removal: values.pos_neg_evidence_removal,
            evidence_pruning: values.pos_neg_evidence_pruning,
            neg_chunk_scoring: values.pos_neg_neg_chunk_scoring,
            skip_easy_negatives: values.pos_neg_skip_easy_negatives,
            skip_perfect_ap: values.pos_neg_skip_perfect_ap,
            skip_zero_ap: values.pos_neg_skip_zero_ap,
          },
          validation: {
            enabled: values.validation_enabled,
          },
        }
      } else if (generationMode === 'qa_to_eval') {
        // qa_to_eval 模式：QA 数据集 → 评估数据
        payload.dataset_id = selectedQaDataset!.dataset_id
        // 无需 steps 配置（仅向量检索）
      } else if (generationMode === 'doc_to_eval') {
        // doc_to_eval 模式：文档 → QA 提取 → 评估数据
        if (inputSource === 'dataset' && selectedDataset) {
          payload.dataset_id = selectedDataset.dataset_id
        } else {
          payload.input_path = values.input_path
          payload.content_field = values.content_field
        }
        payload.steps = {
          doc_quality: {
            enabled: values.doc_quality_enabled,
            min_score: values.doc_quality_min_score,
          },
          keypoint_gen: {
            enabled: values.keypoint_gen_enabled,
            max_keypoints: values.keypoint_gen_max,
          },
          qa_gen: {
            enabled: values.qa_gen_enabled,
            num_qa_per_doc: values.qa_gen_count,
            use_role: values.qa_gen_use_role,
            roles_per_doc: values.qa_gen_roles_per_doc,
          },
        }
      } else {
        // qa_extraction 模式
        if (inputSource === 'dataset' && selectedDataset) {
          payload.dataset_id = selectedDataset.dataset_id
        } else {
          payload.input_path = values.input_path
          payload.content_field = values.content_field
        }
        payload.steps = {
          doc_quality: {
            enabled: values.doc_quality_enabled,
            min_score: values.doc_quality_min_score,
          },
          keypoint_gen: {
            enabled: values.keypoint_gen_enabled,
            max_keypoints: values.keypoint_gen_max,
          },
          qa_gen: {
            enabled: values.qa_gen_enabled,
            num_qa_per_doc: values.qa_gen_count,
            use_role: values.qa_gen_use_role,
            roles_per_doc: values.qa_gen_roles_per_doc,
          },
          validation: {
            enabled: values.validation_enabled,
          },
        }
      }

      payload.post_process = {
        dedup: {
          enabled: values.dedup_enabled,
        },
      }

      if (selectedEmbeddingConfig) {
        payload.embedding_config = {
          config_id: selectedEmbeddingConfig.config_id,
          batch_size: values.embedding_batch_size,
          concurrency: values.embedding_concurrency,
          similarity_threshold: values.similarity_threshold,
          retrieval_top_k: (isEvalMode || posNegMethod === 'retrieval') ? values.retrieval_top_k : undefined,
        }
      }
      if (selectedRerankConfig) {
        payload.rerank_config = {
          config_id: selectedRerankConfig.config_id,
          top_k: values.rerank_top_k,
          batch_size: values.rerank_batch_size,
          concurrency: values.rerank_concurrency,
          rerank_threshold: values.rerank_threshold,
        }
      }
      // 使用已有集合
      if (useExistingCollection && selectedCollection) {
        payload.milvus_collection_name = selectedCollection.name
      }
      await generationApi.createTask(payload)
      message.success(t('create.message.createSuccess'))
      navigate('/datasets?tab=generation')
    } catch (error) {
      message.error(t('create.message.createFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="generation-create-page">
      <div className="page-toolbar">
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/datasets?tab=generation')}
        >
          {t('common:action.back')}
        </Button>
        <Title level={4} style={{ margin: 0 }}>{t('create.title')}</Title>
      </div>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          input_format: 'auto',
          output_format: 'universal',
          generation_mode: 'doc_to_training',
          llm_temperature: 0.7,
          llm_max_tokens: 2048,
          llm_concurrency: 10,
          embedding_batch_size: 32,
          embedding_concurrency: 20,
          rerank_top_k: 10,
          rerank_batch_size: 64,
          rerank_concurrency: 10,
          timeout_per_doc: 300,
          doc_quality_enabled: false,
          doc_quality_min_score: 0.6,
          keypoint_gen_enabled: false,
          keypoint_gen_max: 5,
          qa_gen_enabled: true,
          qa_gen_count: 3,
          qa_gen_use_role: false,
          qa_gen_roles_per_doc: 3,
          pos_neg_enabled: true,
          pos_neg_positive_count: 5,
          pos_neg_negative_count: 20,
          pos_neg_use_role: false,
          pos_neg_roles_per_doc: 3,
          pos_neg_augment: false,
          pos_neg_neg_detection_mode: 'chunk',
          pos_neg_chunk_eval_mode: 'batch',
          pos_neg_supplement_positives: true,
          pos_neg_confirm_positives: false,
          pos_neg_confirm_negatives: false,
          pos_neg_answer_rewrite: false,
          pos_neg_rerank_score_classification: false,
          pos_neg_evidence_removal: false,
          pos_neg_evidence_pruning: false,
          pos_neg_neg_chunk_scoring: false,
          pos_neg_skip_easy_negatives: true,
          pos_neg_skip_perfect_ap: true,
          pos_neg_skip_zero_ap: true,
          validation_enabled: false,
          dedup_enabled: true,
          auto_register_dataset: true,
          similarity_threshold: 0.85,
          retrieval_top_k: 10,
          rerank_threshold: 1,
        }}
      >
        <Card title={t('create.basicConfig')} size="small" style={{ marginBottom: 16 }}>
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item
                name="task_name"
                label={t('create.taskName.label')}
                rules={[{ required: true, message: t('create.taskName.required') }]}
              >
                <Input placeholder={t('create.taskName.placeholder')} />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                name="generation_mode"
                label={t('create.generationMode.label')}
                rules={[{ required: true }]}
              >
                <Select onChange={(v) => setGenerationMode(v)}>
                  <Select.Option value="doc_to_training">{t('modeDescription.doc_to_training')}</Select.Option>
                  <Select.Option value="qa_to_training">{t('modeDescription.qa_to_training')}</Select.Option>
                  <Select.Option value="qa_extraction">{t('modeDescription.qa_extraction')}</Select.Option>
                  <Select.Option value="doc_to_eval">{t('modeDescription.doc_to_eval')}</Select.Option>
                  <Select.Option value="qa_to_eval">{t('modeDescription.qa_to_eval')}</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          {/* qa_to_training / qa_to_eval 模式：选择 QA 数据集 */}
          {needsQaInput ? (
            <Row gutter={16}>
              <Col xs={24} md={12}>
                <Form.Item label={t('create.qaDataset.label')} required>
                  <Select
                    placeholder={t('create.qaDataset.placeholder')}
                    loading={qaLoadingState}
                    showSearch
                    optionFilterProp="children"
                    value={selectedQaDataset?.dataset_id}
                    onChange={(value) => {
                      const ds = qaDatasets.find(d => d.dataset_id === value)
                      setSelectedQaDataset(ds || null)
                    }}
                  >
                    {qaDatasets.map(ds => (
                      <Select.Option key={ds.dataset_id} value={ds.dataset_id}>
                        <Space>
                          <span>{ds.display_name || ds.dataset_name}</span>
                          <Tag color="orange">{t('create.qaDataset.tag')}</Tag>
                        </Space>
                      </Select.Option>
                    ))}
                  </Select>
                  {selectedQaDataset && (
                    <div style={{ marginTop: 8, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4, fontSize: 12 }}>
                      <div><Text type="secondary">{t('create.qaDataset.path', { path: selectedQaDataset.storage_path || selectedQaDataset.storage_uri || '' })}</Text></div>
                      {selectedQaDataset.num_rows != null && (
                        <div><Text type="secondary">{t('create.qaDataset.numRows', { count: selectedQaDataset.num_rows })}</Text></div>
                      )}
                    </div>
                  )}
                </Form.Item>
              </Col>
              {!isEvalMode && (
                <Col xs={24} md={12}>
                  <Form.Item name="output_format" label={t('create.outputFormat.label')}>
                    <Select>
                      {formats?.output_formats.map(f => (
                        <Select.Option key={f.name} value={f.name}>
                          {f.name} - {f.description}
                        </Select.Option>
                      )) || (
                        <>
                          <Select.Option value="universal">universal</Select.Option>
                          <Select.Option value="triplet">triplet</Select.Option>
                          <Select.Option value="pair">pair</Select.Option>
                        </>
                      )}
                    </Select>
                  </Form.Item>
                </Col>
              )}
            </Row>
          ) : (
            <>
              {/* doc_to_training / qa_extraction / doc_to_eval 模式：选择文档输入源 */}
              <Form.Item label={t('create.inputSource.label')}>
                <Radio.Group value={inputSource} onChange={e => setInputSource(e.target.value)}>
                  <Radio.Button value="dataset">
                    <DatabaseOutlined /> {t('create.inputSource.dataset')}
                  </Radio.Button>
                  <Radio.Button value="path">
                    <FolderOutlined /> {t('create.inputSource.path')}
                  </Radio.Button>
                </Radio.Group>
              </Form.Item>

              <Row gutter={16}>
                <Col xs={24} md={12}>
                  {inputSource === 'dataset' ? (
                    <Form.Item label={t('create.docDataset.label')} required>
                      <Select
                        placeholder={t('create.docDataset.placeholder')}
                        loading={datasetsLoading}
                        showSearch
                        optionFilterProp="children"
                        value={selectedDataset?.dataset_id}
                        onChange={(value) => {
                          const ds = datasets.find(d => d.dataset_id === value)
                          setSelectedDataset(ds || null)
                        }}
                      >
                        {datasets.map(ds => (
                          <Select.Option key={ds.dataset_id} value={ds.dataset_id}>
                            <Space>
                              <span>{ds.display_name || ds.dataset_name}</span>
                              <Tag color="purple">{t('create.docDataset.tag')}</Tag>
                            </Space>
                          </Select.Option>
                        ))}
                      </Select>
                      {selectedDataset && (
                        <div style={{ marginTop: 8, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4, fontSize: 12 }}>
                          <div><Text type="secondary">{t('create.docDataset.path', { path: selectedDataset.storage_path || selectedDataset.storage_uri || '' })}</Text></div>
                          {(selectedDataset.extra_metadata as Record<string, string> | undefined)?.content_field && (
                            <div><Text type="secondary">{t('create.docDataset.contentField', { field: (selectedDataset.extra_metadata as Record<string, string>).content_field })}</Text></div>
                          )}
                          {selectedDataset.num_rows != null && (
                            <div><Text type="secondary">{t('create.docDataset.numRows', { count: selectedDataset.num_rows })}</Text></div>
                          )}
                        </div>
                      )}
                    </Form.Item>
                  ) : (
                    <>
                      <Form.Item name="input_path" label={t('create.inputPath.label')} required>
                        <Input placeholder={t('create.inputPath.placeholder')} />
                      </Form.Item>
                      <Form.Item
                        name="content_field"
                        label={t('create.contentField.label')}
                        extra={t('create.contentField.extra')}
                      >
                        <Input placeholder={t('create.contentField.placeholder')} />
                      </Form.Item>
                    </>
                  )}
                </Col>
                <Col xs={24} sm={12} xl={6}>
                  <Form.Item name="input_format" label={t('create.inputFormat.label')}>
                    <Select>
                      <Select.Option value="auto">{t('create.inputFormat.auto')}</Select.Option>
                      <Select.Option value="jsonl">JSONL</Select.Option>
                      <Select.Option value="json">JSON</Select.Option>
                      <Select.Option value="txt">TXT</Select.Option>
                    </Select>
                  </Form.Item>
                </Col>
                {!isEvalMode && (
                  <Col xs={24} sm={12} xl={6}>
                    <Form.Item name="output_format" label={t('create.outputFormat.label')}>
                      <Select>
                        {formats?.output_formats.map(f => (
                          <Select.Option key={f.name} value={f.name}>
                            {f.name} - {f.description}
                          </Select.Option>
                        )) || (
                          <>
                            <Select.Option value="universal">universal</Select.Option>
                            <Select.Option value="triplet">triplet</Select.Option>
                            <Select.Option value="pair">pair</Select.Option>
                          </>
                        )}
                      </Select>
                    </Form.Item>
                  </Col>
                )}
              </Row>
            </>
          )}

          {/* 正负例生成方式（仅 training 模式） */}
          {isTrainingMode && (
            <>
              <Divider style={{ margin: '12px 0' }} />
              <Form.Item label={t('create.posNegMethod.label')}>
                <Radio.Group value={posNegMethod} onChange={e => setPosNegMethod(e.target.value)}>
                  <Radio.Button value="retrieval">{t('posNegMethod.retrievalFull')}</Radio.Button>
                  <Radio.Button value="llm">{t('posNegMethod.llmFull')}</Radio.Button>
                </Radio.Group>
              </Form.Item>

              {posNegMethod === 'retrieval' && (
                <Form.Item label={t('create.vectorCollection.label')} style={{ marginBottom: 8 }}>
                  <Radio.Group
                    value={useExistingCollection ? 'existing' : 'auto'}
                    onChange={e => {
                      const isExisting = e.target.value === 'existing'
                      setUseExistingCollection(isExisting)
                      if (!isExisting) setSelectedCollection(null)
                    }}
                  >
                    <Radio.Button value="auto">{t('create.vectorCollection.auto')}</Radio.Button>
                    <Radio.Button value="existing">{t('create.vectorCollection.existing')}</Radio.Button>
                  </Radio.Group>
                </Form.Item>
              )}
              {posNegMethod === 'retrieval' && useExistingCollection && (
                <Form.Item label={t('create.vectorCollection.selectLabel')} required style={{ marginBottom: 8 }}>
                  <Select
                    placeholder={t('create.vectorCollection.selectPlaceholder')}
                    showSearch
                    optionFilterProp="label"
                    value={selectedCollection?.name}
                    onChange={(name) => {
                      const coll = collections.find(c => c.name === name) || null
                      setSelectedCollection(coll)
                    }}
                    options={collections.filter(c => c.embedding_config_id).map(c => ({
                      value: c.name,
                      label: t('create.vectorCollection.collectionInfo', { name: c.display_name || c.name, model: c.embedding_model || t('create.vectorCollection.unknownModel'), count: c.num_entities }),
                    }))}
                  />
                  {selectedCollection && (
                    <div style={{ marginTop: 4, padding: '6px 12px', background: 'rgba(63, 185, 80, 0.10)', border: '1px solid rgba(63, 185, 80, 0.3)', borderRadius: 4 }}>
                      <Text style={{ fontSize: 12 }}>
                        Embedding: <Tag color="purple">{selectedCollection.embedding_model}</Tag>
                        {t('create.vectorCollection.dimension', { dim: selectedCollection.dim })} · {t('create.vectorCollection.metric', { type: selectedCollection.metric_type })}
                        {selectedCollection.linked_datasets?.length > 0 && (
                          <> · {t('create.vectorCollection.linkedDatasets', { count: selectedCollection.linked_datasets.length })}</>
                        )}
                      </Text>
                    </div>
                  )}
                </Form.Item>
              )}
              {posNegMethod === 'retrieval' && !useExistingCollection && !selectedEmbeddingConfig && (
                <div style={{ padding: '8px 12px', background: 'rgba(248, 81, 73, 0.10)', border: '1px solid rgba(248, 81, 73, 0.3)', borderRadius: 4, marginBottom: 16 }}>
                  <Text type="danger" style={{ fontSize: 12 }}>
                    {t('create.warning.embeddingRequired')}
                  </Text>
                </div>
              )}
              {posNegMethod === 'llm' && !selectedEmbeddingConfig && (
                <div style={{ padding: '8px 12px', background: 'rgba(210, 153, 34, 0.10)', border: '1px solid rgba(210, 153, 34, 0.3)', borderRadius: 4, marginBottom: 16 }}>
                  <Text type="warning" style={{ fontSize: 12 }}>
                    {t('create.warning.embeddingOptional')}
                  </Text>
                </div>
              )}
            </>
          )}

          {/* 评估模式 Embedding 必填提示 */}
          {isEvalMode && !selectedEmbeddingConfig && (
            <div style={{ padding: '8px 12px', background: 'rgba(248, 81, 73, 0.10)', border: '1px solid rgba(248, 81, 73, 0.3)', borderRadius: 4, marginBottom: 16 }}>
              <Text type="danger" style={{ fontSize: 12 }}>
                {t('create.warning.evalEmbeddingRequired')}
              </Text>
            </div>
          )}
        </Card>

        <Card title={t('modelConfig.title')} size="small" style={{ marginBottom: 16 }}>
          {/* LLM 模型 */}
          <Row gutter={16}>
            <Col xs={24} xl={12}>
              <Form.Item label={generationMode === 'qa_to_eval' ? t('modelConfig.llm.labelOptional') : t('modelConfig.llm.label')} required={generationMode !== 'qa_to_eval'}>
                <ModelConfigSelector
                  mode="single"
                  modelType="llm"
                  onChange={(_, config) => setSelectedLLMConfig(config)}
                />
              </Form.Item>
              {selectedLLMConfig && (
                <div style={{ marginBottom: 12, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t('modelConfig.llm.endpoint', { endpoint: selectedLLMConfig.api_endpoint })} | {t('modelConfig.llm.model', { model: selectedLLMConfig.model_name || '-' })}
                  </Text>
                </div>
              )}
            </Col>
            <Col xs={24} sm={8} xl={4}>
              <Form.Item name="llm_temperature" label="Temperature">
                <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8} xl={4}>
              <Form.Item
                name="llm_max_tokens"
                label="Max Tokens"
                extra={t('modelConfig.maxTokensExtra', { max: MAX_LLM_TOKENS })}
                rules={[
                  { type: 'number', max: MAX_LLM_TOKENS, message: t('modelConfig.maxTokensRule', { max: MAX_LLM_TOKENS }) },
                ]}
              >
                <InputNumber min={256} max={MAX_LLM_TOKENS} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={24} sm={8} xl={4}>
              <Form.Item name="llm_concurrency" label={t('modelConfig.concurrency.label')} tooltip={t('modelConfig.concurrency.llmTooltip')}>
                <InputNumber min={1} max={50} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          {/* 评估 LLM（可选，仅训练模式） */}
          {isTrainingMode && (
            <Row gutter={16} style={{ marginTop: 12 }}>
              <Col xs={24} md={12}>
                <Form.Item label={t('modelConfig.evalLlm.label')} tooltip={t('modelConfig.evalLlm.tooltip')}>
                  <ModelConfigSelector
                    mode="single"
                    modelType="llm"
                    allowClear
                    onChange={(_, config) => setSelectedEvalLLMConfig(config)}
                  />
                </Form.Item>
                {selectedEvalLLMConfig && (
                  <div style={{ marginBottom: 12, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {t('modelConfig.llm.endpoint', { endpoint: selectedEvalLLMConfig.api_endpoint })} | {t('modelConfig.llm.model', { model: selectedEvalLLMConfig.model_name || '-' })}
                    </Text>
                  </div>
                )}
              </Col>
            </Row>
          )}

          <Divider style={{ margin: '12px 0' }} />

          {/* Embedding 模型 */}
          <Row gutter={16}>
            <Col span={24}>
              <Form.Item label={isEvalMode ? t('modelConfig.embedding.label') : t('modelConfig.embedding.labelOptional')} required={isEvalMode}>
                <ModelConfigSelector
                  mode="single"
                  modelType="embedding"
                  allowClear
                  onChange={(_, config) => {
                    setSelectedEmbeddingConfig(config)
                    if (!config) setSelectedRerankConfig(null)
                  }}
                />
              </Form.Item>
              {selectedEmbeddingConfig && (
                <div style={{ marginBottom: 12, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {t('modelConfig.llm.endpoint', { endpoint: selectedEmbeddingConfig.api_endpoint })} | {t('modelConfig.llm.model', { model: selectedEmbeddingConfig.model_name || '-' })}
                  </Text>
                </div>
              )}
            </Col>
            {selectedEmbeddingConfig && (
              <>
                <Col xs={24} sm={12} xl={6}>
                  <Form.Item name="embedding_batch_size" label={t('modelConfig.batchSize.label')} tooltip={t('modelConfig.batchSize.embeddingTooltip')}>
                    <InputNumber min={1} max={256} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} xl={6}>
                  <Form.Item name="embedding_concurrency" label={t('modelConfig.concurrency.label')} tooltip={t('modelConfig.concurrency.embeddingTooltip')}>
                    <InputNumber min={1} max={100} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={24} sm={12} xl={6}>
                  <Form.Item name="similarity_threshold" label={t('modelConfig.similarityThreshold.label')} tooltip={t('modelConfig.similarityThreshold.embeddingTooltip')}>
                    <InputNumber min={0.5} max={1} step={0.05} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                {(isEvalMode || posNegMethod === 'retrieval') && (
                  <Col xs={24} sm={12} xl={6}>
                    <Form.Item name="retrieval_top_k" label={t('modelConfig.retrievalTopK.label')} tooltip={t('modelConfig.retrievalTopK.tooltip')}>
                      <InputNumber min={1} max={100} step={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                )}
              </>
            )}
          </Row>

          {/* Rerank 模型（需先选择 Embedding，training 和 eval 模式都支持） */}
          {generationMode !== 'qa_extraction' && selectedEmbeddingConfig && (
            <>
              <Divider style={{ margin: '12px 0' }} />
              <Row gutter={16}>
                <Col span={24}>
                  <Form.Item label={t('modelConfig.rerank.label')}>
                    <ModelConfigSelector
                      mode="single"
                      modelType="rerank"
                      allowClear
                      onChange={(_, config) => setSelectedRerankConfig(config)}
                    />
                  </Form.Item>
                  {selectedRerankConfig && (
                    <div style={{ marginBottom: 12, padding: '8px 12px', background: 'var(--tf-bg-elevated)', borderRadius: 4 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {t('modelConfig.llm.endpoint', { endpoint: selectedRerankConfig.api_endpoint })} | {t('modelConfig.llm.model', { model: selectedRerankConfig.model_name || '-' })}
                      </Text>
                    </div>
                  )}
                </Col>
                {selectedRerankConfig && (
                  <>
                    <Col xs={24} sm={12} xl={6}>
                      <Form.Item name="rerank_top_k" label={t('modelConfig.rerankTopK.label')} tooltip={t('modelConfig.rerankTopK.tooltip')}>
                        <InputNumber min={1} max={100} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12} xl={6}>
                      <Form.Item name="rerank_batch_size" label={t('modelConfig.batchSize.label')} tooltip={t('modelConfig.batchSize.rerankTooltip')}>
                        <InputNumber min={1} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12} xl={6}>
                      <Form.Item name="rerank_concurrency" label={t('modelConfig.concurrency.label')} tooltip={t('modelConfig.concurrency.rerankTooltip')}>
                        <InputNumber min={1} max={50} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col xs={24} sm={12} xl={6}>
                      <Form.Item name="rerank_threshold" label={t('modelConfig.similarityThreshold.label')} tooltip={t('modelConfig.similarityThreshold.rerankTooltip')}>
                        <InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                  </>
                )}
              </Row>
            </>
          )}

          <Divider style={{ margin: '12px 0' }} />
          <Row gutter={16}>
            <Col xs={24} md={12} xl={8}>
              <Form.Item name="timeout_per_doc" label={t('modelConfig.timeout.label')} tooltip={t('modelConfig.timeout.tooltip')}>
                <InputNumber min={60} max={600} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        <Collapse defaultActiveKey={['steps']} style={{ marginBottom: 16 }}>
          <Panel header={t('steps.title')} key="steps">
            {/* qa_to_training 模式：正负例+校验 */}
            {/* doc_to_training 模式：QA 提取步骤 + 正负例+校验 */}
            {/* qa_extraction 模式：文档质量 + 关键点 + 角色 + QA 生成 */}
            {generationMode === 'qa_to_training' ? (
              <>
                <Row gutter={16}>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.qaDedup.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="dedup_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.qaDedup.description')}</Text>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <PosNegCard posNegMethod={posNegMethod} />
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.validation.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="validation_enabled" valuePropName="checked">
                        <Switch checkedChildren={t('steps.validation.enable')} unCheckedChildren={t('steps.validation.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.validation.description')}</Text>
                    </Card>
                  </Col>
                </Row>
              </>
            ) : generationMode === 'doc_to_training' ? (
              <>
                <Row gutter={16}>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.docQuality.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="doc_quality_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="doc_quality_min_score" label={t('steps.docQuality.minScore')}>
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.keypointGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="keypoint_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="keypoint_gen_max" label={t('steps.keypointGen.maxKeypoints')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.qaGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="qa_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="qa_gen_count" label={t('steps.qaGen.countPerDoc')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                      <Tooltip title={t('steps.qaGen.useRoleTooltip')}>
                        <Form.Item name="qa_gen_use_role" valuePropName="checked" style={{ marginBottom: 4 }}>
                          <Switch size="small" checkedChildren={t('steps.qaGen.useRole')} unCheckedChildren={t('steps.qaGen.noRole')} />
                        </Form.Item>
                      </Tooltip>
                      <Form.Item noStyle shouldUpdate={(prev, cur) => prev.qa_gen_use_role !== cur.qa_gen_use_role}>
                        {({ getFieldValue }) => getFieldValue('qa_gen_use_role') ? (
                          <Form.Item name="qa_gen_roles_per_doc" label={t('steps.qaGen.rolesPerDoc')} style={{ marginBottom: 0 }}>
                            <InputNumber min={1} max={10} style={{ width: '100%' }} />
                          </Form.Item>
                        ) : null}
                      </Form.Item>
                    </Card>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.qaDedup.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="dedup_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.qaDedup.description')}</Text>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <PosNegCard posNegMethod={posNegMethod} />
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.validation.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="validation_enabled" valuePropName="checked">
                        <Switch checkedChildren={t('steps.validation.enable')} unCheckedChildren={t('steps.validation.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.validation.description')}</Text>
                    </Card>
                  </Col>
                </Row>
              </>
            ) : generationMode === 'qa_to_eval' ? (
              /* qa_to_eval 模式：仅去重 */
              <Row gutter={16}>
                <Col xs={24} md={12} xl={8}>
                  <Card size="small" title={t('steps.qaDedup.title')} style={{ marginBottom: 16 }}>
                    <Form.Item name="dedup_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                      <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                    </Form.Item>
                    <Text type="secondary">{t('steps.qaDedup.description')}</Text>
                  </Card>
                </Col>
                <Col xs={24} md={12} xl={16}>
                  <Card size="small" style={{ marginBottom: 16, background: 'rgba(63, 185, 80, 0.10)', border: '1px solid rgba(63, 185, 80, 0.3)' }}>
                    <Text type="secondary">
                      {t('steps.qaToEvalDescription')}
                    </Text>
                  </Card>
                </Col>
              </Row>
            ) : generationMode === 'doc_to_eval' ? (
              /* doc_to_eval 模式：QA 提取步骤 + 去重 */
              <>
                <Row gutter={16}>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.docQuality.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="doc_quality_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="doc_quality_min_score" label={t('steps.docQuality.minScore')}>
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.keypointGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="keypoint_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="keypoint_gen_max" label={t('steps.keypointGen.maxKeypoints')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.qaGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="qa_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="qa_gen_count" label={t('steps.qaGen.countPerDoc')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                      <Tooltip title={t('steps.qaGen.useRoleTooltip')}>
                        <Form.Item name="qa_gen_use_role" valuePropName="checked" style={{ marginBottom: 4 }}>
                          <Switch size="small" checkedChildren={t('steps.qaGen.useRole')} unCheckedChildren={t('steps.qaGen.noRole')} />
                        </Form.Item>
                      </Tooltip>
                      <Form.Item noStyle shouldUpdate={(prev, cur) => prev.qa_gen_use_role !== cur.qa_gen_use_role}>
                        {({ getFieldValue }) => getFieldValue('qa_gen_use_role') ? (
                          <Form.Item name="qa_gen_roles_per_doc" label={t('steps.qaGen.rolesPerDoc')} style={{ marginBottom: 0 }}>
                            <InputNumber min={1} max={10} style={{ width: '100%' }} />
                          </Form.Item>
                        ) : null}
                      </Form.Item>
                    </Card>
                  </Col>
                </Row>
                <Row gutter={16}>
                  <Col xs={24} md={12} xl={8}>
                    <Card size="small" title={t('steps.qaDedup.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="dedup_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.qaDedup.description')}</Text>
                    </Card>
                  </Col>
                  <Col xs={24} md={12} xl={16}>
                    <Card size="small" style={{ marginBottom: 16, background: 'rgba(63, 185, 80, 0.10)', border: '1px solid rgba(63, 185, 80, 0.3)' }}>
                      <Text type="secondary">
                        {t('steps.docToEvalDescription')}
                      </Text>
                    </Card>
                  </Col>
                </Row>
              </>
            ) : (
              /* qa_extraction 模式：文档质量 + 关键点 + 角色 + QA 生成 */
              <>
                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Card size="small" title={t('steps.docQuality.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="doc_quality_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="doc_quality_min_score" label={t('steps.docQuality.minScore')}>
                        <InputNumber min={0} max={1} step={0.1} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card size="small" title={t('steps.keypointGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="keypoint_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="keypoint_gen_max" label={t('steps.keypointGen.maxKeypoints')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                    </Card>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Card size="small" title={t('steps.qaGen.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="qa_gen_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Form.Item name="qa_gen_count" label={t('steps.qaGen.countPerDoc')}>
                        <InputNumber min={1} max={10} style={{ width: '100%' }} />
                      </Form.Item>
                      <Tooltip title={t('steps.qaGen.useRoleTooltip')}>
                        <Form.Item name="qa_gen_use_role" valuePropName="checked" style={{ marginBottom: 4 }}>
                          <Switch size="small" checkedChildren={t('steps.qaGen.useRole')} unCheckedChildren={t('steps.qaGen.noRole')} />
                        </Form.Item>
                      </Tooltip>
                      <Form.Item noStyle shouldUpdate={(prev, cur) => prev.qa_gen_use_role !== cur.qa_gen_use_role}>
                        {({ getFieldValue }) => getFieldValue('qa_gen_use_role') ? (
                          <Form.Item name="qa_gen_roles_per_doc" label={t('steps.qaGen.rolesPerDoc')} style={{ marginBottom: 0 }}>
                            <InputNumber min={1} max={10} style={{ width: '100%' }} />
                          </Form.Item>
                        ) : null}
                      </Form.Item>
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card size="small" title={t('steps.qaDedup.title')} style={{ marginBottom: 16 }}>
                      <Form.Item name="dedup_enabled" valuePropName="checked" style={{ marginBottom: 8 }}>
                        <Switch checkedChildren={t('switch.enable')} unCheckedChildren={t('switch.disable')} />
                      </Form.Item>
                      <Text type="secondary">{t('steps.qaDedup.description')}</Text>
                    </Card>
                  </Col>
                </Row>
              </>
            )}
          </Panel>

          <Panel header={t('steps.outputConfig')} key="post_process">
            <Card size="small" title={t('steps.datasetRegister.title')}>
              <Form.Item name="auto_register_dataset" valuePropName="checked">
                <Switch checkedChildren={t('steps.datasetRegister.autoRegister')} unCheckedChildren={t('steps.datasetRegister.noRegister')} />
              </Form.Item>
              <Text type="secondary">{t('steps.datasetRegister.description')}</Text>
            </Card>
          </Panel>
        </Collapse>

        <div className="page-form-actions">
          <Button onClick={() => navigate('/datasets?tab=generation')}>{t('common:action.cancel')}</Button>
          <Button type="primary" htmlType="submit" loading={loading} disabled={loading}>
            {t('create.submitButton')}
          </Button>
        </div>
      </Form>
    </div>
  )
}
