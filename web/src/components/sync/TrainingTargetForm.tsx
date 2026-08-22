import { useTranslation } from 'react-i18next'
import {
  Input, Select, InputNumber, Tabs, Row, Col, Typography, Divider,
} from 'antd'

const { Option } = Select
const { Text } = Typography

const rerankerLossOptions = [
  { label: 'Auto', value: 'auto' },
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

const rerankerScaleLosses = ['MultipleNegativesRankingLoss', 'CachedMultipleNegativesRankingLoss']
const rerankerMiniBatchLosses = ['CachedMultipleNegativesRankingLoss', 'RankNetLoss', 'LambdaLoss', 'ListMLELoss', 'ListNetLoss', 'PListMLELoss']
const rerankerNegativesLosses = ['MultipleNegativesRankingLoss', 'CachedMultipleNegativesRankingLoss']
const rerankerTopKLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerSigmaLosses = ['RankNetLoss', 'LambdaLoss']
const rerankerRespectInputOrderLosses = ['ListMLELoss', 'PListMLELoss']

export interface TrainingTargetFormState {
  key: string
  target_name: string
  model_type: string
  data_phase: 'qa' | 'final'
  training_method: string
  base_model_path: string
  base_model_id: string
  base_model_name: string
  base_deployment_id: string
  training_threshold: number
  priority: number
  sort_order: number
  // LoRA
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  // Training hyperparams
  num_train_epochs: number
  per_device_train_batch_size: number
  learning_rate: number
  warmup_ratio: number
  gradient_accumulation_steps: number
  max_length: number | undefined
  mixed_precision: string
  gpu_ids: number[]
  // Loss
  embedding_loss_name: string
  reranker_loss_name: string
  loss_config: Record<string, unknown> | undefined
  // RL config
  rl_config: {
    beta?: number
    rankings_direction?: 'auto' | 'higher_is_better' | 'lower_is_better'
  } | undefined
}

export const DEFAULT_TARGET: Omit<TrainingTargetFormState, 'key'> = {
  target_name: '',
  model_type: 'embedding',
  data_phase: 'final',
  training_method: 'sft',
  base_model_path: '',
  base_model_id: '',
  base_model_name: '',
  base_deployment_id: '',
  training_threshold: 1000,
  priority: 0,
  sort_order: 0,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.0,
  num_train_epochs: 3,
  per_device_train_batch_size: 16,
  learning_rate: 2e-5,
  warmup_ratio: 0.1,
  gradient_accumulation_steps: 1,
  max_length: undefined,
  mixed_precision: 'none',
  gpu_ids: [],
  embedding_loss_name: '',
  reranker_loss_name: 'auto',
  loss_config: undefined,
  rl_config: undefined,
}

interface TrainingTargetFormProps {
  target: TrainingTargetFormState
  index: number
  registeredModels: Array<{ model_id: string; model_name: string; model_path: string; display_name?: string }>
  onChange: (index: number, patch: Partial<TrainingTargetFormState>) => void
}

export default function TrainingTargetForm({
  target, index, registeredModels, onChange,
}: TrainingTargetFormProps) {
  const { t } = useTranslation(['sync', 'common', 'training'])

  const handleChange = (patch: Partial<TrainingTargetFormState>) => {
    onChange(index, patch)
  }

  const basicTab = (
    <Row gutter={16}>
      <Col span={12}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.targetName')}</Text>
          <Input
            value={target.target_name}
            onChange={e => handleChange({ target_name: e.target.value })}
            placeholder={t('create.fields.targetName')}
            style={{ marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={12}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.dataPhase')}</Text>
          <Select
            value={target.data_phase}
            onChange={v => handleChange({ data_phase: v })}
            style={{ width: '100%', marginTop: 4 }}
          >
            <Option value="qa">{t('create.fields.dataPhaseQa')}</Option>
            <Option value="final">{t('create.fields.dataPhaseFinal')}</Option>
          </Select>
        </div>
      </Col>
      <Col span={12}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.modelType')}</Text>
          <Select
            value={target.model_type}
            onChange={v => handleChange({ model_type: v })}
            style={{ width: '100%', marginTop: 4 }}
          >
            <Option value="llm">LLM</Option>
            <Option value="embedding">Embedding</Option>
            <Option value="reranker">Reranker</Option>
            <Option value="decoder_reranker">Decoder Reranker</Option>
          </Select>
        </div>
      </Col>
      <Col span={12}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('detail.configFields.baseModelPath')}</Text>
          <Select
            showSearch
            value={target.base_model_id || undefined}
            onChange={(v) => {
              const model = registeredModels.find(m => m.model_id === v)
              if (model) {
                handleChange({
                  base_model_id: model.model_id,
                  base_model_name: model.display_name || model.model_name,
                  base_model_path: model.model_path,
                })
              }
            }}
            placeholder={t('detail.configFields.baseModelPath')}
            style={{ width: '100%', marginTop: 4 }}
            filterOption={(input, option) =>
              (option?.children as unknown as string)?.toLowerCase().includes(input.toLowerCase())
            }
          >
            {registeredModels.map(m => (
              <Option key={m.model_id} value={m.model_id}>{m.display_name || m.model_name}</Option>
            ))}
          </Select>
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.trainingThreshold')}</Text>
          <InputNumber
            value={target.training_threshold}
            onChange={v => handleChange({ training_threshold: v ?? 1000 })}
            min={0} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.targetPriority')}</Text>
          <InputNumber
            value={target.priority}
            onChange={v => handleChange({ priority: v ?? 0 })}
            min={0} style={{ width: '100%', marginTop: 4 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>{t('create.fields.targetPriorityHelp')}</Text>
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>LoRA Rank</Text>
          <InputNumber
            value={target.lora_r}
            onChange={v => handleChange({ lora_r: v ?? 16 })}
            min={1} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
    </Row>
  )

  const advancedTab = (
    <Row gutter={16}>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>LoRA Alpha</Text>
          <InputNumber
            value={target.lora_alpha}
            onChange={v => handleChange({ lora_alpha: v ?? 32 })}
            min={1} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.numTrainEpochs')}</Text>
          <InputNumber
            value={target.num_train_epochs}
            onChange={v => handleChange({ num_train_epochs: v ?? 3 })}
            min={1} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.learningRate')}</Text>
          <InputNumber
            value={target.learning_rate}
            onChange={v => handleChange({ learning_rate: v ?? 2e-5 })}
            min={0} step={1e-5} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.batchSize')}</Text>
          <InputNumber
            value={target.per_device_train_batch_size}
            onChange={v => handleChange({ per_device_train_batch_size: v ?? 16 })}
            min={1} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.warmupRatio')}</Text>
          <InputNumber
            value={target.warmup_ratio}
            onChange={v => handleChange({ warmup_ratio: v ?? 0.1 })}
            min={0} max={1} step={0.05} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      <Col span={8}>
        <div style={{ marginBottom: 16 }}>
          <Text strong>{t('create.fields.gradAccumSteps')}</Text>
          <InputNumber
            value={target.gradient_accumulation_steps}
            onChange={v => handleChange({ gradient_accumulation_steps: v ?? 1 })}
            min={1} style={{ width: '100%', marginTop: 4 }}
          />
        </div>
      </Col>
      {target.model_type === 'embedding' && (
        <Col span={12}>
          <div style={{ marginBottom: 16 }}>
            <Text strong>{t('create.fields.lossFunction')}</Text>
            <Select
              value={target.embedding_loss_name || undefined}
              onChange={v => handleChange({ embedding_loss_name: v })}
              allowClear
              placeholder={t('create.fields.lossAuto')}
              style={{ width: '100%', marginTop: 4 }}
            >
              <Option value="MultipleNegativesRankingLoss">MultipleNegativesRankingLoss</Option>
              <Option value="CachedMultipleNegativesRankingLoss">CachedMultipleNegativesRankingLoss</Option>
              <Option value="TripletLoss">TripletLoss</Option>
              <Option value="CoSENTLoss">CoSENTLoss</Option>
              <Option value="ContrastiveLoss">ContrastiveLoss</Option>
              <Option value="DynamicExplicitNegativesRankingLoss">DynamicExplicitNegativesRankingLoss</Option>
            </Select>
          </div>
        </Col>
      )}
      {target.model_type === 'reranker' && (
        <>
          <Col span={12}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>{t('create.fields.lossFunction')}</Text>
              <Select
                value={target.reranker_loss_name}
                onChange={v => handleChange({ reranker_loss_name: v })}
                style={{ width: '100%', marginTop: 4 }}
                options={rerankerLossOptions}
              />
            </div>
          </Col>
          {(() => {
            const lossName = target.reranker_loss_name
            if (!lossName || lossName === 'auto') return null

            const lossConfig = (target.loss_config ?? {}) as Record<string, unknown>
            const showScale = rerankerScaleLosses.includes(lossName)
            const showMiniBatch = rerankerMiniBatchLosses.includes(lossName)
            const showNegatives = rerankerNegativesLosses.includes(lossName)
            const showTopK = rerankerTopKLosses.includes(lossName)
            const showSigma = rerankerSigmaLosses.includes(lossName)
            const showRespectInputOrder = rerankerRespectInputOrderLosses.includes(lossName)

            const patchLossConfig = (patch: Record<string, unknown>) => handleChange({
              loss_config: { ...lossConfig, ...patch },
            })

            return (
              <>
                {showScale && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Scale</Text>
                      <InputNumber
                        value={(lossConfig.scale as number | undefined) ?? 10.0}
                        onChange={v => patchLossConfig({ scale: v ?? 10.0 })}
                        min={0}
                        max={100}
                        step={0.1}
                        style={{ width: '100%', marginTop: 4 }}
                      />
                    </div>
                  </Col>
                )}
                {showMiniBatch && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Mini Batch Size</Text>
                      <InputNumber
                        value={(lossConfig.mini_batch_size as number | undefined) ?? 32}
                        onChange={v => patchLossConfig({ mini_batch_size: v ?? 32 })}
                        min={1}
                        max={512}
                        step={1}
                        style={{ width: '100%', marginTop: 4 }}
                      />
                    </div>
                  </Col>
                )}
                {showNegatives && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Num Negatives</Text>
                      <InputNumber
                        value={(lossConfig.num_negatives as number | undefined) ?? 4}
                        onChange={v => patchLossConfig({ num_negatives: v ?? 4 })}
                        min={1}
                        max={128}
                        step={1}
                        style={{ width: '100%', marginTop: 4 }}
                      />
                    </div>
                  </Col>
                )}
                {showTopK && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Top K</Text>
                      <InputNumber
                        value={lossConfig.k as number | undefined}
                        onChange={v => patchLossConfig({ k: v ?? undefined })}
                        min={1}
                        max={1024}
                        step={1}
                        style={{ width: '100%', marginTop: 4 }}
                      />
                    </div>
                  </Col>
                )}
                {showSigma && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Sigma</Text>
                      <InputNumber
                        value={(lossConfig.sigma as number | undefined) ?? 1.0}
                        onChange={v => patchLossConfig({ sigma: v ?? 1.0 })}
                        min={0}
                        max={100}
                        step={0.1}
                        style={{ width: '100%', marginTop: 4 }}
                      />
                    </div>
                  </Col>
                )}
                {showRespectInputOrder && (
                  <Col span={12}>
                    <div style={{ marginBottom: 16 }}>
                      <Text strong>Respect Input Order</Text>
                      <Select
                        value={(lossConfig.respect_input_order as boolean | undefined) ?? true}
                        onChange={v => patchLossConfig({ respect_input_order: v })}
                        style={{ width: '100%', marginTop: 4 }}
                        options={[
                          { label: 'True', value: true },
                          { label: 'False', value: false },
                        ]}
                      />
                    </div>
                  </Col>
                )}
              </>
            )
          })()}
        </>
      )}
      {target.model_type === 'llm' && (target.training_method === 'dpo' || target.training_method === 'orpo') && (
        <>
          <Divider>{t('create.preference.divider', { ns: 'training' })}</Divider>
          <Col span={12}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>{target.training_method === 'dpo' ? 'DPO Beta' : 'ORPO Beta'}</Text>
              <InputNumber
                value={target.rl_config?.beta ?? 0.1}
                onChange={v => handleChange({ rl_config: { ...target.rl_config, beta: v ?? 0.1 } })}
                min={0.01}
                max={10}
                step={0.01}
                style={{ width: '100%', marginTop: 4 }}
              />
            </div>
          </Col>
          <Col span={12}>
            <div style={{ marginBottom: 16 }}>
              <Text strong>{t('create.preference.rankingsDirection', { ns: 'training' })}</Text>
              <Select
                value={target.rl_config?.rankings_direction ?? 'auto'}
                onChange={v => handleChange({ rl_config: { ...target.rl_config, rankings_direction: v } })}
                style={{ width: '100%', marginTop: 4 }}
                options={[
                  { label: t('create.preference.rankingsDirectionAuto', { ns: 'training' }), value: 'auto' },
                  { label: t('create.preference.rankingsDirectionHigher', { ns: 'training' }), value: 'higher_is_better' },
                  { label: t('create.preference.rankingsDirectionLower', { ns: 'training' }), value: 'lower_is_better' },
                ]}
              />
              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                {t('create.preference.rankingsDirectionExtra', { ns: 'training' })}
              </Text>
            </div>
          </Col>
        </>
      )}
    </Row>
  )

  return (
    <Tabs
      size="small"
      items={[
        { key: 'basic', label: t('create.tabs.basic'), children: basicTab },
        { key: 'advanced', label: t('create.tabs.advanced'), children: advancedTab },
      ]}
    />
  )
}
