import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Form,
  Input,
  Select,
  Checkbox,
  Button,
  Card,
  Space,
  message,
  Typography,
  Collapse,
  InputNumber,
} from 'antd'
import { ArrowLeftOutlined, PlusOutlined, MinusCircleOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { datasetApi } from '@/services/api'
import type { CreateDatasetRequest, DatasetType, DatasetUsage, DatasetModelType, DatasetColumn } from '@/types'

const { Title } = Typography
const { TextArea } = Input

interface DatasetTypeOption {
  label: string
  value: DatasetType
  description: string
  modelType: DatasetModelType
}

const fileFormatOptions = [
  { label: 'JSONL', value: 'jsonl' },
  { label: 'JSON', value: 'json' },
  { label: 'CSV', value: 'csv' },
  { label: 'Parquet', value: 'parquet' },
  { label: 'Arrow', value: 'arrow' },
]

const columnTypeOptions = [
  { label: 'string', value: 'string' },
  { label: 'int', value: 'int' },
  { label: 'float', value: 'float' },
  { label: 'bool', value: 'bool' },
  { label: 'list', value: 'list' },
  { label: 'dict', value: 'dict' },
]

export default function DatasetCreate() {
  const { t } = useTranslation(['datasets', 'common'])
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [selectedModelTypes, setSelectedModelTypes] = useState<DatasetModelType[]>([])

  const usageOptions: { label: string; value: DatasetUsage }[] = [
    { label: t('options.usage.raw'), value: 'raw' },
    { label: t('options.usage.train'), value: 'train' },
    { label: t('options.usage.eval'), value: 'eval' },
    { label: t('options.usage.test'), value: 'test' },
  ]

  const modelTypeOptions: { label: string; value: DatasetModelType }[] = [
    { label: 'Embedding', value: 'embedding' },
    { label: 'Rerank', value: 'rerank' },
    { label: 'LLM', value: 'llm' },
  ]

  // Filter dataset types based on selected model types
  const filteredDatasetTypes = useMemo(() => {
    // Dataset types grouped by model type (10 embedding + 3 rerank + 3 llm)
    const allDatasetTypeOptions: DatasetTypeOption[] = [
      // Embedding types (10 types)
      { label: 'Universal', value: 'embedding_universal', description: t('options.datasetType.embeddingUniversalDesc'), modelType: 'embedding' },
      { label: 'Pair', value: 'embedding_pair', description: t('options.datasetType.embeddingPairDesc'), modelType: 'embedding' },
      { label: 'Triplet', value: 'embedding_triplet', description: t('options.datasetType.embeddingTripletDesc'), modelType: 'embedding' },
      { label: 'Multi Negative', value: 'embedding_multi_neg', description: t('options.datasetType.embeddingMultiNegDesc'), modelType: 'embedding' },
      { label: 'Dynamic Negative', value: 'embedding_dynamic_neg', description: t('options.datasetType.embeddingDynamicNegDesc'), modelType: 'embedding' },
      { label: 'Cosine', value: 'embedding_cosine', description: t('options.datasetType.embeddingCosineDesc'), modelType: 'embedding' },
      { label: 'Margin Triplet', value: 'embedding_margin', description: t('options.datasetType.embeddingMarginDesc'), modelType: 'embedding' },
      { label: 'Margin Multi', value: 'embedding_margin_multi', description: t('options.datasetType.embeddingMarginMultiDesc'), modelType: 'embedding' },
      { label: 'Scored Multi', value: 'embedding_scored', description: t('options.datasetType.embeddingScoredDesc'), modelType: 'embedding' },
      { label: 'Score Triplet', value: 'embedding_score_triplet', description: t('options.datasetType.embeddingScoreTripletDesc'), modelType: 'embedding' },
      // Rerank types
      { label: 'Rerank Pair', value: 'rerank_pair', description: t('options.datasetType.rerankPairDesc'), modelType: 'rerank' },
      { label: 'Rerank Triplet', value: 'rerank_triplet', description: t('options.datasetType.rerankTripletDesc'), modelType: 'rerank' },
      { label: 'Rerank Listwise', value: 'rerank_listwise', description: t('options.datasetType.rerankListwiseDesc'), modelType: 'rerank' },
      // LLM types
      { label: 'SFT Instruct', value: 'sft_instruct', description: t('options.datasetType.sftInstructDesc'), modelType: 'llm' },
      { label: 'DPO Preference', value: 'dpo_preference', description: t('options.datasetType.dpoPreferenceDesc'), modelType: 'llm' },
      { label: 'RL Reward', value: 'rl_reward', description: t('options.datasetType.rlRewardDesc'), modelType: 'llm' },
    ]

    const customOpt = { label: `Custom - ${t('options.datasetType.customDesc')}`, value: 'custom' }

    if (selectedModelTypes.length === 0) {
      return [
        ...allDatasetTypeOptions.map(opt => ({
          label: `${opt.label} - ${opt.description}`,
          value: opt.value,
        })),
        customOpt,
      ]
    }
    return [
      ...allDatasetTypeOptions
        .filter(opt => selectedModelTypes.includes(opt.modelType))
        .map(opt => ({
          label: `${opt.label} - ${opt.description}`,
          value: opt.value,
        })),
      customOpt,
    ]
  }, [selectedModelTypes, t])

  interface FormValues extends Omit<CreateDatasetRequest, 'columns'> {
    columns?: DatasetColumn[]
    tags_str?: string
    content_field?: string
  }

  const handleSubmit = async (values: FormValues) => {
    setLoading(true)
    try {
      const submitData: CreateDatasetRequest = {
        dataset_name: values.dataset_name,
        storage_path: values.storage_path,
        dataset_type: values.dataset_type,
        usage: values.usage,
        model_type: values.model_type,
        display_name: values.display_name,
        description: values.description,
        file_format: values.file_format,
        num_rows: values.num_rows,
        num_train: values.num_train,
        num_eval: values.num_eval,
        num_test: values.num_test,
      }

      if (values.columns && values.columns.length > 0) {
        submitData.columns = values.columns.filter(c => c.name)
      }

      if (values.tags_str) {
        submitData.tags = values.tags_str.split(',').map(t => t.trim()).filter(Boolean)
      }

      if (values.content_field) {
        submitData.extra_metadata = {
          ...submitData.extra_metadata,
          content_field: values.content_field,
        }
      }

      await datasetApi.create(submitData)
      message.success(t('create.message.createSuccess'))
      navigate('/datasets')
    } catch (error) {
      message.error(error instanceof Error ? error.message : t('create.message.createFailed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 600 }}>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets')}>
          {t('common:action.back')}
        </Button>
        <Title level={4} style={{ margin: 0 }}>{t('create.title')}</Title>
      </Space>

      <Card>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            file_format: 'jsonl',
            usage: 'train',
          }}
        >
          <Form.Item
            name="dataset_name"
            label={t('create.form.datasetName.label')}
            rules={[{ required: true, message: t('create.form.datasetName.required') }]}
          >
            <Input placeholder={t('create.form.datasetName.placeholder')} />
          </Form.Item>

          <Form.Item
            name="description"
            label={t('create.form.description.label')}
          >
            <TextArea rows={3} placeholder={t('create.form.description.placeholder')} />
          </Form.Item>

          <Form.Item
            name="model_type"
            label={t('create.form.modelType.label')}
            extra={t('create.form.modelType.extra')}
          >
            <Checkbox.Group
              options={modelTypeOptions}
              onChange={(values) => {
                setSelectedModelTypes((values as DatasetModelType[]) || [])
                form.setFieldValue('dataset_type', undefined)
              }}
            />
          </Form.Item>

          <Form.Item
            name="dataset_type"
            label={t('create.form.datasetType.label')}
            rules={[{ required: true, message: t('create.form.datasetType.required') }]}
          >
            <Select
              placeholder={t('create.form.datasetType.placeholder')}
              options={filteredDatasetTypes}
            />
          </Form.Item>

          <Form.Item
            name="usage"
            label={t('create.form.usage.label')}
          >
            <Select options={usageOptions} allowClear placeholder={t('options.optional')} />
          </Form.Item>

          <Form.Item
            name="storage_path"
            label={t('create.form.storagePath.label')}
            rules={[{ required: true, message: t('create.form.storagePath.required') }]}
            extra={t('create.form.storagePath.extra')}
          >
            <Input placeholder={t('create.form.storagePath.placeholder')} />
          </Form.Item>

          <Form.Item
            name="file_format"
            label={t('create.form.fileFormat.label')}
          >
            <Select options={fileFormatOptions} />
          </Form.Item>

          <Collapse
            ghost
            items={[
              {
                key: 'advanced',
                label: t('create.form.advanced'),
                children: (
                  <>
                    <Form.Item label={t('create.form.columns.label')} extra={t('create.form.columns.extra')}>
                      <Form.List name="columns">
                        {(fields, { add, remove }) => (
                          <>
                            {fields.map(({ key, name, ...restField }) => (
                              <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                                <Form.Item
                                  {...restField}
                                  name={[name, 'name']}
                                  rules={[{ required: true, message: t('create.form.columns.fieldNameRequired') }]}
                                  style={{ marginBottom: 0 }}
                                >
                                  <Input placeholder={t('create.form.columns.fieldNamePlaceholder')} style={{ width: 180 }} />
                                </Form.Item>
                                <Form.Item
                                  {...restField}
                                  name={[name, 'type']}
                                  initialValue="string"
                                  style={{ marginBottom: 0 }}
                                >
                                  <Select options={columnTypeOptions} style={{ width: 100 }} />
                                </Form.Item>
                                <MinusCircleOutlined onClick={() => remove(name)} style={{ color: '#ff4d4f' }} />
                              </Space>
                            ))}
                            <Button type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                              {t('create.form.columns.addField')}
                            </Button>
                          </>
                        )}
                      </Form.List>
                    </Form.Item>

                    <Form.Item label={t('create.form.sampleCount.label')} extra={t('create.form.sampleCount.extra')}>
                      <Space wrap>
                        <Form.Item name="num_rows" noStyle>
                          <InputNumber placeholder={t('create.form.sampleCount.total')} min={0} style={{ width: 100 }} addonBefore={t('create.form.sampleCount.total')} />
                        </Form.Item>
                        <Form.Item name="num_train" noStyle>
                          <InputNumber placeholder={t('create.form.sampleCount.train')} min={0} style={{ width: 100 }} addonBefore={t('create.form.sampleCount.train')} />
                        </Form.Item>
                        <Form.Item name="num_eval" noStyle>
                          <InputNumber placeholder={t('create.form.sampleCount.eval')} min={0} style={{ width: 100 }} addonBefore={t('create.form.sampleCount.eval')} />
                        </Form.Item>
                        <Form.Item name="num_test" noStyle>
                          <InputNumber placeholder={t('create.form.sampleCount.test')} min={0} style={{ width: 100 }} addonBefore={t('create.form.sampleCount.test')} />
                        </Form.Item>
                      </Space>
                    </Form.Item>

                    <Form.Item
                      name="tags_str"
                      label={t('create.form.tags.label')}
                      extra={t('create.form.tags.extra')}
                    >
                      <Input placeholder={t('create.form.tags.placeholder')} />
                    </Form.Item>

                    <Form.Item
                      name="display_name"
                      label={t('create.form.displayName.label')}
                    >
                      <Input placeholder={t('create.form.displayName.placeholder')} />
                    </Form.Item>

                    <Form.Item
                      name="content_field"
                      label={t('create.form.contentField.label')}
                      extra={t('create.form.contentField.extra')}
                    >
                      <Input placeholder={t('create.form.contentField.placeholder')} />
                    </Form.Item>
                  </>
                ),
              },
            ]}
            style={{ marginBottom: 16 }}
          />

          <Form.Item>
            <Space>
              <Button onClick={() => navigate('/datasets')}>{t('common:action.cancel')}</Button>
              <Button type="primary" htmlType="submit" loading={loading} disabled={loading}>
                {t('create.form.submit')}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>
    </div>
  )
}
