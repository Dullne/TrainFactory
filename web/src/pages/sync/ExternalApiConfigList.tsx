import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import {
  Table, Button, Space, Tag, Popconfirm, message, Modal,
  Form, Input, Select, Typography,
} from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined, ReloadOutlined, ApiOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { externalApiConfigApi } from '@/services/api'
import { formatDate } from '@/utils'
import type { ExternalApiConfig } from '@/types'
import { TEXT_SECONDARY } from '@/theme'

const { Title } = Typography

export default function ExternalApiConfigList() {
  const { t } = useTranslation(['sync', 'common'])
  const [configs, setConfigs] = useState<ExternalApiConfig[]>([])
  const [loading, setLoading] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<ExternalApiConfig | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [testing, setTesting] = useState(false)
  const [form] = Form.useForm()

  const fetchConfigs = useCallback(async () => {
    setLoading(true)
    try {
      const res = await externalApiConfigApi.list()
      setConfigs(res.configs || [])
    } catch {
      // handled by interceptor
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchConfigs() }, [fetchConfigs])

  const openCreate = () => {
    setEditing(null)
    form.resetFields()
    setModalOpen(true)
  }

  const openEdit = (record: ExternalApiConfig) => {
    setEditing(record)
    form.setFieldsValue({
      config_name: record.config_name,
      api_url: record.api_url,
      auth_method: record.auth_config?.auth_method || 'cookie',
      description: record.description,
      // Don't pre-fill token for security
    })
    setModalOpen(true)
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      setSubmitting(true)

      if (editing) {
        // Update: only send changed fields
        const updates: Record<string, unknown> = {}
        if (values.config_name !== editing.config_name) updates.config_name = values.config_name
        if (values.api_url !== editing.api_url) updates.api_url = values.api_url
        if (values.description !== editing.description) updates.description = values.description
        if (values.token || values.auth_method !== (editing.auth_config?.auth_method || 'cookie')) {
          updates.auth_config = {
            token: values.token || undefined,
            auth_method: values.auth_method || 'cookie',
          }
        }

        if (Object.keys(updates).length > 0) {
          await externalApiConfigApi.update(editing.config_id, updates)
          message.success(t('apiConfig.message.updateSuccess'))
        }
      } else {
        await externalApiConfigApi.create({
          config_name: values.config_name,
          api_url: values.api_url,
          auth_config: { token: values.token, auth_method: values.auth_method || 'cookie' },
          description: values.description || '',
        })
        message.success(t('apiConfig.message.createSuccess'))
      }

      setModalOpen(false)
      form.resetFields()
      fetchConfigs()
    } catch {
      message.error(editing ? t('apiConfig.message.updateFailed') : t('apiConfig.message.createFailed'))
    } finally {
      setSubmitting(false)
    }
  }

  const handleDelete = async (configId: string) => {
    try {
      await externalApiConfigApi.delete(configId)
      message.success(t('apiConfig.message.deleteSuccess'))
      fetchConfigs()
    } catch {
      message.error(t('apiConfig.message.deleteFailed'))
    }
  }

  const [testingId, setTestingId] = useState<string | null>(null)

  const handleTestById = async (configId: string) => {
    setTestingId(configId)
    try {
      const res = await externalApiConfigApi.testConnectionById(configId)
      if (res.success) {
        message.success(res.message)
      } else {
        message.error(res.message)
      }
      fetchConfigs()
    } catch {
      message.error(t('apiConfig.message.testFailed', { message: '' }))
    } finally {
      setTestingId(null)
    }
  }

  const handleTestConnection = async () => {
    try {
      const values = await form.validateFields(['api_url', 'token', 'auth_method'])
      // For editing without new token, use the saved config's test endpoint
      if (editing && !values.token) {
        setTesting(true)
        const res = await externalApiConfigApi.testConnectionById(editing.config_id)
        if (res.success) {
          message.success(t('apiConfig.message.testSuccess', { message: res.message }))
        } else {
          message.error(t('apiConfig.message.testFailed', { message: res.message }))
        }
        return
      }
      if (!values.api_url || !values.token) {
        message.warning(t('common:form.required'))
        return
      }
      setTesting(true)
      const res = await externalApiConfigApi.testConnection({
        api_url: values.api_url,
        auth_config: { token: values.token, auth_method: values.auth_method || 'cookie' },
      })
      if (res.success) {
        message.success(t('apiConfig.message.testSuccess', { message: res.message }))
      } else {
        message.error(t('apiConfig.message.testFailed', { message: res.message }))
      }
    } catch {
      message.error(t('apiConfig.message.testFailed', { message: 'Request failed' }))
    } finally {
      setTesting(false)
    }
  }

  const columns: ColumnsType<ExternalApiConfig> = [
    {
      title: t('apiConfig.columns.configName'),
      dataIndex: 'config_name',
      key: 'config_name',
    },
    {
      title: t('apiConfig.columns.apiUrl'),
      dataIndex: 'api_url',
      key: 'api_url',
      ellipsis: true,
      render: (url: string) => (
        <span style={{ fontSize: 12, fontFamily: 'monospace' }}>{url}</span>
      ),
    },
    {
      title: t('apiConfig.columns.description'),
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
      render: (v: string) => v || <span style={{ color: TEXT_SECONDARY }}>-</span>,
    },
    {
      title: t('apiConfig.columns.status'),
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : status === 'error' ? 'red' : 'default'}>{status}</Tag>
      ),
    },
    {
      title: t('apiConfig.columns.createdAt'),
      dataIndex: 'created_at',
      key: 'created_at',
      width: 160,
      render: (v: string) => <span style={{ fontSize: 12 }}>{formatDate(v)}</span>,
    },
    {
      title: t('apiConfig.columns.actions'),
      key: 'actions',
      width: 160,
      render: (_, record) => (
        <Space size={4}>
          <Button type="text" size="small" icon={<ApiOutlined />}
            loading={testingId === record.config_id}
            onClick={() => handleTestById(record.config_id)} />
          <Button type="text" size="small" icon={<EditOutlined />} onClick={() => openEdit(record)} />
          <Popconfirm title={t('apiConfig.actions.confirmDelete')} onConfirm={() => handleDelete(record.config_id)}>
            <Button type="text" size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>{t('apiConfig.title')}</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchConfigs} loading={loading}>
            {t('common:action.refresh')}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            {t('apiConfig.actions.create')}
          </Button>
        </Space>
      </div>

      <Table
        rowKey="config_id"
        columns={columns}
        dataSource={configs}
        loading={loading}
        pagination={false}
        locale={{ emptyText: t('apiConfig.empty') }}
      />

      <Modal
        title={editing ? t('apiConfig.actions.edit') : t('apiConfig.actions.create')}
        open={modalOpen}
        onCancel={() => { setModalOpen(false); form.resetFields() }}
        destroyOnClose
        footer={
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button icon={<ApiOutlined />} onClick={handleTestConnection} loading={testing}>
              {t('apiConfig.actions.testConnection')}
            </Button>
            <Space>
              <Button onClick={() => { setModalOpen(false); form.resetFields() }}>
                {t('common:action.cancel')}
              </Button>
              <Button type="primary" onClick={handleSubmit} loading={submitting}>
                {t('common:action.confirm')}
              </Button>
            </Space>
          </div>
        }
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="config_name"
            label={t('apiConfig.fields.configName')}
            rules={[{ required: true, message: t('common:form.required') }]}
          >
            <Input placeholder={t('apiConfig.fields.configNamePlaceholder')} />
          </Form.Item>
          <Form.Item
            name="api_url"
            label={t('apiConfig.fields.apiUrl')}
            rules={[{ required: true, message: t('common:form.required') }]}
          >
            <Input placeholder={t('apiConfig.fields.apiUrlPlaceholder')} />
          </Form.Item>
          <Form.Item
            name="auth_method"
            label={t('apiConfig.fields.authMethod')}
            initialValue="cookie"
            tooltip={t('apiConfig.fields.authMethodHelp')}
          >
            <Select>
              <Select.Option value="cookie">{t('apiConfig.fields.authMethodCookie')}</Select.Option>
              <Select.Option value="bearer">{t('apiConfig.fields.authMethodBearer')}</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="token"
            label={t('apiConfig.fields.token')}
            rules={editing ? [] : [{ required: true, message: t('common:form.required') }]}
            tooltip={t('apiConfig.fields.tokenHelp')}
          >
            <Input.Password placeholder={editing ? '(leave empty to keep current)' : t('apiConfig.fields.tokenPlaceholder')} />
          </Form.Item>
          <Form.Item
            name="description"
            label={t('apiConfig.fields.description')}
          >
            <Input.TextArea rows={2} placeholder={t('apiConfig.fields.descriptionPlaceholder')} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
