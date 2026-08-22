import { useState, useEffect, useCallback } from 'react'
import { Modal, Table, Tooltip, Typography, Button, message } from 'antd'
import { CopyOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import { milvusApi } from '@/services/api'

const { Text } = Typography

interface Props {
  open: boolean
  collectionName: string
  onClose: () => void
}

export default function BrowseEntitiesModal({ open, collectionName, onClose }: Props) {
  const { t } = useTranslation(['vectordb', 'common'])
  const [loading, setLoading] = useState(false)
  const [entities, setEntities] = useState<Record<string, unknown>[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const pageSize = 20

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

  const fetchEntities = useCallback(async (p: number) => {
    setLoading(true)
    try {
      const res = await milvusApi.browseEntities(collectionName, {
        offset: (p - 1) * pageSize,
        limit: pageSize,
      })
      setEntities(res.entities)
      setTotal(res.total)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [collectionName])

  useEffect(() => {
    if (open && collectionName) {
      setPage(1)
      fetchEntities(1)
    }
  }, [open, collectionName, fetchEntities])

  const columns: ColumnsType<Record<string, unknown>> = [
    {
      title: 'chunk_id',
      dataIndex: 'chunk_id',
      key: 'chunk_id',
      width: 220,
      ellipsis: true,
      render: (v: string) => (
        <Tooltip title={v}>
          <Text copyable={{ text: v }} style={{ fontSize: 12 }}>{v}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'chunk_content',
      dataIndex: 'chunk_content',
      key: 'chunk_content',
      ellipsis: true,
      render: (v: string) => (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 4 }}>
          <Tooltip title={t('browse.copyFullText')}>
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
              {v?.slice(0, 120)}{v?.length > 120 ? '...' : ''}
            </span>
          </Tooltip>
        </div>
      ),
    },
  ]

  return (
    <Modal
      title={t('browse.title', { name: collectionName })}
      open={open}
      onCancel={onClose}
      footer={null}
      width={900}
      destroyOnHidden
    >
      <Table
        dataSource={entities}
        columns={columns}
        rowKey="chunk_id"
        loading={loading}
        size="small"
        pagination={{
          current: page,
          pageSize,
          total,
          showTotal: (t_total) => t('common:pagination.total', { total: t_total }),
          onChange: (p) => {
            setPage(p)
            fetchEntities(p)
          },
        }}
      />
    </Modal>
  )
}
