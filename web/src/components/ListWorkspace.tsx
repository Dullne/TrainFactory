import type { ReactNode } from 'react'
import { Empty, Pagination, Spin } from 'antd'
import type { PaginationProps } from 'antd'
import type { ColumnsType, ColumnType } from 'antd/es/table'
import { useTranslation } from 'react-i18next'
import './ListWorkspace.css'

/** Reuse table cells so permissions, confirmations and handlers stay identical on mobile. */
export function renderListCell<T>(columns: ColumnsType<T>, key: string, record: T): ReactNode {
  const column = columns.find((item) => item.key === key) as ColumnType<T> | undefined
  if (!column) return null
  const value =
    typeof column.dataIndex === 'string' ? record[column.dataIndex as keyof T] : undefined
  return column.render ? (column.render(value, record, 0) as ReactNode) : (value as ReactNode)
}

export function MobileList({
  children,
  loading,
  empty,
  pagination,
}: {
  children: ReactNode
  loading: boolean
  empty: boolean
  pagination: PaginationProps
}) {
  const { t } = useTranslation('listWorkspace')
  return (
    <div className="list-workspace-mobile" aria-busy={loading}>
      <Spin spinning={loading} tip={t('loading')}>
        <div className="list-workspace-cards">
          {empty ? <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} /> : children}
        </div>
      </Spin>
      <Pagination
        {...pagination}
        size="small"
        showLessItems
        className="list-workspace-pagination"
      />
    </div>
  )
}
