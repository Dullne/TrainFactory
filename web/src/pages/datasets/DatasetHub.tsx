import { Tabs } from 'antd'
import { useSearchParams } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import DatasetList from './DatasetList'
import GenerationList from '../generation/GenerationList'

export default function DatasetHub() {
  const { t } = useTranslation(['datasets', 'common'])
  const [searchParams, setSearchParams] = useSearchParams()
  const activeKey = searchParams.get('tab') === 'generation' ? 'generation' : 'list'

  const handleTabChange = (key: string) => {
    setSearchParams({ tab: key })
  }

  const tabItems = [
    {
      key: 'list',
      label: t('hub.tabs.list'),
      children: <DatasetList />,
    },
    {
      key: 'generation',
      label: t('hub.tabs.generation'),
      children: <GenerationList />,
    },
  ]

  return (
    <Tabs
      activeKey={activeKey}
      onChange={handleTabChange}
      items={tabItems}
      destroyOnHidden
    />
  )
}
