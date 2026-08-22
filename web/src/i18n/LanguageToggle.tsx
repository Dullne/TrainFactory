import { useTranslation } from 'react-i18next'
import { Button, Tooltip } from 'antd'
import { GlobalOutlined } from '@ant-design/icons'
import { setStoredLanguage } from './index'

export function LanguageToggle() {
  const { t, i18n } = useTranslation('common')
  const isZh = i18n.language === 'zh'

  const handleToggle = () => {
    const newLang = isZh ? 'en' : 'zh'
    i18n.changeLanguage(newLang)
    setStoredLanguage(newLang)
    document.documentElement.lang = newLang === 'zh' ? 'zh-CN' : 'en'
  }

  return (
    <Tooltip title={isZh ? t('language.switchToEn') : t('language.switchToZh')}>
      <Button
        type="text"
        size="small"
        icon={<GlobalOutlined />}
        onClick={handleToggle}
        style={{ color: '#8b949e', fontSize: 14 }}
      >
        {isZh ? 'EN' : '中'}
      </Button>
    </Tooltip>
  )
}
