import React, { useMemo } from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import enUS from 'antd/locale/en_US'
import { useTranslation } from 'react-i18next'
import { createTheme } from './theme'
import { ThemeProvider, useAppearance } from './theme/ThemeProvider'
import { applyAppearance, readAppearance } from './theme/appearance'
import App from './App'
import { AuthProvider } from './auth/AuthContext'
import './i18n'
import './index.css'
import './components/PageLayout.css'

function AppWithLocale() {
  const { i18n } = useTranslation()
  const antdLocale = i18n.language === 'en' ? enUS : zhCN
  const { mode, accent, style } = useAppearance()
  const themeConfig = useMemo(() => createTheme({ mode, accent, style }), [mode, accent, style])

  return (
    <ConfigProvider locale={antdLocale} theme={themeConfig}>
      <AntApp>
        <AuthProvider>
          <App />
        </AuthProvider>
      </AntApp>
    </ConfigProvider>
  )
}

const initialAppearance = readAppearance()
applyAppearance(initialAppearance)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider initialAppearance={initialAppearance}>
      <AppWithLocale />
    </ThemeProvider>
  </React.StrictMode>
)
