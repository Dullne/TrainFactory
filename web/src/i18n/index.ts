import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import zhCommon from './locales/zh/common.json'
import zhTraining from './locales/zh/training.json'
import zhDatasets from './locales/zh/datasets.json'
import zhGeneration from './locales/zh/generation.json'
import zhModels from './locales/zh/models.json'
import zhConfigs from './locales/zh/configs.json'
import zhDeployments from './locales/zh/deployments.json'
import zhEvaluations from './locales/zh/evaluations.json'
import zhVectordb from './locales/zh/vectordb.json'
import zhResources from './locales/zh/resources.json'
import zhSync from './locales/zh/sync.json'

import enCommon from './locales/en/common.json'
import enTraining from './locales/en/training.json'
import enDatasets from './locales/en/datasets.json'
import enGeneration from './locales/en/generation.json'
import enModels from './locales/en/models.json'
import enConfigs from './locales/en/configs.json'
import enDeployments from './locales/en/deployments.json'
import enEvaluations from './locales/en/evaluations.json'
import enVectordb from './locales/en/vectordb.json'
import enResources from './locales/en/resources.json'
import enSync from './locales/en/sync.json'

const LANGUAGE_KEY = 'tf_language'

export const getStoredLanguage = (): string => {
  if (typeof window === 'undefined') {
    return 'zh'
  }
  try {
    const lang = window.localStorage.getItem(LANGUAGE_KEY)
    return lang === 'en' ? 'en' : 'zh'
  } catch {
    return 'zh'
  }
}

export const setStoredLanguage = (lang: string): void => {
  if (typeof window === 'undefined') {
    return
  }
  try {
    window.localStorage.setItem(LANGUAGE_KEY, lang === 'en' ? 'en' : 'zh')
  } catch {
    // Ignore storage write failures (e.g. privacy mode / restricted webview).
  }
}

// Sync HTML lang attribute with stored language preference on initial load
const storedLang = getStoredLanguage()
document.documentElement.lang = storedLang === 'en' ? 'en' : 'zh-CN'

i18n.use(initReactI18next).init({
  resources: {
    zh: {
      common: zhCommon,
      training: zhTraining,
      datasets: zhDatasets,
      generation: zhGeneration,
      models: zhModels,
      configs: zhConfigs,
      deployments: zhDeployments,
      evaluations: zhEvaluations,
      vectordb: zhVectordb,
      resources: zhResources,
      sync: zhSync,
    },
    en: {
      common: enCommon,
      training: enTraining,
      datasets: enDatasets,
      generation: enGeneration,
      models: enModels,
      configs: enConfigs,
      deployments: enDeployments,
      evaluations: enEvaluations,
      vectordb: enVectordb,
      resources: enResources,
      sync: enSync,
    },
  },
  lng: getStoredLanguage(),
  fallbackLng: 'zh',
  defaultNS: 'common',
  ns: [
    'common', 'training', 'datasets', 'generation',
    'models', 'configs', 'deployments', 'evaluations',
    'vectordb', 'resources', 'sync',
  ],
  interpolation: {
    escapeValue: false,
  },
  react: {
    useSuspense: false,
  },
})

export default i18n
