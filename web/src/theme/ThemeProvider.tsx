import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  APPEARANCE_STORAGE_KEY,
  applyAppearance,
  parseAppearance,
  type Appearance,
} from './appearance'

interface AppearanceContextValue extends Appearance {
  setAppearance: (next: Appearance) => void
}

const AppearanceContext = createContext<AppearanceContextValue | null>(null)

export function ThemeProvider({
  children,
  initialAppearance,
}: {
  children: ReactNode
  initialAppearance: Appearance
}) {
  const [appearance, setState] = useState(initialAppearance)

  useLayoutEffect(() => {
    applyAppearance(appearance)
  }, [appearance])

  const setAppearance = useCallback((next: Appearance) => {
    setState(next)
    try {
      window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(next))
    } catch {
      // Theme switching still works when browser storage is unavailable.
    }
  }, [])

  useEffect(() => {
    const syncAppearance = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) return
      if (event.key === APPEARANCE_STORAGE_KEY || event.key === null) {
        setState(parseAppearance(event.newValue))
      }
    }
    window.addEventListener('storage', syncAppearance)
    return () => window.removeEventListener('storage', syncAppearance)
  }, [])

  const value = useMemo(() => ({ ...appearance, setAppearance }), [appearance, setAppearance])
  return <AppearanceContext.Provider value={value}>{children}</AppearanceContext.Provider>
}

export function useAppearance() {
  const context = useContext(AppearanceContext)
  if (!context) throw new Error('useAppearance requires ThemeProvider')
  return context
}
