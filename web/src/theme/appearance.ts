export type ColorMode = 'dark' | 'light'
export type AccentColor = 'blue' | 'purple' | 'cyan' | 'green' | 'rose'
export const THEME_STYLES = ['classic', 'workbench', 'studio'] as const
export type ThemeStyle = (typeof THEME_STYLES)[number]

export interface Appearance {
  mode: ColorMode
  accent: AccentColor
  style: ThemeStyle
}

export const APPEARANCE_STORAGE_KEY = 'tf_appearance:v1'
// Keep the existing storage key so older saved color choices migrate in place.
export const DEFAULT_APPEARANCE: Appearance = { mode: 'dark', accent: 'blue', style: 'classic' }

export const ACCENTS = {
  blue: { primary: '#1677ff', hover: '#4096ff', active: '#0958d9', darkText: '#69b1ff' },
  purple: { primary: '#722ed1', hover: '#9254de', active: '#531dab', darkText: '#b37feb' },
  cyan: { primary: '#08979c', hover: '#13a8a8', active: '#006d75', darkText: '#5cdbd3' },
  green: { primary: '#389e0d', hover: '#52c41a', active: '#237804', darkText: '#95de64' },
  rose: { primary: '#c41d7f', hover: '#eb2f96', active: '#9e1068', darkText: '#ff85c0' },
} satisfies Record<
  AccentColor,
  { primary: string; hover: string; active: string; darkText: string }
>

const surfaces = {
  dark: {
    bgLayout: '#0d1117',
    bgContainer: '#161b22',
    bgElevated: '#21262d',
    bgSpotlight: '#1c2128',
    textPrimary: '#e6edf3',
    textSecondary: '#9ba5b1',
    textTertiary: '#88929e',
    textDisabled: '#626c78',
    borderPrimary: '#424c58',
    borderSecondary: '#30363d',
    authDivider: '#252d37',
    hoverBg: 'rgba(255, 255, 255, 0.06)',
    mask: 'rgba(0, 0, 0, 0.6)',
  },
  light: {
    bgLayout: '#f3f5f8',
    bgContainer: '#ffffff',
    bgElevated: '#f6f8fa',
    bgSpotlight: '#eaeef3',
    textPrimary: '#1f2937',
    textSecondary: '#526071',
    textTertiary: '#627084',
    textDisabled: '#9aa4b2',
    borderPrimary: '#c8d0da',
    borderSecondary: '#e2e7ee',
    authDivider: '#e2e7ee',
    hoverBg: 'rgba(15, 23, 42, 0.04)',
    mask: 'rgba(15, 23, 42, 0.35)',
  },
}

// Each preset owns geometry and density as well as its surface palette.
const styleGeometry = {
  classic: {
    radius: 8,
    radiusLG: 12,
    radiusSM: 6,
    controlHeight: 32,
    cardPadding: 20,
    cardPaddingSM: 16,
    sectionGap: 20,
    formGap: 16,
    sectionHeaderHeight: 54,
    headerHeight: 56,
    contentPadding: 20,
    contentMargin: 16,
    menuItemHeight: 40,
    menuItemGap: 4,
    tablePadding: 16,
    tablePaddingSM: 8,
    headingSize: 24,
    sidebarWidth: 220,
  },
  workbench: {
    radius: 3,
    radiusLG: 4,
    radiusSM: 2,
    controlHeight: 30,
    cardPadding: 16,
    cardPaddingSM: 12,
    sectionGap: 16,
    formGap: 12,
    sectionHeaderHeight: 46,
    headerHeight: 52,
    contentPadding: 16,
    contentMargin: 12,
    menuItemHeight: 36,
    menuItemGap: 2,
    tablePadding: 10,
    tablePaddingSM: 6,
    headingSize: 22,
    sidebarWidth: 208,
  },
  studio: {
    radius: 12,
    radiusLG: 20,
    radiusSM: 8,
    controlHeight: 36,
    cardPadding: 24,
    cardPaddingSM: 20,
    sectionGap: 24,
    formGap: 20,
    sectionHeaderHeight: 60,
    headerHeight: 64,
    contentPadding: 28,
    contentMargin: 20,
    menuItemHeight: 44,
    menuItemGap: 6,
    tablePadding: 18,
    tablePaddingSM: 12,
    headingSize: 26,
    sidebarWidth: 236,
  },
} satisfies Record<ThemeStyle, Record<string, number>>

const styleSurfaces = {
  classic: { dark: {}, light: {} },
  workbench: {
    dark: {
      bgLayout: '#111315',
      bgContainer: '#191c20',
      bgElevated: '#22262b',
      bgSpotlight: '#292e34',
      borderPrimary: '#535b65',
      borderSecondary: '#353b43',
      textPrimary: '#edf0f3',
      textSecondary: '#acb4bf',
      textTertiary: '#929ba7',
    },
    light: {
      bgLayout: '#f2f3f5',
      bgContainer: '#ffffff',
      bgElevated: '#f6f7f8',
      bgSpotlight: '#e9ecf0',
      borderPrimary: '#bcc4ce',
      borderSecondary: '#dce1e7',
      textPrimary: '#20252c',
      textSecondary: '#505a68',
      textTertiary: '#667080',
    },
  },
  studio: {
    dark: {
      bgLayout: '#17151c',
      bgContainer: '#24212b',
      bgElevated: '#302b39',
      bgSpotlight: '#393240',
      borderPrimary: '#60566d',
      borderSecondary: '#423a4c',
      textPrimary: '#f0ebf5',
      textSecondary: '#bdb2c9',
      textTertiary: '#a799b6',
    },
    light: {
      bgLayout: '#f5f2ee',
      bgContainer: '#ffffff',
      bgElevated: '#faf7f3',
      bgSpotlight: '#eee8e2',
      borderPrimary: '#cabfb4',
      borderSecondary: '#e8e0d8',
      textPrimary: '#332f39',
      textSecondary: '#655c6d',
      textTertiary: '#756b7d',
    },
  },
}

export function parseAppearance(value: string | null): Appearance {
  try {
    const parsed: unknown = JSON.parse(value ?? 'null')
    if (!parsed || typeof parsed !== 'object') return DEFAULT_APPEARANCE
    const { mode, accent, style } = parsed as Record<string, unknown>
    return {
      mode: mode === 'light' ? 'light' : 'dark',
      accent:
        typeof accent === 'string' && Object.prototype.hasOwnProperty.call(ACCENTS, accent)
          ? (accent as AccentColor)
          : 'blue',
      style: THEME_STYLES.includes(style as ThemeStyle) ? (style as ThemeStyle) : 'classic',
    }
  } catch {
    return DEFAULT_APPEARANCE
  }
}

export function readAppearance(): Appearance {
  try {
    return parseAppearance(window.localStorage.getItem(APPEARANCE_STORAGE_KEY))
  } catch {
    return DEFAULT_APPEARANCE
  }
}

export function getPalette({ mode, accent, style }: Appearance) {
  const color = ACCENTS[accent]
  return {
    ...surfaces[mode],
    ...styleSurfaces[style][mode],
    primary: color.primary,
    primaryHover: color.hover,
    primaryActive: color.active,
    primaryText: mode === 'dark' ? color.darkText : color.active,
    primarySoft: `${color.primary}${mode === 'dark' ? '26' : '12'}`,
    primaryBorder: `${color.primary}50`,
    accentViolet: mode === 'dark' ? '#b49bea' : '#6e42a0',
    accentTeal: mode === 'dark' ? '#56c9bd' : '#146e69',
    accentAmber: mode === 'dark' ? '#e7b35d' : '#875700',
    accentRose: mode === 'dark' ? '#f28eab' : '#a63867',
    statusSuccess: mode === 'dark' ? '#3fb950' : '#196c2e',
    statusWarning: mode === 'dark' ? '#d29922' : '#785200',
    statusError: mode === 'dark' ? '#ff7b72' : '#b42318',
    statusInfo: mode === 'dark' ? '#58a6ff' : '#0958b5',
    statusDefault: mode === 'dark' ? '#9ba5b1' : '#526071',
    gradientPrimary:
      style === 'classic'
        ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
        : style === 'studio'
          ? 'linear-gradient(120deg, #7560a8 0%, #a16a83 100%)'
          : `linear-gradient(${color.active}, ${color.active})`,
  }
}

export function getStyleTokens(appearance: Appearance) {
  const { style, mode } = appearance
  const colors = getPalette(appearance)
  const studio = style === 'studio'
  const classic = style === 'classic'
  const dark = mode === 'dark'
  const companion = {
    blue: studio ? '#995d76' : '#7046af',
    purple: studio ? '#866052' : '#236c8d',
    cyan: studio ? '#706544' : '#5355a0',
    green: studio ? '#766347' : '#126b77',
    rose: studio ? '#8a6148' : '#7146a5',
  }[appearance.accent]
  return {
    ...styleGeometry[style],
    panelShadow: studio
      ? dark
        ? '0 8px 28px rgba(0, 0, 0, 0.18)'
        : '0 8px 28px rgba(66, 45, 29, 0.06)'
      : 'none',
    sidebarBg: classic ? colors.bgLayout : colors.bgContainer,
    headerBg: colors.bgContainer,
    contentBg: classic ? colors.bgContainer : 'transparent',
    cardHeaderBg: studio ? colors.bgContainer : colors.bgElevated,
    cardBg: colors.bgContainer,
    statBg: classic ? colors.bgElevated : colors.bgContainer,
    inputBg: studio ? colors.bgContainer : colors.bgElevated,
    actionGradient:
      style === 'workbench'
        ? colors.primaryActive
        : `linear-gradient(115deg, ${colors.primaryActive}, ${companion})`,
    navigationGradient:
      style === 'workbench'
        ? 'none'
        : `linear-gradient(100deg, ${colors.primary}${dark ? '26' : '12'}, ${companion}${dark ? '20' : '0d'})`,
    sectionAccent: colors.primaryText,
    brandBg: style === 'workbench' ? 'transparent' : colors.gradientPrimary,
    brandText: style === 'workbench' ? colors.textPrimary : '#ffffff',
    authBg: classic && dark ? '#0b0f14' : colors.bgLayout,
    authPanelBg: classic && dark ? '#10151c' : colors.bgContainer,
    authInputBg: classic && dark ? '#171e27' : colors.bgElevated,
    authSelectedBg: classic && dark ? '#303b47' : colors.primarySoft,
    authSelectedText: classic && dark ? colors.textPrimary : colors.primaryText,
  }
}

// Used by the document and the miniature previews, so they cannot drift apart.
export function getAppearanceVariables(appearance: Appearance): Record<string, string> {
  const values = { ...getPalette(appearance), ...getStyleTokens(appearance) }
  return Object.fromEntries(
    Object.entries(values).map(([name, value]) => [
      `--tf-${name.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase()}`,
      typeof value === 'number' ? `${value}px` : value,
    ])
  )
}

export function applyAppearance(appearance: Appearance) {
  const root = document.documentElement
  root.dataset.theme = appearance.mode
  root.dataset.accent = appearance.accent
  root.dataset.style = appearance.style
  root.style.colorScheme = appearance.mode
  for (const [name, value] of Object.entries(getAppearanceVariables(appearance))) {
    root.style.setProperty(name, value)
  }
}
