import { theme, type ThemeConfig } from 'antd'
import { getPalette, getStyleTokens, type Appearance } from './appearance'

export function createTheme(appearance: Appearance): ThemeConfig {
  const colors = getPalette(appearance)
  const style = getStyleTokens(appearance)
  return {
    algorithm: appearance.mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: colors.primary,
      colorLink: colors.primaryText,
      colorLinkHover: colors.primaryHover,
      colorLinkActive: colors.primaryActive,
      colorBgLayout: colors.bgLayout,
      colorBgContainer: colors.bgContainer,
      colorBgElevated: colors.bgElevated,
      colorBgSpotlight: colors.bgSpotlight,
      colorText: colors.textPrimary,
      colorTextSecondary: colors.textSecondary,
      colorTextTertiary: colors.textTertiary,
      colorTextDisabled: colors.textDisabled,
      colorTextPlaceholder: colors.textTertiary,
      colorBorder: colors.borderPrimary,
      colorBorderSecondary: colors.borderSecondary,
      colorSuccess: colors.statusSuccess,
      colorWarning: colors.statusWarning,
      colorError: colors.statusError,
      colorInfo: colors.statusInfo,
      borderRadius: style.radius,
      borderRadiusLG: style.radiusLG,
      borderRadiusSM: style.radiusSM,
      controlHeight: style.controlHeight,
      fontFamily:
        '-apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Roboto, Arial, sans-serif',
    },
    components: {
      Layout: { siderBg: style.sidebarBg, headerBg: style.headerBg, bodyBg: colors.bgLayout },
      Menu: {
        itemBorderRadius: style.radius,
        itemHeight: style.menuItemHeight,
        itemMarginBlock: style.menuItemGap,
        itemBg: 'transparent',
        subMenuItemBg: 'transparent',
        itemColor: colors.textSecondary,
        itemSelectedColor: colors.primaryText,
        itemSelectedBg: colors.primarySoft,
        itemHoverBg: colors.hoverBg,
        darkItemBg: 'transparent',
        darkSubMenuItemBg: 'transparent',
        darkItemColor: colors.textSecondary,
        darkItemSelectedColor: colors.primaryText,
        darkItemSelectedBg: colors.primarySoft,
        darkItemHoverBg: colors.hoverBg,
      },
      Table: {
        cellPaddingBlock: style.tablePadding,
        cellPaddingBlockSM: style.tablePaddingSM,
        headerBg: colors.bgElevated,
        headerColor: colors.textPrimary,
        rowHoverBg: colors.hoverBg,
        borderColor: colors.borderSecondary,
      },
      Card: {
        borderRadiusLG: style.radiusLG,
        bodyPadding: style.cardPadding,
        bodyPaddingSM: style.cardPaddingSM,
        headerBg: style.cardHeaderBg,
      },
      Form: { itemMarginBottom: style.formGap },
      Modal: {
        contentBg: colors.bgContainer,
        headerBg: colors.bgContainer,
        footerBg: colors.bgContainer,
      },
      Select: {
        optionSelectedBg: colors.primarySoft,
        optionSelectedColor: colors.textPrimary,
        colorBgContainer: style.inputBg,
      },
      Input: {
        colorBgContainer: style.inputBg,
        hoverBorderColor: colors.primaryHover,
        activeBorderColor: colors.primary,
      },
      InputNumber: { colorBgContainer: style.inputBg },
      DatePicker: { colorBgContainer: style.inputBg },
      Button: { primaryShadow: 'none', defaultShadow: 'none' },
      Tag: { defaultBg: colors.bgElevated, defaultColor: colors.textSecondary },
    },
  }
}

export * from './colors'
