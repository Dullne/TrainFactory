import type { ThemeConfig } from 'antd'
import * as colors from './colors'

// 深色主题配置
export const darkTheme: ThemeConfig = {
  token: {
    // 主色
    colorPrimary: colors.PRIMARY,
    colorLink: colors.PRIMARY,
    colorLinkHover: colors.PRIMARY_HOVER,
    colorLinkActive: colors.PRIMARY_ACTIVE,

    // 背景色
    colorBgLayout: colors.BG_LAYOUT,
    colorBgContainer: colors.BG_CONTAINER,
    colorBgElevated: colors.BG_ELEVATED,
    colorBgSpotlight: colors.BG_SPOTLIGHT,

    // 文字色
    colorText: colors.TEXT_PRIMARY,
    colorTextSecondary: colors.TEXT_SECONDARY,
    colorTextTertiary: colors.TEXT_TERTIARY,
    colorTextDisabled: colors.TEXT_DISABLED,

    // 边框色
    colorBorder: colors.BORDER_PRIMARY,
    colorBorderSecondary: colors.BORDER_SECONDARY,

    // 状态色
    colorSuccess: colors.STATUS_SUCCESS,
    colorWarning: colors.STATUS_WARNING,
    colorError: colors.STATUS_ERROR,
    colorInfo: colors.STATUS_INFO,

    // 圆角
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,

    // 字体
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  components: {
    Layout: {
      siderBg: colors.BG_LAYOUT,
      headerBg: colors.BG_CONTAINER,
      bodyBg: colors.BG_LAYOUT,
      triggerBg: colors.BG_ELEVATED,
    },
    Menu: {
      darkItemBg: 'transparent',
      darkSubMenuItemBg: 'transparent',
      darkItemSelectedBg: 'rgba(22, 119, 255, 0.15)',
      darkItemHoverBg: 'rgba(255, 255, 255, 0.08)',
      darkItemColor: colors.TEXT_SECONDARY,
      darkItemSelectedColor: colors.PRIMARY,
    },
    Table: {
      headerBg: colors.BG_ELEVATED,
      headerColor: colors.TEXT_PRIMARY,
      rowHoverBg: 'rgba(255, 255, 255, 0.04)',
      borderColor: colors.BORDER_SECONDARY,
    },
    Card: {
      colorBgContainer: colors.BG_CONTAINER,
      colorBorderSecondary: colors.BORDER_SECONDARY,
    },
    Modal: {
      contentBg: colors.BG_CONTAINER,
      headerBg: colors.BG_CONTAINER,
      footerBg: colors.BG_CONTAINER,
    },
    Select: {
      optionSelectedBg: 'rgba(22, 119, 255, 0.15)',
      colorBgContainer: colors.BG_ELEVATED,
      colorText: colors.TEXT_PRIMARY,
      colorTextPlaceholder: colors.TEXT_TERTIARY,
      colorBgElevated: colors.BG_ELEVATED,
      optionSelectedColor: colors.TEXT_PRIMARY,
    },
    Input: {
      colorBgContainer: colors.BG_ELEVATED,
      hoverBorderColor: colors.PRIMARY,
      activeBorderColor: colors.PRIMARY,
    },
    Button: {
      primaryShadow: 'none',
      defaultShadow: 'none',
    },
    Tag: {
      defaultBg: colors.BG_ELEVATED,
      defaultColor: colors.TEXT_SECONDARY,
    },
    Tooltip: {
      colorBgSpotlight: colors.BG_ELEVATED,
    },
    Alert: {
      colorInfoBg: 'rgba(22, 119, 255, 0.08)',
      colorInfoBorder: 'rgba(22, 119, 255, 0.25)',
      colorSuccessBg: 'rgba(63, 185, 80, 0.08)',
      colorSuccessBorder: 'rgba(63, 185, 80, 0.25)',
      colorWarningBg: 'rgba(210, 153, 34, 0.08)',
      colorWarningBorder: 'rgba(210, 153, 34, 0.25)',
      colorErrorBg: 'rgba(248, 81, 73, 0.08)',
      colorErrorBorder: 'rgba(248, 81, 73, 0.25)',
    },
  },
  algorithm: undefined,
}

// 亮色主题配置（保留作为备选）
export const lightTheme: ThemeConfig = {
  token: {
    colorPrimary: colors.PRIMARY,
    borderRadius: 8,
    borderRadiusLG: 12,
    borderRadiusSM: 6,
  },
}

// 导出颜色常量供其他组件使用
export * from './colors'
