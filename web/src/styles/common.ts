import type { CSSProperties } from 'react'
import { BG_ELEVATED, BORDER_SECONDARY } from '@/theme'

/**
 * 通用样式常量
 */

// Flex 布局
export const flexCenter: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}

export const flexBetween: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
}

export const flexStart: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'flex-start',
}

export const flexColumn: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
}

// 图标容器
export const iconContainer = (
  size: number = 40,
  borderRadius: CSSProperties['borderRadius'] = 'var(--tf-radius)'
): CSSProperties => ({
  width: size,
  height: size,
  borderRadius,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
})

export const coloredIconContainer = (
  color: string,
  size: number = 40,
  borderRadius: CSSProperties['borderRadius'] = 'var(--tf-radius)'
): CSSProperties => ({
  ...iconContainer(size, borderRadius),
  backgroundColor: `color-mix(in srgb, ${color} 12%, transparent)`,
  color,
})

// 卡片样式
export const cardStyle: CSSProperties = {
  backgroundColor: BG_ELEVATED,
  borderColor: BORDER_SECONDARY,
  borderRadius: 'var(--tf-radius-lg)',
}

export const cardBodyStyle: CSSProperties = {
  padding: 'var(--tf-card-padding-sm)',
}

// 文本省略
export const textEllipsis: CSSProperties = {
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
}

// 多行文本省略
export const multiLineEllipsis = (lines: number = 2): CSSProperties => ({
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  display: '-webkit-box',
  WebkitLineClamp: lines,
  WebkitBoxOrient: 'vertical',
})

// 间距
export const gap = (size: number): CSSProperties => ({
  gap: size,
})

// 工具栏样式
export const toolbarStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: 16,
}

// 页面内容间距
export const sectionMargin: CSSProperties = {
  marginBottom: 20,
}

// 全屏居中（用于加载状态）
export const fullCenterStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: '100%',
  minHeight: 300,
}

// 加载容器
export const loadingContainerStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: 400,
}

// 空状态容器
export const emptyContainerStyle: CSSProperties = {
  textAlign: 'center',
  padding: 40,
}

/**
 * 合并样式的工具函数
 */
export function mergeStyles(...styles: (CSSProperties | undefined)[]): CSSProperties {
  return Object.assign({}, ...styles.filter(Boolean))
}

/**
 * 带 gap 的 flex 布局
 */
export const flexWithGap = (
  gapSize: number,
  direction: 'row' | 'column' = 'row'
): CSSProperties => ({
  display: 'flex',
  flexDirection: direction,
  alignItems: direction === 'row' ? 'center' : 'stretch',
  gap: gapSize,
})
