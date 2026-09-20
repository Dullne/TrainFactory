import { useEffect, useMemo, useRef } from 'react'
import { Button, Card, Empty, Spin, Typography, theme } from 'antd'
import { useTranslation } from 'react-i18next'
import type { LineSeriesOption } from 'echarts/charts'
import type {
  AriaComponentOption,
  GridComponentOption,
  LegendComponentOption,
  TooltipComponentOption,
} from 'echarts/components'
import type { TrainingMetricsResponse } from '@/services/api'
import { getPalette } from '@/theme/appearance'
import { useAppearance } from '@/theme/ThemeProvider'
import { init, type ComposeOption, type EChartsType } from './echarts'
import {
  createLossChartModel,
  LossCurveDataTable,
  normalizeLossHistory,
} from './LossCurveDataTable'

interface LossCurveProps {
  metrics: TrainingMetricsResponse | null
  loading?: boolean
  height?: number
  taskStatus?: string
  failed?: boolean
  onRetry?: () => void
}

type LossChartOption = ComposeOption<
  | LineSeriesOption
  | AriaComponentOption
  | GridComponentOption
  | LegendComponentOption
  | TooltipComponentOption
>

export function LossCurve({
  metrics,
  loading = false,
  height = 300,
  taskStatus,
  failed = false,
  onRetry,
}: LossCurveProps) {
  const { t, i18n } = useTranslation('training')
  const { t: extra } = useTranslation('trainingDetailExtras')
  const colors = getPalette(useAppearance())
  const {
    token: { colorText, colorTextSecondary, colorBorder, colorBorderSecondary, colorBgElevated },
  } = theme.useToken()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartInstanceRef = useRef<EChartsType | null>(null)
  const chartData = useMemo(
    () => normalizeLossHistory(metrics?.loss_history),
    [metrics?.loss_history]
  )
  const chartModel = useMemo(
    () =>
      createLossChartModel(
        chartData,
        {
          stepAxis: t('detail.lossCurve.axis.step'),
          trainSeries: t('detail.lossCurve.series.train'),
          evalSeries: t('detail.lossCurve.series.eval'),
          tooltipStep: t('detail.lossCurve.tooltip.step'),
        },
        { train: colors.statusInfo, eval: colors.statusSuccess }
      ),
    [chartData, colors.statusInfo, colors.statusSuccess, t]
  )
  const hasLossData = chartData.some((row) => row.train !== null || row.eval !== null)
  const hasRenderableChart = !loading && hasLossData && chartModel.renderable

  useEffect(() => {
    if (!hasRenderableChart || !containerRef.current) return

    const container = containerRef.current
    const chart = init(container)
    chartInstanceRef.current = chart
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(container)

    return () => {
      observer.disconnect()
      if (chartInstanceRef.current === chart) chartInstanceRef.current = null
      chart.dispose()
    }
  }, [hasRenderableChart])

  useEffect(() => {
    if (!hasRenderableChart) return
    const chart = chartInstanceRef.current
    if (!chart || chart.isDisposed()) return

    const series: LineSeriesOption[] = chartModel.series.map((item) => ({
      name: item.name,
      type: 'line',
      data: item.data,
      smooth: true,
      symbol: item.symbol,
      symbolSize: item.symbolSize,
      lineStyle: { width: 2 },
      itemStyle: { color: item.color },
      showSymbol: item.showSymbol,
    }))

    const ariaDescription = t('detail.lossCurve.ariaDescription')
    const option: LossChartOption = {
      textStyle: { color: colorTextSecondary },
      aria: {
        enabled: true,
        description: ariaDescription,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: colorBgElevated,
        borderColor: colorBorder,
        textStyle: { color: colorText },
        axisPointer: {
          type: 'cross',
          lineStyle: { color: colorTextSecondary },
          crossStyle: { color: colorTextSecondary },
          label: {
            backgroundColor: colorBgElevated,
            borderColor: colorBorder,
            color: colorText,
          },
        },
        formatter: chartModel.formatTooltip,
      },
      legend: {
        data: chartModel.legend,
        top: 4,
        right: 0,
        textStyle: { color: colorTextSecondary },
      },
      grid: {
        left: '3%',
        right: '4%',
        // containLabel reserves tick labels, but not the axis title below them.
        bottom: 32,
        top: 36,
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: chartModel.xAxis.name,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: colorTextSecondary },
        minInterval: 1,
        axisLine: { lineStyle: { color: colorBorder } },
        axisTick: { lineStyle: { color: colorBorder } },
        splitLine: { show: false },
        axisLabel: {
          color: colorTextSecondary,
          formatter: chartModel.xAxis.formatLabel,
        },
      },
      yAxis: {
        type: 'value',
        ...(chartModel.yAxis.min === undefined ? {} : { min: chartModel.yAxis.min }),
        ...(chartModel.yAxis.max === undefined ? {} : { max: chartModel.yAxis.max }),
        ...(chartModel.yAxis.interval === undefined ? {} : { interval: chartModel.yAxis.interval }),
        axisLine: { lineStyle: { color: colorBorder } },
        axisTick: { lineStyle: { color: colorBorder } },
        splitLine: { lineStyle: { color: colorBorderSecondary } },
        axisLabel: {
          color: colorTextSecondary,
          formatter: chartModel.yAxis.formatLabel,
        },
      },
      series,
    }

    if (chartInstanceRef.current === chart && !chart.isDisposed()) {
      chart.setOption(option, true)
    }
  }, [
    chartData,
    chartModel,
    colorBgElevated,
    colorBorder,
    colorBorderSecondary,
    colorText,
    colorTextSecondary,
    hasRenderableChart,
    i18n.resolvedLanguage,
    t,
  ])

  const title = t('detail.lossCurve.title')

  if (loading) {
    return (
      <Card title={title} size="small">
        <div
          role="status"
          style={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 10,
          }}
        >
          <Spin size="small" />
          <span>{t('detail.lossCurve.loading')}</span>
        </div>
      </Card>
    )
  }

  if (!hasLossData) {
    const reason = failed
      ? 'unavailable'
      : taskStatus === 'pending' || taskStatus === 'preparing'
        ? 'pending'
        : taskStatus === 'running' || taskStatus === 'evaluating'
          ? 'active'
          : taskStatus === 'failed'
            ? 'failed'
            : taskStatus
              ? 'finished'
              : null
    return (
      <Card title={title} size="small">
        <Empty
          description={
            <>
              {!failed && <div>{t('detail.lossCurve.empty')}</div>}
              {reason && (
                <Typography.Paragraph type="secondary" style={{ margin: '8px 0 0' }}>
                  {extra(`loss.${reason}`)}
                </Typography.Paragraph>
              )}
            </>
          }
          style={{
            minHeight: height,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
          }}
        >
          {failed && onRetry && <Button onClick={onRetry}>{extra('loss.retry')}</Button>}
        </Empty>
      </Card>
    )
  }

  if (!chartModel.renderable) {
    return (
      <Card title={title} size="small">
        <div
          role="note"
          style={{
            marginBottom: 12,
            padding: '10px 12px',
            borderRadius: 6,
            background: 'rgba(250, 173, 20, 0.12)',
          }}
        >
          {t('detail.lossCurve.rangeUnsupported')}
        </div>
        <LossCurveDataTable rows={chartData} />
      </Card>
    )
  }

  const ariaDescription = t('detail.lossCurve.ariaDescription')
  return (
    <Card title={title} size="small">
      <div
        ref={containerRef}
        role="img"
        aria-label={ariaDescription}
        style={{ height, width: '100%' }}
      />
      <LossCurveDataTable rows={chartData} />
    </Card>
  )
}

export default LossCurve
