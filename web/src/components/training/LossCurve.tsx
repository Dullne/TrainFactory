import { useEffect, useMemo, useRef } from 'react'
import { Card, Empty, Spin } from 'antd'
import { useTranslation } from 'react-i18next'
import type { LineSeriesOption } from 'echarts/charts'
import type {
  AriaComponentOption,
  GridComponentOption,
  LegendComponentOption,
  TooltipComponentOption,
} from 'echarts/components'
import type { TrainingMetricsResponse } from '@/services/api'
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
}

type LossChartOption = ComposeOption<
  | LineSeriesOption
  | AriaComponentOption
  | GridComponentOption
  | LegendComponentOption
  | TooltipComponentOption
>

export function LossCurve({ metrics, loading = false, height = 300 }: LossCurveProps) {
  const { t, i18n } = useTranslation('training')
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartInstanceRef = useRef<EChartsType | null>(null)
  const chartData = useMemo(
    () => normalizeLossHistory(metrics?.loss_history),
    [metrics?.loss_history]
  )
  const chartModel = useMemo(
    () =>
      createLossChartModel(chartData, {
        stepAxis: t('detail.lossCurve.axis.step'),
        trainSeries: t('detail.lossCurve.series.train'),
        evalSeries: t('detail.lossCurve.series.eval'),
        tooltipStep: t('detail.lossCurve.tooltip.step'),
      }),
    [chartData, t]
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
      aria: {
        enabled: true,
        description: ariaDescription,
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: chartModel.formatTooltip,
      },
      legend: {
        data: chartModel.legend,
        top: 4,
        right: 0,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: 36,
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: chartModel.xAxis.name,
        nameLocation: 'end',
        minInterval: 1,
        splitLine: { show: false },
        axisLabel: {
          formatter: chartModel.xAxis.formatLabel,
        },
      },
      yAxis: {
        type: 'value',
        ...(chartModel.yAxis.min === undefined ? {} : { min: chartModel.yAxis.min }),
        ...(chartModel.yAxis.max === undefined ? {} : { max: chartModel.yAxis.max }),
        ...(chartModel.yAxis.interval === undefined ? {} : { interval: chartModel.yAxis.interval }),
        axisLabel: {
          formatter: chartModel.yAxis.formatLabel,
        },
      },
      series,
    }

    if (chartInstanceRef.current === chart && !chart.isDisposed()) {
      chart.setOption(option, true)
    }
  }, [chartData, chartModel, hasRenderableChart, i18n.resolvedLanguage, t])

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
    return (
      <Card title={title} size="small">
        <Empty
          description={t('detail.lossCurve.empty')}
          style={{ height, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}
        />
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
