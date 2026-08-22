import { useTranslation } from 'react-i18next'

export interface LossRow {
  step: number
  train: number | null
  eval: number | null
}

export type LossMetric = Record<string, unknown>

export interface LossChartLabels {
  stepAxis: string
  trainSeries: string
  evalSeries: string
  tooltipStep: string
}

export interface LossSeriesModel {
  name: string
  data: Array<[number, number]>
  symbol: 'circle' | 'none'
  symbolSize: number
  showSymbol: boolean
  color: string
}

export interface LossChartModel {
  renderable: boolean
  legend: string[]
  xAxis: {
    name: string
    formatLabel(value: number): string
  }
  yAxis: {
    min?: number
    max?: number
    interval?: number
    formatLabel(value: number): string
  }
  series: LossSeriesModel[]
  formatTooltip(params: unknown): string
}

interface LossCurveDataTableProps {
  rows: LossRow[]
}

const headerCellStyle = {
  borderBottom: '1px solid rgba(255, 255, 255, 0.16)',
  padding: '8px 12px',
  textAlign: 'left',
} as const

const dataCellStyle = {
  borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
  padding: '8px 12px',
  textAlign: 'left',
} as const

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || typeof value === 'boolean') return null
  if (typeof value !== 'number' && typeof value !== 'string') return null
  if (typeof value === 'string' && value.trim() === '') return null

  const numericValue = Number(value)
  return Number.isFinite(numericValue) ? numericValue : null
}

function firstFinite(record: LossMetric, keys: string[]): number | null {
  for (const key of keys) {
    const value = finiteNumber(record[key])
    if (value !== null) return value
  }
  return null
}

function hasSafeFiniteExtent(values: readonly number[]) {
  if (values.length === 0) return true

  let min = values[0]
  let max = values[0]
  if (!Number.isFinite(min)) return false
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index]
    if (!Number.isFinite(value)) return false
    min = Math.min(min, value)
    max = Math.max(max, value)
  }

  return Number.isFinite(max - min)
}

export function normalizeLossHistory(history: readonly LossMetric[] | undefined): LossRow[] {
  const rowsByStep = new Map<number, LossRow>()

  for (const record of history ?? []) {
    const step = firstFinite(record, ['step', 'global_step', 'current_step'])
    if (step === null) continue

    const train = firstFinite(record, ['train_loss', 'loss'])
    let evaluation = firstFinite(record, ['eval_loss'])
    if (evaluation === null) {
      const legacyEvalKeys = Object.keys(record)
        .filter((key) => key !== 'eval_loss' && key.startsWith('eval_') && key.endsWith('_loss'))
        .sort()
      evaluation = firstFinite(record, legacyEvalKeys)
    }

    const row = rowsByStep.get(step) ?? { step, train: null, eval: null }
    if (train !== null) row.train = train
    if (evaluation !== null) row.eval = evaluation
    rowsByStep.set(step, row)
  }

  return [...rowsByStep.values()].sort((left, right) => left.step - right.step)
}

interface CalculatedAxis {
  min?: number
  max?: number
  interval?: number
  formatInterval: number
}

function representableSpacing(scale: number) {
  if (scale === 0) return Number.MIN_VALUE

  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, scale)
  const bits = view.getBigUint64(0)
  if (scale === Number.MAX_VALUE) {
    view.setBigUint64(0, bits - 1n)
    return scale - view.getFloat64(0)
  }

  view.setBigUint64(0, bits + 1n)
  return view.getFloat64(0) - scale
}

function calculateAxis(lossValues: number[]): CalculatedAxis {
  const finiteValues = lossValues.filter(Number.isFinite)
  if (finiteValues.length === 0) {
    return { min: 0, max: 1, interval: 0.2, formatInterval: 0.2 }
  }

  let minLoss = finiteValues[0]
  let maxLoss = finiteValues[0]
  for (let index = 1; index < finiteValues.length; index += 1) {
    minLoss = Math.min(minLoss, finiteValues[index])
    maxLoss = Math.max(maxLoss, finiteValues[index])
  }

  const rawRange = maxLoss - minLoss
  const scale = Math.max(Math.abs(minLoss), Math.abs(maxLoss))
  const scaleSpacing = representableSpacing(scale)
  if (!Number.isFinite(rawRange)) return { formatInterval: scaleSpacing }

  const fallbackRange = scale === 0 ? 1 : Math.max(scale / 5, Number.MIN_VALUE)
  const effectiveRange = rawRange > 0 ? rawRange : fallbackRange
  const roughInterval = Math.max(effectiveRange / 4, Number.MIN_VALUE)
  const exponent = Math.floor(Math.log10(roughInterval))
  const candidateMagnitude = 10 ** exponent
  const magnitude =
    Number.isFinite(candidateMagnitude) && candidateMagnitude > 0
      ? candidateMagnitude
      : roughInterval
  const normalized = roughInterval / magnitude
  const niceFactor = normalized <= 1.5 ? 1 : normalized <= 3.5 ? 2 : normalized <= 7.5 ? 5 : 10
  const candidateInterval = niceFactor * magnitude
  const interval = Math.max(
    Number.isFinite(candidateInterval) && candidateInterval > 0 ? candidateInterval : roughInterval,
    scaleSpacing
  )

  const subtractFinite = (value: number, amount: number) => {
    const result = value - amount
    return Number.isFinite(result) ? result : -Number.MAX_VALUE
  }
  const addFinite = (value: number, amount: number) => {
    const result = value + amount
    return Number.isFinite(result) ? result : Number.MAX_VALUE
  }

  let min = subtractFinite(minLoss, interval)
  if (minLoss >= 0) min = Math.max(0, min)
  let max = addFinite(maxLoss, interval)

  if (!(min < max)) {
    if (maxLoss > 0) {
      min = Math.max(0, maxLoss / 2)
      max = maxLoss
    } else if (minLoss < 0) {
      min = minLoss
      max = minLoss / 2
    } else {
      min = 0
      max = 1
    }
  }

  const extent = max - min
  if (
    !Number.isFinite(min) ||
    !Number.isFinite(max) ||
    !Number.isFinite(interval) ||
    !Number.isFinite(extent) ||
    interval <= 0 ||
    !(min < max) ||
    !(min + interval > min) ||
    !(max - interval < max)
  ) {
    return { formatInterval: interval }
  }

  return { min, max, interval, formatInterval: interval }
}

function formatAxisValue(value: number, interval: number) {
  if (!Number.isFinite(value)) return ''
  if (value === 0) return '0'

  const absoluteValue = Math.abs(value)
  const absoluteInterval = Math.abs(interval)
  if (
    absoluteInterval < 1e-4 ||
    absoluteInterval >= 1e6 ||
    absoluteValue < 1e-4 ||
    absoluteValue >= 1e6
  ) {
    const relativeMagnitude = Math.log10(absoluteValue) - Math.log10(absoluteInterval)
    const fractionDigits = Number.isFinite(relativeMagnitude)
      ? Math.min(16, Math.max(0, Math.ceil(relativeMagnitude)))
      : 16
    return value
      .toExponential(fractionDigits)
      .replace(/\.0+(?=e)/, '')
      .replace(/(\.\d*?)0+(?=e)/, '$1')
  }

  const decimals = Math.min(12, Math.max(0, -Math.floor(Math.log10(absoluteInterval))))
  return value.toFixed(decimals)
}

export function createLossChartModel(
  rows: readonly LossRow[],
  labels: LossChartLabels
): LossChartModel {
  const trainPoints: Array<[number, number]> = []
  const evalPoints: Array<[number, number]> = []
  for (const row of rows) {
    if (row.train !== null) trainPoints.push([row.step, row.train])
    if (row.eval !== null) evalPoints.push([row.step, row.eval])
  }

  const dataPointCount = Math.max(trainPoints.length, evalPoints.length)
  const showTrainSymbol = dataPointCount <= 50
  const symbolSize = dataPointCount <= 5 ? 8 : dataPointCount <= 20 ? 6 : 4
  const series: LossSeriesModel[] = []

  if (trainPoints.length > 0) {
    series.push({
      name: labels.trainSeries,
      data: trainPoints,
      symbol: showTrainSymbol ? 'circle' : 'none',
      symbolSize,
      showSymbol: showTrainSymbol,
      color: '#1890ff',
    })
  }
  if (evalPoints.length > 0) {
    series.push({
      name: labels.evalSeries,
      data: evalPoints,
      symbol: 'circle',
      symbolSize,
      showSymbol: true,
      color: '#52c41a',
    })
  }

  const lossValues = rows.flatMap((row) =>
    [row.train, row.eval].filter((value): value is number => value !== null)
  )
  const stepValues = [...trainPoints, ...evalPoints].map(([step]) => step)
  const axis = calculateAxis(lossValues)

  return {
    renderable:
      series.length > 0 && hasSafeFiniteExtent(stepValues) && hasSafeFiniteExtent(lossValues),
    legend: series.map((item) => item.name),
    xAxis: {
      name: labels.stepAxis,
      formatLabel: (value) => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value)),
    },
    yAxis: {
      min: axis.min,
      max: axis.max,
      interval: axis.interval,
      formatLabel: (value) => formatAxisValue(value, axis.formatInterval),
    },
    series,
    formatTooltip: (params) => {
      const items = params as Array<{
        name?: string
        seriesName: string
        value: number | [number, number]
      }>
      if (!Array.isArray(items) || items.length === 0) return ''
      const firstValue = items[0].value
      const step = Array.isArray(firstValue) ? firstValue[0] : items[0].name
      const lines = [`${labels.tooltipStep}: ${String(step)}`]
      for (const item of items) {
        const value = Array.isArray(item.value) ? item.value[1] : item.value
        if (typeof value === 'number' && Number.isFinite(value)) {
          lines.push(`${item.seriesName}: ${String(value)}`)
        }
      }
      return lines.join('<br/>')
    },
  }
}

function displayValue(value: number | null, missing: string) {
  return value === null ? missing : String(value)
}

export function LossCurveDataTable({ rows }: LossCurveDataTableProps) {
  const { t } = useTranslation('training')
  const missing = t('detail.lossCurve.dataTable.missing')

  return (
    <details style={{ marginTop: 12 }}>
      <summary style={{ cursor: 'pointer' }}>{t('detail.lossCurve.dataTable.toggle')}</summary>
      <div style={{ marginTop: 8, overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <caption style={{ padding: '4px 0 8px', textAlign: 'left' }}>
            {t('detail.lossCurve.dataTable.caption')}
          </caption>
          <thead>
            <tr>
              <th scope="col" style={headerCellStyle}>
                {t('detail.lossCurve.dataTable.step')}
              </th>
              <th scope="col" style={headerCellStyle}>
                {t('detail.lossCurve.dataTable.train')}
              </th>
              <th scope="col" style={headerCellStyle}>
                {t('detail.lossCurve.dataTable.eval')}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.step}>
                <th scope="row" style={dataCellStyle}>
                  {String(row.step)}
                </th>
                <td style={dataCellStyle}>{displayValue(row.train, missing)}</td>
                <td style={dataCellStyle}>{displayValue(row.eval, missing)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}
