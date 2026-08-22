import { useState, useEffect } from 'react'
import { Select, Tag, Space, Spin } from 'antd'
import { useTranslation } from 'react-i18next'
import { resourceApi } from '@/services/api'

interface GpuInfo {
  id: number
  name: string
  memory_total_gb: number
  memory_used_gb: number
  memory_free_gb: number
  memory_usage_percent: number
  gpu_utilization: number | null
  temperature: number | null
  power_usage_w: number | null
  is_allocated: boolean
  allocated_task: string | null
}

interface GpuSelectProps {
  value?: number[]
  onChange?: (value: number[]) => void
  mode?: 'multiple' | 'single'
  placeholder?: string
  allowClear?: boolean
}

export function GpuSelect({ value, onChange, mode = 'multiple', placeholder, allowClear }: GpuSelectProps) {
  const { t } = useTranslation('common')
  const [gpus, setGpus] = useState<GpuInfo[]>([])
  const [loading, setLoading] = useState(false)

  const fetchGpus = async () => {
    setLoading(true)
    try {
      const res = await resourceApi.getGpuList()
      setGpus(res.gpus || [])
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchGpus()
  }, [])

  const options = gpus.map((gpu) => ({
    value: gpu.id,
    label: (
      <Space size={8} style={{ width: '100%', justifyContent: 'space-between' }}>
        <span>
          GPU {gpu.id}
          <Tag
            color={gpu.is_allocated ? 'orange' : 'green'}
            style={{ marginLeft: 8, fontSize: 10 }}
          >
            {gpu.is_allocated ? t('gpu.occupied') : t('gpu.free')}
          </Tag>
        </span>
        <span style={{ color: '#8b949e', fontSize: 12 }}>
          {t('gpu.memoryInfo', { free: gpu.memory_free_gb.toFixed(0), total: gpu.memory_total_gb.toFixed(0) })}
        </span>
      </Space>
    ),
    disabled: gpu.is_allocated,
    gpu,
  }))

  if (loading && gpus.length === 0) {
    return <Spin size="small" />
  }

  return (
    <Select
      mode={mode === 'multiple' ? 'multiple' : undefined}
      value={value}
      onChange={onChange}
      placeholder={placeholder || t('gpu.selectGpuWithCount', { free: gpus.filter(g => !g.is_allocated).length, total: gpus.length })}
      options={options}
      optionFilterProp="label"
      style={{ width: '100%' }}
      dropdownStyle={{ minWidth: 350 }}
      onDropdownVisibleChange={(open) => { if (open) fetchGpus() }}
      allowClear={allowClear}
    />
  )
}
