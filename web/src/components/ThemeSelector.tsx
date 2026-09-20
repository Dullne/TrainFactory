import { useEffect, useId, useRef, useState, type CSSProperties } from 'react'
import { Button, Popover } from 'antd'
import { BgColorsOutlined, CheckOutlined, MoonOutlined, SunOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import {
  ACCENTS,
  THEME_STYLES,
  getAppearanceVariables,
  type AccentColor,
  type ColorMode,
} from '@/theme/appearance'
import { useAppearance } from '@/theme/ThemeProvider'
import './ThemeSelector.css'

export function ThemeSelector() {
  const { t } = useTranslation('common')
  const { mode, accent, style, setAppearance } = useAppearance()
  const [open, setOpen] = useState(false)
  const id = useId()
  const trigger = useRef<HTMLButtonElement>(null)
  const content = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false)
        trigger.current?.focus()
      }
    }
    document.addEventListener('keydown', closeOnEscape)
    return () => document.removeEventListener('keydown', closeOnEscape)
  }, [open])

  return (
    <Popover
      trigger="click"
      placement="bottom"
      open={open}
      onOpenChange={setOpen}
      afterOpenChange={(visible) => {
        if (visible) content.current?.querySelector<HTMLInputElement>('input:checked')?.focus()
      }}
      content={
        <div
          ref={content}
          id={id}
          className="theme-picker"
          role="dialog"
          aria-label={t('appearance.title')}
        >
          <div className="theme-picker-heading">{t('appearance.title')}</div>
          <fieldset className="theme-picker-section">
            <legend>{t('appearance.style')}</legend>
            <div className="theme-style-options">
              {THEME_STYLES.map((value) => (
                <label key={value} className="theme-option theme-style-option">
                  <input
                    type="radio"
                    name={`${id}-style`}
                    value={value}
                    checked={style === value}
                    aria-label={t(`appearance.styles.${value}.name`)}
                    aria-describedby={`${id}-${value}-description`}
                    onChange={() => setAppearance({ mode, accent, style: value })}
                  />
                  <span className="theme-style-card">
                    <span
                      className="theme-style-preview"
                      data-style={value}
                      style={
                        getAppearanceVariables({ mode, accent, style: value }) as CSSProperties
                      }
                      aria-hidden="true"
                    >
                      <span className="theme-preview-sidebar">
                        <span className="theme-preview-brand" />
                        <span className="theme-preview-nav is-active" />
                        <span className="theme-preview-nav" />
                        <span className="theme-preview-nav" />
                      </span>
                      <span className="theme-preview-workspace">
                        <span className="theme-preview-header">
                          <span />
                          <span />
                        </span>
                        <span className="theme-preview-content">
                          <span className="theme-preview-panel">
                            <span className="theme-preview-panel-header" />
                            <span className="theme-preview-field" />
                            <span className="theme-preview-field is-short" />
                            <span className="theme-preview-action" />
                          </span>
                        </span>
                      </span>
                    </span>
                    <span className="theme-style-name">
                      <span>{t(`appearance.styles.${value}.name`)}</span>
                      <span className="theme-style-check" aria-hidden="true">
                        {style === value ? <CheckOutlined /> : null}
                      </span>
                    </span>
                    <span id={`${id}-${value}-description`} className="theme-style-description">
                      {t(`appearance.styles.${value}.description`)}
                    </span>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className="theme-picker-section">
            <legend>{t('appearance.mode')}</legend>
            <div className="theme-mode-options">
              {(['dark', 'light'] as ColorMode[]).map((value) => (
                <label key={value} className="theme-option">
                  <input
                    type="radio"
                    name={`${id}-mode`}
                    value={value}
                    checked={mode === value}
                    onChange={() => setAppearance({ mode: value, accent, style })}
                  />
                  <span className="theme-mode-label">
                    {value === 'dark' ? <MoonOutlined aria-hidden /> : <SunOutlined aria-hidden />}
                    {t(`appearance.${value}`)}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <details className="theme-picker-advanced">
            <summary>{t('appearance.advanced')}</summary>
            <fieldset className="theme-picker-section">
              <legend>{t('appearance.accent')}</legend>
              <div className="theme-accent-options">
                {(Object.keys(ACCENTS) as AccentColor[]).map((value) => (
                  <label key={value} className="theme-option">
                    <input
                      type="radio"
                      name={`${id}-accent`}
                      value={value}
                      checked={accent === value}
                      onChange={() => setAppearance({ mode, accent: value, style })}
                    />
                    <span className="theme-accent-label">
                      <span className="theme-swatch" style={{ background: ACCENTS[value].primary }}>
                        {accent === value ? <CheckOutlined aria-hidden /> : null}
                      </span>
                      {t(`appearance.${value}`)}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          </details>
          <p className="theme-picker-hint">{t('appearance.hint')}</p>
        </div>
      }
    >
      <Button
        ref={trigger}
        type="text"
        className="theme-selector-trigger"
        icon={<BgColorsOutlined />}
        aria-label={t('appearance.title')}
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-haspopup="dialog"
      >
        <span className="theme-selector-text">{t('appearance.title')}</span>
      </Button>
    </Popover>
  )
}
