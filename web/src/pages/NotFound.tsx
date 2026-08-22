import { useEffect, useRef } from 'react'
import { Button, Result } from 'antd'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'

export default function NotFound() {
  const { t } = useTranslation('common')
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const headingRef = useRef<HTMLHeadingElement | null>(null)

  useEffect(() => {
    headingRef.current?.focus()
  }, [pathname])

  return (
    <section aria-labelledby="not-found-title">
      <Result
        status="info"
        icon={null}
        title={
          <h1 id="not-found-title" ref={headingRef} tabIndex={-1}>
            {`404 · ${t('notFound.title')}`}
          </h1>
        }
        subTitle={t('notFound.description')}
        extra={[
          <Button key="back" onClick={() => navigate(-1)}>
            {t('notFound.back')}
          </Button>,
          <Button key="training" type="primary" href="/training">
            {t('notFound.backToTraining')}
          </Button>,
        ]}
      />
    </section>
  )
}
