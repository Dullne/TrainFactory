import { Component, ReactNode } from 'react'
import { Button, Result } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { withTranslation, WithTranslation } from 'react-i18next'

interface Props extends WithTranslation {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

class ErrorBoundaryInner extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  handleReload = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    const { t } = this.props

    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: 400,
            padding: 24,
          }}
        >
          <Result
            status="error"
            title={t('error.title')}
            subTitle={
              process.env.NODE_ENV === 'development'
                ? this.state.error?.message
                : t('error.description')
            }
            extra={[
              <Button key="retry" onClick={this.handleRetry}>
                {t('action.retry')}
              </Button>,
              <Button
                key="reload"
                type="primary"
                icon={<ReloadOutlined />}
                onClick={this.handleReload}
              >
                {t('action.refreshPage')}
              </Button>,
            ]}
          />
        </div>
      )
    }

    return this.props.children
  }
}

export const ErrorBoundary = withTranslation('common')(ErrorBoundaryInner)
export default ErrorBoundary
