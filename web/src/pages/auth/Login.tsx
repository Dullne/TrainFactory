import { forwardRef, useEffect, useState, type ComponentPropsWithoutRef } from 'react'
import { Alert, Button, Form, Input, Segmented, message, type FormProps } from 'antd'
import { LockOutlined, MailOutlined, UserOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { useAuth } from '@/auth/AuthContext'
import { AccessiblePasswordInput } from '@/components/AccessiblePasswordInput'
import { TrainFactoryWorkflow } from '@/components/brand/TrainFactoryWorkflow'
import { TrainFactoryWordmark } from '@/components/brand/TrainFactoryWordmark'
import { LanguageToggle } from '@/i18n/LanguageToggle'
import { ThemeSelector } from '@/components/ThemeSelector'
import './Login.css'

type AuthMode = 'login' | 'register'

interface AuthFormValues {
  username: string
  email?: string
  password: string
  confirmPassword?: string
}

type NativeAuthFormProps = ComponentPropsWithoutRef<'form'> & {
  'data-native-name'?: string
}

const NativeAuthForm = forwardRef<HTMLFormElement, NativeAuthFormProps>(
  ({ 'data-native-name': nativeName, ...formProps }, ref) => (
    <form {...formProps} ref={ref} name={nativeName} />
  )
)

export default function Login() {
  const { t } = useTranslation('common')
  const { login, register, config } = useAuth()
  const [form] = Form.useForm<AuthFormValues>()
  const [mode, setMode] = useState<AuthMode>('login')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const registrationEnabled = config.self_registration_enabled

  useEffect(() => {
    if (!registrationEnabled && mode === 'register') {
      setMode('login')
      setError(null)
      form.resetFields()
    }
  }, [form, mode, registrationEnabled])

  const changeMode = (nextMode: AuthMode) => {
    if (loading || nextMode === mode) return
    setMode(nextMode)
    setError(null)
    form.resetFields()
  }

  const onFinish: FormProps<AuthFormValues>['onFinish'] = async (values) => {
    if (loading) return

    setLoading(true)
    setError(null)
    try {
      if (mode === 'register') {
        await register({
          username: values.username.trim(),
          password: values.password,
          ...(values.email?.trim() ? { email: values.email.trim() } : {}),
        })
        message.success(t('auth.registerSuccess'))
      } else {
        await login({ username: values.username.trim(), password: values.password })
        message.success(t('auth.success'))
      }
    } catch (requestError) {
      const requestErrorMessage =
        requestError instanceof Error ? requestError.message : t('auth.requestFailed')
      const isIdentityConflict =
        mode === 'register' &&
        requestErrorMessage.toLowerCase().includes('username or email already exists')

      if (isIdentityConflict) {
        form.setFields([
          { name: 'username', errors: [requestErrorMessage] },
          ...(values.email?.trim()
            ? [{ name: 'email' as const, errors: [requestErrorMessage] }]
            : []),
        ])
      } else {
        setError(requestErrorMessage)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <header className="auth-topbar">
        <div className="auth-brand">
          <TrainFactoryWordmark />
        </div>
        <div className="auth-language">
          <ThemeSelector />
          <LanguageToggle />
        </div>
      </header>

      <main className="auth-main">
        <section className="auth-identity" aria-labelledby="workspace-title">
          <div className="auth-identity-content">
            <h1 id="workspace-title">{t('auth.workspaceTitle')}</h1>
            <p>{t('auth.workspaceSubtitle')}</p>
            <TrainFactoryWorkflow alt={t('auth.workflowAlt')} />
          </div>
        </section>

        <section className="auth-panel">
          <div className="auth-form-shell" data-testid="auth-form">
            <div className="auth-mobile-intro">
              <h1>{t('auth.workspaceTitle')}</h1>
              <p>{t('auth.workspaceSubtitle')}</p>
              <TrainFactoryWorkflow alt={t('auth.workflowAlt')} />
            </div>

            {registrationEnabled ? (
              <Segmented<AuthMode>
                aria-label={t('auth.modeLabel')}
                block
                disabled={loading}
                className="auth-mode-switch"
                options={[
                  { label: t('auth.loginMode'), value: 'login' },
                  { label: t('auth.registerMode'), value: 'register' },
                ]}
                value={mode}
                onChange={changeMode}
              />
            ) : null}

            <div className="auth-form-heading">
              <h2>{mode === 'register' ? t('auth.registerTitle') : t('auth.loginTitle')}</h2>
              <p>{mode === 'register' ? t('auth.registerSubtitle') : t('auth.loginSubtitle')}</p>
            </div>

            <Form<AuthFormValues>
              id={mode === 'login' ? 'login-form' : 'registration-form'}
              component={NativeAuthForm}
              data-native-name={mode === 'login' ? 'login' : 'register'}
              noValidate
              form={form}
              layout="vertical"
              requiredMark={false}
              onFinish={onFinish}
              disabled={loading}
              autoComplete="on"
            >
              {error ? (
                <Alert className="auth-error" type="error" message={error} showIcon />
              ) : null}

              <Form.Item
                label={t('auth.username')}
                name="username"
                rules={[
                  { required: true, message: t('auth.requiredField') },
                  ...(mode === 'register'
                    ? [
                        {
                          validator: (_: unknown, value?: string) => {
                            if (!value) return Promise.resolve()
                            const length = value.trim().length
                            return length >= 3 && length <= 64
                              ? Promise.resolve()
                              : Promise.reject(new Error(t('auth.usernameRequirement')))
                          },
                        },
                      ]
                    : []),
                ]}
              >
                <Input
                  name="username"
                  required
                  autoFocus
                  autoComplete="username"
                  maxLength={64}
                  prefix={<UserOutlined aria-hidden />}
                  placeholder={t('auth.usernamePlaceholder')}
                />
              </Form.Item>

              {mode === 'register' ? (
                <Form.Item
                  label={t('auth.email')}
                  name="email"
                  rules={[{ type: 'email', message: t('auth.invalidEmail') }]}
                >
                  <Input
                    name="email"
                    autoComplete="email"
                    maxLength={256}
                    prefix={<MailOutlined aria-hidden />}
                    placeholder={t('auth.emailPlaceholder')}
                  />
                </Form.Item>
              ) : null}

              <Form.Item
                label={t('auth.password')}
                name="password"
                extra={mode === 'register' ? t('auth.passwordRequirement') : undefined}
                rules={[
                  { required: true, message: t('auth.requiredField') },
                  ...(mode === 'register'
                    ? [
                        {
                          min: 10,
                          max: 128,
                          message: t('auth.passwordRequirement'),
                        },
                      ]
                    : []),
                ]}
              >
                <AccessiblePasswordInput
                  key={`password-${mode}`}
                  name="password"
                  required
                  autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                  prefix={<LockOutlined aria-hidden />}
                  placeholder={t('auth.passwordPlaceholder')}
                  disabled={loading}
                  showPasswordLabel={t('auth.showPassword')}
                  hidePasswordLabel={t('auth.hidePassword')}
                />
              </Form.Item>

              {mode === 'register' ? (
                <Form.Item
                  label={t('auth.confirmPassword')}
                  name="confirmPassword"
                  dependencies={['password']}
                  rules={[
                    { required: true, message: t('auth.requiredField') },
                    ({ getFieldValue }) => ({
                      validator(_, value) {
                        if (!value || getFieldValue('password') === value) {
                          return Promise.resolve()
                        }
                        return Promise.reject(new Error(t('auth.passwordsMismatch')))
                      },
                    }),
                  ]}
                >
                  <AccessiblePasswordInput
                    name="confirmPassword"
                    required
                    autoComplete="new-password"
                    prefix={<LockOutlined aria-hidden />}
                    placeholder={t('auth.confirmPasswordPlaceholder')}
                    disabled={loading}
                    showPasswordLabel={t('auth.showConfirmPassword')}
                    hidePasswordLabel={t('auth.hideConfirmPassword')}
                  />
                </Form.Item>
              ) : null}

              <Button
                className="auth-submit"
                type="primary"
                htmlType="submit"
                loading={loading}
                disabled={loading}
                block
              >
                {mode === 'register' ? t('auth.createAccount') : t('auth.submit')}
              </Button>
            </Form>

            <footer className="auth-form-footer">
              <span
                className={`auth-service-status${
                  config.apiAvailable ? ' is-available' : ' is-unavailable'
                }`}
              >
                <span className="auth-service-dot" aria-hidden />
                {config.apiAvailable ? t('auth.serviceAvailable') : t('auth.serviceUnavailable')}
              </span>
              <span>v0.1.0</span>
            </footer>
          </div>
        </section>
      </main>
    </div>
  )
}
