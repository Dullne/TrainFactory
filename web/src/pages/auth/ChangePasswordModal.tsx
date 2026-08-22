import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Alert, Button, Form, Modal, message, type FormProps } from 'antd'
import { LockOutlined } from '@ant-design/icons'
import { useTranslation } from 'react-i18next'
import { useAuth } from '@/auth/AuthContext'
import { AccessiblePasswordInput } from '@/components/AccessiblePasswordInput'
import { authApi } from '@/services/api'

interface ChangePasswordModalProps {
  open: boolean
  onClose: () => void
}

interface ChangePasswordValues {
  currentPassword: string
  newPassword: string
  confirmPassword: string
}

export default function ChangePasswordModal({ open, onClose }: ChangePasswordModalProps) {
  const { t } = useTranslation('common')
  const navigate = useNavigate()
  const { clearLocalSession } = useAuth()
  const [form] = Form.useForm<ChangePasswordValues>()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      form.resetFields()
      setError(null)
    }
  }, [form, open])

  const close = () => {
    if (loading) return
    onClose()
  }

  const onFinish: FormProps<ChangePasswordValues>['onFinish'] = async (values) => {
    if (loading) return

    setLoading(true)
    setError(null)
    try {
      await authApi.changePassword({
        old_password: values.currentPassword,
        new_password: values.newPassword,
      })
      clearLocalSession()

      message.success(t('account.passwordChanged'))
      onClose()
      navigate('/login', { replace: true })
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : t('account.changePasswordFailed')
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <Modal
      title={t('account.changePassword')}
      open={open}
      onCancel={close}
      footer={null}
      width={440}
      closable={!loading}
      keyboard={!loading}
      maskClosable={!loading}
      destroyOnHidden
    >
      <Form<ChangePasswordValues>
        form={form}
        layout="vertical"
        requiredMark={false}
        disabled={loading}
        onFinish={onFinish}
      >
        {error ? (
          <Alert type="error" message={error} showIcon style={{ marginBottom: 16 }} />
        ) : null}

        <Form.Item
          label={t('account.currentPassword')}
          name="currentPassword"
          rules={[{ required: true, message: t('auth.requiredField') }]}
        >
          <AccessiblePasswordInput
            autoComplete="current-password"
            prefix={<LockOutlined aria-hidden />}
            disabled={loading}
            style={{ minHeight: 44 }}
            showPasswordLabel={t('account.showCurrentPassword')}
            hidePasswordLabel={t('account.hideCurrentPassword')}
          />
        </Form.Item>

        <Form.Item
          label={t('account.newPassword')}
          name="newPassword"
          extra={t('auth.passwordRequirement')}
          rules={[
            { required: true, message: t('auth.requiredField') },
            { min: 10, max: 128, message: t('auth.passwordRequirement') },
          ]}
        >
          <AccessiblePasswordInput
            autoComplete="new-password"
            prefix={<LockOutlined aria-hidden />}
            disabled={loading}
            style={{ minHeight: 44 }}
            showPasswordLabel={t('account.showNewPassword')}
            hidePasswordLabel={t('account.hideNewPassword')}
          />
        </Form.Item>

        <Form.Item
          label={t('account.confirmNewPassword')}
          name="confirmPassword"
          dependencies={['newPassword']}
          rules={[
            { required: true, message: t('auth.requiredField') },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue('newPassword') === value) {
                  return Promise.resolve()
                }
                return Promise.reject(new Error(t('auth.passwordsMismatch')))
              },
            }),
          ]}
        >
          <AccessiblePasswordInput
            autoComplete="new-password"
            prefix={<LockOutlined aria-hidden />}
            disabled={loading}
            style={{ minHeight: 44 }}
            showPasswordLabel={t('account.showConfirmNewPassword')}
            hidePasswordLabel={t('account.hideConfirmNewPassword')}
          />
        </Form.Item>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <Button onClick={close} disabled={loading} style={{ minHeight: 40 }}>
            {t('action.cancel')}
          </Button>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            disabled={loading}
            style={{ minHeight: 40 }}
          >
            {t('account.updatePassword')}
          </Button>
        </div>
      </Form>
    </Modal>
  )
}
