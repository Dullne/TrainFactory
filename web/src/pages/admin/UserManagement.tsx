import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Checkbox,
  Form,
  Input,
  Modal,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
  type FormProps,
  type TableColumnsType,
} from 'antd'
import {
  CheckCircleOutlined,
  CrownOutlined,
  KeyOutlined,
  PlusOutlined,
  ReloadOutlined,
  StopOutlined,
  UserSwitchOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/auth/AuthContext'
import { AccessiblePasswordInput } from '@/components/AccessiblePasswordInput'
import { adminUserApi, type AuthUser } from '@/services/api'
import './UserManagement.css'

const DEFAULT_PAGE_SIZE = 20

interface CreateAccountValues {
  username: string
  email?: string
  password: string
  isAdmin?: boolean
}

interface ResetPasswordValues {
  newPassword: string
  confirmPassword: string
}

interface AccountCommand {
  kind: 'status' | 'role'
  account: AuthUser
}

function replaceAccount(accounts: AuthUser[], updated: AuthUser) {
  return accounts.map((account) => (account.user_id === updated.user_id ? updated : account))
}

export default function UserManagement() {
  const { t } = useTranslation('common')
  const navigate = useNavigate()
  const { user: currentUser, refreshUser, logout, clearLocalSession } = useAuth()
  const [users, setUsers] = useState<AuthUser[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [loading, setLoading] = useState(true)
  const [listError, setListError] = useState<string | null>(null)
  const listGenerationRef = useRef(0)
  const lastAutomaticLoadKeyRef = useRef<string | null>(null)

  const [createOpen, setCreateOpen] = useState(false)
  const [createLoading, setCreateLoading] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const createInFlightRef = useRef(false)
  const [createForm] = Form.useForm<CreateAccountValues>()

  const [command, setCommand] = useState<AccountCommand | null>(null)
  const [commandLoading, setCommandLoading] = useState(false)
  const [commandError, setCommandError] = useState<string | null>(null)
  const commandInFlightRef = useRef(false)

  const [resetAccount, setResetAccount] = useState<AuthUser | null>(null)
  const [resetLoading, setResetLoading] = useState(false)
  const [resetError, setResetError] = useState<string | null>(null)
  const resetInFlightRef = useRef(false)
  const [resetForm] = Form.useForm<ResetPasswordValues>()

  const invalidateListRequests = useCallback(() => {
    listGenerationRef.current += 1
    setLoading(false)
    setListError(null)
  }, [])

  const loadUsers = useCallback(async () => {
    const generation = ++listGenerationRef.current
    setLoading(true)
    setListError(null)
    try {
      const response = await adminUserApi.list({
        limit: pageSize,
        offset: (page - 1) * pageSize,
      })
      if (listGenerationRef.current !== generation) return
      setUsers(response.users)
      setTotal(response.total)
    } catch (error) {
      if (listGenerationRef.current !== generation) return
      setUsers([])
      setTotal(0)
      setListError(error instanceof Error ? error.message : t('admin.loadFailed'))
    } finally {
      if (listGenerationRef.current === generation) setLoading(false)
    }
  }, [page, pageSize, t])

  useEffect(() => {
    const loadKey = `${page}:${pageSize}`
    if (lastAutomaticLoadKeyRef.current === loadKey) return
    lastAutomaticLoadKeyRef.current = loadKey
    void loadUsers()
  }, [loadUsers, page, pageSize])

  const openCreate = () => {
    createForm.resetFields()
    setCreateError(null)
    setCreateOpen(true)
  }

  const closeCreate = () => {
    if (createInFlightRef.current) return
    setCreateOpen(false)
    setCreateError(null)
  }

  const submitCreate: FormProps<CreateAccountValues>['onFinish'] = async (values) => {
    if (createInFlightRef.current) return
    createInFlightRef.current = true
    setCreateLoading(true)
    setCreateError(null)
    createForm.setFields([
      { name: 'username', errors: [] },
      { name: 'email', errors: [] },
    ])
    try {
      const created = await adminUserApi.create({
        username: values.username.trim(),
        email: values.email?.trim() || undefined,
        password: values.password,
        is_admin: values.isAdmin ?? false,
      })
      invalidateListRequests()
      if (page === 1) {
        setUsers((current) => [created, ...current].slice(0, pageSize))
      } else {
        setPage(1)
      }
      setTotal((current) => current + 1)
      setCreateOpen(false)
      createForm.resetFields()
      message.success(t('admin.createSuccess'))
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : t('admin.createFailed')
      if (/username or email already exists/i.test(errorMessage)) {
        createForm.setFields([
          { name: 'username', errors: [errorMessage] },
          ...(values.email ? [{ name: 'email' as const, errors: [errorMessage] }] : []),
        ])
      } else {
        setCreateError(errorMessage)
      }
    } finally {
      createInFlightRef.current = false
      setCreateLoading(false)
    }
  }

  const openCommand = (nextCommand: AccountCommand) => {
    setCommandError(null)
    setCommand(nextCommand)
  }

  const closeCommand = () => {
    if (commandInFlightRef.current) return
    setCommand(null)
    setCommandError(null)
  }

  const submitCommand = async () => {
    if (!command || commandInFlightRef.current) return
    commandInFlightRef.current = true
    setCommandLoading(true)
    setCommandError(null)
    try {
      const update =
        command.kind === 'status'
          ? { is_active: !command.account.is_active }
          : { is_admin: !command.account.is_admin }
      const updated = await adminUserApi.updateFlags(command.account.user_id, update)
      invalidateListRequests()
      setUsers((current) => replaceAccount(current, updated))
      setCommand(null)
      message.success(t('admin.updateSuccess'))
      if (
        command.kind === 'role' &&
        updated.user_id === currentUser?.user_id &&
        !updated.is_admin
      ) {
        try {
          await refreshUser()
          navigate('/training', { replace: true })
        } catch {
          // AuthContext clears the identity if the refreshed session is no longer valid.
        }
      }
    } catch (error) {
      setCommandError(error instanceof Error ? error.message : t('admin.updateFailed'))
    } finally {
      commandInFlightRef.current = false
      setCommandLoading(false)
    }
  }

  const openReset = (account: AuthUser) => {
    resetForm.resetFields()
    setResetError(null)
    setResetAccount(account)
  }

  const closeReset = () => {
    if (resetInFlightRef.current) return
    setResetAccount(null)
    setResetError(null)
  }

  const submitReset: FormProps<ResetPasswordValues>['onFinish'] = async (values) => {
    if (!resetAccount || resetInFlightRef.current) return
    const resettingCurrentAccount = resetAccount.user_id === currentUser?.user_id
    resetInFlightRef.current = true
    setResetLoading(true)
    setResetError(null)
    try {
      await adminUserApi.resetPassword(resetAccount.user_id, {
        new_password: values.newPassword,
      })
      invalidateListRequests()
      setResetAccount(null)
      resetForm.resetFields()
      message.success(t('admin.resetSuccess'))
      if (resettingCurrentAccount) {
        try {
          await logout()
        } catch {
          clearLocalSession()
        } finally {
          navigate('/login', { replace: true })
        }
      }
    } catch (error) {
      setResetError(error instanceof Error ? error.message : t('admin.resetFailed'))
    } finally {
      resetInFlightRef.current = false
      setResetLoading(false)
    }
  }

  const commandTitle = command
    ? command.kind === 'status'
      ? command.account.is_active
        ? t('admin.disableTitle', { username: command.account.username })
        : t('admin.enableTitle', { username: command.account.username })
      : command.account.is_admin
        ? t('admin.demoteTitle', { username: command.account.username })
        : t('admin.promoteTitle', { username: command.account.username })
    : ''

  const commandConfirmText = command
    ? command.kind === 'status'
      ? command.account.is_active
        ? t('admin.confirmDisable')
        : t('admin.confirmEnable')
      : command.account.is_admin
        ? t('admin.confirmDemotion')
        : t('admin.confirmPromotion')
    : ''

  const columns: TableColumnsType<AuthUser> = [
    {
      title: t('admin.username'),
      dataIndex: 'username',
      key: 'username',
      width: 180,
      render: (username: string, account) => (
        <div className="admin-account-identity">
          <strong>{username}</strong>
          {account.user_id === currentUser?.user_id ? (
            <span>{t('admin.currentAccount')}</span>
          ) : null}
        </div>
      ),
    },
    {
      title: t('admin.email'),
      dataIndex: 'email',
      key: 'email',
      width: 240,
      render: (email: string | null | undefined) => email || t('admin.notSet'),
    },
    {
      title: t('admin.role'),
      dataIndex: 'is_admin',
      key: 'role',
      width: 140,
      render: (isAdmin: boolean) => (
        <Tag color={isAdmin ? 'blue' : undefined}>
          {isAdmin ? t('admin.administrator') : t('admin.regular')}
        </Tag>
      ),
    },
    {
      title: t('admin.status'),
      dataIndex: 'is_active',
      key: 'status',
      width: 120,
      render: (isActive: boolean) => (
        <Tag color={isActive ? 'green' : 'default'}>
          {isActive ? t('admin.active') : t('admin.disabled')}
        </Tag>
      ),
    },
    {
      title: t('admin.created'),
      dataIndex: 'created_at',
      key: 'createdAt',
      width: 180,
      render: (createdAt: string | null | undefined) =>
        createdAt ? dayjs(createdAt).format('YYYY-MM-DD HH:mm') : t('admin.notSet'),
    },
    {
      title: t('admin.actions'),
      key: 'actions',
      fixed: 'right',
      width: 150,
      render: (_, account) => {
        const isSelf = account.user_id === currentUser?.user_id
        const statusLabel = account.is_active
          ? t('admin.disableAccount', { username: account.username })
          : t('admin.enableAccount', { username: account.username })
        const roleLabel = account.is_admin
          ? t('admin.demoteAccount', { username: account.username })
          : t('admin.promoteAccount', { username: account.username })

        return (
          <Space size={2}>
            <Tooltip title={isSelf ? t('admin.selfDisableBlocked') : statusLabel}>
              <span>
                <Button
                  type="text"
                  aria-label={statusLabel}
                  icon={account.is_active ? <StopOutlined /> : <CheckCircleOutlined />}
                  disabled={isSelf}
                  onClick={() => openCommand({ kind: 'status', account })}
                />
              </span>
            </Tooltip>
            <Tooltip title={roleLabel}>
              <Button
                type="text"
                aria-label={roleLabel}
                icon={account.is_admin ? <UserSwitchOutlined /> : <CrownOutlined />}
                onClick={() => openCommand({ kind: 'role', account })}
              />
            </Tooltip>
            <Tooltip title={t('admin.resetPasswordFor', { username: account.username })}>
              <Button
                type="text"
                aria-label={t('admin.resetPasswordFor', { username: account.username })}
                icon={<KeyOutlined />}
                onClick={() => openReset(account)}
              />
            </Tooltip>
          </Space>
        )
      },
    },
  ]

  return (
    <section className="admin-users-page" aria-labelledby="admin-users-title">
      <div className="admin-users-toolbar">
        <div>
          <Typography.Title id="admin-users-title" level={3}>
            {t('admin.title')}
          </Typography.Title>
          <Typography.Text type="secondary">
            {t('admin.accountCount', { count: total })}
          </Typography.Text>
        </div>
        <Space wrap>
          {loading ? (
            <span className="admin-loading-status" role="status" aria-label={t('admin.loading')}>
              <Spin size="small" />
              {t('admin.loading')}
            </span>
          ) : null}
          <Button icon={<ReloadOutlined />} onClick={() => void loadUsers()} disabled={loading}>
            {t('admin.refresh')}
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={openCreate}>
            {t('admin.createAccount')}
          </Button>
        </Space>
      </div>

      {listError ? (
        <Alert
          className="admin-list-alert"
          type="error"
          showIcon
          message={listError}
          action={
            <Button
              size="small"
              aria-label={t('admin.retryLoading')}
              onClick={() => void loadUsers()}
            >
              {t('admin.retry')}
            </Button>
          }
        />
      ) : null}

      <Table<AuthUser>
        rowKey="user_id"
        columns={columns}
        dataSource={users}
        loading={loading}
        size="middle"
        scroll={{ x: 1050 }}
        locale={{ emptyText: t('admin.empty') }}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          pageSizeOptions: [10, 20, 50, 100],
          showTotal: (count) => t('admin.accountCount', { count }),
          onChange: (nextPage, nextPageSize) => {
            if (nextPageSize !== pageSize) {
              setPage(1)
              setPageSize(nextPageSize)
            } else {
              setPage(nextPage)
            }
          },
        }}
      />

      <Modal
        title={t('admin.createAccount')}
        open={createOpen}
        footer={null}
        width={460}
        onCancel={closeCreate}
        closable={!createLoading}
        keyboard={!createLoading}
        maskClosable={!createLoading}
        destroyOnHidden
      >
        <Form<CreateAccountValues>
          form={createForm}
          layout="vertical"
          requiredMark={false}
          initialValues={{ isAdmin: false }}
          disabled={createLoading}
          onFinish={submitCreate}
        >
          {createError ? (
            <Alert type="error" showIcon message={createError} className="admin-command-alert" />
          ) : null}
          <Form.Item
            name="username"
            label={t('admin.username')}
            rules={[
              { required: true, message: t('auth.requiredField') },
              {
                validator: (_, value: string | undefined) =>
                  value && value.trim().length >= 3 && value.trim().length <= 64
                    ? Promise.resolve()
                    : Promise.reject(new Error(t('auth.usernameRequirement'))),
              },
            ]}
          >
            <Input autoComplete="off" maxLength={64} />
          </Form.Item>
          <Form.Item
            name="email"
            label={t('admin.email')}
            rules={[{ type: 'email', message: t('auth.invalidEmail') }]}
          >
            <Input autoComplete="off" maxLength={256} />
          </Form.Item>
          <Form.Item
            name="password"
            label={t('auth.password')}
            extra={t('auth.passwordRequirement')}
            rules={[
              { required: true, message: t('auth.requiredField') },
              { min: 10, max: 128, message: t('auth.passwordRequirement') },
            ]}
          >
            <AccessiblePasswordInput
              autoComplete="new-password"
              disabled={createLoading}
              showPasswordLabel={t('auth.showPassword')}
              hidePasswordLabel={t('auth.hidePassword')}
            />
          </Form.Item>
          <Form.Item name="isAdmin" valuePropName="checked">
            <Checkbox>{t('account.administratorRole')}</Checkbox>
          </Form.Item>
          <div className="admin-modal-actions">
            <Button onClick={closeCreate} disabled={createLoading}>
              {t('action.cancel')}
            </Button>
            <Button type="primary" htmlType="submit" loading={createLoading}>
              {t('admin.create')}
            </Button>
          </div>
        </Form>
      </Modal>

      <Modal
        title={commandTitle}
        open={command !== null}
        onCancel={closeCommand}
        onOk={() => void submitCommand()}
        okText={commandConfirmText}
        cancelText={t('action.cancel')}
        confirmLoading={commandLoading}
        okButtonProps={{
          danger: command?.kind === 'status' && command.account.is_active,
          disabled: commandLoading,
        }}
        cancelButtonProps={{ disabled: commandLoading }}
        closable={!commandLoading}
        keyboard={!commandLoading}
        maskClosable={!commandLoading}
        destroyOnHidden
      >
        {commandError ? (
          <Alert type="error" showIcon message={commandError} className="admin-command-alert" />
        ) : null}
        <Typography.Paragraph>{t('admin.commandDescription')}</Typography.Paragraph>
      </Modal>

      <Modal
        title={
          resetAccount
            ? t('admin.resetPasswordFor', { username: resetAccount.username })
            : t('admin.resetPassword')
        }
        open={resetAccount !== null}
        footer={null}
        width={440}
        onCancel={closeReset}
        closable={!resetLoading}
        keyboard={!resetLoading}
        maskClosable={!resetLoading}
        destroyOnHidden
      >
        <Typography.Paragraph className="admin-reset-note">
          {t('admin.resetSessionNotice')}
        </Typography.Paragraph>
        <Form<ResetPasswordValues>
          form={resetForm}
          layout="vertical"
          requiredMark={false}
          disabled={resetLoading}
          onFinish={submitReset}
        >
          {resetError ? (
            <Alert type="error" showIcon message={resetError} className="admin-command-alert" />
          ) : null}
          <Form.Item
            name="newPassword"
            label={t('account.newPassword')}
            extra={t('auth.passwordRequirement')}
            rules={[
              { required: true, message: t('auth.requiredField') },
              { min: 10, max: 128, message: t('auth.passwordRequirement') },
            ]}
          >
            <AccessiblePasswordInput
              autoComplete="new-password"
              disabled={resetLoading}
              showPasswordLabel={t('account.showNewPassword')}
              hidePasswordLabel={t('account.hideNewPassword')}
            />
          </Form.Item>
          <Form.Item
            name="confirmPassword"
            label={t('account.confirmNewPassword')}
            dependencies={['newPassword']}
            rules={[
              { required: true, message: t('auth.requiredField') },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  return !value || value === getFieldValue('newPassword')
                    ? Promise.resolve()
                    : Promise.reject(new Error(t('auth.passwordsMismatch')))
                },
              }),
            ]}
          >
            <AccessiblePasswordInput
              autoComplete="new-password"
              disabled={resetLoading}
              showPasswordLabel={t('account.showConfirmNewPassword')}
              hidePasswordLabel={t('account.hideConfirmNewPassword')}
            />
          </Form.Item>
          <div className="admin-modal-actions">
            <Button onClick={closeReset} disabled={resetLoading}>
              {t('action.cancel')}
            </Button>
            <Button type="primary" htmlType="submit" loading={resetLoading}>
              {t('admin.resetPassword')}
            </Button>
          </div>
        </Form>
      </Modal>
    </section>
  )
}
