import { useEffect, useRef, useState } from 'react'
import { Link, Outlet, useNavigate, useLocation } from 'react-router-dom'
import {
  Alert,
  Avatar,
  Breadcrumb,
  Button,
  Drawer,
  Dropdown,
  Layout,
  Menu,
  Modal,
  type MenuProps,
} from 'antd'
import { useTranslation } from 'react-i18next'
import {
  ExperimentOutlined,
  DatabaseOutlined,
  RocketOutlined,
  AppstoreOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  HomeOutlined,
  DesktopOutlined,
  BarChartOutlined,
  CloudServerOutlined,
  SyncOutlined,
  DownOutlined,
  KeyOutlined,
  LogoutOutlined,
  MenuOutlined,
  TeamOutlined,
  UserOutlined,
  CloseOutlined,
} from '@ant-design/icons'
import { BG_LAYOUT, BORDER_SECONDARY, TEXT_PRIMARY, TEXT_SECONDARY } from '@/theme'
import { getStyleTokens } from '@/theme/appearance'
import { LanguageToggle } from '@/i18n/LanguageToggle'
import { ThemeSelector } from './ThemeSelector'
import { useAppearance } from '@/theme/ThemeProvider'
import { useAuth } from '@/auth/AuthContext'
import ChangePasswordModal from '@/pages/auth/ChangePasswordModal'
import './Layout.css'

const { Header, Sider, Content } = Layout

function useMobileLayout() {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia('(max-width: 820px)').matches
  )

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 820px)')
    const updateMobileLayout = () => setIsMobile(mediaQuery.matches)
    updateMobileLayout()
    mediaQuery.addEventListener('change', updateMobileLayout)
    return () => mediaQuery.removeEventListener('change', updateMobileLayout)
  }, [])

  return isMobile
}

export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileNavigationOpen, setMobileNavigationOpen] = useState(false)
  const [changePasswordOpen, setChangePasswordOpen] = useState(false)
  const [signOutOpen, setSignOutOpen] = useState(false)
  const [signingOut, setSigningOut] = useState(false)
  const [signOutFailed, setSignOutFailed] = useState(false)
  const signOutInFlightRef = useRef(false)
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useTranslation('common')
  const { user, logout } = useAuth()
  const isMobile = useMobileLayout()
  const { mode, accent, style } = useAppearance()
  const appearanceTokens = getStyleTokens({ mode, accent, style })
  const accountRole = user
    ? user.is_admin
      ? t('account.administratorRole')
      : t('account.regularRole')
    : ''

  const selectedKey = '/' + location.pathname.split('/')[1]

  useEffect(() => {
    if (!isMobile) setMobileNavigationOpen(false)
  }, [isMobile])

  const menuItems = [
    { key: '/training', icon: <ExperimentOutlined />, label: t('nav.training') },
    { key: '/datasets', icon: <DatabaseOutlined />, label: t('nav.datasets') },
    { key: '/models', icon: <AppstoreOutlined />, label: t('nav.models') },
    { key: '/evaluations', icon: <BarChartOutlined />, label: t('nav.evaluations') },
    { key: '/deployments', icon: <RocketOutlined />, label: t('nav.deployments') },
    { key: '/vectordb', icon: <CloudServerOutlined />, label: t('nav.vectordb') },
    { key: '/sync', icon: <SyncOutlined />, label: t('nav.sync') },
    { key: '/resources', icon: <DesktopOutlined />, label: t('nav.resources') },
    { key: '/configs', icon: <SettingOutlined />, label: t('nav.configs') },
  ]

  const breadcrumbNameMap: Record<string, string> = {
    '/training': t('breadcrumb.training'),
    '/training/create': t('breadcrumb.createTask'),
    '/datasets': t('breadcrumb.datasets'),
    '/datasets/create': t('breadcrumb.addDataset'),
    '/datasets/generation': t('breadcrumb.generation'),
    '/datasets/generation/create': t('breadcrumb.createGeneration'),
    '/models': t('breadcrumb.models'),
    '/evaluations': t('breadcrumb.evaluations'),
    '/deployments': t('breadcrumb.deployments'),
    '/vectordb': t('breadcrumb.vectordb'),
    '/resources': t('breadcrumb.resources'),
    '/sync': t('breadcrumb.sync'),
    '/sync/create': t('breadcrumb.createSync'),
    '/configs': t('breadcrumb.configs'),
    '/admin': t('admin.section'),
    '/admin/users': t('admin.title'),
  }

  // 生成面包屑项
  const pathSnippets = location.pathname.split('/').filter((i) => i)
  const breadcrumbItems = [
    {
      key: 'home',
      title: (
        <Link to="/training" style={{ color: 'inherit', textDecoration: 'none' }}>
          <HomeOutlined style={{ marginRight: 4 }} />
          {t('breadcrumb.home')}
        </Link>
      ),
    },
    ...pathSnippets.map((_, index) => {
      const url = `/${pathSnippets.slice(0, index + 1).join('/')}`
      const name = breadcrumbNameMap[url]
      const isLast = index === pathSnippets.length - 1
      return {
        key: url,
        title: isLast ? (
          <span style={{ color: TEXT_PRIMARY }}>{name || pathSnippets[index]}</span>
        ) : (
          <Link to={url} style={{ color: 'inherit', textDecoration: 'none' }}>
            {name || pathSnippets[index]}
          </Link>
        ),
      }
    }),
  ]

  const accountItems: MenuProps['items'] = user
    ? [
        {
          key: 'identity',
          disabled: true,
          label: (
            <div style={{ minWidth: 180, padding: '2px 0' }}>
              <div style={{ color: TEXT_PRIMARY, fontWeight: 600 }}>{user.username}</div>
              <div style={{ color: TEXT_SECONDARY, fontSize: 12, marginTop: 2 }}>{accountRole}</div>
            </div>
          ),
        },
        { type: 'divider' },
        ...(user.is_admin
          ? [
              {
                key: 'user-management',
                icon: <TeamOutlined />,
                label: t('account.userManagement'),
              },
            ]
          : []),
        {
          key: 'change-password',
          icon: <KeyOutlined />,
          label: t('account.changePassword'),
        },
        { type: 'divider' },
        {
          key: 'sign-out',
          icon: <LogoutOutlined />,
          label: t('account.signOut'),
          danger: true,
        },
      ]
    : []

  const handleAccountAction: MenuProps['onClick'] = ({ key }) => {
    if (key === 'user-management') {
      navigate('/admin/users')
    } else if (key === 'change-password') {
      setChangePasswordOpen(true)
    } else if (key === 'sign-out') {
      setSignOutFailed(false)
      setSignOutOpen(true)
    }
  }

  const handleNavigationAction: MenuProps['onClick'] = ({ key }) => {
    navigate(key)
    setMobileNavigationOpen(false)
  }

  const confirmSignOut = async () => {
    if (signOutInFlightRef.current) return

    signOutInFlightRef.current = true
    setSigningOut(true)
    setSignOutFailed(false)
    try {
      await logout()
      setSignOutOpen(false)
      navigate('/login', { replace: true })
    } catch {
      setSignOutFailed(true)
    } finally {
      signOutInFlightRef.current = false
      setSigningOut(false)
    }
  }

  return (
    <>
      <Layout
        className={isMobile ? 'main-layout is-mobile' : 'main-layout'}
        style={{ minHeight: '100vh', width: '100%', background: BG_LAYOUT }}
      >
        {!isMobile ? (
          <Sider
            className="main-layout-sidebar"
            trigger={null}
            collapsible
            collapsed={collapsed}
            width={appearanceTokens.sidebarWidth}
            collapsedWidth={64}
            style={{
              background: 'var(--tf-sidebar-bg)',
              borderRight: `1px solid ${BORDER_SECONDARY}`,
            }}
          >
            {/* Logo 区域 */}
            <Link
              className="main-layout-brand"
              to="/training"
              aria-label="TrainFactory"
              style={{
                height: 64,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'var(--tf-brand-bg)',
                margin: collapsed ? 8 : 12,
                marginBottom: 16,
                borderRadius: 'var(--tf-radius)',
                cursor: 'pointer',
                textDecoration: 'none',
                transition: 'all 0.2s',
              }}
            >
              <span
                style={{
                  fontSize: collapsed ? 18 : 20,
                  fontWeight: 700,
                  color: 'var(--tf-brand-text)',
                  letterSpacing: collapsed ? 0 : 1,
                  textShadow: style === 'workbench' ? 'none' : '0 2px 4px rgba(0,0,0,0.2)',
                }}
              >
                {collapsed ? 'TF' : 'TrainFactory'}
              </span>
            </Link>

            {/* 菜单 */}
            <Menu
              mode="inline"
              theme={mode}
              selectedKeys={[selectedKey]}
              items={menuItems}
              onClick={handleNavigationAction}
              style={{
                background: 'transparent',
                border: 0,
              }}
            />
          </Sider>
        ) : null}

        <Layout className="main-layout-body" style={{ minWidth: 0, background: BG_LAYOUT }}>
          {/* 顶部栏 */}
          <Header
            className="main-layout-header"
            style={{
              padding: '0 24px',
              background: 'var(--tf-header-bg)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: `1px solid ${BORDER_SECONDARY}`,
              height: 'var(--tf-header-height)',
            }}
          >
            <div className="main-layout-header-left">
              {/* 折叠按钮 */}
              <Button
                type="text"
                className="main-layout-nav-trigger"
                aria-label={
                  isMobile
                    ? t('nav.openNavigation')
                    : collapsed
                      ? t('nav.expandNavigation')
                      : t('nav.collapseNavigation')
                }
                icon={
                  isMobile ? (
                    <MenuOutlined />
                  ) : collapsed ? (
                    <MenuUnfoldOutlined />
                  ) : (
                    <MenuFoldOutlined />
                  )
                }
                onClick={() => {
                  if (isMobile) {
                    setMobileNavigationOpen(true)
                  } else {
                    setCollapsed((current) => !current)
                  }
                }}
              />

              {/* 面包屑 */}
              <Breadcrumb
                className="main-layout-breadcrumb"
                items={breadcrumbItems}
                style={{ color: TEXT_SECONDARY }}
              />
            </div>

            {/* 右侧区域 */}
            <div className="main-layout-header-right">
              <ThemeSelector />
              <LanguageToggle />
              {user ? (
                <Dropdown
                  trigger={['click']}
                  placement="bottomRight"
                  menu={{ items: accountItems, onClick: handleAccountAction }}
                >
                  <Button
                    type="text"
                    className="main-layout-account-button"
                    aria-label={t('account.menuLabel', {
                      username: user.username,
                      role: accountRole,
                    })}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 7,
                      height: 44,
                      maxWidth: 220,
                      padding: '4px 8px',
                      color: TEXT_SECONDARY,
                    }}
                  >
                    <Avatar size={26} icon={<UserOutlined />} style={{ flexShrink: 0 }} />
                    <span
                      className="main-layout-account-identity"
                      style={{
                        display: 'flex',
                        minWidth: 0,
                        maxWidth: 136,
                        flexDirection: 'column',
                        alignItems: 'flex-start',
                        lineHeight: 1.2,
                      }}
                    >
                      <span
                        style={{
                          width: '100%',
                          overflow: 'hidden',
                          color: TEXT_PRIMARY,
                          fontSize: 12,
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {user.username}
                      </span>
                      <span
                        style={{
                          width: '100%',
                          overflow: 'hidden',
                          color: TEXT_SECONDARY,
                          fontSize: 10,
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {accountRole}
                      </span>
                    </span>
                    <DownOutlined
                      className="main-layout-account-chevron"
                      style={{ flexShrink: 0, fontSize: 10 }}
                    />
                  </Button>
                </Dropdown>
              ) : null}
              <span className="main-layout-version" style={{ color: TEXT_SECONDARY, fontSize: 12 }}>
                v0.1.0
              </span>
            </div>
          </Header>

          {/* 内容区域 */}
          <Content
            className="main-layout-content"
            style={{
              margin: 'var(--tf-content-margin)',
              padding: 'var(--tf-content-padding)',
              background: 'var(--tf-content-bg)',
              borderRadius: 'var(--tf-radius-lg)',
              overflow: 'auto',
              minHeight: 'calc(100vh - var(--tf-header-height) - 2 * var(--tf-content-margin))',
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>

      <Drawer
        className="main-layout-navigation-drawer"
        title={t('nav.mainNavigation')}
        placement="left"
        width={280}
        open={mobileNavigationOpen}
        closable={false}
        onClose={() => setMobileNavigationOpen(false)}
        extra={
          <Button
            type="text"
            aria-label={t('nav.closeNavigation')}
            icon={<CloseOutlined />}
            onClick={() => setMobileNavigationOpen(false)}
          />
        }
        styles={{
          header: { background: 'var(--tf-sidebar-bg)', borderBottomColor: BORDER_SECONDARY },
          body: { padding: '12px 0', background: 'var(--tf-sidebar-bg)' },
        }}
      >
        <Menu
          mode="inline"
          theme={mode}
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={handleNavigationAction}
          style={{ background: 'transparent', border: 0 }}
        />
      </Drawer>

      <ChangePasswordModal open={changePasswordOpen} onClose={() => setChangePasswordOpen(false)} />

      <Modal
        title={t('account.signOutTitle')}
        open={signOutOpen}
        onCancel={() => {
          if (!signingOut) {
            setSignOutOpen(false)
            setSignOutFailed(false)
          }
        }}
        onOk={confirmSignOut}
        okText={t('account.confirmSignOut')}
        cancelText={t('action.cancel')}
        confirmLoading={signingOut}
        okButtonProps={{ danger: true, disabled: signingOut }}
        cancelButtonProps={{ disabled: signingOut }}
        closable={!signingOut}
        keyboard={!signingOut}
        maskClosable={!signingOut}
        destroyOnHidden
      >
        {signOutFailed ? (
          <Alert
            type="error"
            showIcon
            message={t('account.signOutFailed')}
            style={{ marginBottom: 16 }}
          />
        ) : null}
        <p style={{ margin: 0 }}>{t('account.signOutDescription')}</p>
      </Modal>
    </>
  )
}
