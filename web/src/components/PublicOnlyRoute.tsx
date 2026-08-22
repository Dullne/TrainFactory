import { useEffect, type CSSProperties, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Spin } from 'antd'
import { useAuth } from '@/auth/AuthContext'
import { safeReturnTo } from '@/auth/returnTo'

const loadingStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: '100vh',
}

export default function PublicOnlyRoute({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  const [searchParams] = useSearchParams()
  const returnTo = safeReturnTo(searchParams.get('redirect'), window.location.origin)

  useEffect(() => {
    if (status === 'authenticated') {
      // 整页跳转而非前端路由：Chrome 只在浏览器认可的"导航"后才会弹出保存密码提示，
      // SPA 的 pushState 路由切换不会稳定触发该提示。
      window.location.replace(returnTo)
    }
  }, [status, returnTo])

  if (status === 'checking') {
    return (
      <div style={loadingStyle}>
        <Spin size="large" />
      </div>
    )
  }

  if (status === 'authenticated') {
    return null
  }

  return <>{children}</>
}
