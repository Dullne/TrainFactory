import type { CSSProperties, ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { Spin } from 'antd'
import { useAuth } from '@/auth/AuthContext'

const loadingStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: '100vh',
}

export default function AdminRoute({ children }: { children: ReactNode }) {
  const { user, status } = useAuth()

  // checking 时先显示 loading，避免未加载完成就把非 admin 用户闪跳走
  if (status === 'checking') {
    return (
      <div style={loadingStyle}>
        <Spin size="large" />
      </div>
    )
  }

  if (status === 'unauthenticated' || !user?.is_admin) {
    return <Navigate to="/training" replace />
  }

  return <>{children}</>
}
