import { useEffect, useRef, type CSSProperties, type ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { Spin } from 'antd'
import { useAuth } from '@/auth/AuthContext'
import { currentReturnTo, loginPathFor } from '@/auth/returnTo'

const loadingStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  height: '100vh',
}

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const { status } = useAuth()
  const location = useLocation()
  const previousStatusRef = useRef(status)

  useEffect(() => {
    previousStatusRef.current = status
  }, [status])

  if (status === 'checking') {
    return (
      <div style={loadingStyle}>
        <Spin size="large" />
      </div>
    )
  }

  if (status === 'unauthenticated') {
    const destination =
      previousStatusRef.current === 'authenticated'
        ? '/login'
        : loginPathFor(currentReturnTo(location))
    return <Navigate to={destination} replace />
  }

  return <>{children}</>
}
