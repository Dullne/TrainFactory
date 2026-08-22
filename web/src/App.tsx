import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Spin } from 'antd'
import ProtectedRoute from './components/ProtectedRoute'
import PublicOnlyRoute from './components/PublicOnlyRoute'
import AdminRoute from './components/AdminRoute'
import NotFound from './pages/NotFound'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useAuth } from './auth/AuthContext'

// Lazy load page components for code splitting
const MainLayout = lazy(() => import('./components/Layout'))
const Login = lazy(() => import('./pages/auth/Login'))
const TrainingList = lazy(() => import('./pages/training/TrainingList'))
const TrainingCreate = lazy(() => import('./pages/training/TrainingCreate'))
const TrainingDetail = lazy(() => import('./pages/training/TrainingDetail'))
const DatasetHub = lazy(() => import('./pages/datasets/DatasetHub'))
const DatasetCreate = lazy(() => import('./pages/datasets/DatasetCreate'))
const DatasetDetail = lazy(() => import('./pages/datasets/DatasetDetail'))
const DeploymentList = lazy(() => import('./pages/deployments/DeploymentList'))
const ModelList = lazy(() => import('./pages/models/ModelList'))
const ConfigList = lazy(() => import('./pages/configs/ConfigList'))
const ResourceMonitor = lazy(() => import('./pages/resources/ResourceMonitor'))
const EvaluationList = lazy(() => import('./pages/evaluations/EvaluationList'))
const GenerationCreate = lazy(() => import('./pages/generation/GenerationCreate'))
const GenerationDetail = lazy(() => import('./pages/generation/GenerationDetail'))
const GenerationList = lazy(() => import('./pages/generation/GenerationList'))
const VectorDBList = lazy(() => import('./pages/vectordb/VectorDBList'))
const SyncConfigList = lazy(() => import('./pages/sync/SyncConfigList'))
const SyncConfigCreate = lazy(() => import('./pages/sync/SyncConfigCreate'))
const SyncConfigDetail = lazy(() => import('./pages/sync/SyncConfigDetail'))
const UserManagement = lazy(() => import('./pages/admin/UserManagement'))

// Loading fallback component
const PageLoading = () => (
  <div
    style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100%',
      minHeight: 300,
    }}
  >
    <Spin size="large" />
  </div>
)

function DirectStorageRegistrationRoute({ children }: { children: ReactNode }) {
  const { config } = useAuth()

  return config.direct_storage_registration_enabled ? (
    <>{children}</>
  ) : (
    <Navigate to="/datasets" replace />
  )
}

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route
            path="/login"
            element={
              <PublicOnlyRoute>
                <Suspense fallback={<PageLoading />}>
                  <Login />
                </Suspense>
              </PublicOnlyRoute>
            }
          />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Suspense fallback={<PageLoading />}>
                  <MainLayout />
                </Suspense>
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/training" replace />} />
            <Route
              path="training"
              element={
                <Suspense fallback={<PageLoading />}>
                  <TrainingList />
                </Suspense>
              }
            />
            <Route
              path="training/create"
              element={
                <Suspense fallback={<PageLoading />}>
                  <TrainingCreate />
                </Suspense>
              }
            />
            <Route
              path="training/:taskId"
              element={
                <Suspense fallback={<PageLoading />}>
                  <TrainingDetail />
                </Suspense>
              }
            />
            <Route
              path="datasets"
              element={
                <Suspense fallback={<PageLoading />}>
                  <DatasetHub />
                </Suspense>
              }
            />
            <Route
              path="datasets/create"
              element={
                <DirectStorageRegistrationRoute>
                  <Suspense fallback={<PageLoading />}>
                    <DatasetCreate />
                  </Suspense>
                </DirectStorageRegistrationRoute>
              }
            />
            <Route
              path="models"
              element={
                <Suspense fallback={<PageLoading />}>
                  <ModelList />
                </Suspense>
              }
            />
            <Route
              path="deployments"
              element={
                <Suspense fallback={<PageLoading />}>
                  <DeploymentList />
                </Suspense>
              }
            />
            <Route
              path="configs"
              element={
                <Suspense fallback={<PageLoading />}>
                  <ConfigList />
                </Suspense>
              }
            />
            <Route
              path="resources"
              element={
                <Suspense fallback={<PageLoading />}>
                  <ResourceMonitor />
                </Suspense>
              }
            />
            <Route
              path="evaluations"
              element={
                <Suspense fallback={<PageLoading />}>
                  <EvaluationList />
                </Suspense>
              }
            />
            <Route
              path="vectordb"
              element={
                <Suspense fallback={<PageLoading />}>
                  <VectorDBList />
                </Suspense>
              }
            />
            <Route
              path="sync"
              element={
                <Suspense fallback={<PageLoading />}>
                  <SyncConfigList />
                </Suspense>
              }
            />
            <Route
              path="sync/create"
              element={
                <Suspense fallback={<PageLoading />}>
                  <SyncConfigCreate />
                </Suspense>
              }
            />
            <Route
              path="sync/:taskId/edit"
              element={
                <Suspense fallback={<PageLoading />}>
                  <SyncConfigCreate />
                </Suspense>
              }
            />
            <Route
              path="sync/:taskId"
              element={
                <Suspense fallback={<PageLoading />}>
                  <SyncConfigDetail />
                </Suspense>
              }
            />
            <Route
              path="datasets/generation"
              element={
                <Suspense fallback={<PageLoading />}>
                  <GenerationList />
                </Suspense>
              }
            />
            <Route
              path="datasets/generation/create"
              element={
                <Suspense fallback={<PageLoading />}>
                  <GenerationCreate />
                </Suspense>
              }
            />
            <Route
              path="datasets/generation/:taskId"
              element={
                <Suspense fallback={<PageLoading />}>
                  <GenerationDetail />
                </Suspense>
              }
            />
            <Route
              path="datasets/:datasetId"
              element={
                <Suspense fallback={<PageLoading />}>
                  <DatasetDetail />
                </Suspense>
              }
            />
            <Route
              path="admin"
              element={
                <AdminRoute>
                  <Navigate to="/admin/users" replace />
                </AdminRoute>
              }
            />
            <Route
              path="admin/users"
              element={
                <AdminRoute>
                  <Suspense fallback={<PageLoading />}>
                    <UserManagement />
                  </Suspense>
                </AdminRoute>
              }
            />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
