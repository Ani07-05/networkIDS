import { SignedIn, SignedOut, RedirectToSignIn } from '@clerk/clerk-react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useAuthToken } from './hooks/useAuthToken'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ToastProvider } from './components/Toast'

// Pages
import LandingPage from './pages/LandingPage'
import SignInPage from './pages/SignInPage'
import SignUpPage from './pages/SignUpPage'
import DashboardLayout from './layouts/DashboardLayout'
import PredictPage from './pages/PredictPage'
import HistoryPage from './pages/HistoryPage'
import ModelInfoPage from './pages/ModelInfoPage'
import ProfilePage from './pages/ProfilePage'

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

// Protected Route Component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  return (
    <>
      <SignedIn>{children}</SignedIn>
      <SignedOut>
        <RedirectToSignIn />
      </SignedOut>
    </>
  )
}

// Auth token handler component
function AuthTokenHandler() {
  useAuthToken()
  return null
}

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ToastProvider>
          <AuthTokenHandler />
          <BrowserRouter>
            <Routes>
              {/* Public routes */}
              <Route path="/" element={<LandingPage />} />
              <Route path="/sign-in/*" element={<SignInPage />} />
              <Route path="/sign-up/*" element={<SignUpPage />} />

              {/* Protected routes */}
              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <DashboardLayout />
                  </ProtectedRoute>
                }
              >
                <Route index element={<Navigate to="/dashboard/predict" replace />} />
                <Route path="predict" element={<PredictPage />} />
                <Route path="history" element={<HistoryPage />} />
                <Route path="models" element={<ModelInfoPage />} />
                <Route path="profile" element={<ProfilePage />} />
              </Route>

              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </ToastProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App
