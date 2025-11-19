import { createBrowserRouter, Navigate } from 'react-router-dom';
import { SignIn, SignUp, useUser } from '@clerk/clerk-react';
import App from './App';
import LandingPage from './pages/LandingPage';
import DashboardPage from './pages/DashboardPage';
import PredictionPage from './pages/PredictionPage';
import HistoryPage from './pages/HistoryPage';
import ModelInfoPage from './pages/ModelInfoPage';
import ProfilePage from './pages/ProfilePage';

// Protected Route wrapper component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isSignedIn, isLoaded } = useUser();

  if (!isLoaded) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
  }

  if (!isSignedIn) {
    return <Navigate to="/sign-in" replace />;
  }

  return <>{children}</>;
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      {
        index: true,
        element: <LandingPage />,
      },
      {
        path: 'sign-in/*',
        element: <SignIn routing="path" path="/sign-in" />,
      },
      {
        path: 'sign-up/*',
        element: <SignUp routing="path" path="/sign-up" />,
      },
      {
        path: 'dashboard',
        element: (
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        ),
      },
      {
        path: 'predict',
        element: (
          <ProtectedRoute>
            <PredictionPage />
          </ProtectedRoute>
        ),
      },
      {
        path: 'history',
        element: (
          <ProtectedRoute>
            <HistoryPage />
          </ProtectedRoute>
        ),
      },
      {
        path: 'models',
        element: (
          <ProtectedRoute>
            <ModelInfoPage />
          </ProtectedRoute>
        ),
      },
      {
        path: 'profile',
        element: (
          <ProtectedRoute>
            <ProfilePage />
          </ProtectedRoute>
        ),
      },
    ],
  },
]);


