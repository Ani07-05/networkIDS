import { useAuth } from '@clerk/clerk-react'
import { useEffect } from 'react'
import { setAuthToken } from '@/lib/api'

/**
 * Custom hook to automatically inject Clerk JWT token into API requests.
 * Uses the "backend" JWT template so the FastAPI service can validate signatures
 * with the configured CLERK_JWT_KEY.
 */
export function useAuthToken() {
  const { getToken, isSignedIn } = useAuth()

  useEffect(() => {
    let isMounted = true

    const updateToken = async () => {
      if (!isSignedIn) {
        setAuthToken(null)
        return
      }

      try {
        const token = await getToken()
        if (isMounted) {
          setAuthToken(token ?? null)
        }
      } catch (error) {
        console.error('Error setting auth token:', error)
        if (isMounted) {
          setAuthToken(null)
        }
      }
    }

    updateToken()

    const refreshInterval = setInterval(updateToken, 5 * 60 * 1000)

    return () => {
      isMounted = false
      clearInterval(refreshInterval)
    }
  }, [getToken, isSignedIn])
}