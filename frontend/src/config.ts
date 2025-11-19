export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  clerkPublishableKey: import.meta.env.VITE_CLERK_PUBLISHABLE_KEY || 'pk_test_placeholder',
} as const
