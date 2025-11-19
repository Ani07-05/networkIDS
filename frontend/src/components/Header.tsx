import { SignedIn, SignedOut, UserButton } from '../lib/clerk-utils';
import { Link } from 'react-router-dom';

export default function Header() {
  return (
    <header className="border-b bg-white">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <Link to="/" className="text-2xl font-bold text-gray-900">
              Network IDS
            </Link>
            <SignedIn>
              <nav className="hidden md:flex space-x-6">
                <Link to="/dashboard" className="text-gray-700 hover:text-gray-900">
                  Dashboard
                </Link>
                <Link to="/predict" className="text-gray-700 hover:text-gray-900">
                  Predict
                </Link>
                <Link to="/history" className="text-gray-700 hover:text-gray-900">
                  History
                </Link>
                <Link to="/models" className="text-gray-700 hover:text-gray-900">
                  Models
                </Link>
              </nav>
            </SignedIn>
          </div>

          <div className="flex items-center space-x-4">
            <SignedOut>
              <Link
                to="/sign-in"
                className="text-gray-700 hover:text-gray-900"
              >
                Sign In
              </Link>
              <Link
                to="/sign-up"
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              >
                Sign Up
              </Link>
            </SignedOut>
            <SignedIn>
              <Link
                to="/profile"
                className="text-gray-700 hover:text-gray-900"
              >
                Profile
              </Link>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </div>
      </div>
    </header>
  );
}

