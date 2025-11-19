import { useState } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { UserButton } from '@clerk/clerk-react';
import { Target, History, BarChart3, User, Menu, X } from 'lucide-react';
import InteractiveBackground from '../components/InteractiveBackground';

const navigation = [
  { name: 'Predict', href: '/dashboard/predict', icon: Target },
  { name: 'History', href: '/dashboard/history', icon: History },
  { name: 'Models', href: '/dashboard/models', icon: BarChart3 },
  { name: 'Profile', href: '/dashboard/profile', icon: User },
];

export default function DashboardLayout() {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);
  const closeMobileMenu = () => setIsMobileMenuOpen(false);

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Interactive Particle Network Background */}
      <InteractiveBackground />
      {/* Top Navigation */}
      <nav className="border-b border-white/10 bg-black/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Desktop Nav */}
            <div className="flex items-center space-x-8">
                  <Link to="/" className="flex items-center space-x-2">
                    <span className="text-xl font-black tracking-wider text-white">
                      NETWORK IDS
                    </span>
                  </Link>
              
              <div className="hidden md:flex space-x-1">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium transition-all ${
                        isActive
                          ? 'text-white border-b-2 border-white'
                          : 'text-white/60 hover:text-white'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{item.name}</span>
                    </Link>
                  );
                })}
              </div>
            </div>

            {/* Right Side - User Button and Mobile Menu Toggle */}
            <div className="flex items-center space-x-4">
              <div className="hidden md:block">
                <UserButton 
                  afterSignOutUrl="/"
                  appearance={{
                    elements: {
                      avatarBox: 'w-9 h-9',
                      userButtonAvatarBox: "w-9 h-9",
                      userButtonPopoverCard: "bg-gray-900 border border-white/10 text-white",
                      userButtonPopoverActionButton: "hover:bg-white/10 text-white",
                      userButtonPopoverActionButtonText: "text-white",
                      userButtonPopoverFooter: "border-t border-white/10",
                    },
                  }}
                />
              </div>
              <button
                onClick={toggleMobileMenu}
                className="md:hidden p-2 text-white/60 hover:text-white transition-colors"
                aria-label="Toggle menu"
              >
                {isMobileMenuOpen ? (
                  <X className="w-6 h-6" />
                ) : (
                  <Menu className="w-6 h-6" />
                )}
              </button>
            </div>
          </div>

          {/* Mobile Navigation */}
          {isMobileMenuOpen && (
            <div className="md:hidden border-t border-white/10 py-4">
              <div className="flex flex-col space-y-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      onClick={closeMobileMenu}
                      className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium transition-all ${
                        isActive
                          ? 'text-white bg-white/10'
                          : 'text-white/60 hover:text-white hover:bg-white/5'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{item.name}</span>
                    </Link>
                  );
                })}
                <div className="pt-4 border-t border-white/10 px-4">
                  <UserButton 
                    afterSignOutUrl="/"
                    appearance={{
                      elements: {
                        avatarBox: 'w-9 h-9',
                      },
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main>
        <Outlet />
      </main>
    </div>
  );
}
