import { UserProfile } from '@clerk/clerk-react';

export default function ProfilePage() {
  return (
    <div className="min-h-screen text-white p-6 relative">
      <div className="max-w-7xl mx-auto space-y-12 relative z-10">
        {/* Header */}
        <div className="border-b border-white/10 pb-8">
          <h1 className="text-6xl font-black mb-4 tracking-tight">
            Profile
          </h1>
          <p className="editorial-caps text-white/60">
            Account Settings
          </p>
        </div>

        {/* Stats will be populated from backend API */}

        {/* Clerk User Profile */}
        <div className="border border-white/10 p-8">
          <h2 className="text-4xl font-bold mb-8 tracking-tight">
            Settings
          </h2>
          <UserProfile
            appearance={{
              elements: {
                rootBox: 'w-full',
                card: 'bg-transparent shadow-none border-0',
                navbar: 'bg-black border border-white/10',
                navbarButton: 'text-white/60 hover:text-white hover:bg-white/5',
                navbarButtonActive: 'text-white bg-white/10 border-l-2 border-white',
                pageScrollBox: 'bg-transparent',
                page: 'bg-transparent',
                profileSection: 'bg-black border border-white/10',
                profileSectionTitle: 'text-white',
                profileSectionContent: 'text-white/80',
                formFieldLabel: 'text-white/60 text-xs uppercase tracking-wider',
                formFieldInput: 'bg-black border border-white/20 text-white focus:border-white/40',
                formButtonPrimary: 'bg-white text-black hover:bg-white/90',
                badge: 'bg-white/10 text-white border-white/20',
                accordionTriggerButton: 'text-white hover:bg-white/5',
                accordionContent: 'text-white/80',
                formFieldInputShowPasswordButton: 'text-white/60 hover:text-white',
              },
            }}
          />
        </div>
      </div>
    </div>
  );
}
