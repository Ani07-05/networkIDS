import { Link } from 'react-router-dom';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { ArrowRight } from 'lucide-react';
import { Button } from '../components/ui/button';
import InteractiveBackground from '../components/InteractiveBackground';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Interactive Particle Network Background */}
      <InteractiveBackground />
      
      {/* Dotted Grid Overlay */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* Grid dots */}
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: 'radial-gradient(circle, rgba(255, 255, 255, 0.3) 1px, transparent 1px)',
            backgroundSize: '32px 32px',
          }}
        />
        {/* Fade from top */}
        <div 
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(to bottom, black 0%, transparent 20%, transparent 80%, black 100%)',
          }}
        />
        {/* Fade from sides */}
        <div 
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(ellipse at center, transparent 0%, transparent 50%, black 100%)',
          }}
        />
      </div>

      {/* Navigation */}
      <nav className="border-b border-white/10 relative z-10">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <Link to="/">
              <span className="text-xl font-black tracking-wider text-white">
                NETWORK IDS
              </span>
            </Link>
            
            <div className="flex items-center gap-6">
              <SignedOut>
                <Link to="/sign-in" className="text-sm text-white/70 hover:text-white transition-colors">
                  Sign In
                </Link>
                <Link to="/sign-up">
                  <Button className="bg-white text-black hover:bg-white/90 h-9 px-4 text-sm">
                    Get Started
                  </Button>
                </Link>
              </SignedOut>
              <SignedIn>
                <Link to="/dashboard/predict">
                  <Button className="bg-white text-black hover:bg-white/90 h-9 px-4 text-sm">
                    Dashboard
                  </Button>
                </Link>
                <UserButton afterSignOutUrl="/" />
              </SignedIn>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10">
        {/* Hero */}
        <section className="container mx-auto px-6 pt-32 pb-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h1 className="text-7xl md:text-8xl font-black text-white mb-8 leading-[0.9] tracking-tight">
                Network
                <br />
                Intrusion
                <br />
                Detection
              </h1>
              <p className="editorial-lead text-white/70 mb-10 max-w-xl">
                Deep learning model for real-time network threat detection.
                Built with PyTorch. Trained on NSL-KDD dataset.
              </p>
              <SignedOut>
                <Link to="/sign-up">
                  <Button className="bg-white text-black hover:bg-white/90 h-11 px-6">
                    Try it now
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </SignedOut>
              <SignedIn>
                <Link to="/dashboard/predict">
                  <Button className="bg-white text-black hover:bg-white/90 h-11 px-6">
                    Open Dashboard
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </Link>
              </SignedIn>
            </div>

            {/* Network Visualization */}
            <div className="hidden lg:block relative h-[500px]">
              <svg width="100%" height="100%" viewBox="0 0 400 500" className="absolute inset-0">
                {/* Connection lines */}
                <line x1="200" y1="250" x2="100" y2="100" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="3s" repeatCount="indefinite" />
                </line>
                <line x1="200" y1="250" x2="300" y2="100" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="3.5s" repeatCount="indefinite" />
                </line>
                <line x1="200" y1="250" x2="100" y2="400" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="2.5s" repeatCount="indefinite" />
                </line>
                <line x1="200" y1="250" x2="300" y2="400" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="4s" repeatCount="indefinite" />
                </line>
                <line x1="200" y1="250" x2="50" y2="250" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="3.2s" repeatCount="indefinite" />
                </line>
                <line x1="200" y1="250" x2="350" y2="250" stroke="rgba(255,255,255,0.1)" strokeWidth="1">
                  <animate attributeName="stroke" values="rgba(255,255,255,0.1);rgba(255,255,255,0.3);rgba(255,255,255,0.1)" dur="2.8s" repeatCount="indefinite" />
                </line>

                {/* Central node (server) */}
                <circle cx="200" cy="250" r="8" fill="white" opacity="0.9" />
                <circle cx="200" cy="250" r="16" fill="none" stroke="white" strokeWidth="1" opacity="0.3">
                  <animate attributeName="r" values="16;24;16" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.3;0;0.3" dur="2s" repeatCount="indefinite" />
                </circle>

                {/* Peripheral nodes */}
                <circle cx="100" cy="100" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="3s" repeatCount="indefinite" />
                </circle>
                <circle cx="300" cy="100" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="3.5s" repeatCount="indefinite" />
                </circle>
                <circle cx="100" cy="400" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite" />
                </circle>
                <circle cx="300" cy="400" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="4s" repeatCount="indefinite" />
                </circle>
                <circle cx="50" cy="250" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="3.2s" repeatCount="indefinite" />
                </circle>
                <circle cx="350" cy="250" r="6" fill="white" opacity="0.6">
                  <animate attributeName="opacity" values="0.6;1;0.6" dur="2.8s" repeatCount="indefinite" />
                </circle>

                {/* Data packets moving along lines */}
                <circle r="3" fill="white" opacity="0.8">
                  <animateMotion dur="3s" repeatCount="indefinite">
                    <mpath href="#path1" />
                  </animateMotion>
                </circle>
                <path id="path1" d="M 100,100 L 200,250" fill="none" stroke="none" />
                
                <circle r="3" fill="white" opacity="0.8">
                  <animateMotion dur="3.5s" repeatCount="indefinite">
                    <mpath href="#path2" />
                  </animateMotion>
                </circle>
                <path id="path2" d="M 300,100 L 200,250" fill="none" stroke="none" />
              </svg>
            </div>
          </div>
        </section>

        {/* Stats */}
        <section className="container mx-auto px-6 py-20 border-t border-white/10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
            <div>
              <div className="text-6xl font-black text-white mb-3 tracking-tight">
                98.2%
              </div>
              <div className="editorial-caps text-white/50">Detection accuracy</div>
            </div>
            <div>
              <div className="text-6xl font-black text-white mb-3 tracking-tight">
                &lt;50ms
              </div>
              <div className="editorial-caps text-white/50">Response time</div>
            </div>
            <div>
              <div className="text-6xl font-black text-white mb-3 tracking-tight">
                5
              </div>
              <div className="editorial-caps text-white/50">Attack categories</div>
            </div>
            <div>
              <div className="text-6xl font-black text-white mb-3 tracking-tight">
                41
              </div>
              <div className="editorial-caps text-white/50">Network features</div>
            </div>
          </div>
        </section>

        {/* Tech */}
        <section className="container mx-auto px-6 py-20 border-t border-white/10">
          <h2 className="editorial-caps text-white/50 mb-12">Technology Stack</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
            <div>
              <div className="text-2xl font-bold text-white mb-2 italic">PyTorch</div>
              <div className="text-sm text-white/40">ML Framework</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white mb-2 italic">FastAPI</div>
              <div className="text-sm text-white/40">Backend</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white mb-2 italic">React</div>
              <div className="text-sm text-white/40">Frontend</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-white mb-2 italic">PostgreSQL</div>
              <div className="text-sm text-white/40">Database</div>
            </div>
          </div>
        </section>

        {/* Architecture */}
        <section className="container mx-auto px-6 py-20 border-t border-white/10">
          <div className="max-w-3xl">
            <h2 className="editorial-caps text-white/50 mb-8">Model Architecture</h2>
            <h3 className="text-5xl font-bold text-white mb-8 leading-tight tracking-tight">
              Hybrid CNN-Transformer Architecture
            </h3>
            <div className="space-y-6 text-white/60 editorial-lead">
              <p>
                Combines convolutional layers for local pattern extraction with 
                transformer attention mechanisms for long-range dependencies in network traffic.
              </p>
              <p>
                Binary classification (normal/attack) and multi-class detection 
                (DoS, Probe, R2L, U2R) with cross-validation and early stopping.
              </p>
              <p>
                Deployed as ONNX runtime for optimized inference. 
                Trained on NSL-KDD benchmark dataset.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="container mx-auto px-6 py-12 border-t border-white/10">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-white/40">
              © 2025 Network IDS
            </div>
            <div className="text-sm text-white/40">
              React · FastAPI · PyTorch
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
