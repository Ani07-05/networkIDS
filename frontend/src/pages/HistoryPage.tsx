import { Button } from '../components/ui/button';
import { Download, Search } from 'lucide-react';

export default function HistoryPage() {
  // Data will be fetched from backend API
  const predictions: any[] = [];

  return (
    <div className="min-h-screen text-white p-6 relative">
      <div className="max-w-7xl mx-auto space-y-12 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 pb-8">
          <div>
            <h1 className="text-6xl font-black mb-4 tracking-tight">
              History
            </h1>
            <p className="editorial-caps text-white/60">
              Past Predictions
            </p>
          </div>
          <Button className="bg-white text-black hover:bg-white/90 h-10 px-4">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>

        {/* Search */}
        <div className="border border-white/10 p-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
              <input
                type="text"
                placeholder="Search predictions..."
                className="w-full pl-10 pr-4 py-2 bg-black border border-white/20 text-white placeholder-white/40 focus:outline-none focus:border-white/40 text-sm"
              />
            </div>
          </div>
        </div>

        {/* Stats will be populated from backend API */}

        {/* Predictions Table */}
        <div className="border border-white/10 p-12 text-center">
          <div className="text-white/40 text-sm">
            No prediction history yet. Run your first analysis to see results here.
          </div>
        </div>
      </div>
    </div>
  );
}
