import { CheckCircle } from 'lucide-react';

export default function ModelInfoPage() {
  return (
    <div className="min-h-screen text-white p-6 relative">
      <div className="max-w-7xl mx-auto space-y-12 relative z-10">
        {/* Header */}
        <div className="border-b border-white/10 pb-8">
          <h1 className="text-6xl font-black mb-4 tracking-tight">
            Models
          </h1>
          <p className="editorial-caps text-white/60">
            Architecture & Performance
          </p>
        </div>

        {/* Model Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="border border-white/10 p-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold mb-2 tracking-tight">Binary Model</h2>
              <p className="editorial-caps text-white/60">Normal vs Attack</p>
            </div>
            <div className="space-y-6">
              <div>
                <div className="text-5xl font-bold text-white mb-2">98.12%</div>
                <div className="text-xs text-white/60 uppercase tracking-wider">Test Accuracy</div>
              </div>
              <div className="grid grid-cols-2 gap-6 pt-6 border-t border-white/10">
                <div>
                  <div className="text-2xl font-bold text-white">0.981</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Precision</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">0.983</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Recall</div>
                </div>
              </div>
            </div>
          </div>

          <div className="border border-white/10 p-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold mb-2 tracking-tight">Multi-class Model</h2>
              <p className="editorial-caps text-white/60">Attack Classification</p>
            </div>
            <div className="space-y-6">
              <div>
                <div className="text-5xl font-bold text-white mb-2">97.85%</div>
                <div className="text-xs text-white/60 uppercase tracking-wider">Test Accuracy</div>
              </div>
              <div className="grid grid-cols-2 gap-6 pt-6 border-t border-white/10">
                <div>
                  <div className="text-2xl font-bold text-white">0.976</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Precision</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">0.979</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Recall</div>
                </div>
              </div>
            </div>
          </div>

          <div className="border border-white/10 p-8">
            <div className="mb-6">
              <h2 className="text-2xl font-bold mb-2 tracking-tight">Performance</h2>
              <p className="editorial-caps text-white/60">Inference Speed</p>
            </div>
            <div className="space-y-6">
              <div>
                <div className="text-5xl font-bold text-white mb-2">&lt;50ms</div>
                <div className="text-xs text-white/60 uppercase tracking-wider">Avg Response</div>
              </div>
              <div className="grid grid-cols-2 gap-6 pt-6 border-t border-white/10">
                <div>
                  <div className="text-2xl font-bold text-white">~10MB</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Size</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-white">ONNX</div>
                  <div className="text-xs text-white/60 uppercase tracking-wider">Format</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Architecture Details */}
        <div className="border border-white/10 p-8">
          <h2 className="text-4xl font-bold mb-8 tracking-tight">
            Architecture
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Type</div>
              <div className="text-sm font-medium text-white">Hybrid CNN + Transformer</div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Dataset</div>
              <div className="text-sm font-medium text-white">NSL-KDD (125,973 samples)</div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Features</div>
              <div className="text-sm font-medium text-white">41 Network Features</div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Validation</div>
              <div className="text-sm font-medium text-white">5-Fold Cross-Validation</div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Framework</div>
              <div className="text-sm font-medium text-white">PyTorch 2.4.0</div>
            </div>
            <div className="space-y-2">
              <div className="text-xs text-white/60 uppercase tracking-wider">Format</div>
              <div className="text-sm font-medium text-white">ONNX Runtime</div>
            </div>
          </div>
        </div>

        {/* Attack Types */}
        <div className="border border-white/10 p-8">
          <h2 className="text-4xl font-bold mb-8 tracking-tight">
            Attack Types
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { name: 'DoS', desc: 'Denial of Service' },
              { name: 'Probe', desc: 'Port Scanning' },
              { name: 'R2L', desc: 'Remote to Local' },
              { name: 'U2R', desc: 'User to Root' },
            ].map((attack, i) => (
              <div key={i} className="border border-white/10 p-4">
                <div className="text-lg font-bold text-white mb-1">
                  {attack.name}
                </div>
                <div className="text-xs text-white/60 uppercase tracking-wider">{attack.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="border border-white/10 p-8">
          <h2 className="text-4xl font-bold mb-8 tracking-tight">
            Technical Specifications
          </h2>
          <div className="space-y-4">
            {[
              { label: 'Model Layers', value: '12 CNN layers + 8 Transformer blocks' },
              { label: 'Attention Mechanism', value: 'Multi-head self-attention (8 heads)' },
              { label: 'Optimizer', value: 'Adam with learning rate scheduling' },
              { label: 'Loss Function', value: 'Weighted Cross-Entropy' },
              { label: 'Regularization', value: 'Dropout (0.3), L2 weight decay' },
              { label: 'Batch Size', value: '128 samples' },
              { label: 'Training Epochs', value: '100 (early stopping enabled)' },
              { label: 'Hardware', value: 'NVIDIA GPU (CUDA optimized)' },
            ].map((spec, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-3 border-b border-white/10 last:border-b-0"
              >
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-4 h-4 text-white" />
                  <span className="text-sm text-white">{spec.label}</span>
                </div>
                <span className="text-sm text-white/60">{spec.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
