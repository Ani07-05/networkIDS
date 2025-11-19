import { useState, useRef, useEffect } from 'react';
import { Button } from '../components/ui/button';
import { Upload, CheckCircle, FileText, X, AlertCircle, Activity, Shield, ShieldAlert, Download, History, Server } from 'lucide-react';
import { useToast } from '../components/Toast';
import api from '../lib/api';
import { parseCSV, SAMPLE_FEATURE } from '../lib/analysis-utils';
import type { PredictionResponse, BatchPredictionResponse } from '../types/api';

export default function PredictPage() {
  const { showToast } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [historyData, setHistoryData] = useState<any>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [modelLogs, setModelLogs] = useState<string[]>([]);
  const [modelStatus, setModelStatus] = useState({ binary: true, multiclass: true, preprocessor: true });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setModelLogs(prev => [...prev, `[${timestamp}] ${message}`].slice(-100));
  };

  const fetchHistory = async () => {
    try {
      const response = await api.get('/api/predictions/history?page=1&page_size=10');
      setHistoryData(response.data);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      // Silently fail if not authenticated yet
    }
  };

  const handleQuickPredict = async () => {
    setIsLoading(true);
    setResult(null);
    setBatchResult(null);
    addLog('Starting single prediction analysis');
    try {
      addLog('Sending features to ML pipeline');
      const response = await api.post<PredictionResponse>('/api/predictions/predict', {
        features: SAMPLE_FEATURE
      });
      setResult(response.data);
      addLog(`Prediction complete: ${response.data.attack_type} (confidence: ${(response.data.multiclass_confidence * 100).toFixed(1)}%)`);
      addLog(`Inference time: ${response.data.inference_time_ms.toFixed(2)}ms`);
      addLog('Saved to database');
      showToast('success', '✓ Analysis complete & saved to history');
      fetchHistory();
    } catch (error) {
      console.error('Prediction error:', error);
      addLog('ERROR: Prediction failed');
      showToast('error', 'Failed to run analysis');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const validateAndSetFile = (file: File) => {
    // Validate file type
    if (!file.type.includes('csv') && !file.name.endsWith('.csv')) {
      showToast('error', 'Please select a CSV file');
      return;
    }
    
    // Validate file size (must be > 0 and < 10MB)
    if (file.size === 0) {
      showToast('error', 'File is empty. Please select a valid CSV file');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      showToast('error', 'File is too large. Maximum size is 10MB');
      return;
    }
    
    setSelectedFile(file);
    showToast('success', `File "${file.name}" selected`);
    // Reset previous results
    setResult(null);
    setBatchResult(null);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setBatchResult(null);
  };

  const handleBatchPredict = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    setResult(null);
    setBatchResult(null);
    addLog(`Starting batch analysis: ${selectedFile.name}`);
    
    try {
      const text = await selectedFile.text();
      addLog(`File loaded: ${(text.length / 1024).toFixed(1)}KB`);
      
      const features = parseCSV(text);
      addLog(`Parsed ${features.length} valid records`);
      
      if (features.length === 0) {
        addLog('ERROR: No valid records found in CSV');
        showToast('error', 'No valid data found in CSV. Check browser console for details.');
        return;
      }
      
      const processCount = Math.min(features.length, 1000);
      if (features.length > 1000) {
        addLog(`Processing first ${processCount} of ${features.length} records`);
        showToast('warning', `Found ${features.length} records - processing first 1000`);
      }

      addLog('Sending batch to ML pipeline');
      const response = await api.post<BatchPredictionResponse>('/api/predictions/predict/batch', {
        predictions: features.slice(0, 1000)
      });
      
      setBatchResult(response.data);
      const attacks = response.data.predictions.filter(p => p.is_attack).length;
      const normal = response.data.predictions.filter(p => !p.is_attack).length;
      addLog(`Batch complete: ${attacks} attacks, ${normal} normal (${response.data.processing_time_ms.toFixed(0)}ms total)`);
      addLog(`Average latency: ${(response.data.processing_time_ms / response.data.total).toFixed(2)}ms per prediction`);
      addLog('Saved to database');
      showToast('success', `✓ Processed ${response.data.total} records & saved to history`);
      fetchHistory();

      
    } catch (error) {
      console.error('Batch prediction error:', error);
      addLog('ERROR: Batch prediction failed');
      showToast('error', 'Failed to process file');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResults = () => {
    if (!batchResult && !result) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    
    if (batchResult) {
      // Header
      csvContent += "ID,Status,Type,Confidence,Inference Time (ms)\n";
      // Rows
      batchResult.predictions.forEach(pred => {
        const row = [
          pred.id,
          pred.is_attack ? "ATTACK" : "NORMAL",
          pred.attack_type,
          pred.multiclass_confidence,
          pred.inference_time_ms
        ].join(",");
        csvContent += row + "\n";
      });
    } else if (result) {
       // Header
       csvContent += "ID,Status,Type,Confidence,Inference Time (ms)\n";
       // Row
       const row = [
         result.id,
         result.is_attack ? "ATTACK" : "NORMAL",
         result.attack_type,
         result.multiclass_confidence,
         result.inference_time_ms
       ].join(",");
       csvContent += row + "\n";
    }

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `prediction_results_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen text-white p-6 relative">
      <div className="max-w-7xl mx-auto space-y-12 relative z-10">
        {/* Header */}
        <div className="border-b border-white/10 pb-8">
          <h1 className="text-6xl font-black mb-4 tracking-tight">
            Predict
          </h1>
          <p className="editorial-caps text-white/60">
            Network Traffic Analysis
          </p>
        </div>

        {/* Main Analysis Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Single Prediction */}
            <div className="border border-white/10 p-8 flex flex-col h-full">
              <div className="mb-8">
                <h2 className="text-3xl font-bold mb-3 tracking-tight">
                  Single Prediction
                </h2>
                <p className="text-white/60">
                  Analyze individual network traffic patterns
                </p>
              </div>

            <div className="space-y-6 flex-grow">
              <div className="border border-white/10 p-4">
                <div className="text-xs text-white/60 uppercase tracking-wider mb-2">Quick Analysis</div>
                <p className="text-sm text-white/80">
                  Analyze sample network traffic with pre-configured features
                </p>
              </div>

              <Button
                onClick={handleQuickPredict}
                disabled={isLoading}
                className="w-full h-12 bg-white text-black hover:bg-white/90 font-medium disabled:opacity-50"
              >
                {isLoading && !selectedFile ? 'Analyzing...' : 'Run Analysis'}
              </Button>

              <div className="text-xs text-center text-white/40 uppercase tracking-wider mt-auto">
                Hybrid CNN + Transformer Model
              </div>
            </div>
          </div>

          {/* Batch Upload */}
          <div className="border border-white/10 p-8 flex flex-col h-full">
            <div className="mb-8">
              <h2 className="text-3xl font-bold mb-3 tracking-tight">
                Batch Analysis
              </h2>
              <p className="text-white/60">
                Upload CSV files for bulk traffic analysis
              </p>
            </div>

            <div className="space-y-6 flex-grow">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={handleFileSelect}
              />

              <div
                className={`border-2 border-dashed p-16 text-center cursor-pointer transition-all duration-300 ${
                  isDragging
                    ? 'border-white bg-white/5 scale-105'
                    : selectedFile
                    ? 'border-white/40 bg-white/5'
                    : 'border-white/20 hover:border-white/40 hover:bg-white/5'
                }`}
                onClick={handleUploadClick}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <div className="flex flex-col items-center space-y-4">
                  {selectedFile ? (
                    <>
                      <FileText className="w-12 h-12 text-white animate-bounce" />
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-white">
                          {selectedFile.name}
                        </div>
                        <div className="text-xs text-white/60">
                          {(selectedFile.size / 1024).toFixed(2)} KB
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          className="mt-2 border-white/20 text-white hover:bg-white/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveFile();
                          }}
                        >
                          <X className="w-3 h-3 mr-1" />
                          Remove
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      <Upload className={`w-12 h-12 text-white/60 transition-transform duration-300 ${isDragging ? 'scale-110' : ''}`} />
                      <div>
                        <div className="text-sm font-medium text-white mb-1">
                          Drop CSV file here
                        </div>
                        <div className="text-xs text-white/60">
                          or click to browse
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {selectedFile && (
                <Button
                  className="w-full bg-white text-black hover:bg-white/90"
                  disabled={!selectedFile || selectedFile.size === 0 || isLoading}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleBatchPredict();
                  }}
                >
                   {isLoading ? 'Processing...' : `Analyze ${selectedFile.name}`}
                </Button>
              )}

                  <Button
                    variant="outline"
                    className="w-full border-white/20 bg-transparent text-white hover:bg-white/10 hover:text-white"
                    onClick={() => {
                      // Download the sample CSV file
                      const link = document.createElement('a');
                      link.href = '/sample_test.csv';
                      link.download = 'sample_test.csv';
                      link.click();
                      showToast('success', 'Sample CSV downloaded! Check your Downloads folder');
                    }}
                  >
                    Download Sample CSV
                  </Button>
            </div>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className="border border-white/10 p-8 animate-in fade-in slide-in-from-bottom-4">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-2xl font-bold">Analysis Result</h3>
                <span className="text-xs text-white/40 font-mono">ID: {result.id}</span>
              </div>
              <Button
                onClick={downloadResults}
                variant="outline"
                className="border-white/20 bg-white/5 text-white hover:bg-white/10"
              >
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className={`p-6 border ${result.is_attack ? 'border-red-500/50 bg-red-500/10' : 'border-green-500/50 bg-green-500/10'}`}>
                <div className="flex items-center gap-3 mb-2">
                  {result.is_attack ? <ShieldAlert className="text-red-500" /> : <Shield className="text-green-500" />}
                  <span className="uppercase tracking-wider text-sm font-bold">Status</span>
                </div>
                <div className="text-4xl font-black">
                  {result.is_attack ? 'THREAT DETECTED' : 'NORMAL TRAFFIC'}
                </div>
              </div>
              
              <div className="p-6 border border-white/10 bg-white/5">
                <div className="flex items-center gap-3 mb-2">
                  <Activity className="text-blue-400" />
                  <span className="uppercase tracking-wider text-sm font-bold">Classification</span>
                </div>
                <div className="text-4xl font-black text-blue-400">
                  {result.attack_type}
                </div>
                <div className="text-sm text-white/60 mt-2">
                  Confidence: {(result.multiclass_confidence * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="p-6 border border-white/10 bg-white/5">
                <div className="flex items-center gap-3 mb-2">
                  <AlertCircle className="text-yellow-400" />
                  <span className="uppercase tracking-wider text-sm font-bold">Inference Time</span>
                </div>
                <div className="text-4xl font-black text-yellow-400">
                  {result.inference_time_ms.toFixed(2)}ms
                </div>
                <div className="text-sm text-white/60 mt-2">
                  Processing latency
                </div>
              </div>
            </div>
          </div>
        )}

        {batchResult && (
          <div className="border border-white/10 p-8 animate-in fade-in slide-in-from-bottom-4">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-2xl font-bold">Batch Analysis Results</h3>
                <span className="text-xs text-white/40 font-mono">Total: {batchResult.total} records</span>
              </div>
              <Button
                onClick={downloadResults}
                variant="outline"
                className="border-white/20 bg-white/5 text-white hover:bg-white/10"
              >
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </Button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
               {/* Summary Stats */}
               <div className="p-4 border border-white/10 bg-white/5">
                 <div className="text-sm text-white/60 uppercase">Attacks Found</div>
                 <div className="text-3xl font-bold text-red-400">
                   {batchResult.predictions.filter(p => p.is_attack).length}
                 </div>
               </div>
               <div className="p-4 border border-white/10 bg-white/5">
                 <div className="text-sm text-white/60 uppercase">Normal Traffic</div>
                 <div className="text-3xl font-bold text-green-400">
                   {batchResult.predictions.filter(p => !p.is_attack).length}
                 </div>
               </div>
               <div className="p-4 border border-white/10 bg-white/5">
                 <div className="text-sm text-white/60 uppercase">Processing Time</div>
                 <div className="text-3xl font-bold text-yellow-400">
                   {batchResult.processing_time_ms.toFixed(0)}ms
                 </div>
               </div>
               <div className="p-4 border border-white/10 bg-white/5">
                 <div className="text-sm text-white/60 uppercase">Avg Latency</div>
                 <div className="text-3xl font-bold text-blue-400">
                   {(batchResult.processing_time_ms / batchResult.total).toFixed(2)}ms
                 </div>
               </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-white/40 uppercase bg-white/5">
                  <tr>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Type</th>
                    <th className="px-4 py-3">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {batchResult.predictions.slice(0, 10).map((pred) => (
                    <tr key={pred.id} className="hover:bg-white/5">
                      <td className="px-4 py-3 font-mono text-xs text-white/60">{pred.id.slice(0,8)}...</td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                          pred.is_attack ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                        }`}>
                          {pred.is_attack ? 'ATTACK' : 'NORMAL'}
                        </span>
                      </td>
                      <td className="px-4 py-3 uppercase">{pred.attack_type}</td>
                      <td className="px-4 py-3">{(pred.multiclass_confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {batchResult.predictions.length > 10 && (
                 <div className="text-center text-xs text-white/40 py-4">
                   Showing first 10 of {batchResult.total} results
                 </div>
              )}
            </div>
          </div>
        )}

        {/* Model Logs, Status & History */}
        <div className="border-t border-white/10 pt-8 space-y-8">
          {/* Model Logs */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Model Logs</h3>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setModelLogs([])}
                  className="text-white border-white/20"
                  disabled={modelLogs.length === 0}
                >
                  Clear Logs
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowLogs(!showLogs)}
                  className="text-white border-white/20"
                >
                  {showLogs ? 'Hide' : 'Show'} Logs
                </Button>
              </div>
            </div>
            
            {showLogs && (
              <div className="bg-black/40 border border-white/10 rounded-lg p-4 h-64 overflow-y-auto font-mono text-xs">
                {modelLogs.length === 0 ? (
                  <div className="text-white/40 text-center py-8">No logs yet. Run a prediction to see activity.</div>
                ) : (
                  <div className="space-y-1">
                    {modelLogs.map((log, idx) => (
                      <div 
                        key={idx} 
                        className={`${
                          log.includes('ERROR') ? 'text-red-400' : 
                          log.includes('complete') || log.includes('Saved') ? 'text-green-400' : 
                          'text-white/70'
                        }`}
                      >
                        {log}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Model Status */}
          <div>
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Server className="w-5 h-5" />
              Model Status
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-white/10 p-4 flex items-center justify-between">
                <span className="text-sm text-white/80">Binary Classifier</span>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="border border-white/10 p-4 flex items-center justify-between">
                <span className="text-sm text-white/80">Multiclass Classifier</span>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="border border-white/10 p-4 flex items-center justify-between">
                <span className="text-sm text-white/80">Preprocessor</span>
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
            </div>
            <div className="mt-4 text-xs text-white/40 uppercase tracking-wider text-center">
              NSL-KDD Dataset • ONNX Runtime • v1.0.0
            </div>
          </div>

          {/* History Section */}
          <div>
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold flex items-center gap-2">
                <History className="w-5 h-5" />
                Recent Predictions
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (!showHistory) fetchHistory(); // Fetch when showing
                  setShowHistory(!showHistory);
                }}
                className="text-white border-white/20"
              >
                {showHistory ? 'Hide' : 'Show'} History
              </Button>
            </div>

            {showHistory && historyData && (
              <div className="border border-white/10 p-6 space-y-4">
                <div className="text-sm text-white/60 mb-4">
                  Total Saved: {historyData.total} predictions
                </div>
                
                {historyData.predictions.length === 0 ? (
                  <div className="text-center py-8 text-white/40">
                    No predictions saved yet. Run an analysis to get started.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {historyData.predictions.map((pred: any) => (
                      <div
                        key={pred.id}
                        className="border border-white/10 p-4 flex items-center justify-between hover:border-white/20 transition-colors"
                      >
                        <div className="flex items-center gap-4">
                          {pred.is_attack ? (
                            <ShieldAlert className="w-5 h-5 text-red-500" />
                          ) : (
                            <Shield className="w-5 h-5 text-green-500" />
                          )}
                          <div>
                            <div className="font-medium">
                              {pred.attack_type}
                            </div>
                            <div className="text-xs text-white/40">
                              {new Date(pred.created_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm">
                            {(pred.multiclass_confidence * 100).toFixed(1)}%
                          </div>
                          <div className="text-xs text-white/40">
                            {pred.inference_time_ms.toFixed(1)}ms
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
