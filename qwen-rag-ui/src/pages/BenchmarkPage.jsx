import React, { useState, useRef } from 'react';
import { chatWithBot } from '../api';
import { Upload, Play, Clock, Zap, Download } from 'lucide-react';

const BenchmarkPage = () => {
  const [questions, setQuestions] = useState([]);
  const [results, setResults] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [concurrency, setConcurrency] = useState(1); 

  const progressRef = useRef(0);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
      setQuestions(lines);
      setResults([]); 
      setProgress({ current: 0, total: lines.length });
    };
    reader.readAsText(file);
  };

  const handleDownloadCSV = () => {
    if (results.length === 0) return;

    const headers = [
      "ID", "Query", "Status", "Response", "Sources", "Safety", 
      "LLM Model", "Embed Model", "Rerank Model", "Guard Model",
      "Server Total (s)", "LLM Gen (s)", "Retrieval (s)", "Rerank (s)", "Embed (s)", 
      "Input Guard (s)", "Output Guard (s)", "Client Latency (s)"
    ];

    const csvRows = results.map(row => {
      const safeQuery = `"${(row.query || "").replace(/"/g, '""')}"`;
      const safeResponse = `"${(row.response || "").replace(/"/g, '""')}"`;
      const safeSources = `"${(row.sources || "").replace(/"/g, '""')}"`;

      return [
        row.id,
        safeQuery,
        row.status,
        safeResponse,
        safeSources,
        row.safety,
        row.llmModel, 
        row.embedModel, 
        row.rerankModel, // Added
        row.guardModel, 
        row.serverTotal,
        row.llmTime,
        row.retrievalTime,
        row.rerankTime, // Added
        row.embedTime,
        row.inputGuardTime,
        row.outputGuardTime,
        row.clientLatency
      ].join(",");
    });

    const csvContent = [headers.join(","), ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `benchmark_results_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const runBenchmark = async () => {
    if (questions.length === 0) return;
    
    setProcessing(true);
    setResults([]);
    progressRef.current = 0;
    setProgress({ current: 0, total: questions.length });

    let globalIndex = 0;

    const worker = async () => {
      while (true) {
        const i = globalIndex++; 
        if (i >= questions.length) break;

        const query = questions[i];
        const clientStart = performance.now();

        try {
          const response = await chatWithBot(query);
          const clientEnd = performance.now();
          const clientLatency = (clientEnd - clientStart) / 1000;

          const data = response.data;
          
          const row = {
            id: i + 1, 
            query: query,
            status: "Success",
            response: data.response,
            sources: data.sources.join(", "),
            safety: data.safety_check,
            
            llmModel: data.llm_model,
            embedModel: data.embed_model,
            rerankModel: data.rerank_model, // Capture Rerank Model
            guardModel: data.guard_model,
            
            serverTotal: data.timings.total_sec,
            llmTime: data.timings.llm_generation_sec,
            retrievalTime: data.timings.retrieval_sec,
            rerankTime: data.timings.reranking_sec || 0, // Capture Rerank Time
            embedTime: data.timings.embedding_sec,
            inputGuardTime: data.timings.input_guardrail_sec,
            outputGuardTime: data.timings.output_guardrail_sec,
            clientLatency: clientLatency.toFixed(4)
          };

          setResults(prev => {
            const newList = [...prev, row];
            return newList.sort((a, b) => a.id - b.id);
          });

        } catch (error) {
          const clientEnd = performance.now();
          const row = {
            id: i + 1,
            query: query,
            status: "Failed",
            response: "Error: " + error.message,
            sources: "-",
            safety: "-",
            llmModel: "-", embedModel: "-", rerankModel: "-", guardModel: "-",
            serverTotal: 0,
            llmTime: 0,
            retrievalTime: 0,
            rerankTime: 0,
            embedTime: 0,
            inputGuardTime: 0,
            outputGuardTime: 0,
            clientLatency: ((clientEnd - clientStart) / 1000).toFixed(4)
          };
          
          setResults(prev => {
            const newList = [...prev, row];
            return newList.sort((a, b) => a.id - b.id);
          });
        }

        progressRef.current += 1;
        setProgress({ current: progressRef.current, total: questions.length });
      }
    };

    const threads = Array.from({ length: concurrency }).map(() => worker());
    await Promise.all(threads);
    setProcessing(false);
  };

  const getStatusColor = (status) => status === "Success" ? "text-green-600 bg-green-50" : "text-red-600 bg-red-50";

  return (
    <div className="p-6 max-w-[95%] mx-auto h-screen flex flex-col">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Performance Benchmark</h1>
          <p className="text-slate-500">Upload a list of questions to analyze latency breakdown.</p>
        </div>

        {/* Action Bar */}
        <div className="flex gap-4 items-center">
          
          <div className="flex items-center gap-2 bg-white border border-slate-300 px-3 py-2 rounded-lg">
            <Zap className="w-4 h-4 text-amber-500" />
            <span className="text-sm text-slate-600 font-medium">Threads:</span>
            <select 
              value={concurrency} 
              onChange={(e) => setConcurrency(Number(e.target.value))}
              disabled={processing}
              className="bg-transparent font-bold text-slate-800 outline-none cursor-pointer"
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
            </select>
          </div>

          <div className="relative">
             <input 
              type="file" 
              accept=".txt" 
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button className="flex items-center gap-2 bg-white border border-slate-300 text-slate-700 px-4 py-2 rounded-lg hover:bg-slate-50">
              <Upload className="w-4 h-4" />
              <span>{questions.length > 0 ? `${questions.length} Questions Loaded` : "Upload .txt"}</span>
            </button>
          </div>

          <button 
            onClick={handleDownloadCSV}
            disabled={processing || results.length === 0}
            className="flex items-center gap-2 bg-white border border-slate-300 text-slate-700 px-4 py-2 rounded-lg hover:bg-slate-50 disabled:opacity-50"
            title="Download Results as CSV"
          >
            <Download className="w-4 h-4" />
            <span>CSV</span>
          </button>

          <button 
            onClick={runBenchmark}
            disabled={processing || questions.length === 0}
            className={`flex items-center gap-2 px-6 py-2 rounded-lg text-white font-medium transition-all ${
              processing || questions.length === 0 
                ? 'bg-slate-300 cursor-not-allowed' 
                : 'bg-primary hover:bg-indigo-700 shadow-lg shadow-indigo-200'
            }`}
          >
            {processing ? (
              <>Processing {progress.current}/{progress.total}...</>
            ) : (
              <><Play className="w-4 h-4" /> Start Run</>
            )}
          </button>
        </div>
      </div>

      {/* Results Table Container */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 overflow-hidden flex flex-col">
        <div className="overflow-auto flex-1">
          <table className="w-full text-sm text-left text-slate-600 whitespace-nowrap">
            <thead className="text-xs text-slate-700 uppercase bg-slate-100 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-3 border-b">#</th>
                <th className="px-4 py-3 border-b">Query</th>
                <th className="px-4 py-3 border-b">Status</th>
                <th className="px-4 py-3 border-b">LLM Model</th>
                <th className="px-4 py-3 border-b">Embed Model</th>
                <th className="px-4 py-3 border-b">Rerank Model</th> {/* Added */}
                <th className="px-4 py-3 border-b bg-indigo-50 text-indigo-800">Total (S)</th>
                <th className="px-4 py-3 border-b">LLM Gen</th>
                <th className="px-4 py-3 border-b">Retrieval</th>
                <th className="px-4 py-3 border-b">Rerank</th> {/* Added */}
                <th className="px-4 py-3 border-b">Embed</th>
                <th className="px-4 py-3 border-b">In Guard</th>
                <th className="px-4 py-3 border-b">Out Guard</th>
                <th className="px-4 py-3 border-b bg-emerald-50 text-emerald-800">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {results.length === 0 ? (
                <tr>
                  <td colSpan="14" className="px-6 py-20 text-center text-slate-400">
                    <div className="flex flex-col items-center gap-2">
                      <Clock className="w-10 h-10 opacity-20" />
                      <p>Upload a .txt file to begin benchmarking</p>
                    </div>
                  </td>
                </tr>
              ) : (
                results.map((row) => (
                  <tr key={row.id} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-mono text-xs">{row.id}</td>
                    <td className="px-4 py-3 max-w-xs truncate" title={row.query}>{row.query}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${getStatusColor(row.status)}`}>
                        {row.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 max-w-[120px] truncate text-xs text-slate-500" title={row.llmModel}>{row.llmModel}</td>
                    <td className="px-4 py-3 max-w-[120px] truncate text-xs text-slate-500" title={row.embedModel}>{row.embedModel}</td>
                    <td className="px-4 py-3 max-w-[120px] truncate text-xs text-slate-500" title={row.rerankModel}>{row.rerankModel}</td>
                    
                    <td className="px-4 py-3 font-mono bg-indigo-50/50 text-indigo-700 font-bold">{row.serverTotal}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.llmTime}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.retrievalTime}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.rerankTime}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.embedTime}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.inputGuardTime}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{row.outputGuardTime}s</td>
                    <td className="px-4 py-3 font-mono bg-emerald-50/50 text-emerald-700 font-bold">{row.clientLatency}s</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default BenchmarkPage;
