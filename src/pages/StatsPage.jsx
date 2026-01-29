import React, { useState, useEffect } from 'react';
import { getLogs, truncateLogs } from '../api';
import { Download, RefreshCcw, Trash2, Filter, Copy, Check } from 'lucide-react';

// Small helper component for the Copy button to handle its own "copied" state
const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button 
      onClick={handleCopy}
      className="p-1 hover:bg-slate-200 rounded text-slate-400 hover:text-indigo-600 transition-colors"
      title="Copy full query"
    >
      {copied ? <Check className="w-3 h-3 text-green-600" /> : <Copy className="w-3 h-3" />}
    </button>
  );
};

const StatsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState("");

  useEffect(() => {
    fetchLogs();
  }, [timeFilter]);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const params = timeFilter ? { hours: parseInt(timeFilter) } : {};
      const res = await getLogs(params);
      setLogs(res.data);
    } catch (err) {
      console.error("Failed to fetch logs", err);
    } finally {
      setLoading(false);
    }
  };

  const handleTruncate = async () => {
    if (!window.confirm("Are you sure? This will delete ALL interaction logs permanently.")) return;
    try {
        await truncateLogs();
        fetchLogs();
    } catch (err) {
        alert("Failed to truncate logs: " + err.message);
    }
  };

  const handleDownloadCSV = () => {
    if (logs.length === 0) return;

    const headers = [
      "Query", "Status", "Response", "Sources", "Safety", 
      "Total (Server)", "LLM Gen", "Retrieval", "Rerank", "Embed", 
      "In Guard", "Out Guard", "Client Latency"
    ];

    const csvRows = logs.map(row => {
      const safeQuery = `"${(row.user_query || "").replace(/"/g, '""')}"`;
      const safeResponse = `"${(row.model_response || "").replace(/"/g, '""')}"`;
      const safeSources = `"${(row.sources?.join(", ") || "").replace(/"/g, '""')}"`;

      return [
        safeQuery,
        row.status,
        safeResponse,
        safeSources,
        row.safety_check_result,
        row.total_duration_sec,
        row.llm_generation_sec,
        row.retrieval_sec,
        row.reranking_sec || 0.0,
        row.embedding_sec,
        row.input_guardrail_sec,
        row.output_guardrail_sec,
        row.client_latency_sec || "N/A"
      ].join(",");
    });

    const csvContent = [headers.join(","), ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `rag_stats_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getStatusColor = (status) => {
    if (status?.includes("Success")) return "text-emerald-600 bg-emerald-50";
    if (status?.includes("Safety")) return "text-amber-600 bg-amber-50";
    return "text-red-600 bg-red-50";
  };

  return (
    <div className="h-full w-full flex flex-col p-6 max-w-[95%] mx-auto">
      <div className="flex-none flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">System Logs</h1>
          <p className="text-slate-500">Live transaction history from PostgreSQL.</p>
        </div>

        <div className="flex gap-3">
          <div className="flex items-center gap-2 bg-white border border-slate-300 px-3 py-2 rounded-lg">
             <Filter className="w-4 h-4 text-slate-500" />
             <select 
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="bg-transparent text-sm text-slate-700 outline-none cursor-pointer"
             >
                <option value="">All Time</option>
                <option value="1">Last 1 Hour</option>
                <option value="6">Last 6 Hours</option>
                <option value="24">Last 24 Hours</option>
                <option value="168">Last 7 Days</option>
             </select>
          </div>

          <button 
            onClick={fetchLogs} 
            className="flex items-center gap-2 bg-white border border-slate-300 text-slate-700 px-4 py-2 rounded-lg hover:bg-slate-50"
          >
            <RefreshCcw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
          
          <button 
            onClick={handleTruncate}
            disabled={logs.length === 0}
            className="flex items-center gap-2 bg-white border border-red-200 text-red-600 px-4 py-2 rounded-lg hover:bg-red-50 disabled:opacity-50"
          >
            <Trash2 className="w-4 h-4" /> Clear Logs
          </button>
          
          <button 
            onClick={handleDownloadCSV}
            disabled={logs.length === 0}
            className="flex items-center gap-2 bg-primary text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50 shadow-sm"
          >
            <Download className="w-4 h-4" /> Export CSV
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-auto w-full">
          <table className="w-full text-sm text-left text-slate-600 whitespace-nowrap">
            <thead className="text-xs text-slate-700 uppercase bg-slate-100 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-3 border-b w-24">Time</th>
                {/* Reduced Width for Query */}
                <th className="px-4 py-3 border-b w-48">Query</th>
                <th className="px-4 py-3 border-b w-24">Status</th>
                 {/* Reduced Width for Response */}
                <th className="px-4 py-3 border-b w-48">Response</th>
                <th className="px-4 py-3 border-b w-20">Sources</th>
                <th className="px-4 py-3 border-b w-24">Safety</th>
                
                {/* Metrics */}
                <th className="px-4 py-3 border-b bg-indigo-50 text-indigo-800">Total</th>
                <th className="px-4 py-3 border-b">LLM</th>
                <th className="px-4 py-3 border-b">RAG</th>
                <th className="px-4 py-3 border-b">Rerank</th> 
                <th className="px-4 py-3 border-b">Embed</th>
                <th className="px-4 py-3 border-b">In Guard</th>
                <th className="px-4 py-3 border-b">Out Guard</th>
                <th className="px-4 py-3 border-b bg-emerald-50 text-emerald-800">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr><td colSpan="15" className="p-10 text-center">Loading logs...</td></tr>
              ) : logs.length === 0 ? (
                <tr><td colSpan="15" className="p-10 text-center text-slate-400">No logs found.</td></tr>
              ) : (
                logs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3 text-xs text-slate-400">{new Date(log.timestamp).toLocaleTimeString()}</td>
                    
                    {/* Fixed Width Query Column with Copy Button */}
                    <td className="px-4 py-3 max-w-[12rem]">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-medium text-slate-700 block" title={log.user_query}>
                          {log.user_query}
                        </span>
                        <CopyButton text={log.user_query} />
                      </div>
                    </td>

                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${getStatusColor(log.status)}`}>{log.status}</span>
                    </td>
                    
                    {/* Fixed Width Response Column */}
                    <td className="px-4 py-3 max-w-[12rem] truncate text-slate-500" title={log.model_response}>
                      {log.model_response}
                    </td>

                    <td className="px-4 py-3 max-w-[100px] truncate text-xs" title={log.sources?.join(", ")}>{log.sources?.length || 0} Sources</td>
                    <td className="px-4 py-3 text-xs">{log.safety_check_result}</td>
                    
                    {/* Metrics */}
                    <td className="px-4 py-3 font-mono bg-indigo-50/50 text-indigo-700 font-bold">{log.total_duration_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.llm_generation_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.retrieval_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.reranking_sec || 0}s</td> 
                    <td className="px-4 py-3 font-mono text-slate-500">{log.embedding_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.input_guardrail_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.output_guardrail_sec}s</td>
                    <td className="px-4 py-3 font-mono bg-emerald-50/50 text-emerald-700 font-bold">{log.client_latency_sec || "-"}</td>
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

export default StatsPage;
