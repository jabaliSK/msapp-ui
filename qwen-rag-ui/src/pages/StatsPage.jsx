import React, { useState, useEffect } from 'react';
import { getLogs, truncateLogs } from '../api';
import { Download, Activity, Clock, RefreshCcw, Trash2, Filter } from 'lucide-react';

const StatsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [timeFilter, setTimeFilter] = useState(""); // Empty = All Time

  useEffect(() => {
    fetchLogs();
  }, [timeFilter]); // Re-fetch when filter changes

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
        fetchLogs(); // Refresh list (should be empty)
    } catch (err) {
        alert("Failed to truncate logs: " + err.message);
    }
  };

  const handleDownloadCSV = () => {
    if (logs.length === 0) return;

    // Ordered Columns as requested
    const headers = [
      "Query", "Status", "Response", "Sources", "Safety", 
      "Total (Server)", "LLM Gen", "Retrieval", "Embed", 
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
    <div className="p-6 max-w-[95%] mx-auto h-screen flex flex-col">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">System Logs</h1>
          <p className="text-slate-500">Live transaction history from PostgreSQL.</p>
        </div>

        <div className="flex gap-3">
          {/* Time Filter */}
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

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 overflow-hidden flex flex-col">
        <div className="overflow-auto flex-1">
          <table className="w-full text-sm text-left text-slate-600 whitespace-nowrap">
            <thead className="text-xs text-slate-700 uppercase bg-slate-100 sticky top-0 z-10">
              <tr>
                <th className="px-4 py-3 border-b">Timestamp</th>
                <th className="px-4 py-3 border-b">Query</th>
                <th className="px-4 py-3 border-b">Status</th>
                <th className="px-4 py-3 border-b">Response</th>
                <th className="px-4 py-3 border-b">Sources</th>
                <th className="px-4 py-3 border-b">Safety</th>
                {/* Metrics */}
                <th className="px-4 py-3 border-b bg-indigo-50 text-indigo-800">Total (S)</th>
                <th className="px-4 py-3 border-b">LLM</th>
                <th className="px-4 py-3 border-b">RAG</th>
                <th className="px-4 py-3 border-b">Embed</th>
                <th className="px-4 py-3 border-b">In Guard</th>
                <th className="px-4 py-3 border-b">Out Guard</th>
                <th className="px-4 py-3 border-b bg-emerald-50 text-emerald-800">Client Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr><td colSpan="14" className="p-10 text-center">Loading logs...</td></tr>
              ) : logs.length === 0 ? (
                <tr><td colSpan="14" className="p-10 text-center text-slate-400">No logs found.</td></tr>
              ) : (
                logs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-4 py-3 text-xs text-slate-400">{new Date(log.timestamp).toLocaleTimeString()}</td>
                    <td className="px-4 py-3 max-w-[200px] truncate font-medium text-slate-700" title={log.user_query}>{log.user_query}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-bold ${getStatusColor(log.status)}`}>{log.status}</span>
                    </td>
                    <td className="px-4 py-3 max-w-[200px] truncate text-slate-500" title={log.model_response}>{log.model_response}</td>
                    <td className="px-4 py-3 max-w-[100px] truncate text-xs" title={log.sources?.join(", ")}>{log.sources?.length || 0} Sources</td>
                    <td className="px-4 py-3 text-xs">{log.safety_check_result}</td>
                    
                    {/* Metrics */}
                    <td className="px-4 py-3 font-mono bg-indigo-50/50 text-indigo-700 font-bold">{log.total_duration_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.llm_generation_sec}s</td>
                    <td className="px-4 py-3 font-mono text-slate-500">{log.retrieval_sec}s</td>
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