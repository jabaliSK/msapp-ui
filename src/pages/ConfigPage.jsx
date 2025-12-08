import React, { useState, useEffect } from 'react';
import { checkModelPort, clearCache } from '../api'; // Import clearCache
import { RefreshCw, Server, CheckCircle, XCircle, Trash2, ShieldAlert } from 'lucide-react';

const PORTS_TO_SCAN = [
  ...Array.from({ length: 11 }, (_, i) => 8000 + i),
  ...Array.from({ length: 11 }, (_, i) => 8100 + i)
];

const ROLES = ["LLM Generation", "Embedding", "Guardrail", "OCR (Vision)"];

const ConfigPage = () => {
  const [activePorts, setActivePorts] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [clearingCache, setClearingCache] = useState(false); // State for button
  const [config, setConfig] = useState({
    llm: { port: "", model: "" },
    embed: { port: "", model: "" },
    guard: { port: "", model: "" },
    ocr: { port: "", model: "" },
  });

  // ... (keep scanNetwork and existing useEffect)
  const scanNetwork = async () => {
    setScanning(true);
    setActivePorts([]);
    const promises = PORTS_TO_SCAN.map(port => checkModelPort(port));
    const results = await Promise.all(promises);
    const alive = results.filter(r => r.alive);
    setActivePorts(alive);
    setScanning(false);
  };

  useEffect(() => {
    scanNetwork();
  }, []);

  const handleAssign = (roleKey, port, modelId) => {
    setConfig(prev => ({
      ...prev,
      [roleKey]: { port, model: modelId }
    }));
  };

  // --- NEW HANDLER ---
  const handleClearCache = async () => {
    if (!window.confirm("Are you sure? This will wipe the semantic search cache.")) return;
    
    setClearingCache(true);
    try {
      await clearCache();
      alert("Semantic Cache Cleared Successfully!");
    } catch (err) {
      alert("Failed to clear cache: " + err.message);
    } finally {
      setClearingCache(false);
    }
  };
  // -------------------

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Model Configuration</h1>
          <p className="text-slate-500">Discover and assign local models to RAG tasks.</p>
        </div>
        <button 
          onClick={scanNetwork}
          disabled={scanning}
          className="flex items-center gap-2 bg-white border border-slate-200 px-4 py-2 rounded-lg hover:bg-slate-50 transition-colors shadow-sm disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${scanning ? 'animate-spin' : ''}`} />
          {scanning ? 'Scanning Ports...' : 'Refresh Network'}
        </button>
      </div>

      {/* Active Ports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {/* ... (Existing Grid Code) ... */}
        {activePorts.length === 0 && !scanning && (
          <div className="col-span-full text-center py-10 bg-slate-100 rounded-xl text-slate-500">
            No active model servers found on ports 8000-8010 or 8100-8110.
          </div>
        )}
        {activePorts.map((node) => (
          <div key={node.port} className="bg-white p-4 rounded-xl border border-emerald-100 shadow-sm flex flex-col gap-2 relative overflow-hidden">
             {/* ... (Existing Node UI) ... */}
            <div className="absolute top-0 right-0 w-16 h-16 bg-emerald-50 rounded-bl-full -mr-8 -mt-8"></div>
            <div className="flex items-center gap-2 z-10">
              <Server className="w-5 h-5 text-emerald-600" />
              <span className="font-mono font-bold text-lg">:{node.port}</span>
            </div>
            <div className="z-10 mt-2">
              <span className="text-xs font-semibold text-slate-400 uppercase">Models Available:</span>
              <ul className="mt-1 text-sm text-slate-700">
                {node.models.map(m => (
                  <li key={m.id} className="truncate" title={m.id}>• {m.id}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      {/* Role Assignment Section */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-8">
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50">
          <h3 className="font-semibold text-slate-700">Assign Roles</h3>
        </div>
        <div className="divide-y divide-slate-100">
          {Object.keys(config).map((roleKey, idx) => (
            <div key={roleKey} className="p-6 flex items-center justify-between hover:bg-slate-50 transition-colors">
              <div className="w-1/3">
                <h4 className="font-medium text-slate-800">{ROLES[idx]}</h4>
                <p className="text-xs text-slate-500 mt-1">Select the port hosting the model for this task.</p>
              </div>
              
              <div className="flex gap-4 w-2/3">
                <select 
                  className="flex-1 bg-white border border-slate-300 text-slate-700 text-sm rounded-lg focus:ring-primary focus:border-primary block p-2.5"
                  onChange={(e) => {
                     const port = e.target.value;
                     const node = activePorts.find(p => p.port.toString() === port);
                     const model = node?.models[0]?.id || "";
                     handleAssign(roleKey, port, model);
                  }}
                  value={config[roleKey].port}
                >
                  <option value="">Select Port</option>
                  {activePorts.map(p => (
                    <option key={p.port} value={p.port}>localhost:{p.port}</option>
                  ))}
                </select>

                <div className="flex items-center justify-center w-10">
                  {config[roleKey].port ? (
                    <CheckCircle className="w-5 h-5 text-emerald-500" />
                  ) : (
                    <XCircle className="w-5 h-5 text-slate-300" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* --- NEW SECTION: SYSTEM ACTIONS --- */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-amber-500" />
          <h3 className="font-semibold text-slate-700">System Actions</h3>
        </div>
        <div className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-slate-800">Clear Semantic Cache</h4>
              <p className="text-xs text-slate-500 mt-1">
                Removes all cached Q&A pairs. The system will force re-generation for all subsequent queries.
              </p>
            </div>
            <button 
              onClick={handleClearCache}
              disabled={clearingCache}
              className="flex items-center gap-2 bg-red-50 text-red-600 border border-red-100 px-4 py-2 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4" />
              {clearingCache ? 'Clearing...' : 'Clear Cache'}
            </button>
          </div>
        </div>
      </div>

    </div>
  );
};

export default ConfigPage;
