import React, { useState, useEffect } from 'react';
import { getModels, getConfig, updateConfig, clearCache } from '../api'; 
import { RefreshCw, Server, CheckCircle, XCircle, Trash2, ShieldAlert, Save } from 'lucide-react';

const ROLE_MAPPING = [
    { label: "LLM Generation", key: "llm" },
    { label: "Embedding", key: "embed" },
    { label: "Guardrail", key: "guard" },
    { label: "OCR (Vision)", key: "ocr" }
];

const ConfigPage = () => {
  const [activePorts, setActivePorts] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [clearingCache, setClearingCache] = useState(false);
  const [saving, setSaving] = useState(false);
  
  const [config, setConfig] = useState({
    llm: { port: "", model: "" },
    embed: { port: "", model: "" },
    guard: { port: "", model: "" },
    ocr: { port: "", model: "" },
  });

  const initializePage = async () => {
    setScanning(true);
    
    try {
      const [modelsRes, currentConfigRes] = await Promise.all([
          getModels(),
          getConfig().catch(() => ({ data: null }))
      ]);

      // 1. Process Active Ports
      const rawModels = modelsRes.data || {};
      const discoveredPorts = Object.entries(rawModels)
        .filter(([model, port]) => model !== "No Model Found")
        .map(([model, port]) => ({
            port: port,
            models: [{ id: model }]
        }));
      
      setActivePorts(discoveredPorts);

      // 2. Process Current Config
      if (currentConfigRes && currentConfigRes.data) {
          const s = currentConfigRes.data;
          setConfig(prev => ({
              llm: { port: s.llm_port, model: s.llm_model_id },
              embed: { port: s.embed_port, model: s.embed_model_id },
              guard: { port: s.guard_port, model: s.guard_model_id },
              ocr: { port: s.vl_port, model: s.vl_model_id },
          }));
      }

    } catch (err) {
      console.error("Failed to load config:", err);
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    initializePage();
  }, []);

  const handleAssign = (roleKey, port, modelId) => {
    setConfig(prev => ({
      ...prev,
      [roleKey]: { port, model: modelId }
    }));
  };

  const handleSaveConfig = async () => {
    setSaving(true);
    
    // Prepare payload
    const payload = {
        llm_port: parseInt(config.llm.port) || 8100,
        llm_model_id: config.llm.model || "Qwen/Qwen3-8B",

        embed_port: parseInt(config.embed.port) || 8002,
        embed_model_id: config.embed.model || "Qwen/Qwen3-Embedding-0.6B",

        guard_port: parseInt(config.guard.port) || 8001,
        guard_model_id: config.guard.model || "Qwen/Qwen3Guard-Gen-0.6B",

        vl_port: parseInt(config.ocr.port) || 8003,
        vl_model_id: config.ocr.model || "Qwen/Qwen3-VL-8B-Instruct-FP8"
    };

    try {
        await updateConfig(payload);
        alert("Configuration saved! Backend clients re-initialized.");
    } catch (err) {
        alert("Failed to save config: " + err.message);
    } finally {
        setSaving(false);
    }
  };

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

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Model Configuration</h1>
          <p className="text-slate-500">Discover and assign local models to RAG tasks.</p>
        </div>
        <div className="flex gap-2">
            <button 
            onClick={initializePage}
            disabled={scanning}
            className="flex items-center gap-2 bg-white border border-slate-200 px-4 py-2 rounded-lg hover:bg-slate-50 transition-colors shadow-sm disabled:opacity-50"
            >
            <RefreshCw className={`w-4 h-4 ${scanning ? 'animate-spin' : ''}`} />
            {scanning ? 'Refreshing...' : 'Refresh'}
            </button>
            
            <button 
            onClick={handleSaveConfig}
            disabled={saving || scanning}
            className="flex items-center gap-2 bg-primary text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50"
            >
            <Save className="w-4 h-4" />
            {saving ? 'Saving...' : 'Save Configuration'}
            </button>
        </div>
      </div>

      {/* Active Ports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {activePorts.length === 0 && !scanning && (
          <div className="col-span-full text-center py-10 bg-slate-100 rounded-xl text-slate-500">
            No active model servers found. Check if vLLM instances are running.
          </div>
        )}
        {activePorts.map((node) => (
          <div key={node.port} className="bg-white p-4 rounded-xl border border-emerald-100 shadow-sm flex flex-col gap-2 relative overflow-hidden">
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
          {ROLE_MAPPING.map((roleObj, idx) => (
            <div key={roleObj.key} className="p-6 flex items-center justify-between hover:bg-slate-50 transition-colors">
              <div className="w-1/3">
                <h4 className="font-medium text-slate-800">{roleObj.label}</h4>
                <p className="text-xs text-slate-500 mt-1">Select the port hosting the model for this task.</p>
              </div>
              
              <div className="flex gap-4 w-2/3">
                <select 
                  className="flex-1 bg-white border border-slate-300 text-slate-700 text-sm rounded-lg focus:ring-primary focus:border-primary block p-2.5"
                  onChange={(e) => {
                     const port = e.target.value;
                     const node = activePorts.find(p => p.port.toString() === port);
                     // Automatically get the model ID associated with this port
                     const model = node?.models[0]?.id || "";
                     handleAssign(roleObj.key, port, model);
                  }}
                  value={config[roleObj.key].port || ""}
                >
                  <option value="">Select Model</option>
                  {activePorts.map(p => (
                    <option key={p.port} value={p.port}>
                      {p.models[0]?.id || `Port ${p.port}`}
                    </option>
                  ))}
                </select>

                <div className="flex items-center justify-center w-10">
                  {config[roleObj.key].port ? (
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

      {/* System Actions */}
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
