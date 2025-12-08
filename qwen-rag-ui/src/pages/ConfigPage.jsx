import React, { useState, useEffect } from 'react';
import { getModels, getConfig, updateConfig, clearCache } from '../api'; 
import { RefreshCw, Server, CheckCircle, XCircle, Trash2, ShieldAlert, Save, MessageSquare, Check, AlertTriangle, Thermometer, Split } from 'lucide-react';

const ROLE_MAPPING = [
    { label: "LLM Generation", key: "llm" },
    { label: "Embedding", key: "embed" },
    { label: "Guardrail", key: "guard" },
    { label: "OCR (Vision)", key: "ocr" },
    { label: "Decision Router", key: "router" } // Added Router Role
];

const ConfigPage = () => {
  const [activePorts, setActivePorts] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [clearingCache, setClearingCache] = useState(false);
  const [saving, setSaving] = useState(false);
  
  // Custom Toast State
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' });

  const [config, setConfig] = useState({
    llm: { port: "", model: "" },
    embed: { port: "", model: "" },
    guard: { port: "", model: "" },
    ocr: { port: "", model: "" },
    router: { port: "", model: "" }, // Added Router State
    cacheThreshold: 0.92,
    systemPrompt: "You are a helpful assistant.",
    llmTemperature: 0.1 
  });

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast(prev => ({ ...prev, show: false })), 3000);
  };

  const initializePage = async () => {
    setScanning(true);
    try {
      const [modelsRes, currentConfigRes] = await Promise.all([
          getModels(),
          getConfig().catch(() => ({ data: null }))
      ]);

      const rawModels = modelsRes.data || {};
      const discoveredPorts = Object.entries(rawModels)
        .filter(([model, port]) => model !== "No Model Found")
        .map(([model, port]) => ({ port: port, models: [{ id: model }] }));
      
      setActivePorts(discoveredPorts);

      if (currentConfigRes && currentConfigRes.data) {
          const s = currentConfigRes.data;
          setConfig(prev => ({
              llm: { port: s.llm_port, model: s.llm_model_id },
              embed: { port: s.embed_port, model: s.embed_model_id },
              guard: { port: s.guard_port, model: s.guard_model_id },
              ocr: { port: s.vl_port, model: s.vl_model_id },
              router: { port: s.router_port, model: s.router_model_id }, // Map Router Config
              cacheThreshold: s.cache_threshold || 0.92,
              systemPrompt: s.system_prompt || prev.systemPrompt,
              llmTemperature: s.llm_temperature !== undefined ? s.llm_temperature : 0.1
          }));
      }
    } catch (err) {
      console.error("Failed to load config:", err);
      showToast("Failed to load configuration.", "error");
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => { initializePage(); }, []);

  const handleAssign = (roleKey, port, modelId) => {
    setConfig(prev => ({ ...prev, [roleKey]: { port, model: modelId } }));
  };

  const handleSaveConfig = async () => {
    setSaving(true);
    const payload = {
        llm_port: parseInt(config.llm.port) || 8100,
        llm_model_id: config.llm.model || "Qwen/Qwen3-8B",
        llm_temperature: parseFloat(config.llmTemperature),

        embed_port: parseInt(config.embed.port) || 8002,
        embed_model_id: config.embed.model || "Qwen/Qwen3-Embedding-0.6B",

        guard_port: parseInt(config.guard.port) || 8001,
        guard_model_id: config.guard.model || "Qwen/Qwen3Guard-Gen-0.6B",

        vl_port: parseInt(config.ocr.port) || 8003,
        vl_model_id: config.ocr.model || "Qwen/Qwen3-VL-8B-Instruct-FP8",

        router_port: parseInt(config.router.port) || 8010, // Save Router Port
        router_model_id: config.router.model || "Qwen/Qwen3-0.6B", // Save Router Model

        cache_threshold: parseFloat(config.cacheThreshold),
        system_prompt: config.systemPrompt
    };

    try {
        await updateConfig(payload);
        showToast("Configuration saved & Models re-bound!");
    } catch (err) {
        showToast("Failed to save config: " + err.message, "error");
    } finally {
        setSaving(false);
    }
  };

  const handleClearCache = async () => {
    if (!window.confirm("Are you sure? This will wipe the semantic search cache.")) return;
    setClearingCache(true);
    try {
      await clearCache();
      showToast("Semantic Cache Cleared Successfully!");
    } catch (err) {
      showToast("Failed to clear cache: " + err.message, "error");
    } finally {
      setClearingCache(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto relative">
      {/* Toast Notification */}
      {toast.show && (
        <div className={`fixed bottom-8 right-8 z-50 px-6 py-4 rounded-xl shadow-2xl flex items-center gap-3 transition-all duration-300 animate-bounce ${
            toast.type === 'success' ? 'bg-white border-l-4 border-emerald-500 text-slate-800' : 'bg-white border-l-4 border-red-500 text-slate-800'
        }`}>
            {toast.type === 'success' ? <Check className="w-5 h-5 text-emerald-500" /> : <AlertTriangle className="w-5 h-5 text-red-500" />}
            <span className="font-medium">{toast.message}</span>
        </div>
      )}

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
            <button onClick={handleSaveConfig} disabled={saving || scanning} className="flex items-center gap-2 bg-primary text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors shadow-sm disabled:opacity-50">
            <Save className="w-4 h-4" />
            {saving ? 'Saving...' : 'Save Configuration'}
            </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {activePorts.length === 0 && !scanning && <div className="col-span-full text-center py-10 bg-slate-100 rounded-xl text-slate-500">No active model servers found.</div>}
        {activePorts.map((node) => (
          <div key={node.port} className="bg-white p-4 rounded-xl border border-emerald-100 shadow-sm flex flex-col gap-2 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-16 h-16 bg-emerald-50 rounded-bl-full -mr-8 -mt-8"></div>
            <div className="flex items-center gap-2 z-10"><Server className="w-5 h-5 text-emerald-600" /><span className="font-mono font-bold text-lg">:{node.port}</span></div>
            <div className="z-10 mt-2"><span className="text-xs font-semibold text-slate-400 uppercase">Models:</span><ul className="mt-1 text-sm text-slate-700">{node.models.map(m => <li key={m.id} className="truncate">• {m.id}</li>)}</ul></div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-8">
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50"><h3 className="font-semibold text-slate-700">Assign Roles</h3></div>
        <div className="divide-y divide-slate-100">
          {ROLE_MAPPING.map((roleObj) => (
            <div key={roleObj.key} className="p-6 flex items-center justify-between hover:bg-slate-50 transition-colors">
              <div className="w-1/3">
                <h4 className="font-medium text-slate-800 flex items-center gap-2">
                    {roleObj.key === 'router' && <Split className="w-4 h-4 text-indigo-500" />}
                    {roleObj.label}
                </h4>
                <p className="text-xs text-slate-500 mt-1">Select the port hosting the model.</p>
              </div>
              <div className="flex gap-4 w-2/3">
                <select className="flex-1 bg-white border border-slate-300 text-slate-700 text-sm rounded-lg p-2.5" onChange={(e) => {
                     const port = e.target.value; const node = activePorts.find(p => p.port.toString() === port);
                     handleAssign(roleObj.key, port, node?.models[0]?.id || "");
                  }} value={config[roleObj.key].port || ""}>
                  <option value="">Select Model</option>
                  {activePorts.map(p => <option key={p.port} value={p.port}>{p.models[0]?.id || `Port ${p.port}`}</option>)}
                </select>
                <div className="flex items-center justify-center w-10">{config[roleObj.key].port ? <CheckCircle className="w-5 h-5 text-emerald-500" /> : <XCircle className="w-5 h-5 text-slate-300" />}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mb-8">
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50"><h3 className="font-semibold text-slate-700">RAG Parameters</h3></div>
        <div className="divide-y divide-slate-100">
            {/* Cache Threshold */}
            <div className="p-6 flex items-center justify-between">
                <div className="w-1/3"><h4 className="font-medium text-slate-800">Semantic Cache Threshold</h4><p className="text-xs text-slate-500 mt-1">Similarity score (0.5 - 1.0) for cache hits.</p></div>
                <div className="w-2/3 flex items-center gap-4">
                    <input type="range" min="0.5" max="1.0" step="0.01" value={config.cacheThreshold} onChange={(e) => setConfig({...config, cacheThreshold: e.target.value})} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary" />
                    <span className="font-mono font-bold text-slate-700 w-12 text-right">{config.cacheThreshold}</span>
                </div>
            </div>
            
            {/* Temperature Slider */}
            <div className="p-6 flex items-center justify-between">
                <div className="w-1/3">
                    <h4 className="font-medium text-slate-800 flex items-center gap-2"><Thermometer className="w-4 h-4 text-slate-500"/> LLM Temperature</h4>
                    <p className="text-xs text-slate-500 mt-1">Controls randomness (0.0 = Deterministic, 1.0 = Creative).</p>
                </div>
                <div className="w-2/3 flex items-center gap-4">
                    <input type="range" min="0.0" max="1.0" step="0.01" value={config.llmTemperature} onChange={(e) => setConfig({...config, llmTemperature: e.target.value})} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-500" />
                    <span className="font-mono font-bold text-slate-700 w-12 text-right">{config.llmTemperature}</span>
                </div>
            </div>

            {/* System Prompt */}
            <div className="p-6 flex items-start justify-between">
                <div className="w-1/3"><h4 className="font-medium text-slate-800 flex items-center gap-2"><MessageSquare className="w-4 h-4 text-slate-500" /> System Prompt</h4><p className="text-xs text-slate-500 mt-1">Define the personality and constraints.</p></div>
                <div className="w-2/3"><textarea value={config.systemPrompt} onChange={(e) => setConfig({...config, systemPrompt: e.target.value})} className="w-full h-32 p-3 bg-slate-50 border border-slate-300 rounded-lg text-sm font-mono" /></div>
            </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex items-center gap-2"><ShieldAlert className="w-4 h-4 text-amber-500" /><h3 className="font-semibold text-slate-700">System Actions</h3></div>
        <div className="p-6">
          <div className="flex items-center justify-between">
            <div><h4 className="font-medium text-slate-800">Clear Semantic Cache</h4><p className="text-xs text-slate-500 mt-1">Removes all cached Q&A pairs.</p></div>
            <button onClick={handleClearCache} disabled={clearingCache} className="flex items-center gap-2 bg-red-50 text-red-600 border border-red-100 px-4 py-2 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50"><Trash2 className="w-4 h-4" />{clearingCache ? 'Clearing...' : 'Clear Cache'}</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfigPage;