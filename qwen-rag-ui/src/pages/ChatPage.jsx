import React, { useState, useRef, useEffect } from 'react';
import { RAG_API_BASE } from '../api'; // Import base URL
import { Send, Bot, User, Clock, ShieldCheck, Database, AlertTriangle, Zap } from 'lucide-react';

const ChatPage = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "Hello! I'm ready to answer questions based on your uploaded documents." }
  ]);
  const [input, setInput] = useState("");
  const [useCache, setUseCache] = useState(true); 
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userQuery = input;
    setInput("");
    
    // Add User Message
    setMessages(prev => [...prev, { role: 'user', text: userQuery }]);
    
    // Add Placeholder Assistant Message
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      text: "", // Will be filled by stream
      sources: [], 
      safety: null 
    }]);

    setLoading(true);

    try {
      const response = await fetch(`${RAG_API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, use_cache: useCache })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value, { stream: true });
        
        // Split by newline to handle NDJSON
        const lines = chunkValue.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const json = JSON.parse(line);

            if (json.type === 'token') {
              setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg.role !== 'assistant') return prev; // Safety check
                
                return [
                  ...prev.slice(0, -1),
                  { ...lastMsg, text: lastMsg.text + json.content }
                ];
              });
            } else if (json.type === 'meta') {
               setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                return [
                  ...prev.slice(0, -1),
                  { 
                    ...lastMsg, 
                    sources: json.sources, 
                    timings: json.timings,
                    safety: json.safety 
                  }
                ];
              });
            }
          } catch (e) {
            console.error("Error parsing stream chunk", e);
          }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: "Sorry, I encountered an error connecting to the server." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 pb-40 space-y-6">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            {/* Avatar */}
            <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-primary text-white' : 'bg-white border border-slate-200 text-slate-600'}`}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>

            {/* Bubble */}
            <div className={`max-w-2xl`}>
              <div className={`p-4 rounded-2xl shadow-sm text-sm leading-relaxed ${msg.role === 'user' ? 'bg-primary text-white rounded-tr-none' : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'}`}>
                {/* Render Text (Whitespace preservation for formatting) */}
                <span className="whitespace-pre-wrap">{msg.text}</span>
                {/* Blinking Cursor for streaming */}
                {loading && idx === messages.length - 1 && (
                  <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-slate-400 animate-pulse"></span>
                )}
              </div>

              {/* Metadata (Only show when stream is done and meta is present) */}
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 space-y-2 fade-in">
                  <div className="flex flex-wrap gap-2">
                    {msg.sources.map((source, i) => (
                      <span key={i} className="inline-flex items-center gap-1 px-2 py-1 bg-slate-200 text-slate-600 text-xs rounded-full">
                        <Database className="w-3 h-3" />
                        {source}
                      </span>
                    ))}
                    {msg.safety && (
                      <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full ${msg.safety === 'PASSED' || msg.safety === 'PASSED_CACHE' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {msg.safety === 'PASSED' || msg.safety === 'PASSED_CACHE' ? <ShieldCheck className="w-3 h-3"/> : <AlertTriangle className="w-3 h-3"/>}
                        {msg.safety}
                      </span>
                    )}
                  </div>
                  
                  {msg.timings && (
                    <div className="text-xs text-slate-400 flex gap-4 mt-1 bg-slate-100 p-2 rounded w-fit">
                      <span className="flex items-center gap-1"><Clock className="w-3 h-3"/> Total: {msg.timings.total_sec.toFixed(2)}s</span>
                      <span>LLM: {msg.timings.llm_generation_sec.toFixed(2)}s</span>
                      <span>RAG: {msg.timings.retrieval_sec.toFixed(2)}s</span>
                      <span>Rerank: {msg.timings.reranking_sec ? msg.timings.reranking_sec.toFixed(2) : '0.00'}s</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 w-full p-6 bg-white/80 backdrop-blur-md border-t border-slate-200 z-10">
        <div className="max-w-4xl mx-auto">
            {/* Controls Bar */}
            <div className="flex justify-end mb-2">
                <label className={`flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full border transition-all cursor-pointer select-none ${useCache ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'}`}>
                    <input 
                        type="checkbox" 
                        checked={useCache}
                        onChange={(e) => setUseCache(e.target.checked)}
                        className="hidden"
                    />
                    <Zap className={`w-3.5 h-3.5 ${useCache ? 'fill-indigo-700' : ''}`} />
                    <span>Semantic Cache {useCache ? 'On' : 'Off'}</span>
                </label>
            </div>

            <form onSubmit={handleSend} className="relative">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your documents..."
                className="w-full pl-6 pr-14 py-4 bg-white border border-slate-200 rounded-2xl shadow-lg focus:outline-none focus:ring-2 focus:ring-primary/50 text-slate-700"
            />
            <button 
                type="submit" 
                disabled={!input.trim() || loading}
                className="absolute right-2 top-2 p-2 bg-primary text-white rounded-xl hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <Send className="w-5 h-5" />
            </button>
            </form>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
