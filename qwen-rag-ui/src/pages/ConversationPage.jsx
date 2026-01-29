import React, { useState, useRef, useEffect } from 'react';
import { RAG_API_BASE } from '../api'; // Import the correct base URL
import { Send, Bot, User, ShieldCheck, Database, AlertTriangle, MessageSquare, Trash2 } from 'lucide-react';

const ConversationPage = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: "Hello! I am in Conversation Mode. I remember context from our previous messages." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const clearConversation = () => {
    if(window.confirm("Clear history and start a new conversation?")) {
        setMessages([{ role: 'assistant', text: "Conversation cleared. Ready for a new topic!" }]);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userQuery = input;
    setInput("");
    
    const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.text
    }));
    historyPayload.push({ role: 'user', content: userQuery });

    setMessages(prev => [...prev, { role: 'user', text: userQuery }]);
    
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      text: "", 
      sources: [], 
      safety: null 
    }]);

    setLoading(true);

    try {
      // FIX: Used RAG_API_BASE here instead of undefined API_BASE
      const response = await fetch(`${RAG_API_BASE}/conversation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: historyPayload })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value, { stream: true });
        
        const lines = chunkValue.split('\n').filter(line => line.trim() !== '');

        for (const line of lines) {
          try {
            const json = JSON.parse(line);

            if (json.type === 'token') {
              setMessages(prev => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg.role !== 'assistant') return prev;
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
            console.error("Stream parse error", e);
          }
        }
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: "Connection Error: " + err.message }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 relative">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex justify-between items-center shadow-sm z-20">
        <div>
            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-indigo-600"/> Conversation Mode
            </h2>
            <p className="text-xs text-slate-500">Multi-turn chat with history memory.</p>
        </div>
        <button 
            onClick={clearConversation}
            className="text-red-500 hover:bg-red-50 p-2 rounded-lg transition-colors flex items-center gap-2 text-sm"
        >
            <Trash2 className="w-4 h-4" /> Reset
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 pb-40 space-y-6">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-600'}`}>
              {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
            </div>

            <div className={`max-w-2xl`}>
              <div className={`p-4 rounded-2xl shadow-sm text-sm leading-relaxed ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-white border border-slate-200 text-slate-700 rounded-tl-none'}`}>
                <span className="whitespace-pre-wrap">{msg.text}</span>
                {loading && idx === messages.length - 1 && (
                  <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-slate-400 animate-pulse"></span>
                )}
              </div>

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
                      <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full ${msg.safety === 'PASSED' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {msg.safety === 'PASSED' ? <ShieldCheck className="w-3 h-3"/> : <AlertTriangle className="w-3 h-3"/>}
                        {msg.safety}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="absolute bottom-0 left-0 w-full p-6 bg-white/80 backdrop-blur-md border-t border-slate-200 z-10">
        <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSend} className="relative">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a follow-up question..."
                className="w-full pl-6 pr-14 py-4 bg-white border border-slate-200 rounded-2xl shadow-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/50 text-slate-700"
            />
            <button 
                type="submit" 
                disabled={!input.trim() || loading}
                className="absolute right-2 top-2 p-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
                <Send className="w-5 h-5" />
            </button>
            </form>
        </div>
      </div>
    </div>
  );
};

export default ConversationPage;