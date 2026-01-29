// Scraper.tsx
import React, { useState, useEffect, useRef } from 'react';
import { Globe, Play, Loader2, CheckCircle, AlertCircle, Link as LinkIcon, FileJson } from 'lucide-react';
import { ApiService, ScrapeStatus } from '../services/apiService';

// Hardcoded from config.py as requested
const TARGET_URLS = [
  "https://www.msheireb.com/",
  "https://www.msheireb.com/ar/",
  "https://www.msheirebproperties.com/",
  "https://www.msheirebproperties.com/ar/",
  "https://msheirebmuseums.com/en/",
  "https://msheirebmuseums.com/ar/"
];

export const Scraper: React.FC = () => {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<ScrapeStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const pollInterval = useRef<NodeJS.Timeout | null>(null);

  // Poll for status when taskId exists and status is running
  useEffect(() => {
    if (taskId) {
      pollInterval.current = setInterval(async () => {
        const data = await ApiService.getScrapeStatus(taskId);
        if (data) {
          setStatus(data);
          if (data.status === 'completed' || data.status === 'failed') {
            if (pollInterval.current) clearInterval(pollInterval.current);
          }
        }
      }, 1000);
    }
    return () => {
      if (pollInterval.current) clearInterval(pollInterval.current);
    };
  }, [taskId]);

  const handleStartScrape = async () => {
    setIsStarting(true);
    try {
      // Default to 100 pages as per config
      const res = await ApiService.startScraping(100);
      setTaskId(res.task_id);
      setStatus({ 
        task_id: res.task_id, 
        status: 'running', 
        progress: 0, 
        pages_scraped: 0, 
        total: 100, 
        current_url: 'Initializing...' 
      });
    } catch (e) {
      alert("Failed to start scraping job.");
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      {/* Header Section */}
      <div className="bg-white p-8 rounded-xl shadow-sm border border-msheireb-sand flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
        <div>
          <h2 className="text-2xl font-serif font-bold text-msheireb-charcoal flex items-center gap-3">
            <Globe className="text-msheireb-gold" />
            Web Crawler
          </h2>
          <p className="text-gray-500 mt-2 text-sm max-w-xl">
            Ingest content directly from Msheireb web properties. The crawler will visit the configured URLs, extract text, and automatically index it into the vector database.
          </p>
        </div>
        
        <button
          onClick={handleStartScrape}
          disabled={isStarting || status?.status === 'running'}
          className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
            status?.status === 'running' 
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-msheireb-charcoal text-white hover:bg-msheireb-gold hover:text-msheireb-charcoal shadow-md hover:shadow-lg'
          }`}
        >
          {isStarting ? <Loader2 className="animate-spin" size={20} /> : <Play size={20} />}
          <span>{status?.status === 'running' ? 'Job Running...' : 'Start Indexing Job'}</span>
        </button>
      </div>

      {/* Active Job Status Card */}
      {status && (
        <div className={`bg-white rounded-xl shadow-sm border overflow-hidden transition-all duration-500 ${status.status === 'failed' ? 'border-red-200' : 'border-msheireb-sand'}`}>
           <div className="px-6 py-4 bg-msheireb-limestone/30 border-b border-msheireb-sand flex justify-between items-center">
              <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">Current Job: {status.task_id.slice(0, 8)}...</span>
              <div className="flex items-center gap-2">
                 {status.status === 'running' && <span className="flex w-3 h-3 bg-blue-500 rounded-full animate-pulse"></span>}
                 <span className={`text-sm font-bold uppercase ${status.status === 'completed' ? 'text-green-600' : status.status === 'failed' ? 'text-red-600' : 'text-blue-600'}`}>
                    {status.status}
                 </span>
              </div>
           </div>
           
           <div className="p-8">
              {/* Progress Bar */}
              <div className="mb-2 flex justify-between text-sm font-medium text-msheireb-charcoal">
                <span>Progress</span>
                <span>{status.progress}%</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-4 mb-6 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-500 ease-out ${status.status === 'failed' ? 'bg-red-500' : 'bg-msheireb-gold'}`}
                  style={{ width: `${status.progress}%` }}
                ></div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                 <div className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                    <p className="text-xs text-gray-400 uppercase">Pages Scraped</p>
                    <p className="text-xl font-bold text-msheireb-charcoal">{status.pages_scraped} <span className="text-sm text-gray-400 font-normal">/ {status.total}</span></p>
                 </div>
                 <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 col-span-2">
                    <p className="text-xs text-gray-400 uppercase">Currently Processing</p>
                    <p className="text-sm font-medium text-msheireb-charcoal truncate" title={status.current_url}>
                      {status.current_url || "Waiting for worker..."}
                    </p>
                 </div>
              </div>

              {/* Completion Message */}
              {status.status === 'completed' && (
                <div className="bg-green-50 border border-green-100 p-4 rounded-lg flex items-start gap-3">
                  <CheckCircle className="text-green-600 mt-0.5" size={20} />
                  <div>
                    <h4 className="font-bold text-green-800">Indexing Complete</h4>
                    <p className="text-sm text-green-700">
                      Data saved to <strong>{status.output_file}</strong> and processed into the vector database.
                    </p>
                  </div>
                </div>
              )}
              
              {status.status === 'failed' && (
                 <div className="bg-red-50 border border-red-100 p-4 rounded-lg flex items-start gap-3">
                    <AlertCircle className="text-red-600 mt-0.5" size={20} />
                    <div>
                       <h4 className="font-bold text-red-800">Job Failed</h4>
                       <p className="text-sm text-red-700">{status.error || "Unknown error occurred"}</p>
                    </div>
                 </div>
              )}
           </div>
        </div>
      )}

      {/* Target URLs List */}
      <div className="bg-white rounded-xl shadow-sm border border-msheireb-sand overflow-hidden">
        <div className="px-6 py-4 border-b border-msheireb-sand bg-gray-50 flex items-center gap-2">
           <LinkIcon size={16} className="text-msheireb-bronze" />
           <h3 className="font-bold text-msheireb-charcoal">Target Data Sources</h3>
        </div>
        <div className="divide-y divide-gray-100">
          {TARGET_URLS.map((url, idx) => (
            <div key={idx} className="px-6 py-4 flex items-center justify-between group hover:bg-msheireb-limestone/30 transition-colors">
              <div className="flex items-center gap-4">
                 <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-400 group-hover:bg-white group-hover:text-msheireb-gold group-hover:shadow-sm transition-all">
                    <Globe size={16} />
                 </div>
                 <a href={url} target="_blank" rel="noreferrer" className="text-sm font-medium text-gray-600 hover:text-msheireb-gold transition-colors">
                    {url}
                 </a>
              </div>
              
              {/* Visual Indicator if this URL is being scraped currently */}
              {status?.status === 'running' && status.current_url.includes(url) && (
                 <div className="flex items-center gap-2 text-xs font-bold text-blue-600 animate-pulse">
                    <Loader2 size={14} className="animate-spin" />
                    Scanning
                 </div>
              )}
            </div>
          ))}
        </div>
        <div className="bg-gray-50 px-6 py-3 text-xs text-gray-400 text-center border-t border-gray-100">
           Configuration is read-only. To modify targets, update <code>config.py</code> on the server.
        </div>
      </div>
    </div>
  );
};