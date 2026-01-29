import React, { useState, useEffect } from 'react';
import { 
  Upload, Trash2, FileText, CheckCircle, Loader2, RefreshCw, X, 
  Cpu, Download, History, Clock, Database, BarChart2
} from 'lucide-react';
import { ApiService, ExtendedDocument } from '../services/apiService';

export const DocumentManager: React.FC = () => {
  const [docs, setDocs] = useState<ExtendedDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{status: string, progress: number} | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<ExtendedDocument | null>(null);

  useEffect(() => { loadDocs(); }, []);

  const loadDocs = async () => {
    setLoading(true);
    const data = await ApiService.getDocuments();
    setDocs(data);
    setLoading(false);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if(!file) return;
    setUploading(true);
    setUploadProgress({ status: 'Uploading...', progress: 30 });
    try {
        await ApiService.uploadDocument(file);
        setUploadProgress({ status: 'Complete', progress: 100 });
        await new Promise(r => setTimeout(r, 500));
        loadDocs();
    } catch(e) { alert("Upload failed"); } 
    finally { setUploading(false); setUploadProgress(null); }
  };

  // Helper to safely calculate totals for the main list
  const getTotalChunks = (doc: ExtendedDocument) => doc.versions?.reduce((acc, v) => acc + (v.chunkCount || 0), 0) || 0;
  const getTotalUsage = (doc: ExtendedDocument) => doc.versions?.reduce((acc, v) => acc + (v.usageCount || 0), 0) || 0;

  const handleDeleteAll = async (filename: string) => {
    if (confirm(`Are you sure you want to delete "${filename}" and ALL its versions? This cannot be undone.`)) {
        try {
            await ApiService.deleteAllVersions(filename);
            setSelectedDoc(null);
            loadDocs();
        } catch (e) { alert("Failed to delete file group."); }
    }
  };

  const handleDeleteVersion = async (versionId: string, versionNum: number) => {
    if (confirm(`Delete version ${versionNum} only?`)) {
        try {
            await ApiService.deleteDocumentVersion(versionId);
            const updatedDocs = await ApiService.getDocuments();
            setDocs(updatedDocs);
            
            // Update the selected view if the document still exists
            if (selectedDoc) {
                const updatedSelected = updatedDocs.find(d => d.title === selectedDoc.title);
                setSelectedDoc(updatedSelected || null);
            }
        } catch (e) { alert("Failed to delete version."); }
    }
  };

  return (
     <div className="flex gap-6 h-full relative">
       {/* List Area */}
       <div className="flex-1 space-y-6 animate-fade-in pb-12 overflow-y-auto">
         {/* Upload Widget */}
         <div className="bg-white rounded-xl border-2 border-dashed border-msheireb-sand p-8 text-center hover:border-msheireb-gold transition-colors relative overflow-hidden group">
            <input type="file" id="fileInput" className="hidden" onChange={handleFileUpload} disabled={uploading} />
            <label htmlFor="fileInput" className="cursor-pointer flex flex-col items-center">
              {uploading ? (
                 <div className="text-msheireb-gold animate-pulse">{uploadProgress?.status}</div>
              ) : (
                <>
                  <Upload className="text-msheireb-gold mb-4" size={32} />
                  <h3 className="text-lg font-bold text-msheireb-charcoal">Upload Document</h3>
                  <p className="text-gray-400 text-sm">Supports Versioning & Auto-Vectorization</p>
                </>
              )}
            </label>
         </div>

         {/* Docs Table */}
         <div className="bg-white rounded-xl shadow-sm border border-msheireb-sand overflow-hidden">
             <div className="px-6 py-4 border-b border-msheireb-sand bg-gray-50/50 flex justify-between">
                <h3 className="font-bold text-msheireb-charcoal">Repository</h3>
                <button onClick={loadDocs} className="hover:bg-gray-200 p-1 rounded-full"><RefreshCw size={16} /></button>
             </div>
             {docs.length === 0 && !loading && (
                 <div className="p-8 text-center text-gray-400">No documents found.</div>
             )}
             {docs.map(doc => (
                 <div 
                   key={doc.id} 
                   onClick={() => setSelectedDoc(doc)}
                   className={`px-6 py-4 border-b border-msheireb-sand/50 hover:bg-msheireb-limestone/50 cursor-pointer flex items-center justify-between group ${selectedDoc?.id === doc.id ? 'bg-msheireb-limestone' : ''}`}
                 >
                    <div className="flex items-center gap-3">
                        <FileText className="text-msheireb-charcoal" size={20} />
                        <div>
                            <p className="font-medium text-msheireb-charcoal">{doc.title}</p>
                            <p className="text-xs text-gray-400">v{doc.versions?.[0]?.version || 1} • {(doc.size/1024/1024).toFixed(2)} MB</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-6">
                        <div className="hidden md:flex items-center gap-4 text-xs text-gray-400">
                             <div className="flex items-center gap-1" title="Total Chunks (All Versions)">
                                <Database size={12} /> {getTotalChunks(doc)}
                             </div>
                             <div className="flex items-center gap-1" title="Total Retrievals (All Versions)">
                                <BarChart2 size={12} /> {getTotalUsage(doc)}
                             </div>
                        </div>
                        <div className="flex flex-col items-end">
                             <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">Versions</span>
                             <span className="text-sm font-medium text-msheireb-gold">{doc.versions?.length || 1}</span>
                        </div>
                    </div>
                 </div>
             ))}
         </div>
       </div>

       {/* File Details Sidebar */}
       {/* FIX: Changed 'inset-y-0' to 'top-20 bottom-0' to avoid overlapping with the main header */}
       <div 
         className={`fixed top-20 bottom-0 right-0 w-[500px] bg-white shadow-2xl transform transition-transform duration-300 z-40 flex flex-col border-l border-msheireb-sand ${selectedDoc ? 'translate-x-0' : 'translate-x-full'}`}
       >
         {selectedDoc && (
            <>
                {/* Fixed Header with Close Button */}
                <div className="p-6 border-b border-msheireb-sand flex justify-between items-start bg-msheireb-limestone/30 flex-shrink-0">
                    <div>
                        <h2 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Filename</h2>
                        <h3 className="text-xl font-serif font-bold text-msheireb-charcoal break-all leading-tight">
                            {selectedDoc.title}
                        </h3>
                    </div>
                    {/* Explicit Close Button */}
                    <button 
                        onClick={() => setSelectedDoc(null)} 
                        className="p-2 ml-4 hover:bg-gray-200 rounded-full text-gray-500 transition-colors flex-shrink-0"
                        title="Close Sidebar"
                    >
                        <X size={24} />
                    </button>
                </div>

                {/* Scrollable Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    
                    {/* Version History & Stats */}
                    <div className="bg-white border border-msheireb-sand rounded-xl overflow-hidden shadow-sm">
                        <div className="bg-msheireb-limestone/50 px-4 py-3 border-b border-msheireb-sand flex items-center gap-2">
                            <History size={16} className="text-msheireb-gold" />
                            <span className="text-sm font-bold text-msheireb-charcoal">Version History & Stats</span>
                        </div>
                        <div className="divide-y divide-msheireb-sand/30">
                            {selectedDoc.versions?.map((v) => (
                                <div key={v.id} className="p-4 hover:bg-gray-50 transition-colors group">
                                    {/* Version Header */}
                                    <div className="flex items-center justify-between mb-2">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${v.version === selectedDoc.versions[0].version ? 'bg-msheireb-gold text-white' : 'bg-gray-200 text-gray-600'}`}>
                                                v{v.version}
                                            </div>
                                            <div>
                                                <p className="text-xs text-gray-500 flex items-center gap-1">
                                                    <Clock size={10} />
                                                    {new Date(v.uploadDate).toLocaleDateString()}
                                                    <span className="text-gray-300">|</span>
                                                    {(v.size / 1024).toFixed(1)} KB
                                                </p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                            <button 
                                                onClick={() => ApiService.downloadDocument(v.id, `v${v.version}_${selectedDoc.title}`)}
                                                className="p-2 text-gray-400 hover:text-msheireb-gold hover:bg-msheireb-sand/20 rounded-full"
                                                title="Download"
                                            >
                                                <Download size={16} />
                                            </button>
                                            <button 
                                                onClick={() => handleDeleteVersion(v.id, v.version)}
                                                className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full"
                                                title="Delete this version"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </div>

                                    {/* Granular Stats Per Version */}
                                    <div className="flex items-center gap-2 mt-2">
                                        <div className="flex-1 bg-msheireb-limestone/50 rounded px-2 py-1.5 flex items-center justify-between border border-msheireb-sand/30">
                                            <span className="text-[10px] uppercase text-gray-500 font-bold flex items-center gap-1">
                                                <Database size={10} /> Chunks
                                            </span>
                                            <span className="text-xs font-mono font-bold text-msheireb-charcoal">{v.chunkCount}</span>
                                        </div>
                                        <div className="flex-1 bg-msheireb-limestone/50 rounded px-2 py-1.5 flex items-center justify-between border border-msheireb-sand/30">
                                            <span className="text-[10px] uppercase text-gray-500 font-bold flex items-center gap-1">
                                                <BarChart2 size={10} /> Retrievals
                                            </span>
                                            <span className="text-xs font-mono font-bold text-blue-600">{v.usageCount}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Insight Box */}
                    <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                        <div className="flex items-center gap-2 mb-2">
                            <Cpu size={16} className="text-blue-500" />
                            <span className="font-bold text-sm text-blue-800">RAG Engine Insight</span>
                        </div>
                        <p className="text-xs text-blue-600 leading-relaxed">
                            Retrieval count indicates how often chunks from a specific version were used to answer user queries. High retrieval counts on older versions may indicate they contain enduring knowledge.
                        </p>
                    </div>
                </div>

                {/* Footer Action */}
                <div className="p-6 border-t border-msheireb-sand bg-gray-50 flex-shrink-0">
                    <button 
                        onClick={() => handleDeleteAll(selectedDoc.title)} 
                        className="w-full py-3 border border-red-200 text-red-600 font-medium rounded-lg hover:bg-red-50 flex items-center justify-center gap-2 transition-colors"
                    >
                        <Trash2 size={18} /> Delete File & All Versions
                    </button>
                </div>
            </>
         )}
       </div>
     </div>
  );
};