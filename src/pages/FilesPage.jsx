import React, { useState, useEffect } from 'react';
import { getFiles, uploadFile, deleteFile, getDownloadUrl } from '../api';
import { UploadCloud, Trash2, FileText, Check, AlertCircle, FolderOpen, ChevronRight, ChevronDown, Download, Layers, Clock, Eye } from 'lucide-react';

const FilesPage = () => {
  const [fileGroups, setFileGroups] = useState([]);
  const [expandedFiles, setExpandedFiles] = useState(new Set());
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const fetchList = async () => {
    try {
      const res = await getFiles();
      setFileGroups(res.data);
    } catch (err) {
      console.error("Failed to load files", err);
    }
  };

  useEffect(() => {
    fetchList();
  }, []);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setStatusMsg(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await uploadFile(formData);
      setStatusMsg({ type: 'success', text: `Uploaded ${res.data.filename} (Version ${res.data.version}) successfully.` });
      fetchList();
    } catch (err) {
      if (err.response && err.response.status === 409) {
        setStatusMsg({ type: 'error', text: 'Duplicate file content detected. Upload rejected.' });
      } else {
        setStatusMsg({ type: 'error', text: 'Upload failed. Check server logs.' });
      }
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (file_id, filename) => {
    if (!window.confirm(`Delete ${filename}? This cannot be undone.`)) return;
    try {
        await deleteFile(file_id);
        fetchList();
    } catch (err) {
        alert("Failed to delete file");
    }
  };

  const toggleExpand = (filename) => {
    const newSet = new Set(expandedFiles);
    if (newSet.has(filename)) newSet.delete(filename);
    else newSet.add(filename);
    setExpandedFiles(newSet);
  };

  const formatBytes = (bytes, decimals = 2) => {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
  };

  const formatDate = (isoString) => {
      return new Date(isoString).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  };

  // --- FIXED FileRow Component ---
  const FileRow = ({ file, isVersion = false, hasVersions = false, versionCount = 0, isExpanded = false, onExpand }) => (
    <div className={`grid grid-cols-12 p-4 border-b border-slate-100 items-center transition-colors ${isVersion ? 'bg-slate-50/50 hover:bg-slate-100' : 'hover:bg-slate-50'}`}>
        {/* Name Column */}
        <div className="col-span-5 flex items-center gap-3 overflow-hidden">
            {hasVersions && !isVersion ? (
                <button onClick={onExpand} className="text-slate-400 hover:text-primary transition-colors">
                    {isExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                </button>
            ) : <div className="w-5" />} {/* Spacer */}
            
            <div className={`w-8 h-8 rounded flex items-center justify-center ${isVersion ? 'bg-slate-200 text-slate-500' : 'bg-blue-50 text-blue-500'}`}>
                {isVersion ? <Layers className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
            </div>
            
            <div className="flex flex-col truncate pr-4">
                <span className={`truncate font-medium ${isVersion ? 'text-slate-600' : 'text-slate-800'}`}>
                    {file.filename} {isVersion && <span className="text-xs bg-slate-200 px-1.5 py-0.5 rounded text-slate-600 ml-1">v{file.version}</span>}
                </span>
                {/* FIXED: Use versionCount prop instead of file.versions.length */}
                {!isVersion && hasVersions && <span className="text-xs text-slate-400">{versionCount} versions available</span>}
            </div>
        </div>

        {/* Metadata Columns */}
        <div className="col-span-1 text-slate-500 text-sm">{file.file_type}</div>
        <div className="col-span-1 text-slate-500 text-sm font-mono">{formatBytes(file.file_size)}</div>
        <div className="col-span-1 text-slate-500 text-sm font-mono">{file.chunk_count} Chunks</div>
        <div className="col-span-1 flex items-center gap-1 text-slate-600 text-sm">
            <Eye className="w-3 h-3 text-slate-400" /> {file.usage_count}
        </div>
        <div className="col-span-2 text-slate-400 text-xs flex items-center gap-1">
            <Clock className="w-3 h-3" /> {formatDate(file.upload_date)}
        </div>

        {/* Actions */}
        <div className="col-span-1 flex justify-end gap-2">
            <a 
                href={getDownloadUrl(file.id)} 
                target="_blank" 
                rel="noreferrer"
                className="p-1.5 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded transition-colors"
                title="Download"
            >
                <Download className="w-4 h-4" />
            </a>
            <button 
                onClick={() => handleDelete(file.id, file.filename)} 
                className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                title="Delete"
            >
                <Trash2 className="w-4 h-4" />
            </button>
        </div>
    </div>
  );

  return (
    <div className="p-8 max-w-7xl mx-auto h-screen flex flex-col">
      <div className="flex justify-between items-end mb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Knowledge Base</h1>
          <p className="text-slate-500 mt-1">Manage documents. Uploading a file with the same name creates a new version.</p>
        </div>
        
        <label className={`cursor-pointer flex items-center gap-2 bg-primary text-white px-5 py-2 rounded-lg hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-200 ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
          <UploadCloud className="w-5 h-5" />
          <span>{uploading ? 'Uploading...' : 'Upload File'}</span>
          <input type="file" className="hidden" onChange={handleUpload} accept=".pdf,.docx,.txt,.jpg,.png,.jpeg" />
        </label>
      </div>

      {statusMsg && (
        <div className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${statusMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
          {statusMsg.type === 'success' ? <Check className="w-4 h-4"/> : <AlertCircle className="w-4 h-4"/>}
          {statusMsg.text}
        </div>
      )}

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 overflow-hidden flex flex-col">
        {/* Header */}
        <div className="grid grid-cols-12 bg-slate-50 p-4 border-b border-slate-200 font-semibold text-slate-600 text-sm">
          <div className="col-span-5 pl-8">File Name</div>
          <div className="col-span-1">Type</div>
          <div className="col-span-1">Size</div>
          <div className="col-span-1">Chunks</div>
          <div className="col-span-1">Usage</div>
          <div className="col-span-2">Uploaded</div>
          <div className="col-span-1 text-right">Actions</div>
        </div>
        
        <div className="overflow-y-auto flex-1">
          {fileGroups.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <FolderOpen className="w-16 h-16 mb-4 opacity-20" />
              <p>No files indexed yet.</p>
            </div>
          ) : (
            fileGroups.map((group) => (
              <div key={group.filename} className="border-b border-slate-100 last:border-0">
                  {/* Latest Version (Parent) */}
                  <FileRow 
                    file={group.latest} 
                    hasVersions={group.versions.length > 0}
                    versionCount={group.versions.length} // Pass count explicitly
                    isExpanded={expandedFiles.has(group.filename)}
                    onExpand={() => toggleExpand(group.filename)}
                  />
                  
                  {/* Version History (Children) */}
                  {expandedFiles.has(group.filename) && group.versions.map(ver => (
                      <FileRow key={ver.id} file={ver} isVersion={true} />
                  ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default FilesPage;
