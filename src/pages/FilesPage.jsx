import React, { useState, useEffect } from 'react';
import { getFiles, uploadFile, deleteFile } from '../api';
// Added FolderOpen to the top import list here:
import { UploadCloud, Trash2, FileText, Check, AlertCircle, FolderOpen } from 'lucide-react';

const FilesPage = () => {
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [uploading, setUploading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const fetchList = async () => {
    try {
      const res = await getFiles();
      setFiles(res.data);
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
      await uploadFile(formData);
      setStatusMsg({ type: 'success', text: `Uploaded ${file.name} successfully.` });
      fetchList();
    } catch (err) {
      setStatusMsg({ type: 'error', text: 'Upload failed.' });
    } finally {
      setUploading(false);
    }
  };

  const toggleSelect = (filename) => {
    const newSet = new Set(selectedFiles);
    if (newSet.has(filename)) newSet.delete(filename);
    else newSet.add(filename);
    setSelectedFiles(newSet);
  };

  const handleBulkDelete = async () => {
    if (!window.confirm(`Delete ${selectedFiles.size} files?`)) return;
    
    // API only supports single delete, so we loop
    const promises = Array.from(selectedFiles).map(name => deleteFile(name));
    await Promise.all(promises);
    setSelectedFiles(new Set());
    fetchList();
  };

  const handleDeleteOne = async (filename) => {
    if (!window.confirm(`Delete ${filename}?`)) return;
    await deleteFile(filename);
    fetchList();
  };

  return (
    <div className="p-8 max-w-6xl mx-auto h-screen flex flex-col">
      <div className="flex justify-between items-end mb-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-800">Knowledge Base</h1>
          <p className="text-slate-500 mt-1">Manage documents, PDFs, and images for RAG retrieval.</p>
        </div>
        
        <div className="flex gap-3">
          {selectedFiles.size > 0 && (
            <button 
              onClick={handleBulkDelete}
              className="flex items-center gap-2 bg-red-50 text-red-600 px-4 py-2 rounded-lg hover:bg-red-100 transition-colors"
            >
              <Trash2 className="w-4 h-4" /> Delete ({selectedFiles.size})
            </button>
          )}
          
          <label className={`cursor-pointer flex items-center gap-2 bg-primary text-white px-5 py-2 rounded-lg hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-200 ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
            <UploadCloud className="w-5 h-5" />
            <span>{uploading ? 'Uploading...' : 'Upload File'}</span>
            <input type="file" className="hidden" onChange={handleUpload} accept=".pdf,.docx,.txt,.jpg,.png,.jpeg" />
          </label>
        </div>
      </div>

      {statusMsg && (
        <div className={`mb-4 p-3 rounded-lg flex items-center gap-2 ${statusMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
          {statusMsg.type === 'success' ? <Check className="w-4 h-4"/> : <AlertCircle className="w-4 h-4"/>}
          {statusMsg.text}
        </div>
      )}

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex-1 overflow-hidden flex flex-col">
        <div className="grid grid-cols-12 bg-slate-50 p-4 border-b border-slate-200 font-medium text-slate-600 text-sm">
          <div className="col-span-1 text-center">Select</div>
          <div className="col-span-10">Filename</div>
          <div className="col-span-1 text-center">Action</div>
        </div>
        
        <div className="overflow-y-auto flex-1">
          {files.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <FolderOpen className="w-16 h-16 mb-4 opacity-20" />
              <p>No files indexed yet.</p>
            </div>
          ) : (
            files.map((file, idx) => (
              <div key={file} className="grid grid-cols-12 p-4 border-b border-slate-100 hover:bg-slate-50 items-center transition-colors">
                <div className="col-span-1 flex justify-center">
                  <input 
                    type="checkbox" 
                    checked={selectedFiles.has(file)}
                    onChange={() => toggleSelect(file)}
                    className="w-4 h-4 text-primary rounded border-slate-300 focus:ring-primary"
                  />
                </div>
                <div className="col-span-10 flex items-center gap-3">
                  <div className="w-8 h-8 rounded bg-blue-50 flex items-center justify-center text-blue-500">
                    <FileText className="w-4 h-4" />
                  </div>
                  <span className="text-slate-700 font-medium truncate">{file}</span>
                </div>
                <div className="col-span-1 flex justify-center">
                  <button onClick={() => handleDeleteOne(file)} className="text-slate-400 hover:text-red-500">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default FilesPage;
