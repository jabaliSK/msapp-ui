import axios from 'axios';

// --- DYNAMIC CONFIGURATION ---
// Automatically detects the host (localhost or network IP) running the UI
const HOST = window.location.hostname; 

// Service 1: RAG & Chat Service (Port 8050)
export const RAG_API_BASE = `http://${HOST}:8060`;

// Service 2: File Management Service (Port 8051)
export const FILE_API_BASE = `http://${HOST}:8061`;

// --- FILES SERVICE ENDPOINTS (Port 8051) ---
export const uploadFile = (formData) => axios.post(`${FILE_API_BASE}/upload`, formData);
export const getFiles = () => axios.get(`${FILE_API_BASE}/files`);
export const deleteFile = (fileId) => axios.delete(`${FILE_API_BASE}/files/${fileId}`);
export const getDownloadUrl = (fileId) => `${FILE_API_BASE}/files/download/${fileId}`;

// --- RAG SERVICE ENDPOINTS (Port 8050) ---

// Chat & Benchmarks
export const chatWithBot = (query, useCache=true) => axios.post(`${RAG_API_BASE}/chat`, { query, use_cache: useCache });

// Conversation (Streaming fetch wrapper)
export const conversationWithBot = (messages) =>
  fetch(`${RAG_API_BASE}/conversation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, use_cache: false }) 
  });

// Configuration
export const getModels = () => axios.get(`${RAG_API_BASE}/models`);
export const getConfig = () => axios.get(`${RAG_API_BASE}/config`);
export const updateConfig = (config) => axios.post(`${RAG_API_BASE}/config`, config);

// Logs & Stats
export const getLogs = (params) => axios.get(`${RAG_API_BASE}/logs`, { params });
export const truncateLogs = () => axios.delete(`${RAG_API_BASE}/logs`);

// Cache
export const clearCache = () => axios.delete(`${RAG_API_BASE}/milvus/cache/clear`);

// --- HELPER: MODEL SCANNING ---
// Scans ports on the same host to see if models are running
export const checkModelPort = async (port) => {
  try {
    const response = await axios.get(`http://${HOST}:${port}/v1/models`, { 
      timeout: 500 
    });
    return { port, alive: true, models: response.data.data };
  } catch (error) {
    return { port, alive: false, models: [] };
  }
};