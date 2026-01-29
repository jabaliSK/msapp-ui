import axios from 'axios';

// The Main Python API
export const API_BASE = "http://10.25.73.101:8050";

// --- FILES ---
export const uploadFile = (formData) => axios.post(`${API_BASE}/upload`, formData);
export const getFiles = () => axios.get(`${API_BASE}/files`);
export const deleteFile = (fileId) => axios.delete(`${API_BASE}/files/${fileId}`);
export const getDownloadUrl = (fileId) => `${API_BASE}/files/download/${fileId}`;

// --- CHAT & BENCHMARK ---
// Standard non-streaming chat (used by BenchmarkPage)
export const chatWithBot = (query, useCache=true) => axios.post(`${API_BASE}/chat`, { query, use_cache: useCache });

// --- CONFIGURATION ---
export const getModels = () => axios.get(`${API_BASE}/models`);
export const getConfig = () => axios.get(`${API_BASE}/config`);
export const updateConfig = (config) => axios.post(`${API_BASE}/config`, config);

// --- LOGS & STATS ---
export const getLogs = (params) => axios.get(`${API_BASE}/logs`, { params });
export const truncateLogs = () => axios.delete(`${API_BASE}/logs`);

// --- CACHE ---
export const clearCache = () => axios.delete(`${API_BASE}/milvus/cache/clear`);

// --- HELPER: LOCAL MODEL SCANNING ---
export const checkModelPort = async (port) => {
  try {
    const response = await axios.get(`http://localhost:${port}/v1/models`, { 
      timeout: 500 
    });
    return { port, alive: true, models: response.data.data };
  } catch (error) {
    return { port, alive: false, models: [] };
  }
};


export const conversationWithBot = (messages) =>
  fetch(`${API_BASE}/conversation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, use_cache: false }) // Cache usually off for convo to force context processing
  });
