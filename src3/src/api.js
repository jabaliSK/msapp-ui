import axios from 'axios';

// The Main Python API
const API_BASE = "http://10.25.73.101:8050"; 

export const getLogs = () => axios.get(`${API_BASE}/logs`);
export const uploadFile = (formData) => axios.post(`${API_BASE}/upload`, formData);
export const getFiles = () => axios.get(`${API_BASE}/files`);
export const deleteFile = (filename) => axios.delete(`${API_BASE}/files/${filename}`);
export const chatWithBot = (query) => axios.post(`${API_BASE}/chat`, { query });
export const clearCache = () => axios.delete(`${API_BASE}/milvus/cache/clear`);

// --- NEW CONFIG API CALLS ---
export const getModels = () => axios.get(`${API_BASE}/models`);
export const getConfig = () => axios.get(`${API_BASE}/config`);
export const updateConfig = (configData) => axios.post(`${API_BASE}/config`, configData);
