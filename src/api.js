import axios from 'axios';

// The Main Python API
const API_BASE = "http://10.25.73.101:8050";

export const uploadFile = (formData) => axios.post(`${API_BASE}/upload`, formData);
export const getFiles = () => axios.get(`${API_BASE}/files`);
export const deleteFile = (filename) => axios.delete(`${API_BASE}/files/${filename}`);
export const chatWithBot = (query) => axios.post(`${API_BASE}/chat`, { query });
export const clearCache = () => axios.delete(`${API_BASE}/milvus/cache/clear`);
// Helper to scan a specific model port
export const checkModelPort = async (port) => {
  try {
    // We assume standard OpenAI compatible /v1/models endpoint
    const response = await axios.get(`http://localhost:${port}/v1/models`, { 
      timeout: 500 // 0.5s timeout requirement
    });
    return { port, alive: true, models: response.data.data };
  } catch (error) {
    return { port, alive: false, models: [] };
  }
};
