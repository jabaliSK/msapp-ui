import { CouchbaseDocument, SystemStats } from '../types';

const API_BASE_URL = 'http://localhost:8061';

export interface DashboardData {
  stats: SystemStats;
  distribution: { name: string; value: number }[];
  activity: { name: string; docs: number; vectors: number }[];
}

export interface ExtendedDocument extends CouchbaseDocument {
  versions: {
    version: number;
    id: string;
    uploadDate: string;
    size: number;
    chunkCount: number;
    usageCount: number;
  }[];
}

export interface ScrapeStatus {
  task_id: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  pages_scraped: number;
  total: number;
  current_url: string;
  output_file?: string;
  error?: string;
}

export const ApiService = {
  getDashboardStats: async (): Promise<DashboardData | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      return await response.json();
    } catch (error) {
      console.error("Stats API Error:", error);
      return null;
    }
  },

  getDocuments: async (): Promise<ExtendedDocument[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/files`);
      if (!response.ok) throw new Error('Failed to fetch files');
      
      const fileGroups: any[] = await response.json();
      
      return fileGroups.flatMap(group => {
        const latestMeta = group.latest;
        
        // FIX: Combine 'latest' + 'versions' so the UI sees ALL versions including the current one.
        const allVersionsRaw = [latestMeta, ...group.versions];

        // Map to UI format and Sort descending by version number
        const versionHistory = allVersionsRaw.map((v: any) => ({
            version: v.version,
            id: v.id,
            uploadDate: v.upload_date,
            size: v.file_size,
            chunkCount: v.chunk_count || 0,
            usageCount: v.usage_count || 0
        })).sort((a: any, b: any) => b.version - a.version);

        return {
          id: latestMeta.id,
          title: latestMeta.filename,
          type: latestMeta.file_type.toLowerCase(),
          size: latestMeta.file_size,
          uploadedAt: latestMeta.upload_date,
          updatedAt: latestMeta.upload_date,
          author: 'System', 
          // Summary now reflects the true latest version from the history array
          summary: `Current: v${latestMeta.version} • ${latestMeta.chunk_count} chunks`, 
          tags: [latestMeta.file_type, `v${latestMeta.version}`], 
          status: 'synced',
          contentUrl: `${API_BASE_URL}/files/download/${latestMeta.id}`,
          versions: versionHistory
        } as ExtendedDocument;
      });
    } catch (error) {
      console.error("API Error:", error);
      return [];
    }
  },

  uploadDocument: async (file: File): Promise<void> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData });
    if (!response.ok) throw new Error(await response.text() || 'Upload failed');
  },

  deleteDocumentVersion: async (id: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/files/${id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Delete failed');
  },

  deleteAllVersions: async (filename: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/files/all/${filename}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Delete failed');
  },

  downloadDocument: (id: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `${API_BASE_URL}/files/download/${id}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  },

  startScraping: async (maxPages: number = 100): Promise<{ task_id: string }> => {
    try {
      const response = await fetch(`${API_BASE_URL}/scrape/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_pages: maxPages, store_result: true })
      });
      if (!response.ok) throw new Error('Failed to start scraper');
      return await response.json();
    } catch (error) {
      console.error("Scrape API Error:", error);
      throw error;
    }
  },

  getScrapeStatus: async (taskId: string): Promise<ScrapeStatus | null> => {
    try {
      const response = await fetch(`${API_BASE_URL}/scrape/status/${taskId}`);
      if (!response.ok) throw new Error('Failed to fetch status');
      return await response.json();
    } catch (error) {
      console.error("Scrape Status Error:", error);
      return null;
    }
  }
};