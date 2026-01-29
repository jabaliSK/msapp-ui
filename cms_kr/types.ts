// Data Models reflecting the Dual-DB Architecture

// Represents the Document Metadata stored in Couchbase
export interface CouchbaseDocument {
  id: string;
  title: string;
  type: string;
  size: number;
  uploadedAt: string;
  updatedAt: string;
  author: string;
  summary?: string;
  tags: string[];
  status: 'synced' | 'processing' | 'error';
  contentUrl?: string; // Mock URL
}

// Represents the Vector Chunk stored in Milvus
export interface MilvusChunk {
  id: string; // Unique Chunk ID
  docId: string; // Foreign Key to Couchbase Document
  vector: number[]; // The embedding (mocked)
  text: string; // The actual chunk text
  sequence: number; // Order in document
}

export interface SystemStats {
  totalDocuments: number;
  totalChunks: number;
  storageUsed: string;
  lastSync: string;
}

export type ViewState = 'login' | 'dashboard' | 'documents' | 'settings';
