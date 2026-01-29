import { CouchbaseDocument, MilvusChunk } from '../types';

// In-memory storage simulation
let couchbaseStore: CouchbaseDocument[] = [
  {
    id: 'doc-001',
    title: 'Msheireb Sustainability Report 2024',
    type: 'pdf',
    size: 2450000,
    uploadedAt: new Date(Date.now() - 86400000 * 2).toISOString(),
    updatedAt: new Date(Date.now() - 86400000 * 2).toISOString(),
    author: 'Admin',
    summary: 'Detailed analysis of LEED Platinum certifications and water conservation strategies in Downtown Doha.',
    tags: ['sustainability', 'LEED', 'water'],
    status: 'synced',
    contentUrl: 'mock-url-1'
  },
  {
    id: 'doc-002',
    title: 'Architectural Blueprint: Barahat',
    type: 'cad',
    size: 15400000,
    uploadedAt: new Date(Date.now() - 86400000 * 5).toISOString(),
    updatedAt: new Date(Date.now() - 86400000 * 1).toISOString(),
    author: 'Eng. Sarah',
    summary: 'Technical drawings for the Barahat Msheireb town square cooling systems.',
    tags: ['architecture', 'blueprint', 'cooling'],
    status: 'synced',
    contentUrl: 'mock-url-2'
  }
];

let milvusStore: MilvusChunk[] = [];

// Helper to generate mock vectors
const generateMockVector = (dim = 128): number[] => {
  return Array.from({ length: dim }, () => Math.random());
};

// SIMULATED TRANSACTION MANAGER
// Ensures Atomicity between Couchbase and Milvus
export const MockBackend = {
  getDocuments: async (): Promise<CouchbaseDocument[]> => {
    await new Promise(r => setTimeout(r, 500)); // Network delay
    return [...couchbaseStore];
  },

  getStats: async () => {
    await new Promise(r => setTimeout(r, 300));
    return {
      totalDocuments: couchbaseStore.length,
      totalChunks: milvusStore.length + (couchbaseStore.length * 12), // Mocking chunk count if empty
      storageUsed: `${(couchbaseStore.reduce((acc, doc) => acc + doc.size, 0) / 1024 / 1024).toFixed(2)} MB`,
      lastSync: new Date().toISOString()
    };
  },

  // The "Upload" process
  uploadDocument: async (
    file: File, 
    generatedMetadata: Partial<CouchbaseDocument>, 
    chunks: string[]
  ): Promise<CouchbaseDocument> => {
    console.log('Starting Distributed Transaction...');
    
    await new Promise(r => setTimeout(r, 1500)); // Simulate processing time

    const newDocId = `doc-${Date.now()}`;
    
    // 1. Prepare Couchbase Document
    const newDoc: CouchbaseDocument = {
      id: newDocId,
      title: file.name,
      type: file.name.split('.').pop() || 'unknown',
      size: file.size,
      uploadedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      author: 'CurrentUser',
      status: 'processing', // Initially processing
      tags: [],
      ...generatedMetadata
    };

    // 2. Prepare Milvus Chunks
    const newChunks: MilvusChunk[] = chunks.map((text, index) => ({
      id: `chunk-${newDocId}-${index}`,
      docId: newDocId,
      text: text,
      sequence: index,
      vector: generateMockVector() // Simulating embedding generation
    }));

    try {
      // 3. Commit to Couchbase (Mock)
      couchbaseStore = [newDoc, ...couchbaseStore];
      console.log(`[Couchbase] Document ${newDocId} inserted.`);

      // 4. Commit to Milvus (Mock)
      // In a real app, if this fails, we roll back Couchbase
      milvusStore = [...milvusStore, ...newChunks];
      console.log(`[Milvus] ${newChunks.length} vectors inserted for ${newDocId}.`);

      // 5. Update Status to Synced
      couchbaseStore = couchbaseStore.map(d => d.id === newDocId ? { ...d, status: 'synced' } : d);
      
      return { ...newDoc, status: 'synced' };

    } catch (error) {
      console.error("Transaction failed", error);
      // Rollback logic would go here
      throw new Error("Distributed Transaction Failed");
    }
  },

  updateDocument: async (doc: CouchbaseDocument): Promise<void> => {
    console.log(`[Update] Updating document ${doc.id}`);
    await new Promise(r => setTimeout(r, 800)); // Network delay

    // Update locally
    couchbaseStore = couchbaseStore.map(d => d.id === doc.id ? { ...doc, updatedAt: new Date().toISOString() } : d);
    
    console.log(`[Couchbase] Metadata updated for ${doc.id}`);
    // Note: In a full implementation, if content changed, we would update Milvus here too.
  },

  deleteDocument: async (docId: string): Promise<void> => {
    await new Promise(r => setTimeout(r, 800));
    
    // 1. Delete from Couchbase
    couchbaseStore = couchbaseStore.filter(d => d.id !== docId);
    console.log(`[Couchbase] Document ${docId} deleted.`);

    // 2. Delete from Milvus
    const initialCount = milvusStore.length;
    milvusStore = milvusStore.filter(c => c.docId !== docId);
    console.log(`[Milvus] ${initialCount - milvusStore.length} vectors deleted for ${docId}.`);
  },

  search: async (query: string): Promise<CouchbaseDocument[]> => {
    await new Promise(r => setTimeout(r, 600));
    // Simple mock search on title or tags
    const lowerQ = query.toLowerCase();
    return couchbaseStore.filter(d => 
      d.title.toLowerCase().includes(lowerQ) || 
      d.tags.some(t => t.toLowerCase().includes(lowerQ)) ||
      (d.summary && d.summary.toLowerCase().includes(lowerQ))
    );
  }
};