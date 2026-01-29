# config.py
import os
import psycopg2
from pymongo import MongoClient
from pymilvus import MilvusClient, DataType
from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime
from psycopg2.extras import Json

# --- CONSTANTS & NETWORK ---
TOOL_HOST_IP = "http://10.25.73.101"
TOOL_HOST_IP_ONLY = "10.25.73.101"

MILVUS_URI = f"{TOOL_HOST_IP}:19530"
COLLECTION_NAME = "qwen_rag_collection"
CACHE_COLLECTION = "qwen_rag_cache"

MONGO_URI = f"mongodb://admin:admin@{TOOL_HOST_IP_ONLY}:27017/"
MONGO_DB_NAME = "rag-files"
MONGO_COLL_NAME = "rag-files"

PG_CONFIG = {
    "dbname": "rag_stats",
    "user": "jabali",
    "password": "jabali",
    "host": TOOL_HOST_IP_ONLY,
    "port": "5432"
}

# --- SHARED STATE ---
# Note: In a production split, configuration updates should be stored in a DB 
# or Redis so both services stay in sync. Here we define defaults.
default_config = {
    "llm_port": 8000,
    "llm_model_id": "Qwen/Qwen3-8B-FP8",
    "embed_port": 8006,
    "embed_model_id": "ibm-granite/granite-embedding-278m-multilingual",
    "guard_port": 8001,
    "guard_model_id": "Qwen/Qwen3Guard-Gen-0.6B",
    "vl_port": 8003,
    "vl_model_id": "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "rerank_port": 8007,
    "rerank_model_id": "Qwen/Qwen3-Reranker-0.6B",
    "system_prompt": "You are a helpful assistant. Answer the user's question using the provided context. If the context lacks the answer, say I don't know.",
    "cache_threshold": 0.92,
    "llm_temperature": 0.1,
}

# --- SCRAPER CONFIGURATION ---
SCRAPER_SETTINGS = {
    "PRIMARY_URLS": [
        "https://www.msheireb.com/",
        "https://www.msheireb.com/ar/",
        "https://www.msheirebproperties.com/",
        "https://www.msheirebproperties.com/ar/",
        "https://msheirebmuseums.com/en/",
        "https://msheirebmuseums.com/ar/",
    ],
    "ALLOWED_DOMAINS": [
        'msheireb.com',
        'msheirebproperties.com', 
        'msheirebmuseums.com'
    ],
    "SKIP_EXTENSIONS": [
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
        '.zip', '.rar', '.tar', '.gz', '.7z',
        '.mp4', '.mp3', '.avi', '.mov', '.wmv', '.flv',
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.exe', '.dmg', '.pkg', '.deb', '.rpm'
    ],
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "MAX_PAGES_DEFAULT": 100, # Reduced default for API safety
    "DELAY_BETWEEN_REQUESTS": 0.5,
    "TIMEOUT": 30
}

# --- DATABASE CLIENTS ---
# Initialize generic clients used by both apps
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_coll = mongo_db[MONGO_COLL_NAME]

milvus = MilvusClient(uri=MILVUS_URI)

def get_pg_connection():
    try:
        return psycopg2.connect(**PG_CONFIG)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

# --- PYDANTIC MODELS ---

class ProcessTimings(BaseModel):
    input_guardrail_sec: float
    embedding_sec: float
    retrieval_sec: float
    reranking_sec: float
    llm_generation_sec: float
    output_guardrail_sec: float
    total_sec: float

class FileMetadata(BaseModel):
    id: str
    filename: str
    version: int
    upload_date: datetime
    file_size: int
    file_type: str
    chunk_count: int
    usage_count: int

class FileGroup(BaseModel):
    filename: str
    latest: FileMetadata
    versions: List[FileMetadata]

class FileUploadResponse(BaseModel):
    filename: str
    version: int
    status: str
    extracted_text_preview: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    use_cache: Optional[bool] = True

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]
    use_cache: Optional[bool] = True

class LogEntry(BaseModel):
    id: int
    timestamp: datetime
    user_query: str
    model_response: Optional[str] = None
    status: Optional[str] = None
    safety_check_result: Optional[str] = None
    sources: List[str] = []
    model_id_llm: Optional[str] = None
    model_id_embed: Optional[str] = None
    model_id_rerank: Optional[str] = None
    total_duration_sec: Optional[float] = 0.0
    llm_generation_sec: Optional[float] = 0.0
    retrieval_sec: Optional[float] = 0.0
    reranking_sec: Optional[float] = 0.0
    embedding_sec: Optional[float] = 0.0
    input_guardrail_sec: Optional[float] = 0.0
    output_guardrail_sec: Optional[float] = 0.0

class ConfigUpdate(BaseModel):
    llm_port: int
    llm_model_id: str
    llm_temperature: float
    embed_port: int
    embed_model_id: str
    guard_port: int
    guard_model_id: str
    vl_port: int
    vl_model_id: str
    rerank_port: int
    rerank_model_id: str
    system_prompt: str
    cache_threshold: float


class ScrapeRequest(BaseModel):
    max_pages: Optional[int] = 100
    store_result: Optional[bool] = True