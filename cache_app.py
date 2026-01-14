import os
import io
import time
import json
import random 
import asyncio 
from datetime import datetime, timedelta
import base64
import zipfile
import hashlib
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
from pymilvus import MilvusClient, DataType
from openai import OpenAI
import uvicorn
from pypdf import PdfReader
from docx import Document
from PIL import Image
import itertools
import requests
import psycopg2 
from psycopg2.extras import Json
from pymongo import MongoClient
from bson.binary import Binary
from bson.objectid import ObjectId
import urllib.parse

# --- CONFIGURATION ---
TOOL_HOST_IP = "http://10.25.73.101"
TOOL_HOST_IP_ONLY = "10.25.73.101"

MILVUS_URI = TOOL_HOST_IP+":19530"
COLLECTION_NAME = "qwen_rag_collection"
CACHE_COLLECTION = "qwen_rag_cache"

# MongoDB Config
MONGO_URI = f"mongodb://admin:admin@{TOOL_HOST_IP_ONLY}:27017/"
MONGO_DB_NAME = "rag-files"
MONGO_COLL_NAME = "rag-files"

# Global Dimension State
CURRENT_DIMENSION = 1024

PG_CONFIG = {
    "dbname": "rag_stats",
    "user": "jabali",      
    "password": "jabali",  
    "host": TOOL_HOST_IP_ONLY,
    "port": "5432"
}

# Default Configuration State
current_config = {
    "llm_port": 8000,
    "llm_model_id": "Qwen/Qwen3-8B-FP8",
    "embed_port": 8006,
    "embed_model_id": "ibm-granite/granite-embedding-278m-multilingual",
    "guard_port": 8001,
    "guard_model_id": "Qwen/Qwen3Guard-Gen-0.6B",
    "vl_port": 8003,
    "vl_model_id": "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "rerank_port": 8007,                                # <--- NEW
    "rerank_model_id": "Qwen/Qwen3-Reranker-0.6B",      # <--- NEW
    "system_prompt": "You are a helpful assistant. Answer the user's question using the provided context. If the context lacks the answer, say I don't know.",
    "cache_threshold": 0.92,
    "llm_temperature": 0.1,
}

# Clients
client_llm = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['llm_port']}/v1", api_key="EMPTY")
client_guard = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['guard_port']}/v1", api_key="EMPTY")
client_embed = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['embed_port']}/v1", api_key="EMPTY")
client_vl = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['vl_port']}/v1", api_key="EMPTY")

# Initialize Milvus
milvus = MilvusClient(uri=MILVUS_URI)

# Initialize MongoDB
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
mongo_coll = mongo_db[MONGO_COLL_NAME]

# --- FILLER PHRASES ---
FILLER_PHRASES = [
    "Let me take a brief moment to check on those details for you right now.",
    "Please bear with me for just a moment while I look that up for you.",
    "I'm going to quickly consult the knowledge base to find the exact answer.",
    "I am scanning through the documents right now to find what you need.",
    "Give me just one second while I locate the relevant information for this request."
]

# --- DYNAMIC DIMENSION DETECTION ---
def detect_embedding_dimension():
    global CURRENT_DIMENSION
    print(f"Attempting to detect embedding dimension from {current_config['embed_model_id']}...")
    try:
        response = client_embed.embeddings.create(
            input=["warmup_ping"],
            model=current_config["embed_model_id"]
        )
        if response.data and len(response.data) > 0:
            detected_dim = len(response.data[0].embedding)
            if detected_dim != CURRENT_DIMENSION:
                print(f" [INFO] Detected Dimension Change: {CURRENT_DIMENSION} -> {detected_dim}")
                CURRENT_DIMENSION = detected_dim
            else:
                print(f" [INFO] Dimension verified: {CURRENT_DIMENSION}")
        else:
            print(" [WARN] Embedding response empty. Keeping default dimension.")
    except Exception as e:
        print(f" [ERROR] Failed to detect embedding dimension: {e}. Keeping default {CURRENT_DIMENSION}.")

# --- DATABASE SETUP ---
def init_db():
    detect_embedding_dimension()

    if not milvus.has_collection(COLLECTION_NAME):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=CURRENT_DIMENSION)
        schema.add_field(field_name="chunk_md5", datatype=DataType.VARCHAR, max_length=64)

        index_params = milvus.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="chunk_md5", index_type="STL_SORT")

        milvus.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
        print(f"Collection {COLLECTION_NAME} created with dimension {CURRENT_DIMENSION}.")

    if not milvus.has_collection(CACHE_COLLECTION):
        milvus.create_collection(
            collection_name=CACHE_COLLECTION,
            dimension=CURRENT_DIMENSION, 
            metric_type="COSINE", 
            auto_id=True,
            enable_dynamic_field=True 
        )
        print(f"Cache Collection {CACHE_COLLECTION} created with dimension {CURRENT_DIMENSION}.")
    
    try: milvus.load_collection(CACHE_COLLECTION)
    except Exception as e: print(f"Loading notice: {e}")

# --- Pydantic Models ---
class ProcessTimings(BaseModel):
    input_guardrail_sec: float
    embedding_sec: float
    retrieval_sec: float
    reranking_sec: float    # <--- NEW
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
    model_id_rerank: Optional[str] = None # <--- NEW
    total_duration_sec: Optional[float] = 0.0
    llm_generation_sec: Optional[float] = 0.0
    retrieval_sec: Optional[float] = 0.0
    reranking_sec: Optional[float] = 0.0  # <--- NEW
    embedding_sec: Optional[float] = 0.0
    input_guardrail_sec: Optional[float] = 0.0
    output_guardrail_sec: Optional[float] = 0.0

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]
    use_cache: Optional[bool] = True

# --- OCR & VISION SERVICES (Unchanged) ---
def encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

def perform_ocr(image_bytes: bytes, filename_context: str) -> str:
    # ... [Same as original] ...
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB': img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_image = encode_image_base64(buffered.getvalue())
        response = client_vl.chat.completions.create(
            model=current_config["vl_model_id"],
            messages=[{"role": "user", "content": [{"type": "text", "text": "Transcribe the text in this image word for word."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}],
            max_tokens=512, temperature=0.1
        )
        extracted = response.choices[0].message.content
        return f"\n[OCR Content from Image in {filename_context}]:\n{extracted}\n"
    except Exception as e:
        print(f"OCR Failed for {filename_context}: {e}")
        return ""

def parse_file_bytes(file_bytes: bytes, filename: str) -> str:
    # ... [Same as original] ...
    content = ""
    extension = filename.split(".")[-1].lower()
    if extension in ["jpg", "jpeg", "png"]:
        content += perform_ocr(file_bytes, filename)
    elif extension == "pdf":
        try:
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(pdf_reader.pages):
                txt = page.extract_text()
                if txt: content += txt + "\n"
                for img_file_obj in page.images:
                    content += perform_ocr(img_file_obj.data, f"{filename} page {i+1}")
        except Exception as e: print(f"PDF Parse Error: {e}")
    elif extension in ["docx", "doc"]:
        try:
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs: content += para.text + "\n"
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    if name.startswith("word/media/"):
                        content += perform_ocr(z.read(name), f"{filename} embedded {name}")
        except Exception as e: print(f"Could not extract from docx: {e}")
    elif extension == "txt":
        content = file_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return content

# --- DB & LOGGING ---
def get_db_connection():
    try: return psycopg2.connect(**PG_CONFIG)
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def log_interaction_to_db(query, response, status, safety_check, sources, timings, config_snapshot):
    conn = get_db_connection()
    if not conn: return
    try:
        cur = conn.cursor()
        # Updated SQL to include reranking columns
        insert_query = """
        INSERT INTO interaction_logs (
            user_query, model_response, status, safety_check_result, sources, 
            model_id_llm, model_id_embed, model_id_rerank, 
            total_duration_sec, input_guardrail_sec, embedding_sec, 
            retrieval_sec, reranking_sec, llm_generation_sec, output_guardrail_sec,
            client_latency_sec, config_snapshot
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
            query, response, status, safety_check, sources, 
            config_snapshot.get("llm_model_id"), config_snapshot.get("embed_model_id"), config_snapshot.get("rerank_model_id"),
            timings.get("total_sec", 0), timings.get("input_guardrail_sec", 0), timings.get("embedding_sec", 0), 
            timings.get("retrieval_sec", 0), timings.get("reranking_sec", 0), timings.get("llm_generation_sec", 0), timings.get("output_guardrail_sec", 0),
            0.0, Json(config_snapshot)
        ))
        conn.commit()
        cur.close()
    except Exception as e: print(f"Failed to log interaction: {e}")
    finally: conn.close()

# --- RAG CORE ---

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    response = client_embed.embeddings.create(input=[text], model=current_config["embed_model_id"])
    return response.data[0].embedding

def check_safety(text: str, context_type: str = "User Query") -> bool:
    prompt = f"You are a safety moderation tool. Analyze the following {context_type}. If it contains harmful/illegal content, respond 'UNSAFE'. Otherwise 'SAFE'. {context_type}: {text}"
    try:
        response = client_guard.chat.completions.create(
            model=current_config["guard_model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        if "UNSAFE" in response.choices[0].message.content.strip().upper(): return False
        return True
    except Exception: return True

# --- NEW RERANKING FUNCTION ---
def perform_reranking(query: str, documents: List[dict]) -> List[dict]:
    """
    Sends documents to the local reranker service and returns them sorted by relevance.
    Input 'documents' is a list of dicts from Milvus hits (must contain 'text').
    """
    if not documents: return []
    
    url = f"{TOOL_HOST_IP}:{current_config['rerank_port']}/v1/rerank"
    
    # Extract just text for the model
    doc_texts = [d["text"] for d in documents]
    
    payload = {
        "model": current_config["rerank_model_id"],
        "query": query,
        "documents": doc_texts,
        "top_n": len(documents) # Get scores for all, we slice later
    }
    
    try:
        # Standard TEI/vLLM/OpenAI-compatible Rerank API
        response = requests.post(url, json=payload, timeout=2.0)
        response.raise_for_status()
        results = response.json()
        
        # Expected response: {"results": [{"index": 0, "relevance_score": 0.99}, ...]}
        if "results" in results:
            # Sort original documents based on returned indices and scores
            sorted_docs = []
            for item in results["results"]:
                idx = item["index"]
                # Attach score for debugging if needed
                documents[idx]["score"] = item["relevance_score"]
                sorted_docs.append(documents[idx])
            return sorted_docs
        else:
            return documents # Fallback: return as is
            
    except Exception as e:
        print(f"Rerank failed (using retrieval order): {e}")
        return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_index(filename: str, version: int, text: str):
    chunks = chunk_text(text)
    versioned_filename = f"{filename} (v{version})"
    print(f"Processing {len(chunks)} chunks for {versioned_filename}...")
    
    chunk_hashes = []
    for chunk in chunks:
        c_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
        chunk_hashes.append(c_hash)
        try:
            res = milvus.query(collection_name=COLLECTION_NAME, filter=f'chunk_md5 == "{c_hash}"', output_fields=["id", "filenames"])
            if res:
                existing_entity = res[0]
                existing_files = existing_entity.get("filenames", [])
                if isinstance(existing_files, str): existing_files = [existing_files]
                if versioned_filename not in existing_files:
                    existing_files.append(versioned_filename)
                    milvus.upsert(collection_name=COLLECTION_NAME, data=[{"id": existing_entity["id"], "vector": get_embedding(chunk), "text": chunk, "filenames": existing_files, "chunk_md5": c_hash}])
            else:
                milvus.insert(collection_name=COLLECTION_NAME, data=[{"vector": get_embedding(chunk), "text": chunk, "filenames": [versioned_filename], "chunk_md5": c_hash}])
        except Exception as e: print(f"Error chunk {c_hash[:8]}: {e}")

    try:
        mongo_coll.update_one({"filename": filename, "version": version}, {"$set": {"chunk_hashes": chunk_hashes, "chunk_count": len(chunks)}})
    except Exception as e: print(f"Failed to update MongoDB: {e}")


def contextualize_query(history: List[Message], latest_query: str) -> str:
    """
    Uses the LLM to rewrite the latest query into a standalone question 
    based on the conversation history.
    """
    if not history:
        return latest_query

    # Create a concise history string
    history_str = "\n".join([f"{msg.role.title()}: {msg.content}" for msg in history[-4:]]) # Limit to last 4 turns for speed

    prompt = (
        f"Given the chat history and the latest user question which might reference context in the chat history, "
        f"formulate a standalone question which can be understood without the chat history. "
        f"Do NOT answer the question, just reformulate it if needed and otherwise return it as is.\n\n"
        f"Chat History:\n{history_str}\n\n"
        f"User Question: {latest_query}\n\n"
        f"Standalone Question:"
    )

    try:
        response = client_llm.chat.completions.create(
            model=current_config["llm_model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=128,
            stream=False
        )
        reformulated = response.choices[0].message.content.strip()
        print(f" [INFO] Query Reformulated: '{latest_query}' -> '{reformulated}'")
        return reformulated
    except Exception as e:
        print(f" [WARN] Contextualization failed: {e}")
        return latest_query


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
    rerank_port: int      # <--- NEW
    rerank_model_id: str  # <--- NEW
    system_prompt: str
    cache_threshold: float

# --- API ---

app = FastAPI(title="Qwen Multi-Modal RAG API", on_startup=[init_db])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/logs", response_model=List[LogEntry])
def get_logs(limit: int = 100, hours: Optional[int] = None):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cur = conn.cursor()
        # Updated SELECT query
        sql = """SELECT id, timestamp, user_query, model_response, status, safety_check_result, sources,
                model_id_llm, model_id_embed, model_id_rerank, 
                total_duration_sec, llm_generation_sec, retrieval_sec, reranking_sec, embedding_sec,
                input_guardrail_sec, output_guardrail_sec, client_latency_sec, config_snapshot
            FROM interaction_logs """
        params = []
        if hours:
            sql += " WHERE timestamp >= NOW() - INTERVAL '%s hours'"
            params.append(hours)
        sql += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        logs = []
        for r in rows:
            logs.append({
                "id": r[0], "timestamp": r[1], "user_query": r[2], "model_response": r[3],
                "status": r[4], "safety_check_result": r[5], "sources": r[6] if r[6] else [],
                "model_id_llm": r[7], "model_id_embed": r[8], "model_id_rerank": r[9],
                "total_duration_sec": r[10], "llm_generation_sec": r[11],
                "retrieval_sec": r[12], "reranking_sec": r[13], "embedding_sec": r[14],
                "input_guardrail_sec": r[15], "output_guardrail_sec": r[16],
            })
        return logs
    finally:
        if conn: conn.close()

# ... [truncate_logs, upload_file, list_files, download_file, delete_file remain UNCHANGED] ...
@app.delete("/logs")
def truncate_logs():
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE interaction_logs")
        conn.commit()
        return {"status": "Logs truncated"}
    finally: conn.close()

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        md5_hash = hashlib.md5(file_bytes).hexdigest()
        siblings = list(mongo_coll.find({"filename": file.filename}))
        for sib in siblings:
            if sib.get('md5') == md5_hash:
                raise HTTPException(status_code=409, detail=f"Duplicate content found in version {sib.get('version')}")
        new_version = 1
        if siblings: new_version = max([s.get('version', 1) for s in siblings]) + 1

        extracted_content = parse_file_bytes(file_bytes, file.filename)
        file_type = file.filename.split('.')[-1].upper()

        doc = {"filename": file.filename, "version": new_version, "upload_date": datetime.utcnow(), "md5": md5_hash, "file_size": len(file_bytes), "file_type": file_type, "chunk_count": 0, "usage_count": 0, "content": Binary(file_bytes), "chunk_hashes": []}
        mongo_coll.insert_one(doc)

        if extracted_content.strip():
            background_tasks.add_task(process_and_index, file.filename, new_version, extracted_content)
        
        return {"filename": file.filename, "version": new_version, "status": "File stored and processing", "extracted_text_preview": extracted_content[:50] + "..."}
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/files", response_model=List[FileGroup])
def list_files():
    try:
        all_docs = list(mongo_coll.find({}, {"content": 0}).sort([("filename", 1), ("version", -1)]))
        groups = {}
        for doc in all_docs:
            fname = doc["filename"]
            meta = FileMetadata(id=str(doc["_id"]), filename=fname, version=doc.get("version", 1), upload_date=doc["upload_date"], file_size=doc.get("file_size", 0), file_type=doc.get("file_type", "UNK"), chunk_count=doc.get("chunk_count", 0), usage_count=doc.get("usage_count", 0))
            if fname not in groups: groups[fname] = {"filename": fname, "latest": meta, "versions": []}
            else: groups[fname]["versions"].append(meta)
        return list(groups.values())
    except Exception as e:
        print(f"Error listing files: {e}"); return []

@app.get("/files/download/{file_id}")
def download_file(file_id: str):
    try:
        doc = mongo_coll.find_one({"_id": ObjectId(file_id)})
        if not doc: raise HTTPException(status_code=404, detail="File not found")
        file_stream = io.BytesIO(doc["content"])
        safe_filename = urllib.parse.quote(f"{doc['filename']}")
        return StreamingResponse(file_stream, media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"})
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    try:
        doc = mongo_coll.find_one({"_id": ObjectId(file_id)})
        if not doc: raise HTTPException(status_code=404, detail="File not found")
        filename = doc["filename"]
        version = doc.get("version", 1)
        versioned_name = f"{filename} (v{version})"
        chunk_hashes = doc.get("chunk_hashes", [])
        for c_hash in chunk_hashes:
            try:
                res = milvus.query(collection_name=COLLECTION_NAME, filter=f'chunk_md5 == "{c_hash}"', output_fields=["id", "filenames"])
                if res:
                    entity = res[0]
                    curr = entity.get("filenames", [])
                    if isinstance(curr, str): curr = [curr]
                    if versioned_name in curr:
                        curr.remove(versioned_name)
                        if not curr: milvus.delete(collection_name=COLLECTION_NAME, ids=[entity["id"]])
                        else: milvus.upsert(collection_name=COLLECTION_NAME, data=[{**entity, "filenames": curr}])
            except Exception: pass
        mongo_coll.delete_one({"_id": ObjectId(file_id)})
        return {"status": "deleted", "id": file_id}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models():
    mods = {}
    for port in itertools.chain(range(8000, 8012), [8100, 8200]):
        try:
            m = requests.get(TOOL_HOST_IP+f":{port}/v1/models", timeout=.2).json()["data"][0]["id"]
            mods[m] = port
        except: pass 
    return(mods)

@app.get("/config")
def get_config(): return current_config

@app.post("/config")
def update_config(cfg: ConfigUpdate):
    global client_llm, client_guard, client_embed, client_vl, current_config
    old_embed_model = current_config.get("embed_model_id")
    try:
        current_config.update(cfg.dict())
        client_llm = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.llm_port}/v1", api_key="EMPTY")
        client_guard = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.guard_port}/v1", api_key="EMPTY")
        client_embed = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.embed_port}/v1", api_key="EMPTY")
        client_vl = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.vl_port}/v1", api_key="EMPTY")
        
        if old_embed_model != cfg.embed_model_id:
            print(f"Embedding model changed. Re-checking dimension...")
            milvus.drop_collection(COLLECTION_NAME)
            milvus.drop_collection(CACHE_COLLECTION)
            init_db()
        return {"status": "Updated", "config": current_config}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.delete("/milvus/cache/clear")
def clear_milvus_cache():
    milvus.drop_collection(CACHE_COLLECTION)
    init_db()
    return {"status": "Cache cleared"}

@app.post("/chat")
async def chat_rag_stream(request: ChatRequest):
    async def response_generator():
        t_start_total = time.perf_counter()
        timings = {
            "input_guardrail_sec": 0.0, "embedding_sec": 0.0, 
            "retrieval_sec": 0.0, "reranking_sec": 0.0, # <--- NEW
            "llm_generation_sec": 0.0, "output_guardrail_sec": 0.0, 
            "total_sec": 0.0
        }
        
        filler = random.choice(FILLER_PHRASES)
        yield json.dumps({"type": "token", "content": f"{filler}\n\n"}) + "\n"
        
        # --- 2. INPUT GUARD ---
        t_start = time.perf_counter()
        if not check_safety(request.query):
            timings["input_guardrail_sec"] = time.perf_counter() - t_start
            timings["total_sec"] = time.perf_counter() - t_start_total
            yield json.dumps({"type": "token", "content": "I cannot answer this query due to safety policies."}) + "\n"
            yield json.dumps({"type": "meta", "safety": "FAILED_INPUT", "timings": timings, "sources": []}) + "\n"
            log_interaction_to_db(request.query, "Safety Violation", "Safety Violation", "FAILED_INPUT", [], timings, current_config)
            return
        timings["input_guardrail_sec"] = time.perf_counter() - t_start

        # --- 3. EMBEDDING ---
        t_start = time.perf_counter()
        query_vector = get_embedding(request.query)
        timings["embedding_sec"] = time.perf_counter() - t_start
        
        # --- 4. CACHE CHECK ---
        if request.use_cache:
            t_start = time.perf_counter()
            cache_res = milvus.search(collection_name=CACHE_COLLECTION, data=[query_vector], limit=1, output_fields=["response"])
            if cache_res and cache_res[0] and cache_res[0][0]["distance"] >= current_config["cache_threshold"]:
                timings["retrieval_sec"] = time.perf_counter() - t_start
                cached_resp = cache_res[0][0]["entity"]["response"]
                timings["total_sec"] = time.perf_counter() - t_start_total
                yield json.dumps({"type": "token", "content": cached_resp}) + "\n"
                yield json.dumps({"type": "meta", "safety": "PASSED_CACHE", "timings": timings, "sources": ["SEMANTIC_CACHE"]}) + "\n"
                log_interaction_to_db(request.query, cached_resp, "Success (Cache)", "PASSED_CACHE", ["SEMANTIC_CACHE"], timings, current_config)
                return

        # --- 5. RETRIEVAL (Fetch 15 candidates) ---
        t_start = time.perf_counter()
        # Fetch more candidates for the reranker
        search_res = milvus.search(collection_name=COLLECTION_NAME, data=[query_vector], limit=15, output_fields=["text", "filenames"])
        timings["retrieval_sec"] = (time.perf_counter() - t_start)
        
        candidate_chunks = []
        if search_res and search_res[0]:
            for hit in search_res[0]:
                candidate_chunks.append({
                    "text": hit["entity"]["text"],
                    "filenames": hit["entity"].get("filenames", [])
                })
        
        # --- 5.1 RERANKING ---
        t_start_rerank = time.perf_counter()
        # Rerank and slice to Top 3
        reranked_chunks = perform_reranking(request.query, candidate_chunks)
        top_chunks = reranked_chunks[:3]
        timings["reranking_sec"] = time.perf_counter() - t_start_rerank # <--- Record Time

        retrieved_text = [c["text"] for c in top_chunks]
        sources = set()
        for c in top_chunks:
            fnames = c.get("filenames", [])
            if isinstance(fnames, list): 
                for f in fnames: sources.add(f)
            else: sources.add(fnames)
        
        if sources:
             for s in sources:
                try:
                    if " (v" in s:
                        name_part = s.rsplit(" (v", 1)[0]
                        ver_part = int(s.rsplit(" (v", 1)[1].replace(")", ""))
                        mongo_coll.update_one({"filename": name_part, "version": ver_part}, {"$inc": {"usage_count": 1}})
                    else:
                        mongo_coll.update_one({"filename": s}, {"$inc": {"usage_count": 1}})
                except: pass

        # --- 6. LLM GENERATION ---
        context = "\n\n".join(retrieved_text)
        t_start_gen = time.perf_counter()
        full_response_accumulator = ""
        
        try:
            stream = client_llm.chat.completions.create(
                model=current_config["llm_model_id"], 
                messages=[
                    {"role": "system", "content": current_config["system_prompt"]}, 
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
                ],
                temperature=current_config["llm_temperature"], 
                max_tokens=300, stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response_accumulator += content
                    yield json.dumps({"type": "token", "content": content}) + "\n"
            
        except Exception as e:
            error_msg = f"\nError generating response: {str(e)}"
            full_response_accumulator += error_msg
            yield json.dumps({"type": "token", "content": error_msg}) + "\n"

        timings["llm_generation_sec"] = time.perf_counter() - t_start_gen

        # --- 7. GUARDRAIL & LOGGING ---
        t_start_guard = time.perf_counter()
        is_safe = check_safety(full_response_accumulator, "AI Response")
        timings["output_guardrail_sec"] = time.perf_counter() - t_start_guard
        status = "PASSED" if is_safe else "FAILED_OUTPUT"
        
        if is_safe and "i don't know" not in full_response_accumulator.lower():
            milvus.insert(collection_name=CACHE_COLLECTION, data=[{"vector": query_vector, "response": full_response_accumulator}])
        
        timings["total_sec"] = time.perf_counter() - t_start_total

        log_interaction_to_db(request.query, full_response_accumulator, "Success" if is_safe else "Unsafe Content", status, list(sources), timings, current_config)

        yield json.dumps({
            "type": "meta",
            "sources": list(sources),
            "safety": status,
            "timings": timings,
            "llm_model": current_config["llm_model_id"],
            "embed_model": current_config["embed_model_id"],
            "rerank_model": current_config["rerank_model_id"],
            "guard_model": current_config["guard_model_id"]
        }) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")


@app.post("/conversation")
async def conversation_rag_stream(request: ConversationRequest):
    async def response_generator():
        t_start_total = time.perf_counter()
        timings = {
            "input_guardrail_sec": 0.0, "embedding_sec": 0.0, 
            "retrieval_sec": 0.0, "reranking_sec": 0.0,
            "llm_generation_sec": 0.0, "output_guardrail_sec": 0.0, 
            "total_sec": 0.0
        }
        
        # --- 1. CLEAN & VALIDATE MESSAGES ---
        # Filter out specific frontend artifacts or empty messages
        clean_messages = [
            m for m in request.messages 
            if m.content.strip() and "Conversation cleared" not in m.content
        ]

        if not clean_messages or clean_messages[-1].role != 'user':
            yield json.dumps({"type": "token", "content": "Error: Valid user query not found."}) + "\n"
            return

        last_query = clean_messages[-1].content
        history = clean_messages[:-1] 

        # Send Filler
        filler = random.choice(FILLER_PHRASES)
        yield json.dumps({"type": "token", "content": f"{filler}\n\n"}) + "\n"
        
        # --- 2. INPUT GUARD ---
        t_start = time.perf_counter()
        if not check_safety(last_query):
            timings["input_guardrail_sec"] = time.perf_counter() - t_start
            yield json.dumps({"type": "token", "content": "I cannot answer this query due to safety policies."}) + "\n"
            yield json.dumps({"type": "meta", "safety": "FAILED_INPUT", "timings": timings, "sources": []}) + "\n"
            log_interaction_to_db(last_query, "Safety Violation", "Safety Violation", "FAILED_INPUT", [], timings, current_config)
            return
        timings["input_guardrail_sec"] = time.perf_counter() - t_start

        # --- 3. CONTEXTUALIZATION (New Step) ---
        # We use the REFORMULATED query for Embedding/Retrieval, but the ORIGINAL query for the final generation
        search_query = last_query
        if history:
            search_query = contextualize_query(history, last_query)

        # --- 4. EMBEDDING (Use Search Query) ---
        t_start = time.perf_counter()
        query_vector = get_embedding(search_query)
        timings["embedding_sec"] = time.perf_counter() - t_start
        
        # --- 5. RETRIEVAL (Use Search Query) ---
        t_start = time.perf_counter()
        search_res = milvus.search(collection_name=COLLECTION_NAME, data=[query_vector], limit=15, output_fields=["text", "filenames"])
        timings["retrieval_sec"] = (time.perf_counter() - t_start)
        
        candidate_chunks = []
        if search_res and search_res[0]:
            for hit in search_res[0]:
                candidate_chunks.append({
                    "text": hit["entity"]["text"],
                    "filenames": hit["entity"].get("filenames", [])
                })
        
        # --- 5.1 RERANKING (Use Search Query) ---
        t_start_rerank = time.perf_counter()
        # Rerank based on the standalone reformulated query
        reranked_chunks = perform_reranking(search_query, candidate_chunks)
        top_chunks = reranked_chunks[:3]
        timings["reranking_sec"] = time.perf_counter() - t_start_rerank

        retrieved_text = [c["text"] for c in top_chunks]
        sources = set()
        for c in top_chunks:
            fnames = c.get("filenames", [])
            if isinstance(fnames, list): 
                for f in fnames: sources.add(f)
            else: sources.add(fnames)

        # --- 6. LLM GENERATION ---
        context_block = "\n\n".join(retrieved_text)
        
        # Build messages: System -> History -> (Context + Latest Question)
        llm_messages = [{"role": "system", "content": current_config["system_prompt"]}]
        
        # Append cleaned history
        for msg in history:
            llm_messages.append(msg.dict())
            
        # Inject Context into the final user prompt
        # We use the ORIGINAL 'last_query' here so the tone matches the conversation flow
        final_prompt = f"Reference Context:\n{context_block}\n\nUser Question: {last_query}"
        llm_messages.append({"role": "user", "content": final_prompt})

        t_start_gen = time.perf_counter()
        full_response_accumulator = ""
        
        try:
            stream = client_llm.chat.completions.create(
                model=current_config["llm_model_id"], 
                messages=llm_messages,
                temperature=current_config["llm_temperature"], 
                max_tokens=512, stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response_accumulator += content
                    yield json.dumps({"type": "token", "content": content}) + "\n"
            
        except Exception as e:
            error_msg = f"\nError generating response: {str(e)}"
            full_response_accumulator += error_msg
            yield json.dumps({"type": "token", "content": error_msg}) + "\n"

        timings["llm_generation_sec"] = time.perf_counter() - t_start_gen

        # --- 7. LOGGING ---
        t_start_guard = time.perf_counter()
        is_safe = check_safety(full_response_accumulator, "AI Response")
        timings["output_guardrail_sec"] = time.perf_counter() - t_start_guard
        status = "PASSED" if is_safe else "FAILED_OUTPUT"
        
        timings["total_sec"] = time.perf_counter() - t_start_total

        # Log to DB (Storing the reformulated query in logs is often helpful for debugging)
        log_query_text = f"[{search_query}] " + last_query 
        log_interaction_to_db(log_query_text, full_response_accumulator, "Conversation Turn", status, list(sources), timings, current_config)

        yield json.dumps({
            "type": "meta",
            "sources": list(sources),
            "safety": status,
            "timings": timings
        }) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)