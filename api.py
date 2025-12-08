import os
import io
from datetime import datetime, timedelta
import base64
import zipfile
from typing import List, Optional, Dict, Any
from enum import Enum
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymilvus import MilvusClient
from openai import OpenAI
import uvicorn
from pypdf import PdfReader
from docx import Document
from PIL import Image
import itertools
import requests
import psycopg2 
from psycopg2.extras import Json
from fastapi.responses import StreamingResponse
import json

# --- CONFIGURATION ---
TOOL_HOST_IP = "http://10.25.73.101"
TOOL_HOST_IP_ONLY = "10.25.73.101"
EXTERNAL_API_BASE = "http://10.25.73.101:9000" 

MILVUS_URI = TOOL_HOST_IP+":19530"
COLLECTION_NAME = "qwen_rag_collection"
CACHE_COLLECTION = "qwen_rag_cache"

# Global Dimension State (defaults to 1024, updates dynamically)
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
    
    # New Router Config
    "router_port": 8010,
    "router_model_id": "Qwen/Qwen3-0.6B",

    "system_prompt": "You are a helpful assistant. Answer the user's question using the provided context. If the context lacks the answer, say I don't know.",
    "cache_threshold": 0.92,
    "llm_temperature": 0.1,
}

# Clients - Initialized with defaults
client_llm = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['llm_port']}/v1", api_key="EMPTY")
client_guard = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['guard_port']}/v1", api_key="EMPTY")
client_embed = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['embed_port']}/v1", api_key="EMPTY")
client_vl = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['vl_port']}/v1", api_key="EMPTY")
client_router = OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['router_port']}/v1", api_key="EMPTY")

# Initialize Milvus
milvus = MilvusClient(uri=MILVUS_URI)

# --- DATABASE SETUP ---
def init_db():
    # 1. Main RAG Collection
    if not milvus.has_collection(COLLECTION_NAME):
        milvus.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=CURRENT_DIMENSION,
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True 
        )
        print(f"Collection {COLLECTION_NAME} created.")

    # 2. Semantic Cache Collection
    if not milvus.has_collection(CACHE_COLLECTION):
        milvus.create_collection(
            collection_name=CACHE_COLLECTION,
            dimension=CURRENT_DIMENSION,
            metric_type="COSINE", 
            auto_id=True,
            enable_dynamic_field=True 
        )
        print(f"Cache Collection {CACHE_COLLECTION} created.")
    
    try:
        milvus.load_collection(CACHE_COLLECTION)
    except Exception as e:
        print(f"Loading notice: {e}")

# --- HELPER CLASSES ---

class ProcessTimings(BaseModel):
    input_guardrail_sec: float
    embedding_sec: float
    retrieval_sec: float
    llm_generation_sec: float
    output_guardrail_sec: float
    total_sec: float

class FileResponse(BaseModel):
    filename: str
    status: str
    extracted_text_preview: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    use_cache: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    safety_check: str
    timings: ProcessTimings
    llm_model: str
    embed_model: str
    guard_model: str

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
    router_port: int        # Added
    router_model_id: str    # Added
    system_prompt: str
    cache_threshold: float

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
    total_duration_sec: Optional[float] = 0.0
    llm_generation_sec: Optional[float] = 0.0
    retrieval_sec: Optional[float] = 0.0
    embedding_sec: Optional[float] = 0.0
    input_guardrail_sec: Optional[float] = 0.0
    output_guardrail_sec: Optional[float] = 0.0


# --- OCR & VISION SERVICES ---

def encode_image_base64(image_bytes: bytes, format: str = "JPEG") -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

def perform_ocr(image_bytes: bytes, filename_context: str) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_image = encode_image_base64(buffered.getvalue())

        print(f"Sending image from {filename_context} to Qwen-VL...")

        response = client_vl.chat.completions.create(
            model=current_config["vl_model_id"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the text in this image word for word."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.1
        )
        extracted = response.choices[0].message.content
        return f"\n[OCR Content from Image in {filename_context}]:\n{extracted}\n"
    except Exception as e:
        print(f"OCR Failed for {filename_context}: {e}")
        return ""

# --- FILE PARSING SERVICES ---

async def parse_file(file: UploadFile) -> str:
    content = ""
    filename = file.filename
    extension = filename.split(".")[-1].lower()
    file_bytes = await file.read()

    if extension in ["jpg", "jpeg", "png"]:
        content += perform_ocr(file_bytes, filename)

    elif extension == "pdf":
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                content += page_text + "\n"
            for img_file_obj in page.images:
                content += perform_ocr(img_file_obj.data, f"{filename} page {i+1}")

    elif extension in ["docx", "doc"]:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            content += para.text + "\n"
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    if name.startswith("word/media/"):
                        img_data = z.read(name)
                        content += perform_ocr(img_data, f"{filename} embedded {name}")
        except Exception as e:
            print(f"Could not extract images from docx: {e}")

    elif extension == "txt":
        content = file_bytes.decode("utf-8")
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
        
    return content


def get_db_connection():
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def log_interaction_to_db(
    query: str, 
    response: str, 
    status: str, 
    safety_check: str, 
    sources: List[str], 
    timings: Dict[str, float],
    config_snapshot: Dict[str, Any]
):
    conn = get_db_connection()
    if not conn: return

    try:
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO interaction_logs (
            user_query, 
            model_response, 
            status, 
            safety_check_result, 
            sources, 
            model_id_llm, 
            model_id_embed,
            total_duration_sec,
            input_guardrail_sec,
            embedding_sec,
            retrieval_sec,
            llm_generation_sec,
            output_guardrail_sec,
            client_latency_sec,
            config_snapshot
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(insert_query, (
            query,
            response,
            status,
            safety_check,
            sources, 
            config_snapshot.get("llm_model_id"),
            config_snapshot.get("embed_model_id"),
            timings.get("total_sec", 0),
            timings.get("input_guardrail_sec", 0),
            timings.get("embedding_sec", 0),
            timings.get("retrieval_sec", 0),
            timings.get("llm_generation_sec", 0),
            timings.get("output_guardrail_sec", 0),
            0.0, 
            Json(config_snapshot)
        ))
        
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"Failed to log interaction: {e}")
    finally:
        conn.close()


# --- STANDARD RAG SERVICES ---

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    response = client_embed.embeddings.create(
        input=[text],
        model=current_config["embed_model_id"]
    )
    return response.data[0].embedding

def check_safety(text: str, context_type: str = "User Query") -> bool:
    prompt = f"""
    You are a safety moderation tool. Analyze the following {context_type}. 
    If it contains harmful, illegal, or unsafe content, respond with 'UNSAFE'. 
    If it is safe, respond with 'SAFE'.
    {context_type}: {text}
    Assessment:"""
    
    try:
        response = client_guard.chat.completions.create(
            model=current_config["guard_model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        safety_assessment = response.choices[0].message.content.strip().upper()
        if "UNSAFE" in safety_assessment:
            return False
        return True
    except Exception as e:
        print(f"Guardrail check failed: {e}")
        return True

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_index(filename: str, text: str):
    chunks = chunk_text(text)
    data_rows = []
    print(f"Indexing {len(chunks)} chunks for {filename}...")
    try:
        for chunk in chunks:
            vector = get_embedding(chunk)
            data_rows.append({"vector": vector, "text": chunk, "filename": filename})
        
        if data_rows:
            milvus.insert(collection_name=COLLECTION_NAME, data=data_rows)
        print(f"Finished indexing {filename}")
    except Exception as e:
        print(f"Indexing error (Dimension Mismatch?): {e}")

# --- DECISION ENGINE & EXTERNAL APIS ---

def classify_intent(query: str) -> str:
    """Uses Qwen 0.6B to decide if the query is WEATHER, PARKING, or RAG."""
    system_prompt = """You are a precise intent classification engine. 
    Classify the user's query into exactly one of these three categories:
    1. WEATHER: For questions about weather, rain, temperature, or forecast.
    2. PARKING: For questions about finding parking, garages, or parking availability.
    3. RAG: For all other general questions, document lookups, or knowledge base queries.
    Reply ONLY with the category name (WEATHER, PARKING, or RAG). Do not explain."""
    
    try:
        response = client_router.chat.completions.create(
            model=current_config["router_model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=10,
			extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        intent = response.choices[0].message.content.strip().upper()
        
        if "WEATHER" in intent: return "WEATHER"
        if "PARKING" in intent: return "PARKING"
        return "RAG"
    except Exception as e:
        print(f"Intent classification failed: {e}. Defaulting to RAG.")
        return "RAG"

def fetch_weather_data() -> str:
    try:
        #resp = requests.get(f"{EXTERNAL_API_BASE}/weather", timeout=2)
        resp = requests.get("http://api.weatherapi.com/v1/current.json?key=b883b8fe23af4ef2b2395630250812&q=Doha&aqi=no")
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

def fetch_parking_data() -> str:
    try:
        resp = requests.get("http://10.25.73.101:9090/parking", timeout=2)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        print(f"Error fetching parking data: {str(e)}")


# --- FASTAPI APP ---

app = FastAPI(title="Qwen Multi-Modal RAG API", on_startup=[init_db])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/logs", response_model=List[LogEntry])
def get_logs(limit: int = 100, hours: Optional[int] = None):
    """Fetch logs from Postgres with optional time filtering"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cur = conn.cursor()
        sql = """
            SELECT 
                id, timestamp, 
                user_query, model_response, status, safety_check_result, sources,
                model_id_llm, model_id_embed,
                total_duration_sec, llm_generation_sec, retrieval_sec, embedding_sec,
                input_guardrail_sec, output_guardrail_sec, client_latency_sec,
                config_snapshot
            FROM interaction_logs 
        """
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
                "model_id_llm": r[7], "model_id_embed": r[8],
                "total_duration_sec": r[9] or 0.0, "llm_generation_sec": r[10] or 0.0,
                "retrieval_sec": r[11] or 0.0, "embedding_sec": r[12] or 0.0,
                "input_guardrail_sec": r[13] or 0.0, "output_guardrail_sec": r[14] or 0.0,
            })
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()

@app.delete("/logs")
def truncate_logs():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE interaction_logs")
        conn.commit()
        return {"status": "Logs truncated successfully"}
    finally:
        if conn: conn.close()

@app.post("/upload", response_model=FileResponse)
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        content = await parse_file(file)
        if not content.strip():
            return {"filename": file.filename, "status": "Warning: No text or OCR content found"}
        background_tasks.add_task(process_and_index, file.filename, content)
        return {
            "filename": file.filename, 
            "status": "Processing background OCR and Embedding",
            "extracted_text_preview": content[:100] + "..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files", response_model=List[str])
def list_files():
    results = milvus.query(collection_name=COLLECTION_NAME, filter='filename != ""', output_fields=["filename"])
    return list(set([res["filename"] for res in results]))

@app.delete("/files/{filename}")
def delete_file(filename: str):
    milvus.delete(collection_name=COLLECTION_NAME, filter=f'filename == "{filename}"')
    return {"status": "deleted", "filename": filename}

@app.get("/models")
def list_models():
    VLLM_API_BASE_IP = TOOL_HOST_IP
    mods = {}
    # Scan range including standard ports + router port + debug ports
    for port in itertools.chain(range(8000, 8012), [8100, 8200]):
        try:
            m = requests.get(VLLM_API_BASE_IP+f":{port}/v1/models", timeout=.2).json()["data"][0]["id"]
            mods[m] = port
        except:
            pass 
    return(mods)

# --- CONFIG ENDPOINTS ---

@app.get("/config")
def get_config():
    return current_config

@app.post("/config")
def update_config(cfg: ConfigUpdate):
    global client_llm, client_guard, client_embed, client_vl, client_router, current_config, CURRENT_DIMENSION
    
    old_embed_model = current_config.get("embed_model_id")
    
    try:
        current_config.update({
            "llm_port": cfg.llm_port,
            "llm_model_id": cfg.llm_model_id,
            "llm_temperature": cfg.llm_temperature,
            "embed_port": cfg.embed_port,
            "embed_model_id": cfg.embed_model_id,
            "guard_port": cfg.guard_port,
            "guard_model_id": cfg.guard_model_id,
            "vl_port": cfg.vl_port,
            "vl_model_id": cfg.vl_model_id,
            "router_port": cfg.router_port,        # Update Router Config
            "router_model_id": cfg.router_model_id, # Update Router Config
            "system_prompt": cfg.system_prompt,
            "cache_threshold": cfg.cache_threshold
        })

        # Re-bind Clients
        client_llm = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.llm_port}/v1", api_key="EMPTY")
        client_guard = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.guard_port}/v1", api_key="EMPTY")
        client_embed = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.embed_port}/v1", api_key="EMPTY")
        client_vl = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.vl_port}/v1", api_key="EMPTY")
        client_router = OpenAI(base_url=f"{TOOL_HOST_IP}:{cfg.router_port}/v1", api_key="EMPTY") # Re-bind Router

        # --- EMBEDDING MODEL CHANGED LOGIC ---
        if old_embed_model != cfg.embed_model_id:
            print(f"Detected Embedding Model Change: {old_embed_model} -> {cfg.embed_model_id}")
            # 1. Backup existing data
            try:
                existing_data = milvus.query(collection_name=COLLECTION_NAME, filter="id >= 0", output_fields=["text", "filename"])
            except:
                existing_data = []

            # 2. Get New Dimension
            test_embed = client_embed.embeddings.create(input=["test"], model=cfg.embed_model_id).data[0].embedding
            CURRENT_DIMENSION = len(test_embed)
            
            # 3. Drop and Recreate
            if milvus.has_collection(COLLECTION_NAME): milvus.drop_collection(COLLECTION_NAME)
            if milvus.has_collection(CACHE_COLLECTION): milvus.drop_collection(CACHE_COLLECTION)
            
            milvus.create_collection(collection_name=COLLECTION_NAME, dimension=CURRENT_DIMENSION, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)
            milvus.create_collection(collection_name=CACHE_COLLECTION, dimension=CURRENT_DIMENSION, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)
            milvus.load_collection(CACHE_COLLECTION)
            
            # 4. Re-index
            if existing_data:
                new_rows = []
                for item in existing_data:
                    try:
                        vector = client_embed.embeddings.create(input=[item["text"].replace("\n", " ")], model=cfg.embed_model_id).data[0].embedding
                        new_rows.append({"vector": vector, "text": item["text"], "filename": item["filename"]})
                    except: pass
                if new_rows:
                    for i in range(0, len(new_rows), 100):
                        milvus.insert(collection_name=COLLECTION_NAME, data=new_rows[i:i+100])

        return {"status": "Configuration updated successfully", "config": current_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@app.delete("/milvus/cache/clear")
def clear_milvus_cache():
    try:
        milvus.drop_collection("qwen_rag_cache")
        milvus.create_collection(
            collection_name=CACHE_COLLECTION,
            dimension=CURRENT_DIMENSION,
            metric_type="COSINE", 
            auto_id=True,
            enable_dynamic_field=True 
        )
        milvus.load_collection(CACHE_COLLECTION)
        return {"status": "Cache cleared successfully", "collection": CACHE_COLLECTION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_rag(request: ChatRequest, background_tasks: BackgroundTasks):
    t_start_total = time.perf_counter()
    timings = {
        "input_guardrail_sec": 0.0, "embedding_sec": 0.0, "retrieval_sec": 0.0,
        "llm_generation_sec": 0.0, "output_guardrail_sec": 0.0, "total_sec": 0.0
    }

    # Used Models Snapshot
    used_models = {
        "llm": current_config["llm_model_id"],
        "embed": current_config["embed_model_id"],
        "guard": current_config["guard_model_id"]
    }

    # --- STEP 1: INTENT CLASSIFICATION ---
    intent = classify_intent(request.query)
    print(f"Detected Intent: {intent}")

    if intent in ["WEATHER", "PARKING"]:
        # --- API FLOW ---
        t_start = time.perf_counter()
        
        if intent == "WEATHER":
            context_data = fetch_weather_data()
            source_label = "EXTERNAL_API:Weather"
        else:
            context_data = fetch_parking_data()
            source_label = "EXTERNAL_API:Parking"
        
        # We attribute API fetch time to "retrieval_sec" for metrics consistency
        timings["retrieval_sec"] = time.perf_counter() - t_start 

        # Generate Response using LLM
        t_start = time.perf_counter()
        system_prompt = "You are a helpful assistant. Use the provided JSON data to answer the user's question clearly and conversationally."
        user_prompt = f"Data:\n{context_data}\n\nQuestion: {request.query}"
        
        try:
            completion = client_llm.chat.completions.create(
                model=current_config["llm_model_id"],
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=current_config["llm_temperature"],
                max_tokens=300,
				extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            final_response = completion.choices[0].message.content
        except Exception as e:
            final_response = f"Error generating response: {e}"
        
        timings["llm_generation_sec"] = time.perf_counter() - t_start
        timings["total_sec"] = time.perf_counter() - t_start_total

        # Log & Return
        background_tasks.add_task(
            log_interaction_to_db, 
            query=request.query, 
            response=final_response, 
            status="Success (API)", 
            safety_check="PASSED", 
            sources=[source_label], 
            timings=timings, 
            config_snapshot=current_config
        )
        return {
            "response": final_response, "sources": [source_label], "safety_check": "PASSED",
            "timings": timings, "llm_model": used_models["llm"], 
            "embed_model": "N/A", "guard_model": "N/A"
        }

    # --- STANDARD RAG FLOW ---

    # 1. Input Guardrail
    t_start = time.perf_counter()
    if not check_safety(request.query, context_type="User Query"):
        timings["input_guardrail_sec"] = time.perf_counter() - t_start
        timings["total_sec"] = time.perf_counter() - t_start_total
        
        background_tasks.add_task(
            log_interaction_to_db, request.query, "I cannot process this request due to safety guidelines.", 
            "Safety Violation", "FAILED_INPUT", [], timings, current_config
        )
        return {
            "response": "I cannot process this request due to safety guidelines.", "sources": [], 
            "safety_check": "FAILED_INPUT", "timings": timings, 
            "llm_model": used_models["llm"], "embed_model": used_models["embed"], "guard_model": used_models["guard"]
        }
    timings["input_guardrail_sec"] = time.perf_counter() - t_start

    # 2. Embedding
    t_start = time.perf_counter()
    query_vector = get_embedding(request.query)
    timings["embedding_sec"] = time.perf_counter() - t_start
    
    # 3. Cache Check
    if request.use_cache:
        t_start = time.perf_counter()
        cache_res = milvus.search(
            collection_name=CACHE_COLLECTION, data=[query_vector], limit=1, 
            output_fields=["response", "original_query"], search_params={"metric_type": "COSINE", "params": {}}
        )
        timings["retrieval_sec"] = time.perf_counter() - t_start

        if cache_res and cache_res[0]:
            top_hit = cache_res[0][0]
            if top_hit["distance"] >= current_config["cache_threshold"]:
                timings["total_sec"] = time.perf_counter() - t_start_total
                
                background_tasks.add_task(
                    log_interaction_to_db, request.query, top_hit["entity"]["response"], 
                    "Success (Cache)", "PASSED_CACHE", ["SEMANTIC_CACHE"], timings, current_config
                )
                return {
                    "response": top_hit["entity"]["response"], "sources": ["SEMANTIC_CACHE"], 
                    "safety_check": "PASSED_CACHE", "timings": timings,
                    "llm_model": used_models["llm"], "embed_model": used_models["embed"], "guard_model": used_models["guard"]
                }

    # 4. Vector DB Retrieval
    t_start = time.perf_counter()
    search_res = milvus.search(
        collection_name=COLLECTION_NAME, data=[query_vector], limit=3, 
        output_fields=["text", "filename"], search_params={"metric_type": "COSINE", "params": {}}
    )
    # Add to retrieval time (cumulative if cache checked)
    timings["retrieval_sec"] += (time.perf_counter() - t_start)
    
    retrieved_chunks = []
    sources = []
    if search_res and search_res[0]:
        for hit in search_res[0]:
            retrieved_chunks.append(hit["entity"]["text"])
            sources.append(hit["entity"]["filename"])
    
    context = "\n\n".join(retrieved_chunks)
    
    # 5. LLM Generation
    t_start = time.perf_counter()
    try:
        completion = client_llm.chat.completions.create(
            model=current_config["llm_model_id"], 
            messages=[
                {"role": "system", "content": current_config["system_prompt"]},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
            ],
            temperature=current_config["llm_temperature"],
            max_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        raw_response = completion.choices[0].message.content
    except Exception as e:
        raw_response = f"Error: {str(e)}"
    timings["llm_generation_sec"] = time.perf_counter() - t_start

    # 6. Output Guardrail
    t_start = time.perf_counter()
    is_safe_output = check_safety(raw_response, context_type="AI Response")
    timings["output_guardrail_sec"] = time.perf_counter() - t_start

    final_response = raw_response
    safety_status = "PASSED"
    status_label = "Success"
    
    if not is_safe_output:
        final_response = "The generated response was withheld because it violated safety policies."
        safety_status = "FAILED_OUTPUT"
        status_label = "Safety Violation"
    else:
        # Cache logic: Only cache if not "I don't know"
        response_lower = final_response.lower().strip()
        is_unknown = "i don't know" in response_lower or "i do not know" in response_lower
        
        if not is_unknown:
            milvus.insert(
                collection_name=CACHE_COLLECTION,
                data=[{
                    "vector": query_vector,
                    "response": final_response,
                    "original_query": request.query
                }]
            )

    timings["total_sec"] = time.perf_counter() - t_start_total

    background_tasks.add_task(
        log_interaction_to_db, request.query, final_response, status_label, 
        safety_status, list(set(sources)), timings, current_config
    )

    return {
        "response": final_response,
        "sources": list(set(sources)),
        "safety_check": safety_status,
        "timings": timings,
        "llm_model": used_models["llm"],
        "embed_model": used_models["embed"],
        "guard_model": used_models["guard"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)