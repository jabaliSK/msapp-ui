# rag_service.py
import time
import json
import random
import requests
import itertools
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from psycopg2.extras import Json

# Import from common config
from config import (
    TOOL_HOST_IP, MILVUS_URI, COLLECTION_NAME, CACHE_COLLECTION,
    PG_CONFIG, default_config, mongo_coll, milvus, get_pg_connection,
    ChatRequest, ConversationRequest, ConfigUpdate, LogEntry, Message
)

# --- APP SETUP ---
app = FastAPI(title="RAG Inference Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Local Config State
current_config = default_config.copy()

# --- CLIENTS ---
def init_clients():
    return {
        "llm": OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['llm_port']}/v1", api_key="EMPTY"),
        "guard": OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['guard_port']}/v1", api_key="EMPTY"),
        "embed": OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['embed_port']}/v1", api_key="EMPTY"),
    }

clients = init_clients()

FILLER_PHRASES = [
    "Let me take a brief moment to check on those details for you right now.",
    "Please bear with me for just a moment while I look that up for you.",
    "I'm going to quickly consult the knowledge base to find the exact answer.",
    "I am scanning through the documents right now to find what you need.",
    "Give me just one second while I locate the relevant information for this request."
]

# --- HELPER FUNCTIONS ---

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    response = clients["embed"].embeddings.create(input=[text], model=current_config["embed_model_id"])
    return response.data[0].embedding

def check_safety(text: str, context_type: str = "User Query") -> bool:
    prompt = f"You are a safety moderation tool. Analyze the following {context_type}. If it contains harmful/illegal content, respond 'UNSAFE'. Otherwise 'SAFE'. {context_type}: {text}"
    try:
        response = clients["guard"].chat.completions.create(
            model=current_config["guard_model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        if "UNSAFE" in response.choices[0].message.content.strip().upper(): return False
        return True
    except Exception: return True

def perform_reranking(query: str, documents: List[dict]) -> List[dict]:
    if not documents: return []
    url = f"{TOOL_HOST_IP}:{current_config['rerank_port']}/v1/rerank"
    doc_texts = [d["text"] for d in documents]
    payload = {
        "model": current_config["rerank_model_id"],
        "query": query,
        "documents": doc_texts,
        "top_n": len(documents)
    }
    try:
        response = requests.post(url, json=payload, timeout=2.0)
        response.raise_for_status()
        results = response.json()
        if "results" in results:
            sorted_docs = []
            for item in results["results"]:
                idx = item["index"]
                documents[idx]["score"] = item["relevance_score"]
                sorted_docs.append(documents[idx])
            return sorted_docs
        else:
            return documents
    except Exception as e:
        print(f"Rerank failed: {e}")
        return documents

def contextualize_query(history: List[Message], latest_query: str) -> str:
    if not history: return latest_query
    history_str = "\n".join([f"{msg.role.title()}: {msg.content}" for msg in history[-4:]])
    prompt = (
        f"Given the chat history and the latest user question, formulate a standalone question. "
        f"Do NOT answer the question.\n\nChat History:\n{history_str}\n\n"
        f"User Question: {latest_query}\n\nStandalone Question:"
    )
    try:
        response = clients["llm"].chat.completions.create(
            model=current_config["llm_model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=128, stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Contextualization failed: {e}")
        return latest_query

def log_interaction_to_db(query, response, status, safety_check, sources, timings, config_snapshot):
    conn = get_pg_connection()
    if not conn: return
    try:
        cur = conn.cursor()
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

# --- ENDPOINTS ---

@app.get("/stats")
def get_dashboard_stats():
    try:
        # Totals
        pipeline_totals = [{"$group": {"_id": None, "total_docs": {"$sum": 1}, "total_size": {"$sum": "$file_size"}, "total_chunks": {"$sum": "$chunk_count"}}}]
        totals_res = list(mongo_coll.aggregate(pipeline_totals))
        stats = {"totalDocuments": 0, "storageUsed": 0, "totalChunks": 0}
        if totals_res:
            res = totals_res[0]
            stats = {"totalDocuments": res.get("total_docs", 0), "storageUsed": res.get("total_size", 0), "totalChunks": res.get("total_chunks", 0)}

        # Distribution
        pipeline_dist = [{"$group": {"_id": "$file_type", "value": {"$sum": 1}}}]
        dist_res = list(mongo_coll.aggregate(pipeline_dist))
        content_distribution = [{"name": (d["_id"] or "UNK"), "value": d["value"]} for d in dist_res]

        # Activity
        end_date = datetime.utcnow()
        activity_map = {}
        for i in range(6, -1, -1):
            d = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            activity_map[d] = {"docs": 0, "vectors": 0}
        start_date = end_date - timedelta(days=7)
        pipeline_activity = [{"$match": {"upload_date": {"$gte": start_date}}}, {"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$upload_date"}}, "docs": {"$sum": 1}, "vectors": {"$sum": "$chunk_count"}}}]
        activity_res = list(mongo_coll.aggregate(pipeline_activity))
        for entry in activity_res:
            if entry["_id"] in activity_map:
                activity_map[entry["_id"]]["docs"] = entry["docs"]
                activity_map[entry["_id"]]["vectors"] = entry["vectors"]
        ingestion_activity = [{"name": datetime.strptime(d, "%Y-%m-%d").strftime("%a"), "docs": v["docs"], "vectors": v["vectors"]} for d, v in sorted(activity_map.items())]

        return {"stats": stats, "distribution": content_distribution, "activity": ingestion_activity}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs", response_model=List[LogEntry])
def get_logs(limit: int = 100, hours: Optional[int] = None):
    conn = get_pg_connection()
    if not conn: raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        cur = conn.cursor()
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

@app.delete("/logs")
def truncate_logs():
    conn = get_pg_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE interaction_logs")
        conn.commit()
        return {"status": "Logs truncated"}
    finally: conn.close()

@app.get("/config")
def get_config_endpoint(): return current_config

@app.post("/config")
def update_config_endpoint(cfg: ConfigUpdate):
    global clients, current_config
    try:
        current_config.update(cfg.dict())
        clients = init_clients() # Re-init clients with new ports
        return {"status": "Updated", "config": current_config}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.delete("/milvus/cache/clear")
def clear_milvus_cache():
    milvus.drop_collection(CACHE_COLLECTION)
    # Recreate is tricky here without schema, assuming calling init_db logic or similar
    # For simplicity, we just return status, usually schema is recreated on next use or app startup
    return {"status": "Cache cleared"}

@app.get("/models")
def list_models():
    mods = {}
    for port in itertools.chain(range(8000, 8012), range(8100, 8110)):
        try:
            m = requests.get(TOOL_HOST_IP+f":{port}/v1/models", timeout=.2).json()["data"][0]["id"]
            if(m in mods.keys()): m = "dock_" + m
            mods[m] = port
        except: pass 
    return(mods)

@app.post("/chat")
async def chat_rag_stream(request: ChatRequest):
    async def response_generator():
        t_start_total = time.perf_counter()
        timings = {"input_guardrail_sec": 0.0, "embedding_sec": 0.0, "retrieval_sec": 0.0, "reranking_sec": 0.0, "llm_generation_sec": 0.0, "output_guardrail_sec": 0.0, "total_sec": 0.0}
        
        filler = random.choice(FILLER_PHRASES)
        yield json.dumps({"type": "token", "content": f"{filler}\n\n"}) + "\n"
        
        # 1. Input Guard
        t_start = time.perf_counter()
        if not check_safety(request.query):
            timings["input_guardrail_sec"] = time.perf_counter() - t_start
            yield json.dumps({"type": "token", "content": "I cannot answer this query due to safety policies."}) + "\n"
            yield json.dumps({"type": "meta", "safety": "FAILED_INPUT", "timings": timings, "sources": []}) + "\n"
            log_interaction_to_db(request.query, "Safety Violation", "Safety Violation", "FAILED_INPUT", [], timings, current_config)
            return
        timings["input_guardrail_sec"] = time.perf_counter() - t_start

        # 2. Embedding
        t_start = time.perf_counter()
        query_vector = get_embedding(request.query)
        timings["embedding_sec"] = time.perf_counter() - t_start
        
        # 3. Cache
        if request.use_cache:
            t_start = time.perf_counter()
            cache_res = milvus.search(collection_name=CACHE_COLLECTION, data=[query_vector], limit=1, output_fields=["response"])
            if cache_res and cache_res[0] and cache_res[0][0]["distance"] >= current_config["cache_threshold"]:
                timings["retrieval_sec"] = time.perf_counter() - t_start
                cached_resp = cache_res[0][0]["entity"]["response"]
                yield json.dumps({"type": "token", "content": cached_resp}) + "\n"
                yield json.dumps({"type": "meta", "safety": "PASSED_CACHE", "timings": timings, "sources": ["SEMANTIC_CACHE"]}) + "\n"
                log_interaction_to_db(request.query, cached_resp, "Success (Cache)", "PASSED_CACHE", ["SEMANTIC_CACHE"], timings, current_config)
                return

        # 4. Retrieval
        t_start = time.perf_counter()
        search_res = milvus.search(collection_name=COLLECTION_NAME, data=[query_vector], limit=15, output_fields=["text", "filenames"])
        timings["retrieval_sec"] = (time.perf_counter() - t_start)
        
        candidate_chunks = []
        if search_res and search_res[0]:
            for hit in search_res[0]:
                candidate_chunks.append({"text": hit["entity"]["text"], "filenames": hit["entity"].get("filenames", [])})
        
        # 5. Rerank
        t_start_rerank = time.perf_counter()
        top_chunks = perform_reranking(request.query, candidate_chunks)[:3]
        timings["reranking_sec"] = time.perf_counter() - t_start_rerank

        retrieved_text = [c["text"] for c in top_chunks]
        sources = set()
        for c in top_chunks:
            fnames = c.get("filenames", [])
            if isinstance(fnames, list): 
                for f in fnames: sources.add(f)
            else: sources.add(fnames)
        
        # Update usage count in Mongo (Best effort)
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

        # 6. LLM
        context = "\n\n".join(retrieved_text)
        t_start_gen = time.perf_counter()
        full_response_accumulator = ""
        
        try:
            stream = clients["llm"].chat.completions.create(
                model=current_config["llm_model_id"], 
                messages=[{"role": "system", "content": current_config["system_prompt"]}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}],
                temperature=current_config["llm_temperature"], max_tokens=300, stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response_accumulator += content
                    yield json.dumps({"type": "token", "content": content}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "token", "content": f"\nError: {e}"}) + "\n"

        timings["llm_generation_sec"] = time.perf_counter() - t_start_gen

        # 7. Output Guard & Log
        t_start_guard = time.perf_counter()
        is_safe = check_safety(full_response_accumulator, "AI Response")
        timings["output_guardrail_sec"] = time.perf_counter() - t_start_guard
        status = "PASSED" if is_safe else "FAILED_OUTPUT"
        
        if is_safe and "i don't know" not in full_response_accumulator.lower():
            milvus.insert(collection_name=CACHE_COLLECTION, data=[{"vector": query_vector, "response": full_response_accumulator}])
        
        timings["total_sec"] = time.perf_counter() - t_start_total
        log_interaction_to_db(request.query, full_response_accumulator, "Success" if is_safe else "Unsafe Content", status, list(sources), timings, current_config)

        yield json.dumps({"type": "meta", "sources": list(sources), "safety": status, "timings": timings}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

@app.post("/conversation")
async def conversation_rag_stream(request: ConversationRequest):

    pass 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)