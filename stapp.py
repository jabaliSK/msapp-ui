import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any

import requests
import streamlit as st
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# =========================
# Configuration
# =========================

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GUARDRAIL_MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"
LLM_MODEL_NAME = "Qwen/Qwen3-32B-FP8"

EMBEDDING_URL = "http://localhost:8002/v1/embeddings"
GUARDRAIL_URL = "http://localhost:8001/v1/chat/completions"
LLM_URL = "http://localhost:8000/v1/chat/completions"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "rag_documents"

DOCS_DIR = "docs"
DOC_INDEX_FILE = "documents_index.json"
STATS_FILE = "rag_timings.jsonl"
MAX_EMBED_CHARS = 480
TOP_K = 5
CHUNK_SIZE_CHARS = 400
CHUNK_OVERLAP_CHARS = 100


# =========================
# Utility: Files & JSON
# =========================

def ensure_dirs():
    os.makedirs(DOCS_DIR, exist_ok=True)


def load_doc_index() -> List[Dict[str, Any]]:
    if not os.path.exists(DOC_INDEX_FILE):
        return []
    with open(DOC_INDEX_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_doc_index(index: List[Dict[str, Any]]):
    with open(DOC_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def log_timing(record: Dict[str, Any]):
    with open(STATS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_timing_records() -> List[Dict[str, Any]]:
    if not os.path.exists(STATS_FILE):
        return []
    records = []
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# =========================
# Milvus Helpers
# =========================

def connect_milvus():
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


def ensure_collection(embedding_dim: int) -> Collection:
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim
            ),
        ]
        schema = CollectionSchema(fields=fields, description="RAG documents")
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        # Create index on embedding
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
    else:
        collection = Collection(name=MILVUS_COLLECTION)
        collection.load()
    return collection


def delete_doc_from_milvus(doc_id: str):
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        return
    collection = Collection(MILVUS_COLLECTION)
    expr = f"doc_id == '{doc_id}'"
    collection.delete(expr)


def search_milvus(query_embedding: List[float], top_k: int = TOP_K):
    connect_milvus()
    if not utility.has_collection(MILVUS_COLLECTION):
        return []
    collection = Collection(MILVUS_COLLECTION)
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "doc_id"],
    )
    if not results:
        return []
    hits = results[0]
    contexts = []
    for hit in hits:
        text = hit.entity.get("text")
        contexts.append(text)
    return contexts


# =========================
# Models via vLLM (OpenAI-style)
# =========================

def get_query_embedding(text: str) -> List[float]:
    # Make sure we don't exceed the model's context limit
    text = text[:MAX_EMBED_CHARS]

    payload = {
        "model": EMBEDDING_MODEL_NAME,
        "input": text,
    }

    resp = requests.post(EMBEDDING_URL, json=payload, timeout=60)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(
            f"Embedding endpoint HTTP error: {e}\n"
            f"Status: {resp.status_code}\n"
            f"Body: {resp.text[:500]}"
        )

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(
            "Embedding endpoint did not return JSON. "
            f"Status: {resp.status_code}, body: {resp.text[:500]}"
        )

    if isinstance(data, dict) and "data" in data:
        try:
            return data["data"][0]["embedding"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Embedding response had 'data' but unexpected structure: {data}"
            ) from e

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Embedding endpoint returned error: {data['error']}")

    raise RuntimeError(
        f"Unexpected embedding response format: {data}"
    )



def run_guardrail(text: str, stage: str = "input") -> str:
    system_prompt = (
        f"Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a \"safe\" or \"unsafe\" rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violence.\nS2: Sexual.\nS3: Criminal Planning/Confessions.\nS4: Guns and Illegal Weapons.\nS5: Controlled/Regulated Substances.\nS6: Suicide and Self Harm.\nS7: Sexual (minor).\nS8: Hate/Identity Hate.\nS9: PII/Privacy.\nS10: Harassment.\nS11: Threat.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\nuser: {text}?\n<END CONVERSATION>\n\nProvide your safety assessment for the above conversation in the below output JSON format:\n{{\"User Safety\": safe or unsafe rating for the user message, \"Response Safety\": safe or unsafe rating for the agent response. Omit if no agent response present. \"Safety Categories\": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}}\n\nDo not include anything other than the output JSON in your response.\n\nOutput JSON:"
        f"This content is from the {stage} stage."
    )
    payload = {
        "model": GUARDRAIL_MODEL_NAME,
        "messages": [
            {"role": "user", "content": text},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    resp = requests.post(GUARDRAIL_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def run_llm(context: str, question: str) -> str:
    system_prompt = (
        "You are a fast RAG assistant. Answer the user's question using the provided context. If the context lacks the answer, say I don't know."
    )
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
		"chat_template_kwargs": {"enable_thinking": False}
    }
    resp = requests.post(LLM_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# =========================
# Chunking Helpers
# =========================

def simple_chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def index_document(filename: str, content: str):
    # Get embedding dimension by embedding the first chunk
    chunks = simple_chunk_text(content)
    if not chunks:
        return 0

    first_embedding = get_query_embedding(chunks[0])
    embedding_dim = len(first_embedding)

    collection = ensure_collection(embedding_dim=embedding_dim)

    # Build doc_id from hash of filename
    doc_id = hashlib.sha256(filename.encode("utf-8")).hexdigest()

    # Embed all chunks
    chunk_embeddings = [first_embedding]
    for chunk in chunks[1:]:
        emb = get_query_embedding(chunk)
        chunk_embeddings.append(emb)

    doc_ids = [doc_id] * len(chunks)

    # Insert into Milvus (pk is auto_id)
    data = [doc_ids, chunks, chunk_embeddings]
    # Order must match fields: doc_id, text, embedding
    collection.insert(data)
    collection.flush()

    # Save document to disk
    path = os.path.join(DOCS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    # Update index file
    index = load_doc_index()
    # Avoid duplicates
    if not any(d["doc_id"] == doc_id for d in index):
        index.append({"doc_id": doc_id, "filename": filename})
        save_doc_index(index)

    return len(chunks)


# =========================
# RAG Pipeline (with timing)
# =========================

def run_rag_pipeline(user_query: str) -> str:
    # Timers
    t0 = time.perf_counter()

    # 1. Guardrail on input
    t_guard_in_start = time.perf_counter()
    guard_in_result = run_guardrail(user_query, stage="input")
    t_guard_in_end = time.perf_counter()
    guard_in_ms = (t_guard_in_end - t_guard_in_start) * 1000

    is_input_unsafe = "unsafe" in guard_in_result.lower()

    # Base record
    timing_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": user_query,
        "status": "ok",
        "steps": {
            "input_guardrail_ms": guard_in_ms,
            "query_embedding_ms": None,
            "vector_search_ms": None,
            "llm_generation_ms": None,
            "output_guardrail_ms": None,
        },
        "num_context_chunks": 0,
        "response_length": 0,
        "guardrail_input_result": guard_in_result,
        "guardrail_output_result": None,
    }

    if is_input_unsafe:
        timing_record["status"] = "blocked-input"
        total_ms = (time.perf_counter() - t0) * 1000
        timing_record["total_ms"] = total_ms
        log_timing(timing_record)
        return f"⚠️ Your query was flagged as unsafe by the safety guard:\n\n{guard_in_result}"

    # 2. Query embedding
    t_emb_start = time.perf_counter()
    query_embedding = get_query_embedding(user_query)
    t_emb_end = time.perf_counter()
    query_emb_ms = (t_emb_end - t_emb_start) * 1000
    timing_record["steps"]["query_embedding_ms"] = query_emb_ms

    # 3. Vector search
    t_search_start = time.perf_counter()
    contexts = search_milvus(query_embedding, top_k=TOP_K)
    t_search_end = time.perf_counter()
    search_ms = (t_search_end - t_search_start) * 1000
    timing_record["steps"]["vector_search_ms"] = search_ms
    timing_record["num_context_chunks"] = len(contexts)

    context_str = "\n\n---\n\n".join(contexts) if contexts else "No relevant context found in the knowledge base."

    # 4. LLM generation
    t_llm_start = time.perf_counter()
    answer = run_llm(context=context_str, question=user_query)
    t_llm_end = time.perf_counter()
    llm_ms = (t_llm_end - t_llm_start) * 1000
    timing_record["steps"]["llm_generation_ms"] = llm_ms
    timing_record["response_length"] = len(answer)

    # 5. Guardrail on output
    t_guard_out_start = time.perf_counter()
    guard_out_result = run_guardrail(answer, stage="output")
    t_guard_out_end = time.perf_counter()
    guard_out_ms = (t_guard_out_end - t_guard_out_start) * 1000
    timing_record["steps"]["output_guardrail_ms"] = guard_out_ms
    timing_record["guardrail_output_result"] = guard_out_result

    is_output_unsafe = "unsafe" in guard_out_result.lower()

    total_ms = (time.perf_counter() - t0) * 1000
    timing_record["total_ms"] = total_ms

    if is_output_unsafe:
        timing_record["status"] = "blocked-output"
        log_timing(timing_record)
        return (
            "⚠️ The generated response was flagged as unsafe by the safety guard:\n\n"
            f"{guard_out_result}\n\n"
            "No unsafe content has been shown."
        )

    # Log final record
    log_timing(timing_record)
    return answer


# =========================
# Streamlit UI
# =========================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of dicts: {"role": "user"/"assistant", "content": str}


def page_chat():
    st.header("💬 RAG Chat")

    init_session_state()

    # Show chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    user_query = st.text_input("Ask a question using the indexed documents:")
    if st.button("Send") and user_query.strip():
        with st.spinner("Running RAG pipeline..."):
            answer = run_rag_pipeline(user_query.strip())

        st.session_state["messages"].append({"role": "user", "content": user_query.strip()})
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        st.rerun()


def page_documents():
    st.header("📄 Document Management")

    ensure_dirs()
    init_session_state()

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload .txt or .md files to index into Milvus",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process & Index Uploaded Files") and uploaded_files:
        for file in uploaded_files:
            filename = file.name
            content = file.read().decode("utf-8", errors="ignore")
            with st.spinner(f"Indexing {filename}..."):
                num_chunks = index_document(filename, content)
                st.success(f"Indexed {filename} into {num_chunks} chunks.")
        st.rerun()

    st.subheader("Existing documents")
    index = load_doc_index()
    if not index:
        st.info("No documents indexed yet.")
        return

    filenames = [d["filename"] for d in index]
    selection = st.multiselect(
        "Select documents to delete from Milvus and local storage:",
        options=filenames,
    )

    if st.button("Delete Selected Documents") and selection:
        index_map = {d["filename"]: d for d in index}
        new_index = []
        for entry in index:
            if entry["filename"] in selection:
                doc_id = entry["doc_id"]
                delete_doc_from_milvus(doc_id)
                local_path = os.path.join(DOCS_DIR, entry["filename"])
                if os.path.exists(local_path):
                    os.remove(local_path)
                st.success(f"Deleted {entry['filename']} and its vectors.")
            else:
                new_index.append(entry)
        save_doc_index(new_index)
        st.rerun()

    st.subheader("Document list")
    for entry in index:
        st.markdown(f"- **{entry['filename']}**  (`doc_id`: `{entry['doc_id'][:8]}…`)")



def page_statistics():
    st.header("📊 Timing Statistics")

    records = load_timing_records()
    if not records:
        st.info("No timing data recorded yet. Run some chat queries first.")
        return

    # Show raw records
    st.subheader("Raw timing records")
    st.json(records[-20:])  # last 20 records

    # Aggregate metrics
    st.subheader("Aggregated metrics (milliseconds)")
    steps = [
        "input_guardrail_ms",
        "query_embedding_ms",
        "vector_search_ms",
        "llm_generation_ms",
        "output_guardrail_ms",
        "total_ms",
    ]

    # Compute averages
    aggregates = {}
    for step in steps:
        values = []
        for rec in records:
            if step == "total_ms":
                v = rec.get("total_ms")
            else:
                v = rec.get("steps", {}).get(step)
            if v is not None:
                values.append(v)
        if values:
            aggregates[step] = {
                "avg_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "count": len(values),
            }

    if not aggregates:
        st.info("No complete timing data to aggregate yet.")
        return

    # Display as table
    import pandas as pd

    rows = []
    for step, stats in aggregates.items():
        rows.append(
            {
                "Step": step,
                "Count": stats["count"],
                "Avg (ms)": round(stats["avg_ms"], 2),
                "Min (ms)": round(stats["min_ms"], 2),
                "Max (ms)": round(stats["max_ms"], 2),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# =========================
# Main App
# =========================

def main():
    st.set_page_config(page_title="RAG with vLLM & Milvus", layout="wide")

    st.sidebar.title("RAG App")
    page = st.sidebar.radio(
        "Navigate",
        ["Chat", "Documents", "Statistics"],
        index=0,
    )

    if page == "Chat":
        page_chat()
    elif page == "Documents":
        page_documents()
    elif page == "Statistics":
        page_statistics()


if __name__ == "__main__":
    main()

