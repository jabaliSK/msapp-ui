# file_service.py
import io
import hashlib
import zipfile
import base64
import urllib.parse
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from PIL import Image
from pymilvus import DataType, MilvusClient
from bson.binary import Binary
from bson.objectid import ObjectId

# Import from common config
from config import (
    TOOL_HOST_IP, MILVUS_URI, COLLECTION_NAME, CACHE_COLLECTION,
    default_config, mongo_coll, milvus, 
    FileMetadata, FileGroup, FileUploadResponse
)

app = FastAPI(title="File Management & Ingestion Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Local copy of config for client init
current_config = default_config.copy()

# --- CLIENTS (OCR & EMBEDDING) ---
clients = {
    "embed": OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['embed_port']}/v1", api_key="EMPTY"),
    "vl": OpenAI(base_url=f"{TOOL_HOST_IP}:{current_config['vl_port']}/v1", api_key="EMPTY")
}

# Global Dimension State
CURRENT_DIMENSION = 1024

# --- HELPERS ---

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    response = clients["embed"].embeddings.create(input=[text], model=current_config["embed_model_id"])
    return response.data[0].embedding

def detect_embedding_dimension():
    global CURRENT_DIMENSION
    try:
        response = clients["embed"].embeddings.create(input=["warmup"], model=current_config["embed_model_id"])
        if response.data:
            CURRENT_DIMENSION = len(response.data[0].embedding)
            print(f"Dimension detected: {CURRENT_DIMENSION}")
    except Exception as e: print(f"Dim detection failed: {e}")

# Call on startup
@app.on_event("startup")
def startup_event():
    detect_embedding_dimension()
    # Init Milvus Collections if needed
    if not milvus.has_collection(COLLECTION_NAME):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=CURRENT_DIMENSION)
        schema.add_field(field_name="chunk_md5", datatype=DataType.VARCHAR, max_length=64)
        index_params = milvus.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="chunk_md5", index_type="STL_SORT")
        milvus.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

    if not milvus.has_collection(CACHE_COLLECTION):
         milvus.create_collection(collection_name=CACHE_COLLECTION, dimension=CURRENT_DIMENSION, metric_type="COSINE", auto_id=True, enable_dynamic_field=True)


def encode_image_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

def perform_ocr(image_bytes: bytes, filename_context: str) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB': img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_image = encode_image_base64(buffered.getvalue())
        response = clients["vl"].chat.completions.create(
            model=current_config["vl_model_id"],
            messages=[{"role": "user", "content": [{"type": "text", "text": "Transcribe the text in this image word for word."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}],
            max_tokens=512, temperature=0.1
        )
        return f"\n[OCR Content {filename_context}]:\n{response.choices[0].message.content}\n"
    except Exception as e:
        print(f"OCR Failed for {filename_context}: {e}")
        return ""

def parse_file_bytes(file_bytes: bytes, filename: str) -> str:
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
        except Exception as e: print(f"Docx Error: {e}")
    elif extension == "txt":
        content = file_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return content

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

# --- API ENDPOINTS ---

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
    # (Implementation matches original logic, checking chunk hashes in Milvus and removing references)
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

@app.delete("/files/all/{filename}")
def delete_all_file_versions(filename: str):
    # Logic to loop through all versions and delete them (same as original code)
    try:
        docs = list(mongo_coll.find({"filename": filename}))
        if not docs: raise HTTPException(status_code=404, detail="File not found")
        deleted_count = 0
        for doc in docs:
            delete_file(str(doc["_id"])) # Reuse the single delete logic for cleanliness
            deleted_count += 1
        return {"status": "deleted", "count": deleted_count, "filename": filename}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8051)