
from pymilvus import MilvusClient
from app.config import settings

milvus = MilvusClient(uri=f"{settings.TOOL_HOST_IP}:{settings.MILVUS_PORT}")

def init_milvus():
    if not milvus.has_collection(settings.COLLECTION_NAME):
        milvus.create_collection(
            collection_name=settings.COLLECTION_NAME,
            dimension=1024,
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True,
        )
    if not milvus.has_collection(settings.CACHE_COLLECTION):
        milvus.create_collection(
            collection_name=settings.CACHE_COLLECTION,
            dimension=1024,
            metric_type="COSINE",
            auto_id=True,
            enable_dynamic_field=True,
        )
    milvus.load_collection(settings.CACHE_COLLECTION)

def search_cache(vec):
    res = milvus.search(
        collection_name=settings.CACHE_COLLECTION,
        data=[vec], limit=1,
        output_fields=["response", "original_query"]
    )
    if res and res[0]:
        return res[0][0]["entity"]
    return None

def search_collection(vec):
    res = milvus.search(
        collection_name=settings.COLLECTION_NAME,
        data=[vec], limit=3,
        output_fields=["text", "filename"]
    )
    chunks, sources = [], []
    if res and res[0]:
        for hit in res[0]:
            chunks.append(hit["entity"]["text"])
            sources.append(hit["entity"]["filename"])
    return chunks, list(set(sources))

def insert_cache(vec, response, q):
    milvus.insert(
        collection_name=settings.CACHE_COLLECTION,
        data=[{"vector": vec, "response": response, "original_query": q}]
    )
