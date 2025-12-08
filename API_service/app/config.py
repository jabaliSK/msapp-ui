
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    TOOL_HOST_IP: str = "http://10.25.73.101"
    TOOL_HOST_IP_ONLY: str = "10.25.73.101"
    MILVUS_PORT: int = 19530

    COLLECTION_NAME: str = "qwen_rag_collection"
    CACHE_COLLECTION: str = "qwen_rag_cache"

    PG_DB: str = "rag_stats"
    PG_USER: str = "jabali"
    PG_PASSWORD: str = "jabali"
    PG_HOST: str = TOOL_HOST_IP_ONLY
    PG_PORT: str = "5432"

    SYSTEM_PROMPT: str = "You are a helpful assistant..."

settings = Settings()
