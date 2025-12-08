
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProcessTimings(BaseModel):
    input_guardrail_sec: float
    embedding_sec: float
    retrieval_sec: float
    llm_generation_sec: float
    output_guardrail_sec: float
    total_sec: float

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

class FileResponse(BaseModel):
    filename: str
    status: str
    extracted_text_preview: Optional[str] = None
