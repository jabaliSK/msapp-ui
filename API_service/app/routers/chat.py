
from fastapi import APIRouter, BackgroundTasks
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import rag_chat

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
def chat_route(req: ChatRequest, bg: BackgroundTasks):
    return rag_chat(req, bg)
