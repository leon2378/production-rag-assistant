from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.rag.pipeline import answer_question


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await answer_question(
        question=request.question,
        top_k=request.top_k,
    )

    return result