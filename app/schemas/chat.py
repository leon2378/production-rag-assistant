from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    source_file: str
    chunk_index: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
