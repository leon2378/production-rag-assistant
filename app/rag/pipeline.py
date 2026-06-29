import uuid 

from app.rag.document_loader import load_pdf_text
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.llm import generate_answer

from langfuse import Langfuse
from app.core.config import settings

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_private_key,
    host=settings.langfuse_host,
)


embedding_model = EmbeddingModel()
vector_store = VectorStore(vector_size=384)

def ingest_document(file_path: str, source_file: str) -> dict:
    document_id = str(uuid.uuid4())

    text = load_pdf_text(file_path)
    chunks = chunk_text(text)

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.embed_texts(chunk_texts)

    vector_store.add_chunks(
        document_id=document_id,
        chunks=chunk_texts,
        embeddings=embeddings,
        source_file=source_file,
    )

    return {
        "document_id": document_id,
        "source_file": source_file,
        "num_chunks": len(chunks),
    }


async def answer_question(question: str, top_k: int = 5) -> dict:
    query_embedding = embedding_model.embed_query(question)
    search_results = vector_store.search(query_embedding, top_k=top_k)

    context_chunks = []
    sources = []

    for result in search_results:
        payload = result.payload or {}

        chunk_text = payload.get("text", "")

        if chunk_text:
            context_chunks.append(chunk_text)

        sources.append(
            {
                "source_file": payload.get("source_file", "unknown"),
                "chunk_index": payload.get("chunk_index", 0),
                "score": float(result.score),
                "preview": chunk_text[:250],
            }
        )

    if not context_chunks:
        return{
            "question": question,
            "answer": "I could not find relevant information in the uploaded documents.",
            "sources": sources,
        }

    answer = await generate_answer(question, context_chunks)


    trace = langfuse.trace(
        name="rag-question-answering",
        input={"question": question},
    )

    trace.update(
        output={
            "answer": answer,
            "sources": sources,
        }
    )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }