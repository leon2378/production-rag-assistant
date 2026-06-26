import uuid 

from app.rag.document_loader import load_pdf_text
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore
from app.rag.llm import generate_answer


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
        payload = result.payload
        context_chunks.append(payload["text"])

        sources.append(
            {
                "source_file": payload["source_file"],
                "chunk_index": payload["chunk_index"],
                "score": result.score,
                "preview": payload["text"][:250],
            }
        )

    answer = await generate_answer(question, context_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }