from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    chunk_id: int


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(TextChunk(text=chunk, chunk_id=chunk_id))
            chunk_id += 1

        start += chunk_size - overlap

    return chunks