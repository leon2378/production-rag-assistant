from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.tolist()