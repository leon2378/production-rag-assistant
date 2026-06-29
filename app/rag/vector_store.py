from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings


class VectorStore:
    def __init__(self, vector_size: int = 384):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.qdrant_collection
        self.vector_size = vector_size

    def ensure_collection(self):
        collections = self.client.get_collections().collections
        existing_names = [collection.name for collection in collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, 
                    distance=Distance.COSINE,
                ),
            )

    def add_chunks(
            self, 
            document_id: str,
            chunks: list[str],
            embeddings: list[list[float]],
            source_file: str,
    ):
        self.ensure_collection()

        points = []

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = abs(hash(f"{document_id}-{index}"))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_index": index,
                        "text": chunk,
                        "source_file": source_file,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return response.points