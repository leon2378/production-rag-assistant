from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Production RAG Assistant"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    langfuse_public_key: str | None = None
    langfuse_private_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"

settings = Settings()