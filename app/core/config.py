from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "gdrive-assistant"
    APP_ENV: str = "development"
    SECRET_KEY: str
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    DATABASE_URL: str
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    ENCRYPTION_KEY: str

    GOOGLE_SCOPES: list[str] = [
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "openid",
    ]

    # ---------- LLM / AI ----------
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    DEFAULT_LLM_TEMPERATURE: float = 0.2
    DEFAULT_LLM_MAX_TOKENS: int = 2048

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_BASE_URL: str | None = None

    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_APP_NAME: str = "gdrive-assistant"
    OPENROUTER_SITE_URL: str = "http://localhost:3000"

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ---------- Vector store ----------
    VECTOR_STORE_PROVIDER: str = "chroma_cloud"
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_PREFIX: str = "user_"

    # Chroma Cloud
    CHROMA_API_KEY: str | None = None
    CHROMA_TENANT: str | None = None
    CHROMA_DATABASE: str | None = None
    CHROMA_HOST: str = "api.trychroma.com"

    # ---------- RAG ----------
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_TOP_K: int = 5
    RAG_MAX_FILE_SIZE_MB: int = 25
    RAG_SUPPORTED_MIMETYPES: list[str] = [
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "application/json",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
    ]

    # ---------- Chat / Agent ----------
    AGENT_MAX_ITERATIONS: int = 12
    CHAT_HISTORY_WINDOW: int = 20
    CHAT_TITLE_MAX_LEN: int = 80

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def getSettings() -> Settings:
    return Settings()
