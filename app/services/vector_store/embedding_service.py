from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import getSettings
from app.core.exceptions import AppException
from app.core.logging import logger


class EmbeddingService:
    """Wrapper that produces a LangChain Embeddings model from configuration."""

    @classmethod
    def build(cls) -> Embeddings:
        settings = getSettings()
        provider = settings.EMBEDDING_PROVIDER.lower().strip()
        return cls._cachedBuild(provider, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS)

    @staticmethod
    @lru_cache(maxsize=4)
    def _cachedBuild(provider: str, model: str, dimensions: int) -> Embeddings:
        settings = getSettings()
        logger.info("embeddings_init", provider=provider, model=model)

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise AppException(400, "EMBEDDING_CONFIG_ERROR", "OPENAI_API_KEY not configured for embeddings")
            return OpenAIEmbeddings(
                model=model,
                dimensions=dimensions,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                timeout=60,
                max_retries=2,
            )

        if provider == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise AppException(400, "EMBEDDING_CONFIG_ERROR", "OPENROUTER_API_KEY not configured")
            return OpenAIEmbeddings(
                model=model,
                dimensions=dimensions,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                timeout=60,
                max_retries=2,
            )

        raise AppException(400, "EMBEDDING_CONFIG_ERROR", f"Unsupported embedding provider: {provider}")
