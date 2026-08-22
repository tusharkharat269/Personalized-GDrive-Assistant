from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import getSettings
from app.core.exceptions import AppException
from app.core.logging import logger


@dataclass(frozen=True)
class LLMSpec:
    provider: str
    model: str
    temperature: float
    maxTokens: int

    @classmethod
    def default(cls) -> "LLMSpec":
        s = getSettings()
        return cls(
            provider=s.DEFAULT_LLM_PROVIDER,
            model=s.DEFAULT_LLM_MODEL,
            temperature=s.DEFAULT_LLM_TEMPERATURE,
            maxTokens=s.DEFAULT_LLM_MAX_TOKENS,
        )


class LLMFactory:
    """Produces LangChain chat models for OpenAI, Anthropic and OpenRouter."""

    _SUPPORTED_PROVIDERS = {"openai", "anthropic", "openrouter"}

    @classmethod
    def build(cls, spec: LLMSpec | None = None) -> BaseChatModel:
        spec = spec or LLMSpec.default()
        provider = spec.provider.lower().strip()
        if provider not in cls._SUPPORTED_PROVIDERS:
            raise AppException(400, "LLM_UNSUPPORTED_PROVIDER", f"Unsupported LLM provider: {provider}")
        return cls._cachedBuild(provider, spec.model, spec.temperature, spec.maxTokens)

    @classmethod
    @lru_cache(maxsize=32)
    def _cachedBuild(cls, provider: str, model: str, temperature: float, maxTokens: int) -> BaseChatModel:
        settings = getSettings()
        logger.info("llm_init", provider=provider, model=model, temperature=temperature)

        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise AppException(400, "LLM_CONFIG_ERROR", "OPENAI_API_KEY not configured")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=maxTokens,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                timeout=60,
                max_retries=2,
            )

        if provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise AppException(400, "LLM_CONFIG_ERROR", "ANTHROPIC_API_KEY not configured")
            kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "max_tokens": maxTokens,
                "api_key": settings.ANTHROPIC_API_KEY,
                "timeout": 60,
                "max_retries": 2,
            }
            if settings.ANTHROPIC_BASE_URL:
                kwargs["base_url"] = settings.ANTHROPIC_BASE_URL
            return ChatAnthropic(**kwargs)

        # openrouter — uses OpenAI-compatible chat completions
        if not settings.OPENROUTER_API_KEY:
            raise AppException(400, "LLM_CONFIG_ERROR", "OPENROUTER_API_KEY not configured")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=maxTokens,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": settings.OPENROUTER_SITE_URL,
                "X-Title": settings.OPENROUTER_APP_NAME,
            },
            timeout=60,
            max_retries=2,
        )
