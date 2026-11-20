"""Chat model factory. All pipelines get their LLM from here so model settings stay in one place."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.errors import LLMNotConfiguredError

MISSING_KEY_MESSAGE = (
    "No LLM API key is configured: set OPENAI_API_KEY (or GEMINI_API_KEY with LLM_PROVIDER=gemini or auto) "
    "in .env and restart the backend."
)


def create_chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    if not settings.llm_configured:
        raise LLMNotConfiguredError(MISSING_KEY_MESSAGE)
    if settings.resolved_llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries makes the client back off and retry on rate limits (HTTP 429) of the free tier.
        name = model or settings.gemini_model
        # Gemini 2.5 Flash "thinks" before answering by default; a budget of 0 makes replies much faster.
        extra = {"thinking_budget": 0} if "2.5-flash" in name and "lite" not in name else {}
        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_retries=max(settings.llm_max_retries, 5),
            **extra,
        )
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def is_rate_limit(exc: BaseException) -> bool:
    """True for provider quota / rate-limit errors (HTTP 429), whichever client raised them."""
    return "RateLimit" in type(exc).__name__ or "RESOURCE_EXHAUSTED" in str(exc)[:400] or "429" in str(exc)[:120]


def is_gemini(model: Any) -> bool:
    return "GoogleGenerativeAI" in type(model).__name__


def limit_kwargs(model: Any, max_tokens: int) -> dict[str, Any]:
    """The provider's name for the answer-length limit."""
    return {"max_output_tokens": max_tokens} if is_gemini(model) else {"max_tokens": max_tokens}


def json_kwargs(model: Any) -> dict[str, Any]:
    """The provider's switch for JSON-only replies."""
    if is_gemini(model):
        return {"response_mime_type": "application/json"}
    return {"response_format": {"type": "json_object"}}


def message_text(message: BaseMessage) -> str:
    """Plain text of a chat message whose content may be a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
