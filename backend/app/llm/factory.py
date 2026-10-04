from typing import Optional
from app.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama_client import OllamaClient
from app.llm.openai_provider import OpenAIProvider

def get_llm_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> LLMProvider:
    provider = provider_name or settings.LLM_PROVIDER

    if provider.lower() == "openai":
        return OpenAIProvider(model=model_name or settings.OPENAI_MODEL)
    else:
        return OllamaClient(model=model_name or settings.OLLAMA_MODEL)

def get_evaluator_provider() -> LLMProvider:
    """Returns the LLM provider configured specifically for evaluation."""
    provider_name = settings.EVALUATOR_PROVIDER
    model_name = settings.EVALUATOR_MODEL

    # If using OpenAI for evaluation and no specific model is set, default to OPENAI_MODEL
    if provider_name.lower() == "openai" and not model_name:
        model_name = settings.OPENAI_MODEL

    return get_llm_provider(provider_name=provider_name, model_name=model_name)

def get_orchestrator_provider() -> LLMProvider:
    """Returns the LLM provider configured specifically for orchestration."""
    provider_name = settings.ORCHESTRATOR_PROVIDER
    model_name = settings.ORCHESTRATOR_MODEL

    if provider_name.lower() == "openai" and not model_name:
        model_name = settings.OPENAI_MODEL

    return get_llm_provider(provider_name=provider_name, model_name=model_name)
