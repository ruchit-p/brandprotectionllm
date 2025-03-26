from typing import Optional
from .base import LLMProvider
from .factory import LLMProviderFactory
from ...config import settings

_provider_instance: Optional[LLMProvider] = None

def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider instance.
    
    Returns:
        LLMProvider: The configured provider instance
    
    Raises:
        ValueError: If the provider configuration is invalid
    """
    global _provider_instance
    
    if _provider_instance is None:
        provider_type = settings.LLM_PROVIDER.lower()
        
        if provider_type == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
            _provider_instance = LLMProviderFactory.create_provider(
                provider_type="anthropic",
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.LLM_MODEL
            )
        
        elif provider_type == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            _provider_instance = LLMProviderFactory.create_provider(
                provider_type="openai",
                api_key=settings.OPENAI_API_KEY,
                model=settings.LLM_MODEL
            )
        
        elif provider_type == "ollama":
            _provider_instance = LLMProviderFactory.create_provider(
                provider_type="ollama",
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL or settings.LLM_MODEL
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
    
    return _provider_instance

def reset_llm_provider():
    """Reset the LLM provider instance.
    
    This is useful for testing or when configuration changes.
    """
    global _provider_instance
    _provider_instance = None 