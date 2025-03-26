from typing import Optional
from .base import LLMProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance.
        
        Args:
            provider_type: Type of provider ("anthropic", "openai", or "ollama")
            api_key: API key for cloud providers
            model: Model name to use
            base_url: Base URL for self-hosted providers
            **kwargs: Additional provider-specific arguments
        
        Returns:
            An instance of LLMProvider
        
        Raises:
            ValueError: If provider_type is invalid or required arguments are missing
        """
        provider_type = provider_type.lower()
        
        if provider_type == "anthropic":
            if not api_key:
                raise ValueError("API key is required for Anthropic")
            return AnthropicProvider(
                api_key=api_key,
                model=model or "claude-3-opus-20240229",
                **kwargs
            )
        
        elif provider_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI")
            return OpenAIProvider(
                api_key=api_key,
                model=model or "gpt-4-turbo-preview",
                **kwargs
            )
        
        elif provider_type == "ollama":
            return OllamaProvider(
                base_url=base_url or "http://ollama:11434",
                model=model or "llama2",
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get a list of available provider types."""
        return ["anthropic", "openai", "ollama"] 