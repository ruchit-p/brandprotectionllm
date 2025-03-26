from typing import Dict, List, Optional, Union
import anthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic Claude."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stop_sequences=stop_sequences,
            **kwargs
        )
        return response.content[0].text

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Anthropic Claude."""
        # Note: As of now, Anthropic doesn't provide a dedicated embeddings API
        # This is a placeholder for when they do, or we could use a different provider for embeddings
        raise NotImplementedError("Anthropic does not currently support embeddings")

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze an image using Anthropic Claude."""
        message = anthropic.Message(
            role="user",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data.decode('utf-8')
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )
        
        response = await self.client.messages.create(
            model=self.model,
            messages=[message],
            max_tokens=1000,
            **kwargs
        )
        return response.content[0].text

    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        return anthropic.count_tokens(text)

    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "capabilities": ["text", "vision"],
            "max_tokens": 4096  # This varies by model
        } 