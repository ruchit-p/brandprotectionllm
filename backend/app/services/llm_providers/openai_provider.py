from typing import Dict, List, Optional, Union
import base64
from openai import AsyncOpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = "text-embedding-3-small"

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
            **kwargs
        )
        return response.choices[0].message.content

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using OpenAI."""
        if isinstance(texts, str):
            texts = [texts]
        
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            **kwargs
        )
        
        embeddings = [data.embedding for data in response.data]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze an image using OpenAI Vision."""
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            **kwargs
        )
        return response.choices[0].message.content

    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # Note: This is a simplified implementation
        # For production, use tiktoken for accurate counts
        return len(text.split()) * 1.3

    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        return {
            "provider": "openai",
            "model": self.model,
            "embedding_model": self.embedding_model,
            "capabilities": ["text", "vision", "embeddings"],
            "max_tokens": 4096  # This varies by model
        } 