from typing import Dict, List, Optional, Union
import aiohttp
import json
from .base import LLMProvider

class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""
    
    def __init__(self, base_url: str = "http://ollama:11434", model: str = "llama2"):
        self.base_url = base_url.rstrip('/')
        self.model = model

    async def _post_request(self, endpoint: str, data: dict) -> dict:
        """Helper method to make POST requests to Ollama API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/{endpoint}", json=data) as response:
                return await response.json()

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using Ollama."""
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        if max_tokens:
            data["num_predict"] = max_tokens
        if stop_sequences:
            data["stop"] = stop_sequences

        response = await self._post_request("api/generate", data)
        return response.get("response", "")

    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = await self._post_request("api/embeddings", {
                "model": self.model,
                "prompt": text
            })
            embeddings.append(response.get("embedding", []))
        
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze an image using Ollama."""
        # Note: This assumes using a multimodal model like llava
        import base64
        
        data = {
            "model": "llava",  # Override model for image analysis
            "prompt": prompt,
            "images": [base64.b64encode(image_data).decode('utf-8')]
        }
        
        response = await self._post_request("api/generate", data)
        return response.get("response", "")

    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # Simple approximation as Ollama doesn't provide token counting
        return len(text.split()) * 1.3

    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        try:
            response = await self._post_request("api/show", {"name": self.model})
            return {
                "provider": "ollama",
                "model": self.model,
                "capabilities": ["text", "embeddings"],
                "details": response
            }
        except Exception:
            return {
                "provider": "ollama",
                "model": self.model,
                "capabilities": ["text", "embeddings"],
                "max_tokens": 2048  # Default approximation
            } 