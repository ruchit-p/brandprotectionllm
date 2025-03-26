from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)."""
        pass

    @abstractmethod
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        **kwargs
    ) -> str:
        """Analyze an image with a prompt."""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        pass 