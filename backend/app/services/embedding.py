import os
import base64
import asyncio
from typing import Dict, List, Any, Optional
import anthropic
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.db import get_qdrant_client
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    """
    Service for creating and managing vector embeddings using Claude and Qdrant
    """
    
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.qdrant_client = get_qdrant_client()
    
    async def create_text_embedding(self, text: str) -> List[float]:
        """Create vector embedding for text"""
        # Truncate text if too long
        if len(text) > 25000:
            text = text[:25000]
        
        # Generate embedding using Claude API
        response = await self.anthropic_client.embeddings.create(
            model="claude-3-haiku-20240307",
            input=text,
            dimensions=1536
        )
        
        return response.embeddings[0]
    
    async def create_image_embedding(self, image_path: str) -> List[float]:
        """Create vector embedding for image via Claude"""
        # Read image file
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Get image file extension (for mime type)
        file_ext = os.path.splitext(image_path)[1][1:].lower()
        mime_type = f"image/{file_ext}"
        
        # Have Claude describe the image
        response = await self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64.b64encode(image_data).decode("utf-8")
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail, focusing on visual elements, logos, products, and brand-related features."
                        }
                    ]
                }
            ]
        )
        
        # Get the text description from the response
        description = response.content[0].text
        
        # Generate embedding of the description
        embedding_response = await self.anthropic_client.embeddings.create(
            model="claude-3-haiku-20240307",
            input=description,
            dimensions=1536
        )
        
        return embedding_response.embeddings[0]
    
    async def store_logo_embedding(self, logo_path: str, brand_id: str) -> str:
        """Store logo embedding in Qdrant"""
        # Generate embedding
        embedding = await self.create_image_embedding(logo_path)
        
        # Generate a unique ID for the vector
        vector_id = f"logo_{brand_id}"
        
        # Store in Qdrant
        self.qdrant_client.upsert(
            collection_name="brand_logos",
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        "brand_id": brand_id,
                        "file_path": logo_path,
                        "type": "logo"
                    }
                )
            ]
        )
        
        return vector_id
    
    async def store_website_embeddings(self, snapshot_id: str, html_path: str, text_content: str, screenshot_path: str) -> Dict[str, str]:
        """Create and store embeddings for website HTML, text and screenshot"""
        # Generate embeddings
        html_embedding = None
        text_embedding = None
        screenshot_embedding = None
        
        # Read HTML file
        if html_path and os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            html_embedding = await self.create_text_embedding(html_content)
        
        # Process text content
        if text_content:
            text_embedding = await self.create_text_embedding(text_content)
        
        # Process screenshot
        if screenshot_path and os.path.exists(screenshot_path):
            screenshot_embedding = await self.create_image_embedding(screenshot_path)
        
        # Store embeddings
        results = {}
        
        if html_embedding:
            html_vector_id = f"html_{snapshot_id}"
            self.qdrant_client.upsert(
                collection_name="site_html",
                points=[
                    models.PointStruct(
                        id=html_vector_id,
                        vector=html_embedding,
                        payload={
                            "snapshot_id": snapshot_id,
                            "file_path": html_path,
                            "type": "html"
                        }
                    )
                ]
            )
            results["html_vector_id"] = html_vector_id
        
        if text_embedding:
            text_vector_id = f"text_{snapshot_id}"
            self.qdrant_client.upsert(
                collection_name="site_text",
                points=[
                    models.PointStruct(
                        id=text_vector_id,
                        vector=text_embedding,
                        payload={
                            "snapshot_id": snapshot_id,
                            "type": "text"
                        }
                    )
                ]
            )
            results["text_vector_id"] = text_vector_id
        
        if screenshot_embedding:
            screenshot_vector_id = f"screenshot_{snapshot_id}"
            self.qdrant_client.upsert(
                collection_name="site_images",
                points=[
                    models.PointStruct(
                        id=screenshot_vector_id,
                        vector=screenshot_embedding,
                        payload={
                            "snapshot_id": snapshot_id,
                            "file_path": screenshot_path,
                            "type": "screenshot"
                        }
                    )
                ]
            )
            results["screenshot_vector_id"] = screenshot_vector_id
        
        return results
    
    async def search_similar_images(self, image_path: str, collection_name: str = "site_images", limit: int = 10) -> List[Dict]:
        """Search for similar images based on embedding similarity"""
        # Generate embedding for the query image
        embedding = await self.create_image_embedding(image_path)
        
        # Search for similar images
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        
        return results
    
    async def search_similar_text(self, text: str, collection_name: str = "site_text", limit: int = 10) -> List[Dict]:
        """Search for similar text based on embedding similarity"""
        # Generate embedding for the query text
        embedding = await self.create_text_embedding(text)
        
        # Search for similar text
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        
        return results
