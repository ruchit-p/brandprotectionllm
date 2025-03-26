import os
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional
import anthropic
from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.memory import ConversationBufferMemory
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import httpx
from app.config import get_settings

load_dotenv()

class OnboardingService:
    """
    Service for client onboarding using LangChain and Claude
    """
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.sessions = {}
        self.settings = get_settings()
        
        # Initialize LLM based on configuration
        self._initialize_llm()
        
        # Setup output parser for structured brand data
        self.output_parser = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="brand_name", description="Name of the brand"),
            ResponseSchema(name="website_url", description="Official website URL"),
            ResponseSchema(name="description", description="Brand description"),
            ResponseSchema(name="social_media", description="List of social media handles as objects with platform and handle"),
            ResponseSchema(name="key_terms", description="List of key brand terms and phrases")
        ])
        
        self.format_instructions = self.output_parser.get_format_instructions()
        
        # Create conversation prompt
        self.prompt = PromptTemplate(
            template="""
            You are a brand protection assistant helping a client set up protection for their brand.
            
            Your task is to collect the following details through a friendly, conversational interaction:
            - Brand name
            - Official website URL
            - Brief brand description
            - Social media handles (ask for platforms like Twitter/X, Facebook, Instagram, LinkedIn, etc.)
            - Key brand terms and phrases (including product names, slogans, etc.)
            
            Important instructions:
            1. Ask one question at a time, focusing on collecting each piece of information separately
            2. Keep track of what information you've already collected
            3. Be friendly and conversational
            4. Once you have collected ALL the required information, summarize what you've learned
            5. Then, and ONLY then, provide the structured data in JSON format according to the format instructions below
            6. After providing the JSON, tell the user that onboarding is complete and they will proceed to the next step
            
            {format_instructions}
            
            Previous conversation:
            {chat_history}
            
            Human: {input}
            AI: """,
            input_variables=["chat_history", "input"],
            partial_variables={"format_instructions": self.format_instructions}
        )
    
    def _initialize_llm(self):
        """
        Initialize the appropriate LLM based on configuration
        """
        if self.settings.LLM_PROVIDER.lower() == "anthropic":
            self.llm = ChatAnthropic(
                model=self.settings.LLM_MODEL or "claude-3-7-sonnet-20250219",
                anthropic_api_key=self.settings.ANTHROPIC_API_KEY,
                temperature=0.2
            )
        elif self.settings.LLM_PROVIDER.lower() == "openai":
            self.llm = ChatOpenAI(
                model=self.settings.LLM_MODEL or "gpt-4",
                api_key=self.settings.OPENAI_API_KEY,
                temperature=0.2
            )
        elif self.settings.LLM_PROVIDER.lower() == "ollama":
            # Import inside method to avoid unnecessary dependencies
            from langchain_community.llms import Ollama
            self.llm = Ollama(
                model=self.settings.OLLAMA_MODEL or "llama2",
                base_url=self.settings.OLLAMA_BASE_URL
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.settings.LLM_PROVIDER}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models from the configured provider
        """
        models = []
        
        try:
            if self.settings.LLM_PROVIDER.lower() == "anthropic":
                # Get Anthropic models
                if self.settings.ANTHROPIC_API_KEY:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            "https://api.anthropic.com/v1/models",
                            headers={
                                "x-api-key": self.settings.ANTHROPIC_API_KEY,
                                "anthropic-version": "2023-06-01"
                            }
                        )
                        if response.status_code == 200:
                            for model in response.json().get("data", []):
                                models.append({
                                    "id": model.get("id"),
                                    "name": model.get("id"),
                                    "provider": "anthropic",
                                    "description": f"Context: {model.get('context_window')} tokens"
                                })
            elif self.settings.LLM_PROVIDER.lower() == "openai":
                # Get OpenAI models
                if self.settings.OPENAI_API_KEY:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            "https://api.openai.com/v1/models",
                            headers={"Authorization": f"Bearer {self.settings.OPENAI_API_KEY}"}
                        )
                        if response.status_code == 200:
                            for model in response.json().get("data", []):
                                if model.get("id").startswith("gpt"):
                                    models.append({
                                        "id": model.get("id"),
                                        "name": model.get("id"),
                                        "provider": "openai",
                                        "description": "OpenAI model"
                                    })
            elif self.settings.LLM_PROVIDER.lower() == "ollama":
                # Get Ollama models
                if self.settings.OLLAMA_BASE_URL:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.settings.OLLAMA_BASE_URL}/api/tags")
                        if response.status_code == 200:
                            for model in response.json().get("models", []):
                                models.append({
                                    "id": model.get("name"),
                                    "name": model.get("name"),
                                    "provider": "ollama",
                                    "description": "Local Ollama model"
                                })
        except Exception as e:
            print(f"Error fetching models: {e}")
        
        # If no models found, add default ones
        if not models:
            if self.settings.LLM_PROVIDER.lower() == "anthropic":
                models = [
                    {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet", "provider": "anthropic", "description": "Latest Claude model"},
                    {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "provider": "anthropic", "description": "Highest capability Claude model"},
                    {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet", "provider": "anthropic", "description": "Balanced Claude model"}
                ]
            elif self.settings.LLM_PROVIDER.lower() == "openai":
                models = [
                    {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "description": "Highest capability OpenAI model"},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai", "description": "Fast OpenAI model"}
                ]
            elif self.settings.LLM_PROVIDER.lower() == "ollama":
                models = [
                    {"id": "llama2", "name": "Llama 2", "provider": "ollama", "description": "Meta's Llama 2 model"},
                    {"id": "mistral", "name": "Mistral", "provider": "ollama", "description": "Mistral 7B model"}
                ]
        
        return models

    def start_session(self) -> str:
        """
        Start a new onboarding session
        """
        session_id = str(uuid.uuid4())
        
        # Create memory to track conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Initialize the conversation with the first LLM message
        conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=memory,
            verbose=True
        )
        
        # Get the initial greeting from the LLM
        initial_prompt = "Hi, I'd like to set up brand protection."
        initial_response = conversation.predict(input=initial_prompt)
        
        # Store initial exchange in memory
        memory.save_context({"input": initial_prompt}, {"output": initial_response})
        
        self.sessions[session_id] = {
            "conversation": conversation,
            "memory": memory,
            "completed": False,
            "brand_data": {},
            "chat_history": [
                {"role": "human", "content": initial_prompt},
                {"role": "ai", "content": initial_response}
            ]
        }
        
        return session_id
    
    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a message in the onboarding conversation
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        # If session is already completed, return stored data
        if session["completed"]:
            return {
                "message": "Onboarding is already completed!",
                "completed": True,
                "brand_data": session["brand_data"]
            }
        
        # Process with LangChain conversation
        response = await asyncio.to_thread(
            session["conversation"].predict,
            input=message
        )
        
        # Manually update chat history for frontend display
        session["chat_history"].append({"role": "human", "content": message})
        session["chat_history"].append({"role": "ai", "content": response})
        
        # Try to parse structured data
        try:
            # Check if the response contains JSON
            if "```json" in response:
                # Extract JSON
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                
                # Parse JSON
                brand_data = json.loads(json_str)
                
                # Check if all required fields are present
                required_fields = ["brand_name", "website_url", "description", "social_media", "key_terms"]
                if all(field in brand_data for field in required_fields):
                    session["completed"] = True
                    session["brand_data"] = brand_data
                    
                    # Store in database
                    brand_id = self._store_brand_data(brand_data)
                    brand_data["brand_id"] = brand_id
                    
                    return {
                        "message": response.replace(json_str, "[Structured data recorded]"),
                        "completed": True,
                        "brand_data": brand_data
                    }
        except Exception as e:
            print(f"Error parsing brand data: {e}")
        
        # If not completed, return normal response
        return {
            "message": response,
            "completed": False
        }
    
    def _store_brand_data(self, brand_data: Dict[str, Any]) -> str:
        """
        Store brand data in PostgreSQL
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Insert brand
            cursor.execute(
                """
                INSERT INTO brands (id, name, website_url, description)
                VALUES (uuid_generate_v4(), %s, %s, %s)
                RETURNING id
                """,
                (brand_data["brand_name"], brand_data["website_url"], brand_data["description"])
            )
            
            brand_id = cursor.fetchone()["id"]
            
            # Insert social media
            for social in brand_data["social_media"]:
                cursor.execute(
                    """
                    INSERT INTO brand_social_media (brand_id, platform, handle, url)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        brand_id, 
                        social["platform"], 
                        social["handle"],
                        f"https://{social['platform'].lower()}.com/{social['handle'].replace('@', '')}"
                    )
                )
            
            # Insert keywords
            for keyword in brand_data["key_terms"]:
                cursor.execute(
                    """
                    INSERT INTO brand_keywords (brand_id, keyword)
                    VALUES (%s, %s)
                    """,
                    (brand_id, keyword)
                )
            
            self.db_connection.commit()
            return brand_id
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error storing brand data: {e}")
            raise
    
    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """
        Get data for a session
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        return {
            "completed": session["completed"],
            "brand_data": session["brand_data"],
            "chat_history": session["chat_history"]
        }
