from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import os
import uuid
import shutil
from app.db import get_db_connection
from app.services.onboarding import OnboardingService
from app.services.embedding import EmbeddingService
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

def get_onboarding_service(conn = Depends(get_db_connection)):
    return OnboardingService(conn)

@router.get("/models")
async def get_available_models(
    onboarding_service: OnboardingService = Depends(get_onboarding_service),
):
    """
    Get available LLM models for onboarding
    """
    try:
        models = await onboarding_service.get_available_models()
        return {"models": models, "provider": onboarding_service.settings.LLM_PROVIDER}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

def get_embedding_service():
    return EmbeddingService()

@router.post("/start")
async def start_onboarding(
    onboarding_service: OnboardingService = Depends(get_onboarding_service)
):
    """
    Start a new onboarding session
    """
    session_id = onboarding_service.start_session()
    return {"session_id": session_id, "status": "started"}

@router.post("/message/{session_id}")
async def process_message(
    session_id: str,
    message: Dict[str, str],
    onboarding_service: OnboardingService = Depends(get_onboarding_service)
):
    """
    Process a message in the onboarding conversation
    """
    if "message" not in message:
        raise HTTPException(status_code=400, detail="Message field is required")
    
    try:
        result = await onboarding_service.process_message(session_id, message["message"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    onboarding_service: OnboardingService = Depends(get_onboarding_service)
):
    """
    Get data for a session
    """
    try:
        result = onboarding_service.get_session_data(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@router.post("/upload/logo/{session_id}")
async def upload_logo(
    session_id: str,
    file: UploadFile = File(...),
    brand_id: str = Form(...),
    conn = Depends(get_db_connection),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Upload a logo for a brand
    """
    # Verify file type
    allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".svg"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Please upload {', '.join(allowed_extensions)}"
        )
    
    # Create a unique filename
    storage_path = os.getenv("STORAGE_PATH", "./storage")
    logos_dir = os.path.join(storage_path, "logos")
    os.makedirs(logos_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(logos_dir, f"{file_id}{file_ext}")
    
    # Save the file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        # Create vector embedding for logo
        vector_id = await embedding_service.store_logo_embedding(file_path, brand_id)
        
        # Store logo info in database
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO brand_assets (id, brand_id, asset_type, file_path, mime_type, vector_id)
            VALUES (uuid_generate_v4(), %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (brand_id, "logo", file_path, file.content_type, vector_id)
        )
        
        asset_id = cursor.fetchone()[0]
        conn.commit()
        
        return {
            "asset_id": asset_id,
            "brand_id": brand_id,
            "file_path": file_path,
            "status": "uploaded"
        }
        
    except Exception as e:
        # Clean up file if database operation fails
        if os.path.exists(file_path):
            os.remove(file_path)
        
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error storing logo: {str(e)}")
