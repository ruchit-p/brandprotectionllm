from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import os
import anthropic

from app.db import get_db_connection
from app.services.analysis import AnalysisService
from app.services.embedding import EmbeddingService
from app.services.rekognition import RekognitionService
from app.tasks.rekognition_tasks import (
    create_and_train_custom_dataset,
    start_custom_model,
    stop_custom_model,
    check_model_status
)
from app.tasks.analysis_tasks import (
    analyze_website_for_brand as analyze_website_task,
    search_similar_content as search_similar_content_task
)
from app.config import get_settings

settings = get_settings()
router = APIRouter()

def get_analysis_service(conn = Depends(get_db_connection)):
    embedding_service = EmbeddingService()
    return AnalysisService(conn, embedding_service)

def get_rekognition_service():
    return RekognitionService()

@router.post("/website")
async def analyze_website(
    data: Dict[str, Any],
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Analyze a website for a specific brand
    """
    if "brand_id" not in data or "website_id" not in data:
        raise HTTPException(status_code=400, detail="Brand ID and Website ID are required")
    
    try:
        # Start analysis as a Celery task
        task = analyze_website_task.delay(data["brand_id"], data["website_id"])
        
        return {
            "status": "analysis_started",
            "brand_id": data["brand_id"],
            "website_id": data["website_id"],
            "task_id": task.id,
            "message": "Website analysis has been started. Results will be available shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

@router.get("/website/{brand_id}/{website_id}")
async def get_website_analysis(
    brand_id: str,
    website_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis results for a website
    """
    try:
        # For now, this still runs synchronously since we're just fetching results
        analysis_result = await analysis_service.get_analysis_results(brand_id, website_id)
        return analysis_result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis results: {str(e)}")

@router.post("/brand/monitor/{brand_id}")
async def start_brand_monitoring(
    brand_id: str,
    conn = Depends(get_db_connection)
):
    """
    Start monitoring for a brand (placeholder for future implementation)
    """
    # This would typically be a background task that continuously monitors for new sites
    # For now, it's just a placeholder
    
    return {
        "status": "monitoring_started",
        "brand_id": brand_id,
        "message": "Brand monitoring has been started."
    }

@router.post("/search-similar")
async def search_similar_content(
    data: Dict[str, Any]
):
    """
    Search for similar content based on text or image
    """
    if "type" not in data or data["type"] not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="Type must be 'text' or 'image'")
    
    try:
        # Start search as a Celery task
        task = search_similar_content_task.delay(
            data["type"],
            data,
            data.get("limit", 10)
        )
        
        return {
            "status": "search_started",
            "task_id": task.id,
            "message": "Search for similar content has been started."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching for similar content: {str(e)}")

@router.get("/search-result/{task_id}")
async def get_search_result(
    task_id: str
):
    """
    Get the result of a search task
    """
    try:
        task_result = search_similar_content_task.AsyncResult(task_id)
        
        if task_result.ready():
            result = task_result.get()
            return result
        else:
            return {"status": "pending", "message": "Search is still in progress"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving search result: {str(e)}")

def get_embedding_service():
    return EmbeddingService()

# Custom Model Management Endpoints

class ModelTrainingRequest(BaseModel):
    brand_id: str
    force_retrain: bool = False

@router.post("/custom-model/train")
async def train_custom_model(
    request: ModelTrainingRequest,
    conn = Depends(get_db_connection),
    rekognition_service: RekognitionService = Depends(get_rekognition_service)
):
    """
    Start training a custom model for a brand
    """
    try:
        cursor = conn.cursor()
        
        # Check if brand exists
        cursor.execute(
            """
            SELECT id, name FROM brands WHERE id = %s
            """,
            (request.brand_id,)
        )
        brand = cursor.fetchone()
        
        if not brand:
            raise HTTPException(status_code=404, detail=f"Brand with ID {request.brand_id} not found")
        
        # Get brand logo
        cursor.execute(
            """
            SELECT file_path FROM brand_assets 
            WHERE brand_id = %s AND asset_type = 'logo'
            LIMIT 1
            """,
            (request.brand_id,)
        )
        logo = cursor.fetchone()
        
        if not logo or not logo[0]:
            raise HTTPException(status_code=400, detail="Brand logo not found. Logo is required for model training")
        
        # Get latest snapshots to use for training
        cursor.execute(
            """
            SELECT ws.id, ws.html_path, ws.text_content, ws.screenshot_path
            FROM website_snapshots ws
            JOIN websites w ON ws.website_id = w.id
            WHERE w.is_flagged = TRUE
            ORDER BY ws.created_at DESC
            LIMIT 5
            """
        )
        snapshots = cursor.fetchall()
        
        # Convert snapshots to dictionaries
        snapshot_dicts = []
        for snapshot in snapshots:
            snapshot_dicts.append({
                "id": snapshot[0],
                "html_path": snapshot[1],
                "text_content": snapshot[2],
                "screenshot_path": snapshot[3]
            })
        
        # Create a project if it doesn't exist
        project_arn = await rekognition_service.get_or_create_custom_label_project(
            request.brand_id, 
            brand[1]
        )
        
        # Check if a model is already training (unless force retrain is set)
        if not request.force_retrain:
            cursor.execute(
                """
                SELECT id FROM brand_custom_models
                WHERE brand_id = %s AND status IN ('TRAINING', 'TRAINING_IN_PROGRESS', 'CREATING_ANNOTATIONS', 'PROCESSING_IMAGES')
                """,
                (request.brand_id,)
            )
            if cursor.fetchone():
                return {
                    "status": "already_training",
                    "message": "A model is already being trained for this brand"
                }
        
        # Create a new model record
        cursor.execute(
            """
            INSERT INTO brand_custom_models
            (brand_id, project_arn, status, status_message)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                request.brand_id,
                project_arn,
                "TRAINING_IN_PROGRESS",
                "Model training has been initiated"
            )
        )
        model_id = cursor.fetchone()[0]
        conn.commit()
        
        # Start training as a Celery task
        task = create_and_train_custom_dataset.delay(
            request.brand_id,
            logo[0],
            snapshot_dicts,
            model_id
        )
        
        # Update with task ID
        cursor.execute(
            """
            UPDATE brand_custom_models
            SET task_id = %s
            WHERE id = %s
            """,
            (task.id, model_id)
        )
        conn.commit()
        
        return {
            "status": "training_started",
            "model_id": model_id,
            "task_id": task.id,
            "project_arn": project_arn,
            "message": "Custom model training has been started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting model training: {str(e)}")

@router.get("/custom-model/status/{brand_id}")
async def get_custom_model_status(
    brand_id: str,
    conn = Depends(get_db_connection),
    rekognition_service: RekognitionService = Depends(get_rekognition_service)
):
    """
    Get the status of a brand's custom model
    """
    try:
        cursor = conn.cursor()
        
        # Check if brand exists
        cursor.execute(
            """
            SELECT id FROM brands WHERE id = %s
            """,
            (brand_id,)
        )
        
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Brand with ID {brand_id} not found")
        
        # Get custom model info
        cursor.execute(
            """
            SELECT id, project_arn, model_version_arn, status, status_message, 
                   task_id, created_at, updated_at
            FROM brand_custom_models
            WHERE brand_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (brand_id,)
        )
        model = cursor.fetchone()
        
        if not model:
            return {
                "status": "no_model",
                "brand_id": brand_id,
                "message": "No custom model found for this brand"
            }
        
        # Schedule a task to check the current status from AWS
        if model[2]:  # If model_version_arn exists
            check_model_status.delay(model[0])
        
        return {
            "model_id": model[0],
            "brand_id": brand_id,
            "project_arn": model[1],
            "model_version_arn": model[2],
            "status": model[3],
            "status_message": model[4],
            "task_id": model[5],
            "created_at": model[6].isoformat() if model[6] else None,
            "updated_at": model[7].isoformat() if model[7] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

@router.post("/custom-model/retrain/{brand_id}")
async def retrain_custom_model(
    brand_id: str,
    conn = Depends(get_db_connection),
    rekognition_service: RekognitionService = Depends(get_rekognition_service)
):
    """
    Manually retrain a brand's custom model
    """
    # This is essentially the same as train_custom_model but with force_retrain=True
    try:
        request = ModelTrainingRequest(brand_id=brand_id, force_retrain=True)
        return await train_custom_model(request, conn, rekognition_service)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")

@router.post("/custom-model/start/{brand_id}")
async def start_custom_model_endpoint(
    brand_id: str,
    min_inference_units: int = 1,
    conn = Depends(get_db_connection),
    rekognition_service: RekognitionService = Depends(get_rekognition_service)
):
    """
    Start a trained custom model for inference
    """
    try:
        cursor = conn.cursor()
        
        # Get model info
        cursor.execute(
            """
            SELECT id, model_version_arn, status 
            FROM brand_custom_models
            WHERE brand_id = %s AND status = 'TRAINING_COMPLETED'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (brand_id,)
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(
                status_code=404, 
                detail="No trained model found for this brand. Train a model first before starting it."
            )
        
        # Start the model using Celery task
        task = start_custom_model.delay(brand_id, model[0], min_inference_units)
        
        # Update model record with task ID
        cursor.execute(
            """
            UPDATE brand_custom_models
            SET task_id = %s
            WHERE id = %s
            """,
            (task.id, model[0])
        )
        conn.commit()
        
        return {
            "status": "starting",
            "model_id": model[0],
            "brand_id": brand_id,
            "task_id": task.id,
            "model_version_arn": model[1],
            "message": "Model is starting"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting model: {str(e)}")

@router.post("/custom-model/stop/{brand_id}")
async def stop_custom_model_endpoint(
    brand_id: str,
    conn = Depends(get_db_connection),
    rekognition_service: RekognitionService = Depends(get_rekognition_service)
):
    """
    Stop a running custom model
    """
    try:
        cursor = conn.cursor()
        
        # Get model info
        cursor.execute(
            """
            SELECT id, model_version_arn, status 
            FROM brand_custom_models
            WHERE brand_id = %s AND status = 'RUNNING'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (brand_id,)
        )
        model = cursor.fetchone()
        
        if not model:
            raise HTTPException(
                status_code=404, 
                detail="No running model found for this brand."
            )
        
        # Stop the model using Celery task
        task = stop_custom_model.delay(brand_id, model[0])
        
        # Update model record with task ID
        cursor.execute(
            """
            UPDATE brand_custom_models
            SET task_id = %s
            WHERE id = %s
            """,
            (task.id, model[0])
        )
        conn.commit()
        
        return {
            "status": "stopping",
            "model_id": model[0],
            "brand_id": brand_id,
            "task_id": task.id,
            "message": "Model is stopping"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping model: {str(e)}")

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a Celery task
    """
    try:
        # This will check the task status in any of our task modules
        from celery.result import AsyncResult
        task_result = AsyncResult(task_id)
        
        if task_result.ready():
            if task_result.successful():
                return {
                    "status": "completed",
                    "result": task_result.get(),
                    "task_id": task_id
                }
            else:
                return {
                    "status": "failed",
                    "error": str(task_result.result),
                    "task_id": task_id
                }
        else:
            return {
                "status": "pending",
                "task_id": task_id,
                "message": "Task is still running"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking task status: {str(e)}")
