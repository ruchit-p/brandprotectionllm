from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import uuid
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from app.services.onboarding import OnboardingService
from app.services.embedding import EmbeddingService
from app.services.crawler import FirecrawlService
from app.services.analysis import AnalysisService
from app.db import get_db_connection
from app.routes import onboarding, websites, detections, analysis
from app.config import get_settings
import logging

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Brand Protection API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(onboarding.router, prefix="/api/onboarding", tags=["onboarding"])
app.include_router(websites.router, prefix="/api/websites", tags=["websites"])
app.include_router(detections.router, prefix="/api/detections", tags=["detections"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

# Storage directories are created by the settings validator
logger.info(f"Storage path: {settings.STORAGE_PATH}")

@app.get("/")
async def read_root():
    return {
        "status": "ok", 
        "service": "Brand Protection API",
        "celery": {
            "broker": settings.CELERY_BROKER_URL,
            "backend": settings.CELERY_RESULT_BACKEND
        }
    }

@app.get("/api/health")
async def health_check(conn = Depends(get_db_connection)):
    """
    Health check endpoint to verify database connection
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        # Check Redis connection
        from redis import Redis
        redis_client = Redis.from_url(settings.CELERY_BROKER_URL)
        redis_ping = redis_client.ping()
        
        return {
            "status": "ok", 
            "db_connection": True,
            "redis_connection": redis_ping,
            "env": {
                "storage_path": settings.STORAGE_PATH,
                "aws_region": settings.AWS_REGION
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e), "db_connection": False}
