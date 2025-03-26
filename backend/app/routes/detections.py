from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from app.db import get_db_connection
from app.services.analysis import AnalysisService
from app.services.embedding import EmbeddingService

router = APIRouter()

def get_analysis_service(conn = Depends(get_db_connection)):
    embedding_service = EmbeddingService()
    return AnalysisService(conn, embedding_service)

@router.get("/flagged")
async def get_flagged_sites(
    conn = Depends(get_db_connection),
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
):
    """
    Get list of flagged websites with detection information
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = """
            SELECT d.id, d.brand_id, d.website_id, d.detection_type, d.confidence, d.status,
                   d.created_at, w.url, w.domain
            FROM detections d
            JOIN websites w ON d.website_id = w.id
            WHERE w.is_flagged = TRUE
        """
        
        params = []
        
        if status:
            query += " AND d.status = %s"
            params.append(status)
        
        query += " ORDER BY d.confidence DESC, d.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, skip])
        
        cursor.execute(query, params)
        
        detections = cursor.fetchall()
        
        return {"detections": detections, "count": len(detections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving flagged sites: {str(e)}")

@router.get("/{detection_id}")
async def get_detection(
    detection_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get detection details with evidence
    """
    try:
        detection = await analysis_service.get_detection_with_evidence(detection_id)
        return detection
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving detection: {str(e)}")

@router.get("/{detection_id}/evidence")
async def get_detection_evidence(
    detection_id: str,
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get evidence for a detection
    """
    try:
        detection = await analysis_service.get_detection_with_evidence(detection_id)
        
        # Organize evidence by type
        evidence_by_type = {}
        
        for evidence in detection.get("evidence", []):
            evidence_type = evidence.get("evidence_type")
            if evidence_type:
                evidence_by_type[evidence_type] = evidence
        
        return {"evidence": evidence_by_type}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving evidence: {str(e)}")

@router.put("/{detection_id}")
async def update_detection(
    detection_id: str,
    data: Dict[str, Any],
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Update detection status (reviewed, dismissed, confirmed)
    """
    if "status" not in data:
        raise HTTPException(status_code=400, detail="Status field is required")
    
    if data["status"] not in ["new", "reviewed", "dismissed", "confirmed"]:
        raise HTTPException(status_code=400, detail="Invalid status value")
    
    try:
        updated = await analysis_service.update_detection_status(detection_id, data["status"])
        return updated
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating detection: {str(e)}")

@router.get("/brand/{brand_id}")
async def get_brand_detections(
    brand_id: str,
    conn = Depends(get_db_connection),
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None
):
    """
    Get all detections for a specific brand
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = """
            SELECT d.id, d.brand_id, d.website_id, d.detection_type, d.confidence, d.status,
                   d.created_at, w.url, w.domain
            FROM detections d
            JOIN websites w ON d.website_id = w.id
            WHERE d.brand_id = %s
        """
        
        params = [brand_id]
        
        if status:
            query += " AND d.status = %s"
            params.append(status)
        
        query += " ORDER BY d.confidence DESC, d.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, skip])
        
        cursor.execute(query, params)
        
        detections = cursor.fetchall()
        
        return {"detections": detections, "count": len(detections)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving brand detections: {str(e)}")
