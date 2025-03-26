import logging
from celery import shared_task
import psycopg2
from app.services.analysis import AnalysisService
from app.services.embedding import EmbeddingService
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def get_db():
    """Get a database connection"""
    conn = psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )
    conn.autocommit = False
    return conn


@shared_task(bind=True, max_retries=3)
def analyze_website_for_brand(self, brand_id: str, website_id: str):
    """
    Analyze a website for a specific brand
    
    Args:
        brand_id: Brand ID
        website_id: Website ID
        
    Returns:
        Analysis results
    """
    conn = None
    try:
        conn = get_db()
        embedding_service = EmbeddingService()
        analysis_service = AnalysisService(conn, embedding_service)
        
        # Update website status to analyzing
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE websites 
            SET analysis_status = 'ANALYZING',
                analysis_status_message = 'Analysis in progress',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            (website_id,)
        )
        conn.commit()
        
        # Perform analysis
        try:
            result = analysis_service.analyze_website_for_brand(brand_id, website_id)
            
            # Update website status to analyzed
            cursor.execute(
                """
                UPDATE websites 
                SET analysis_status = 'ANALYZED',
                    analysis_status_message = 'Analysis completed',
                    updated_at = CURRENT_TIMESTAMP,
                    analysis_completed_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (website_id,)
            )
            conn.commit()
            
            return {
                "status": "success",
                "message": "Website analysis completed",
                "brand_id": brand_id,
                "website_id": website_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error analyzing website: {e}")
            
            # Update website status to error
            cursor.execute(
                """
                UPDATE websites 
                SET analysis_status = 'ERROR',
                    analysis_status_message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (f"Analysis error: {str(e)}", website_id)
            )
            conn.commit()
            
            raise self.retry(exc=e)
            
    except Exception as e:
        logger.error(f"Error in analyze_website_for_brand: {e}")
        
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE websites 
                    SET analysis_status = 'ERROR',
                        analysis_status_message = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (f"Analysis error: {str(e)}", website_id)
                )
                conn.commit()
            except Exception as db_error:
                logger.error(f"Error updating website status: {db_error}")
        
        return {
            "status": "error",
            "message": str(e),
            "brand_id": brand_id,
            "website_id": website_id
        }
    finally:
        if conn:
            conn.close()


@shared_task(bind=True)
def search_similar_content(self, content_type: str, search_data: dict, limit: int = 10):
    """
    Search for similar content in vector store
    
    Args:
        content_type: Type of content ('text' or 'image')
        search_data: Search data (text or image path)
        limit: Maximum number of results to return
        
    Returns:
        Search results
    """
    try:
        embedding_service = EmbeddingService()
        
        if content_type == "text":
            if "text" not in search_data:
                return {"status": "error", "message": "Text field is required"}
            
            results = embedding_service.search_similar_text(
                search_data["text"],
                collection_name=search_data.get("collection", "site_text"),
                limit=limit
            )
            
            return {"status": "success", "results": results, "type": "text"}
            
        elif content_type == "image":
            if "image_path" not in search_data:
                return {"status": "error", "message": "Image path is required"}
            
            results = embedding_service.search_similar_images(
                search_data["image_path"],
                collection_name=search_data.get("collection", "site_images"),
                limit=limit
            )
            
            return {"status": "success", "results": results, "type": "image"}
            
        else:
            return {"status": "error", "message": "Invalid content type. Must be 'text' or 'image'"}
            
    except Exception as e:
        logger.error(f"Error searching for similar content: {e}")
        return {"status": "error", "message": str(e)} 