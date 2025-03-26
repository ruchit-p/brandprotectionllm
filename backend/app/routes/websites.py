from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

from app.db import get_db_connection
from app.services.crawler import FirecrawlService

router = APIRouter()

def get_crawler_service(conn = Depends(get_db_connection)):
    return FirecrawlService(conn)

@router.post("/crawl")
async def crawl_website(
    data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    crawler_service: FirecrawlService = Depends(get_crawler_service),
):
    """
    Crawl a website and store data for analysis
    """
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL is required")
    
    # Set up max pages
    max_pages = data.get("max_pages", 50)
    
    try:
        # Start crawling in the background
        background_tasks.add_task(crawler_service.crawl_website, data["url"], max_pages)
        
        return {
            "status": "crawling_started",
            "url": data["url"],
            "max_pages": max_pages,
            "message": "Website crawling has been started. Results will be stored for analysis."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting crawl: {str(e)}")

@router.post("/scrape")
async def scrape_page(
    data: Dict[str, Any],
    crawler_service: FirecrawlService = Depends(get_crawler_service),
):
    """
    Scrape a single page and store data for analysis
    """
    if "url" not in data:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        result = await crawler_service.scrape_page(data["url"])
        
        return {
            "status": "scraped",
            "website_id": result["website_id"],
            "snapshot_id": result["snapshot_id"],
            "url": result["url"],
            "domain": result["domain"],
            "message": "Page scraped successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping page: {str(e)}")

@router.get("/")
async def get_websites(
    conn = Depends(get_db_connection),
    skip: int = 0, 
    limit: int = 100,
    flagged: Optional[bool] = None
):
    """
    Get list of monitored websites
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        query = """
            SELECT id, url, domain, title, first_discovered_at, last_checked_at, is_flagged
            FROM websites
        """
        
        params = []
        
        if flagged is not None:
            query += " WHERE is_flagged = %s"
            params.append(flagged)
        
        query += " ORDER BY last_checked_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, skip])
        
        cursor.execute(query, params)
        
        websites = cursor.fetchall()
        
        return {"websites": websites, "count": len(websites)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving websites: {str(e)}")

@router.get("/{website_id}")
async def get_website(
    website_id: str,
    conn = Depends(get_db_connection)
):
    """
    Get website details
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get website
        cursor.execute(
            """
            SELECT id, url, domain, title, first_discovered_at, last_checked_at, is_flagged
            FROM websites
            WHERE id = %s
            """,
            (website_id,)
        )
        
        website = cursor.fetchone()
        
        if not website:
            raise HTTPException(status_code=404, detail=f"Website {website_id} not found")
        
        # Get snapshots
        cursor.execute(
            """
            SELECT id, created_at, html_path, screenshot_path
            FROM website_snapshots
            WHERE website_id = %s
            ORDER BY created_at DESC
            """,
            (website_id,)
        )
        
        snapshots = cursor.fetchall()
        
        # Get latest website assets
        if snapshots:
            latest_snapshot_id = snapshots[0]["id"]
            
            cursor.execute(
                """
                SELECT id, asset_type, url, file_path
                FROM website_assets
                WHERE website_id = %s AND snapshot_id = %s
                """,
                (website_id, latest_snapshot_id)
            )
            
            assets = cursor.fetchall()
        else:
            assets = []
        
        return {
            "website": website,
            "snapshots": snapshots,
            "assets": assets
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving website details: {str(e)}")
