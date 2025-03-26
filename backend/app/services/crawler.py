import os
import httpx
import json
import uuid
import base64
from typing import Dict, List, Any, Optional
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

class FirecrawlService:
    """
    Service for website crawling and data extraction using Firecrawl
    """
    
    def __init__(self, db_connection, base_url="http://localhost:3000"):
        self.db_connection = db_connection
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.storage_path = os.getenv("STORAGE_PATH", "./storage")
    
    async def crawl_website(self, url: str, max_pages: int = 50) -> Dict[str, Any]:
        """
        Crawl a website using local Firecrawl instance
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/tools/firecrawl_crawl",
                json={
                    "arguments": {
                        "url": url,
                        "maxDepth": 2,
                        "maxPages": max_pages,
                        "includeScreenshots": True,
                        "preserveImages": True,
                        "includeScripts": True  # Enable script collection
                    }
                }
            )
            
            result = response.json()
            
            # Parse domain from URL
            domain = urlparse(url).netloc
            
            # Store website in database
            website_id = await self._store_website(url, domain)
            
            # Process crawl results
            pages_data = []
            
            for page in result.get('pages', []):
                page_url = page.get('url')
                html_content = page.get('html')
                text_content = page.get('markdown')
                screenshot = page.get('screenshot')
                scripts = page.get('scripts', [])  # Get scripts from page
                
                # Save data
                snapshot_id = str(uuid.uuid4())
                
                # Save HTML content
                html_path = os.path.join(
                    self.storage_path, 
                    "snapshots", 
                    f"{snapshot_id}_html.html"
                )
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Save screenshot if available
                screenshot_path = None
                if screenshot:
                    screenshot_path = os.path.join(
                        self.storage_path, 
                        "snapshots", 
                        f"{snapshot_id}_screenshot.png"
                    )
                    
                    # Decode base64 screenshot
                    screenshot_data = base64.b64decode(screenshot.split(',')[1])
                    
                    with open(screenshot_path, 'wb') as f:
                        f.write(screenshot_data)
                
                # Store snapshot in database
                snapshot_data = await self._store_snapshot(
                    website_id, 
                    html_path, 
                    text_content, 
                    screenshot_path
                )
                
                # Process images
                images = page.get('images', [])
                for image in images:
                    image_url = image.get('url')
                    image_data = image.get('data')
                    
                    if image_url and image_data:
                        # Save image
                        parsed_url = urlparse(image_url)
                        file_name = os.path.basename(parsed_url.path)
                        
                        image_path = os.path.join(
                            self.storage_path, 
                            "assets", 
                            f"{snapshot_id}_{file_name}"
                        )
                        
                        # Decode base64 image
                        if image_data.startswith('data:'):
                            image_binary = base64.b64decode(image_data.split(',')[1])
                        else:
                            image_binary = base64.b64decode(image_data)
                        
                        with open(image_path, 'wb') as f:
                            f.write(image_binary)
                        
                        # Store image in database
                        await self._store_website_asset(
                            website_id,
                            snapshot_data["id"],
                            "image",
                            image_url,
                            image_path
                        )
                
                # Process scripts
                for script in scripts:
                    script_url = script.get('url')
                    script_content = script.get('content')
                    script_type = script.get('type', 'javascript')  # Default to javascript
                    
                    if script_url and script_content:
                        # Determine file extension based on type
                        ext = '.ts' if script_type == 'typescript' else '.js'
                        
                        # Create filename from URL or generate one
                        parsed_url = urlparse(script_url)
                        file_name = os.path.basename(parsed_url.path)
                        if not file_name or not file_name.endswith((ext, '.js', '.ts')):
                            file_name = f"script_{str(uuid.uuid4())[:8]}{ext}"
                        
                        # Save script file
                        script_path = os.path.join(
                            self.storage_path,
                            "assets",
                            f"{snapshot_id}_{file_name}"
                        )
                        
                        with open(script_path, 'w', encoding='utf-8') as f:
                            f.write(script_content)
                        
                        # Store script in database
                        await self._store_website_asset(
                            website_id,
                            snapshot_data["id"],
                            script_type,  # 'javascript' or 'typescript'
                            script_url,
                            script_path
                        )
                
                pages_data.append({
                    "url": page_url,
                    "snapshot_id": snapshot_data["id"],
                    "html_path": html_path,
                    "text_content": text_content[:100] + "..." if text_content else None,
                    "screenshot_path": screenshot_path,
                    "images_count": len(images),
                    "scripts_count": len(scripts)
                })
            
            return {
                "website_id": website_id,
                "url": url,
                "domain": domain,
                "pages_count": len(pages_data),
                "pages": pages_data
            }
            
        except Exception as e:
            print(f"Error crawling website {url}: {e}")
            raise
    
    async def scrape_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single page with Firecrawl
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/tools/firecrawl_scrape",
                json={
                    "arguments": {
                        "url": url,
                        "formats": ["html", "markdown"],
                        "screenshot": True,
                        "includeImages": True
                    }
                }
            )
            
            result = response.json()
            
            # Parse domain from URL
            domain = urlparse(url).netloc
            
            # Store website in database
            website_id = await self._store_website(url, domain)
            
            # Process scrape results
            html_content = result.get('html')
            text_content = result.get('markdown')
            screenshot = result.get('screenshot')
            title = result.get('title')
            
            # Save data
            snapshot_id = str(uuid.uuid4())
            
            # Save HTML content
            html_path = os.path.join(
                self.storage_path, 
                "snapshots", 
                f"{snapshot_id}_html.html"
            )
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save screenshot if available
            screenshot_path = None
            if screenshot:
                screenshot_path = os.path.join(
                    self.storage_path, 
                    "snapshots", 
                    f"{snapshot_id}_screenshot.png"
                )
                
                # Decode base64 screenshot
                screenshot_data = base64.b64decode(screenshot.split(',')[1])
                
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_data)
            
            # Store snapshot in database
            snapshot_data = await self._store_snapshot(
                website_id, 
                html_path, 
                text_content, 
                screenshot_path
            )
            
            # Process images
            images = result.get('images', [])
            for image in images:
                image_url = image.get('url')
                image_data = image.get('data')
                
                if image_url and image_data:
                    # Save image
                    parsed_url = urlparse(image_url)
                    file_name = os.path.basename(parsed_url.path)
                    
                    image_path = os.path.join(
                        self.storage_path, 
                        "assets", 
                        f"{snapshot_id}_{file_name}"
                    )
                    
                    # Decode base64 image
                    if image_data.startswith('data:'):
                        image_binary = base64.b64decode(image_data.split(',')[1])
                    else:
                        image_binary = base64.b64decode(image_data)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_binary)
                    
                    # Store image in database
                    await self._store_website_asset(
                        website_id,
                        snapshot_data["id"],
                        "image",
                        image_url,
                        image_path
                    )
            
            return {
                "website_id": website_id,
                "snapshot_id": snapshot_data["id"],
                "url": url,
                "domain": domain,
                "title": title,
                "html_path": html_path,
                "text_content": text_content[:100] + "..." if text_content else None,
                "screenshot_path": screenshot_path,
                "images_count": len(images)
            }
            
        except Exception as e:
            print(f"Error scraping page {url}: {e}")
            raise
    
    async def _store_website(self, url: str, domain: str) -> str:
        """
        Store website information in database
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Check if website already exists
            cursor.execute(
                """
                SELECT id FROM websites WHERE url = %s
                """,
                (url,)
            )
            
            existing = cursor.fetchone()
            
            if existing:
                # Update last checked timestamp
                cursor.execute(
                    """
                    UPDATE websites 
                    SET last_checked_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id
                    """,
                    (existing["id"],)
                )
                
                website_id = cursor.fetchone()["id"]
            else:
                # Insert new website
                cursor.execute(
                    """
                    INSERT INTO websites (id, url, domain)
                    VALUES (uuid_generate_v4(), %s, %s)
                    RETURNING id
                    """,
                    (url, domain)
                )
                
                website_id = cursor.fetchone()["id"]
            
            self.db_connection.commit()
            return website_id
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error storing website data: {e}")
            raise
    
    async def _store_snapshot(
        self, 
        website_id: str, 
        html_path: str, 
        text_content: str, 
        screenshot_path: str
    ) -> Dict[str, Any]:
        """
        Store website snapshot in database
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute(
                """
                INSERT INTO website_snapshots 
                (id, website_id, html_path, text_content, screenshot_path)
                VALUES (uuid_generate_v4(), %s, %s, %s, %s)
                RETURNING id
                """,
                (website_id, html_path, text_content, screenshot_path)
            )
            
            snapshot_id = cursor.fetchone()["id"]
            
            self.db_connection.commit()
            return {
                "id": snapshot_id,
                "website_id": website_id,
                "html_path": html_path,
                "text_content": text_content[:100] + "..." if text_content else None,
                "screenshot_path": screenshot_path
            }
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error storing snapshot data: {e}")
            raise
    
    async def _store_website_asset(
        self, 
        website_id: str, 
        snapshot_id: str, 
        asset_type: str, 
        url: str, 
        file_path: str
    ) -> str:
        """
        Store website asset in database
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute(
                """
                INSERT INTO website_assets 
                (id, website_id, snapshot_id, asset_type, url, file_path)
                VALUES (uuid_generate_v4(), %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (website_id, snapshot_id, asset_type, url, file_path)
            )
            
            asset_id = cursor.fetchone()["id"]
            
            self.db_connection.commit()
            return asset_id
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error storing asset data: {e}")
            raise
