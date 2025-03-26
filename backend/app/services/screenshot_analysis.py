import os
import json
import uuid
from typing import Dict, List, Any, Optional

class ScreenshotAnalysisService:
    """
    Service for analyzing screenshots for brand elements
    """
    
    def __init__(self, rekognition_service):
        self.rekognition_service = rekognition_service
        self.storage_path = os.getenv("STORAGE_PATH", "./storage")
        self.evidence_path = os.path.join(self.storage_path, "evidence")
        os.makedirs(self.evidence_path, exist_ok=True)
    
    async def analyze_screenshots(self, brand_info: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze screenshots for brand elements using AWS Rekognition
        """
        if not snapshot.get("screenshot_path") or not os.path.exists(snapshot["screenshot_path"]):
            return None
        
        # Get client's logo path
        logo_path = brand_info["logo"]["file_path"]
        if not logo_path or not os.path.exists(logo_path):
            return None
        
        # Use Rekognition service to analyze the screenshot
        return await self.rekognition_service.analyze_image_with_rekognition(
            brand_info, 
            snapshot["screenshot_path"],
            self.evidence_path
        )
