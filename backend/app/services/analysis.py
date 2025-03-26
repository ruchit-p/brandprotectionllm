import os
import json
import uuid
import subprocess
import shutil
import tempfile
import base64
from typing import Dict, List, Any, Optional
import asyncio
import anthropic
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import BackgroundTasks
from PIL import Image
import numpy as np
from app.services.embedding import EmbeddingService
from app.services.rekognition import RekognitionService
from app.services.screenshot_analysis import ScreenshotAnalysisService
from dotenv import load_dotenv

load_dotenv()

class AnalysisService:
    """
    Service for analyzing websites for brand infringement
    """
    
    def __init__(self, db_connection, embedding_service: EmbeddingService):
        self.db_connection = db_connection
        self.embedding_service = embedding_service
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize service components
        self.rekognition_service = RekognitionService()
        self.screenshot_analysis_service = ScreenshotAnalysisService(self.rekognition_service)
        
        # Set path to JPlag JAR file (this would need to be downloaded)
        self.jplag_jar_path = os.getenv("JPLAG_JAR_PATH", "./lib/jplag.jar")
        
        # Set storage path
        self.storage_path = os.getenv("STORAGE_PATH", "./storage")
        self.evidence_path = os.path.join(self.storage_path, "evidence")
        self.custom_datasets_path = os.path.join(self.storage_path, "custom_datasets")
        os.makedirs(self.evidence_path, exist_ok=True)
        os.makedirs(self.custom_datasets_path, exist_ok=True)
    
    async def analyze_website_for_brand(self, brand_id: str, website_id: str) -> Dict[str, Any]:
        """
        Run all analysis methods on a website for a specific brand
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        # Get brand information
        cursor.execute(
            """
            SELECT b.id, b.name, b.website_url, b.description 
            FROM brands b WHERE b.id = %s
            """,
            (brand_id,)
        )
        brand = cursor.fetchone()
        
        if not brand:
            raise ValueError(f"Brand with ID {brand_id} not found")
        
        # Get brand keywords
        cursor.execute(
            """
            SELECT keyword FROM brand_keywords WHERE brand_id = %s
            """,
            (brand_id,)
        )
        keywords = [row["keyword"] for row in cursor.fetchall()]
        
        # Get brand social media
        cursor.execute(
            """
            SELECT platform, handle, url FROM brand_social_media WHERE brand_id = %s
            """,
            (brand_id,)
        )
        social_media = [dict(row) for row in cursor.fetchall()]
        
        # Get brand logo
        cursor.execute(
            """
            SELECT id, file_path, vector_id FROM brand_assets 
            WHERE brand_id = %s AND asset_type = 'logo'
            LIMIT 1
            """,
            (brand_id,)
        )
        logo = cursor.fetchone()
        
        # Complete brand information
        brand_info = {
            **brand,
            "keywords": keywords,
            "social_media": social_media,
            "logo": logo
        }
        
        # Get website information
        cursor.execute(
            """
            SELECT id, url, domain, title 
            FROM websites WHERE id = %s
            """,
            (website_id,)
        )
        website = cursor.fetchone()
        
        if not website:
            raise ValueError(f"Website with ID {website_id} not found")
        
        # Get the latest snapshot
        cursor.execute(
            """
            SELECT id, html_path, text_content, screenshot_path, 
                   html_vector_id, text_vector_id, screenshot_vector_id
            FROM website_snapshots
            WHERE website_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (website_id,)
        )
        snapshot = cursor.fetchone()
        
        if not snapshot:
            raise ValueError(f"No snapshot found for website {website_id}")
        
        # Get website assets (images)
        cursor.execute(
            """
            SELECT id, asset_type, url, file_path, vector_id
            FROM website_assets
            WHERE website_id = %s AND snapshot_id = %s
            """,
            (website_id, snapshot["id"])
        )
        assets = [dict(row) for row in cursor.fetchall()]
        
        # Check if we need to create a custom dataset for this brand
        if not await self.rekognition_service.has_custom_label_model(brand_id):
            try:
                # Get project ARN
                project_arn = await self.rekognition_service.get_or_create_custom_label_project(brand_id, brand_info['name'])
                
                # Create custom dataset asynchronously (don't wait for completion)
                background_tasks = BackgroundTasks()
                background_tasks.add_task(
                    self.rekognition_service.create_and_train_custom_dataset,
                    brand_id,
                    brand_info['logo']['file_path'],
                    [snapshot],
                    self.anthropic_client,
                    self.db_connection
                )
            except Exception as e:
                print(f"Error initiating custom dataset creation: {e}")
        
        # Run analysis methods in parallel
        results = await asyncio.gather(
            self.analyze_text_similarity(brand_info, website, snapshot),
            self.analyze_html_similarity(brand_info, website, snapshot),
            self.analyze_image_similarity(brand_info, website, snapshot, assets),
            self.analyze_code_similarity(brand_info, website, snapshot),
            self.screenshot_analysis_service.analyze_screenshots(brand_info, snapshot),
            return_exceptions=True
        )
        
        # Process results and store detections
        detections = []
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Analysis error: {str(result)}")
                continue
            
            if result and result.get('confidence', 0) > 0.3:  # Threshold for flagging
                detection_id = await self._store_detection(
                    brand_id,
                    website_id,
                    result.get('type'),
                    result.get('confidence'),
                    result.get('evidence')
                )
                
                detections.append({
                    'id': detection_id,
                    'type': result.get('type'),
                    'confidence': result.get('confidence')
                })
        
        # If any detection was found, flag the website
        if detections:
            cursor.execute(
                """
                UPDATE websites
                SET is_flagged = TRUE
                WHERE id = %s
                """,
                (website_id,)
            )
            self.db_connection.commit()
        
        return {
            'website_id': website_id,
            'website_url': website['url'],
            'detections': detections,
            'is_flagged': len(detections) > 0
        }
    
    async def analyze_text_similarity(
        self, 
        brand_info: Dict[str, Any], 
        website: Dict[str, Any], 
        snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze text content for brand impersonation or misuse
        """
        if not snapshot.get("text_content"):
            return None
        
        # Prepare brand context
        brand_context = {
            "name": brand_info["name"],
            "website_url": brand_info["website_url"],
            "description": brand_info["description"],
            "keywords": brand_info["keywords"]
        }
        
        # Call Claude for analysis
        prompt = f"""
        You are an expert brand protection analyst. Your task is to analyze text 
        for possible brand impersonation, trademark infringement, or unauthorized use of brand assets.
        
        Brand Information:
        - Name: {brand_context['name']}
        - Website: {brand_context['website_url']}
        - Description: {brand_context['description']}
        - Key Terms: {', '.join(brand_context['keywords'])}
        
        Analyze the given text for:
        1. Mentions of the brand name or similar variations
        2. Use of product names, slogans, or descriptions similar to the brand's
        3. Attempts to impersonate the brand's voice or style
        4. Misleading claims about affiliation with the brand
        5. Any other suspicious content related to brand impersonation or infringement
        
        Provide an analysis with:
        - A confidence score (0.0 to 1.0) indicating how likely this text represents a brand infringement
        - Specific findings with examples from the text
        - Evidence that can be shown to the brand owner
        
        Format your response as a JSON object with these fields:
        - confidence_score: float between 0 and 1
        - findings: list of specific issues found
        - evidence: map of issue to specific text evidence
        - analysis: brief explanation of your assessment
        - recommendation: "flag" or "ignore" based on confidence
        """
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            temperature=0.2,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this text from {website['url']} for potential brand infringement:\n\n{snapshot['text_content']}"
                }
            ]
        )
        
        # Extract and parse JSON from response
        try:
            result_text = response.content[0].text
            # Find JSON in the response (Claude usually wraps it in ```json ... ```)
            import re
            json_match = re.search(r'```json\s+(.*?)\s+```', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(1))
            else:
                # Try to parse the entire response as JSON
                result_json = json.loads(result_text)
            
            # Prepare evidence for storage
            evidence_id = str(uuid.uuid4())
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_text.json")
            
            with open(evidence_file, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2)
            
            # Prepare highlighted text for evidence
            highlighted_text = snapshot['text_content']
            for finding, evidence in result_json.get('evidence', {}).items():
                if evidence in highlighted_text:
                    highlighted_text = highlighted_text.replace(
                        evidence, 
                        f'<mark style="background-color: #ffff00;">{evidence}</mark>'
                    )
            
            # Create evidence metadata
            evidence_metadata = {
                "analysis": result_json.get('analysis', ''),
                "findings": result_json.get('findings', []),
                "highlighted_text": highlighted_text
            }
            
            return {
                "type": "text_similarity",
                "confidence": result_json.get('confidence_score', 0.0),
                "evidence": {
                    "type": "text",
                    "description": "Text similarity analysis",
                    "file_path": evidence_file,
                    "metadata": evidence_metadata
                }
            }
            
        except Exception as e:
            print(f"Error parsing Claude response for text analysis: {e}")
            # Return a minimal result if parsing fails
            return {
                "type": "text_similarity",
                "confidence": 0.5,  # Moderate confidence as fallback
                "evidence": {
                    "type": "text",
                    "description": "Error in text analysis",
                    "file_path": None,
                    "metadata": {"error": str(e)}
                }
            }
    
    async def analyze_html_similarity(
        self, 
        brand_info: Dict[str, Any], 
        website: Dict[str, Any], 
        snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze HTML structure for similarity with brand website
        """
        if not snapshot.get("html_path") or not os.path.exists(snapshot["html_path"]):
            return None
        
        # Get brand website snapshot
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT ws.id, ws.html_path, ws.screenshot_path
            FROM website_snapshots ws
            JOIN websites w ON ws.website_id = w.id
            WHERE w.url = %s
            ORDER BY ws.created_at DESC
            LIMIT 1
            """,
            (brand_info["website_url"],)
        )
        
        brand_snapshot = cursor.fetchone()
        
        if not brand_snapshot or not brand_snapshot.get("html_path") or not os.path.exists(brand_snapshot["html_path"]):
            # Can't compare if we don't have the brand's HTML
            return None
        
        # Create temp directory for JPlag comparison
        temp_dir = tempfile.mkdtemp(prefix="jplag_")
        try:
            # Copy HTML files to temp directories
            os.makedirs(os.path.join(temp_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "suspected"), exist_ok=True)
            
            shutil.copy(brand_snapshot["html_path"], os.path.join(temp_dir, "original", "index.html"))
            shutil.copy(snapshot["html_path"], os.path.join(temp_dir, "suspected", "index.html"))
            
            # Run JPlag comparison
            result_dir = os.path.join(temp_dir, "results")
            os.makedirs(result_dir, exist_ok=True)
            
            # Check if JPlag JAR exists
            if not os.path.exists(self.jplag_jar_path):
                # Fallback to vector similarity if JPlag is not available
                return await self._html_vector_similarity(brand_snapshot, snapshot)
            
            # Run JPlag with v6.0.0 options
            process = subprocess.run([
                "java", "-jar", self.jplag_jar_path,
                "-l", "web",  # Language option
                "-M", "RUN",  # Run mode
                "-r", os.path.join(result_dir, "results"),  # Result file
                "--normalize",  # Enable token normalization
                "-n", "-1",  # Show all comparisons
                temp_dir  # Root directory containing submissions
            ], capture_output=True, text=True)
            
            # Parse results
            similarity_score = 0.0
            
            # Look for results in the JPlag output directory
            result_zip = os.path.join(result_dir, "results.zip")
            if os.path.exists(result_zip):
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(result_zip, 'r') as zip_ref:
                    zip_ref.extractall(result_dir)
                
                # Read the index.html file
                if os.path.exists(os.path.join(result_dir, "index.html")):
                    with open(os.path.join(result_dir, "index.html"), "r") as f:
                        content = f.read()
                        import re
                        # Look for similarity score in the new format
                        match = re.search(r'Average similarity:\s*(\d+\.\d+)%', content)
                        if match:
                            similarity_score = float(match.group(1)) / 100
                        else:
                            # Try legacy format
                            match = re.search(r'Similarity:\s+(\d+\.\d+)%', content)
                            if match:
                                similarity_score = float(match.group(1)) / 100
            
            # Create evidence
            evidence_id = str(uuid.uuid4())
            
            # Copy result HTML and screenshots
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_jplag_result.html")
            if os.path.exists(os.path.join(result_dir, "index.html")):
                shutil.copy(os.path.join(result_dir, "index.html"), evidence_file)
            
            # Create evidence metadata with screenshots
            evidence_metadata = {
                "score": similarity_score,
                "raw_output": process.stdout,
                "original_screenshot_url": brand_snapshot.get("screenshot_path"),
                "suspected_screenshot_url": snapshot.get("screenshot_path"),
                "analysis": f"HTML structure similarity analysis found {similarity_score * 100:.1f}% similarity between the brand website and the suspected website."
            }
            
            # Only flag if similarity is high enough
            if similarity_score > 0.5:
                return {
                    "type": "html_similarity",
                    "confidence": similarity_score,
                    "evidence": {
                        "type": "html",
                        "description": "HTML structure similarity",
                        "file_path": evidence_file,
                        "metadata": evidence_metadata
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Error in JPlag analysis: {e}")
            # Fallback to vector similarity
            return await self._html_vector_similarity(brand_snapshot, snapshot)
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    async def _html_vector_similarity(
        self, 
        brand_snapshot: Dict[str, Any], 
        suspect_snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback method to compare HTML using vector similarity
        """
        try:
            # Read HTML files
            with open(brand_snapshot["html_path"], 'r', encoding='utf-8') as f:
                brand_html = f.read()
            
            with open(suspect_snapshot["html_path"], 'r', encoding='utf-8') as f:
                suspect_html = f.read()
            
            # Create embeddings
            brand_embedding = await self.embedding_service.create_text_embedding(brand_html)
            suspect_embedding = await self.embedding_service.create_text_embedding(suspect_html)
            
            # Calculate cosine similarity
            import numpy as np
            from numpy.linalg import norm
            
            similarity_score = np.dot(brand_embedding, suspect_embedding) / (norm(brand_embedding) * norm(suspect_embedding))
            
            # Create evidence
            evidence_id = str(uuid.uuid4())
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_html_sim.json")
            
            # Create evidence metadata with screenshots
            evidence_metadata = {
                "score": float(similarity_score),
                "original_screenshot_url": brand_snapshot.get("screenshot_path"),
                "suspected_screenshot_url": suspect_snapshot.get("screenshot_path"),
                "analysis": f"Vector-based HTML similarity analysis found {similarity_score * 100:.1f}% similarity between the brand website and the suspected website."
            }
            
            with open(evidence_file, 'w', encoding='utf-8') as f:
                json.dump(evidence_metadata, f, indent=2)
            
            # Only flag if similarity is high enough
            if similarity_score > 0.7:
                return {
                    "type": "html_similarity",
                    "confidence": float(similarity_score),
                    "evidence": {
                        "type": "html",
                        "description": "HTML structure similarity (vector-based)",
                        "file_path": evidence_file,
                        "metadata": evidence_metadata
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Error in HTML vector similarity analysis: {e}")
            return None
    
    async def analyze_image_similarity(
        self, 
        brand_info: Dict[str, Any], 
        website: Dict[str, Any], 
        snapshot: Dict[str, Any],
        assets: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze images for brand logo or asset usage
        """
        if not brand_info.get("logo") or not assets:
            return None
        
        logo_path = brand_info["logo"]["file_path"]
        if not logo_path or not os.path.exists(logo_path):
            return None
        
        # Load logo image
        with open(logo_path, 'rb') as logo_file:
            logo_bytes = logo_file.read()
        
        # Check each image for logo similarity
        best_match = None
        best_score = 0.0
        
        for asset in assets:
            if asset.get("asset_type") != "image" or not asset.get("file_path"):
                continue
            
            if not os.path.exists(asset["file_path"]):
                continue
            
            try:
                # Compare using AWS Rekognition
                with open(asset["file_path"], 'rb') as asset_file:
                    asset_bytes = asset_file.read()
                
                response = self.rekognition_service.rekognition_client.compare_faces(
                    SourceImage={'Bytes': logo_bytes},
                    TargetImage={'Bytes': asset_bytes},
                    SimilarityThreshold=0
                )
                
                # Check matches
                for match in response.get('FaceMatches', []):
                    similarity = match['Similarity'] / 100.0
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            "asset": asset,
                            "similarity": similarity,
                            "bounding_box": match['Face']['BoundingBox']
                        }
                
                # If no face matches, try detect labels
                if not best_match:
                    # Get logo labels
                    logo_response = self.rekognition_service.rekognition_client.detect_labels(
                        Image={'Bytes': logo_bytes},
                        MaxLabels=10,
                        MinConfidence=70
                    )
                    
                    logo_labels = [label['Name'].lower() for label in logo_response['Labels']]
                    
                    # Get target image labels
                    target_response = self.rekognition_service.rekognition_client.detect_labels(
                        Image={'Bytes': asset_bytes},
                        MaxLabels=10,
                        MinConfidence=70
                    )
                    
                    target_labels = [label['Name'].lower() for label in target_response['Labels']]
                    
                    # Calculate label similarity
                    common_labels = set(logo_labels) & set(target_labels)
                    if common_labels:
                        similarity = len(common_labels) / max(len(logo_labels), len(target_labels))
                        if similarity > best_score:
                            best_score = similarity
                            best_match = {
                                "asset": asset,
                                "similarity": similarity,
                                "common_labels": list(common_labels)
                            }
                
                # If still no match, try text detection (for logos with text)
                if not best_match or best_score < 0.3:
                    # Get text in logo
                    logo_text_response = self.rekognition_service.rekognition_client.detect_text(
                        Image={'Bytes': logo_bytes}
                    )
                    
                    logo_texts = [text['DetectedText'].lower() for text in logo_text_response['TextDetections']]
                    
                    # Get text in target image
                    target_text_response = self.rekognition_service.rekognition_client.detect_text(
                        Image={'Bytes': asset_bytes}
                    )
                    
                    target_texts = [text['DetectedText'].lower() for text in target_text_response['TextDetections']]
                    
                    # Check for brand name or keywords in detected text
                    brand_terms = [brand_info["name"].lower()] + [kw.lower() for kw in brand_info["keywords"]]
                    
                    for brand_term in brand_terms:
                        for target_text in target_texts:
                            if brand_term in target_text:
                                similarity = 0.8  # High confidence if brand name/term is found
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = {
                                        "asset": asset,
                                        "similarity": similarity,
                                        "detected_text": target_text,
                                        "brand_term": brand_term
                                    }
                
            except Exception as e:
                print(f"Error comparing images with Rekognition: {e}")
                # Try vector-based similarity as fallback
                try:
                    # Create embeddings using Claude
                    logo_embedding = await self.embedding_service.create_image_embedding(logo_path)
                    asset_embedding = await self.embedding_service.create_image_embedding(asset["file_path"])
                    
                    # Calculate cosine similarity
                    import numpy as np
                    from numpy.linalg import norm
                    
                    similarity = np.dot(logo_embedding, asset_embedding) / (norm(logo_embedding) * norm(asset_embedding))
                    
                    if similarity > best_score:
                        best_score = float(similarity)
                        best_match = {
                            "asset": asset,
                            "similarity": float(similarity),
                            "method": "vector_similarity"
                        }
                except Exception as inner_e:
                    print(f"Error calculating vector similarity: {inner_e}")
        
        # Create evidence if a good match was found
        if best_match and best_score > 0.5:
            evidence_id = str(uuid.uuid4())
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_image_match.json")
            
            # Create evidence metadata
            evidence_metadata = {
                "score": best_score,
                "original_image_url": logo_path,
                "detected_image_url": best_match["asset"]["file_path"],
                "analysis": f"Image analysis found {best_score * 100:.1f}% similarity with the brand logo or elements.",
                "match_details": best_match
            }
            
            with open(evidence_file, 'w', encoding='utf-8') as f:
                json.dump(evidence_metadata, f, indent=2)
            
            return {
                "type": "image_match",
                "confidence": best_score,
                "evidence": {
                    "type": "image",
                    "description": "Image similarity with brand assets",
                    "file_path": evidence_file,
                    "metadata": evidence_metadata
                }
            }
        
        return None
    
    async def analyze_code_similarity(
        self, 
        brand_info: Dict[str, Any], 
        website: Dict[str, Any], 
        snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze JavaScript/TypeScript code similarity with brand website
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        # Get brand website snapshot
        cursor.execute(
            """
            SELECT ws.id, wa.file_path, wa.asset_type
            FROM website_snapshots ws
            JOIN websites w ON ws.website_id = w.id
            JOIN website_assets wa ON wa.snapshot_id = ws.id
            WHERE w.url = %s AND wa.asset_type IN ('javascript', 'typescript')
            ORDER BY ws.created_at DESC
            """,
            (brand_info["website_url"],)
        )
        
        brand_scripts = cursor.fetchall()
        
        if not brand_scripts:
            # No scripts to compare
            return None
        
        # Get suspected website scripts
        cursor.execute(
            """
            SELECT id, file_path, asset_type
            FROM website_assets
            WHERE snapshot_id = %s AND asset_type IN ('javascript', 'typescript')
            """,
            (snapshot["id"],)
        )
        
        suspect_scripts = cursor.fetchall()
        
        if not suspect_scripts:
            # No scripts to compare
            return None
        
        # Create temp directory for JPlag comparison
        temp_dir = tempfile.mkdtemp(prefix="jplag_code_")
        try:
            # Create directories for original and suspected code
            os.makedirs(os.path.join(temp_dir, "original"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "suspected"), exist_ok=True)
            
            # Copy brand scripts to original directory
            for script in brand_scripts:
                if os.path.exists(script["file_path"]):
                    ext = '.ts' if script["asset_type"] == 'typescript' else '.js'
                    dest_path = os.path.join(temp_dir, "original", f"script_{str(uuid.uuid4())[:8]}{ext}")
                    shutil.copy(script["file_path"], dest_path)
            
            # Copy suspected scripts to suspected directory
            for script in suspect_scripts:
                if os.path.exists(script["file_path"]):
                    ext = '.ts' if script["asset_type"] == 'typescript' else '.js'
                    dest_path = os.path.join(temp_dir, "suspected", f"script_{str(uuid.uuid4())[:8]}{ext}")
                    shutil.copy(script["file_path"], dest_path)
            
            # Check if JPlag JAR exists
            if not os.path.exists(self.jplag_jar_path):
                # Fallback to vector similarity if JPlag is not available
                return await self._code_vector_similarity(brand_scripts, suspect_scripts)
            
            # Run JPlag comparison
            result_dir = os.path.join(temp_dir, "results")
            os.makedirs(result_dir, exist_ok=True)
            
            # Run JPlag with appropriate language setting
            process = subprocess.run([
                "java", "-jar", self.jplag_jar_path,
                "-l", "javascript",  # Language option for JS/TS
                "-M", "RUN",  # Run mode
                "-r", os.path.join(result_dir, "results"),  # Result file
                "--normalize",  # Enable token normalization
                "-n", "-1",  # Show all comparisons
                temp_dir  # Root directory containing submissions
            ], capture_output=True, text=True)
            
            # Parse results
            similarity_score = 0.0
            
            # Look for results in the JPlag output directory
            result_zip = os.path.join(result_dir, "results.zip")
            if os.path.exists(result_zip):
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(result_zip, 'r') as zip_ref:
                    zip_ref.extractall(result_dir)
                
                # Read the index.html file
                if os.path.exists(os.path.join(result_dir, "index.html")):
                    with open(os.path.join(result_dir, "index.html"), "r") as f:
                        content = f.read()
                        import re
                        # Look for similarity score in the new format
                        match = re.search(r'Average similarity:\s*(\d+\.\d+)%', content)
                        if match:
                            similarity_score = float(match.group(1)) / 100
                        else:
                            # Try legacy format
                            match = re.search(r'Similarity:\s+(\d+\.\d+)%', content)
                            if match:
                                similarity_score = float(match.group(1)) / 100
            
            # Create evidence
            evidence_id = str(uuid.uuid4())
            
            # Copy result HTML
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_jplag_code_result.html")
            if os.path.exists(os.path.join(result_dir, "index.html")):
                shutil.copy(os.path.join(result_dir, "index.html"), evidence_file)
            
            # Create evidence metadata
            evidence_metadata = {
                "score": similarity_score,
                "raw_output": process.stdout,
                "brand_scripts": [
                    {
                        "type": script["asset_type"],
                        "file_path": script["file_path"]
                    }
                    for script in brand_scripts
                ],
                "suspected_scripts": [
                    {
                        "type": script["asset_type"],
                        "file_path": script["file_path"]
                    }
                    for script in suspect_scripts
                ],
                "analysis": f"Code similarity analysis found {similarity_score * 100:.1f}% similarity between the brand website and the suspected website's JavaScript/TypeScript code."
            }
            
            # Store evidence
            await self._store_detection_evidence(
                evidence_id,
                "code_similarity",
                evidence_metadata["analysis"],
                evidence_file,
                evidence_metadata
            )
            
            if similarity_score > 0.5:  # Threshold for code similarity
                return {
                    "type": "code_similarity",
                    "confidence": similarity_score,
                    "evidence": {
                        "id": evidence_id,
                        "type": "code_similarity",
                        "description": evidence_metadata["analysis"],
                        "file_path": evidence_file,
                        "metadata": evidence_metadata
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing code similarity: {e}")
            return None
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def _code_vector_similarity(
        self, 
        brand_scripts: List[Dict[str, Any]], 
        suspect_scripts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Fallback method to compare code using vector similarity
        """
        try:
            # Combine all brand scripts
            brand_code = ""
            for script in brand_scripts:
                if os.path.exists(script["file_path"]):
                    with open(script["file_path"], 'r', encoding='utf-8') as f:
                        brand_code += f.read() + "\n"
            
            # Combine all suspect scripts
            suspect_code = ""
            for script in suspect_scripts:
                if os.path.exists(script["file_path"]):
                    with open(script["file_path"], 'r', encoding='utf-8') as f:
                        suspect_code += f.read() + "\n"
            
            if not brand_code or not suspect_code:
                return None
            
            # Create embeddings
            brand_embedding = await self.embedding_service.create_text_embedding(brand_code)
            suspect_embedding = await self.embedding_service.create_text_embedding(suspect_code)
            
            # Calculate cosine similarity
            import numpy as np
            from numpy.linalg import norm
            
            similarity_score = np.dot(brand_embedding, suspect_embedding) / (norm(brand_embedding) * norm(suspect_embedding))
            
            # Create evidence
            evidence_id = str(uuid.uuid4())
            evidence_file = os.path.join(self.evidence_path, f"{evidence_id}_code_sim.json")
            
            # Create evidence metadata
            evidence_metadata = {
                "score": float(similarity_score),
                "brand_scripts": [
                    {
                        "type": script["asset_type"],
                        "file_path": script["file_path"]
                    }
                    for script in brand_scripts
                ],
                "suspected_scripts": [
                    {
                        "type": script["asset_type"],
                        "file_path": script["file_path"]
                    }
                    for script in suspect_scripts
                ],
                "analysis": f"Vector-based code similarity analysis found {similarity_score * 100:.1f}% similarity between the brand website and the suspected website's JavaScript/TypeScript code."
            }
            
            with open(evidence_file, 'w', encoding='utf-8') as f:
                json.dump(evidence_metadata, f, indent=2)
            
            if similarity_score > 0.5:  # Threshold for code similarity
                return {
                    "type": "code_similarity",
                    "confidence": similarity_score,
                    "evidence": {
                        "id": evidence_id,
                        "type": "code_similarity",
                        "description": evidence_metadata["analysis"],
                        "file_path": evidence_file,
                        "metadata": evidence_metadata
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Error in code vector similarity: {e}")
            return None
    
    async def _store_detection(
        self, 
        brand_id: str, 
        website_id: str, 
        detection_type: str, 
        confidence: float,
        evidence: Dict[str, Any]
    ) -> str:
        """
        Store detection and evidence in the database
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Insert detection
            cursor.execute(
                """
                INSERT INTO detections 
                (id, brand_id, website_id, detection_type, confidence)
                VALUES (uuid_generate_v4(), %s, %s, %s, %s)
                RETURNING id
                """,
                (brand_id, website_id, detection_type, confidence)
            )
            
            detection_id = cursor.fetchone()["id"]
            
            # Insert evidence
            cursor.execute(
                """
                INSERT INTO detection_evidence
                (id, detection_id, evidence_type, description, file_path, metadata)
                VALUES (uuid_generate_v4(), %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    detection_id, 
                    evidence.get("type"), 
                    evidence.get("description"), 
                    evidence.get("file_path"),
                    json.dumps(evidence.get("metadata", {}))
                )
            )
            
            self.db_connection.commit()
            return detection_id
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error storing detection: {e}")
            raise
    
    async def get_detection_with_evidence(self, detection_id: str) -> Dict[str, Any]:
        """
        Get detection and evidence details
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            SELECT d.id, d.brand_id, d.website_id, d.detection_type, d.confidence, d.status,
                   d.created_at, w.url, w.domain
            FROM detections d
            JOIN websites w ON d.website_id = w.id
            WHERE d.id = %s
            """,
            (detection_id,)
        )
        
        detection = cursor.fetchone()
        
        if not detection:
            raise ValueError(f"Detection with ID {detection_id} not found")
        
        # Get evidence
        cursor.execute(
            """
            SELECT id, evidence_type, description, file_path, metadata
            FROM detection_evidence
            WHERE detection_id = %s
            """,
            (detection_id,)
        )
        
        evidence_items = [dict(row) for row in cursor.fetchall()]
        
        # Format evidence
        for evidence in evidence_items:
            if evidence.get("metadata"):
                evidence["metadata"] = json.loads(evidence["metadata"])
        
        detection["evidence"] = evidence_items
        
        return detection
    
    async def update_detection_status(self, detection_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a detection
        """
        cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            """
            UPDATE detections
            SET status = %s
            WHERE id = %s
            RETURNING id, detection_type, confidence, status
            """,
            (status, detection_id)
        )
        
        detection = cursor.fetchone()
        if not detection:
            raise ValueError(f"Detection with ID {detection_id} not found")
        
        self.db_connection.commit()
        return dict(detection)

    async def get_analysis_results(self, brand_id: str, website_id: str) -> Dict[str, Any]:
        """
        Get analysis results for a website and brand
        
        Args:
            brand_id: Brand ID
            website_id: Website ID
            
        Returns:
            Analysis results dictionary
        """
        cursor = self.db_connection.cursor()
        
        # Check if website exists
        cursor.execute(
            """
            SELECT id, domain, url, analysis_status, analysis_status_message, analysis_completed_at, updated_at
            FROM websites
            WHERE id = %s
            """,
            (website_id,)
        )
        website = cursor.fetchone()
        
        if not website:
            raise ValueError(f"Website with ID {website_id} not found")
        
        # Check if brand exists
        cursor.execute(
            """
            SELECT id, name
            FROM brands
            WHERE id = %s
            """,
            (brand_id,)
        )
        brand = cursor.fetchone()
        
        if not brand:
            raise ValueError(f"Brand with ID {brand_id} not found")
        
        # Get latest analysis results
        cursor.execute(
            """
            SELECT id, similarity_score, text_similarity, html_similarity, image_similarity, 
                   evidence_path, created_at
            FROM analysis_results
            WHERE brand_id = %s AND website_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (brand_id, website_id)
        )
        analysis = cursor.fetchone()
        
        # Get website snapshots
        cursor.execute(
            """
            SELECT id, screenshot_path, html_path, created_at
            FROM website_snapshots
            WHERE website_id = %s
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (website_id,)
        )
        snapshots = cursor.fetchall()
        
        # Get assets for the website
        cursor.execute(
            """
            SELECT id, snapshot_id, asset_type, file_path, similarity_score
            FROM website_assets
            WHERE snapshot_id IN (
                SELECT id FROM website_snapshots 
                WHERE website_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            )
            AND asset_type = 'image'
            """,
            (website_id,)
        )
        assets = cursor.fetchall()
        
        # Format results
        result = {
            "website": dict(website) if website else None,
            "brand": dict(brand) if brand else None,
            "analysis": dict(analysis) if analysis else None,
            "snapshots": [dict(snapshot) for snapshot in snapshots],
            "assets": [dict(asset) for asset in assets],
            "status": website["analysis_status"] if website else "UNKNOWN"
        }
        
        # Add status message based on analysis state
        if not analysis and website["analysis_status"] == "PENDING":
            result["message"] = "Analysis has not been started yet"
        elif not analysis and website["analysis_status"] == "ANALYZING":
            result["message"] = "Analysis is in progress"
        elif not analysis and website["analysis_status"] == "ERROR":
            result["message"] = f"Analysis failed: {website['analysis_status_message']}"
        elif not analysis:
            result["message"] = "No analysis results found"
        
        return result

    # Custom Model methods are now in RekognitionService
