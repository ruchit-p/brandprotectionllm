import os
import json
import uuid
import datetime
import base64
from typing import Dict, List, Any, Optional
import boto3
from PIL import Image
from dotenv import load_dotenv
from app.config import get_settings

settings = get_settings()

class RekognitionService:
    """
    Service for AWS Rekognition operations including custom label management
    """
    
    def __init__(self):
        # Initialize AWS Rekognition client
        self.rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        
        # Initialize AWS S3 client for storing custom label images
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION
        )
        
        # S3 bucket for custom label datasets
        self.custom_labels_bucket = settings.AWS_CUSTOM_LABELS_BUCKET
        
        # Storage paths
        self.storage_path = settings.STORAGE_PATH
        self.custom_datasets_path = os.path.join(self.storage_path, "custom_datasets")
        os.makedirs(self.custom_datasets_path, exist_ok=True)
    
    # Direct AWS Rekognition API access methods
    
    def describe_projects(self, max_results: int = 100, next_token: str = None) -> Dict:
        """
        Direct access to AWS Rekognition describe_projects operation
        
        Args:
            max_results: Maximum number of results to return
            next_token: Token for pagination
            
        Returns:
            Response from AWS Rekognition API
        """
        params = {'MaxResults': max_results}
        if next_token:
            params['NextToken'] = next_token
            
        try:
            return self.rekognition_client.describe_projects(**params)
        except Exception as e:
            print(f"Error describing projects: {e}")
            raise
    
    def create_project(self, project_name: str) -> Dict:
        """
        Direct access to AWS Rekognition create_project operation
        
        Args:
            project_name: Name for the new project
            
        Returns:
            Response from AWS Rekognition API with project ARN
        """
        try:
            return self.rekognition_client.create_project(ProjectName=project_name)
        except Exception as e:
            print(f"Error creating project: {e}")
            raise
    
    def describe_project_versions(
        self, 
        project_arn: str, 
        version_names: List[str] = None,
        max_results: int = 100,
        next_token: str = None
    ) -> Dict:
        """
        Direct access to AWS Rekognition describe_project_versions operation
        
        Args:
            project_arn: ARN of the project
            version_names: List of version names to describe
            max_results: Maximum number of results to return
            next_token: Token for pagination
            
        Returns:
            Response from AWS Rekognition API
        """
        params = {
            'ProjectArn': project_arn,
            'MaxResults': max_results
        }
        
        if version_names:
            params['VersionNames'] = version_names
            
        if next_token:
            params['NextToken'] = next_token
            
        try:
            return self.rekognition_client.describe_project_versions(**params)
        except Exception as e:
            print(f"Error describing project versions: {e}")
            raise
    
    def create_project_version(
        self,
        project_arn: str,
        version_name: str,
        output_config: Dict,
        training_data: Dict,
        testing_data: Dict = None,
        max_results: int = 1
    ) -> Dict:
        """
        Direct access to AWS Rekognition create_project_version operation
        
        Args:
            project_arn: ARN of the project
            version_name: Name for the version
            output_config: Output configuration
            training_data: Training data configuration
            testing_data: Testing data configuration
            max_results: Maximum number of inference units
            
        Returns:
            Response from AWS Rekognition API
        """
        params = {
            'ProjectArn': project_arn,
            'VersionName': version_name,
            'OutputConfig': output_config,
            'TrainingData': training_data,
        }
        
        if testing_data:
            params['TestingData'] = testing_data
            
        try:
            return self.rekognition_client.create_project_version(**params)
        except Exception as e:
            print(f"Error creating project version: {e}")
            raise
    
    def start_project_version(self, project_version_arn: str, min_inference_units: int = 1) -> Dict:
        """
        Direct access to AWS Rekognition start_project_version operation
        
        Args:
            project_version_arn: ARN of the project version
            min_inference_units: Minimum number of inference units
            
        Returns:
            Response from AWS Rekognition API
        """
        try:
            return self.rekognition_client.start_project_version(
                ProjectVersionArn=project_version_arn,
                MinInferenceUnits=min_inference_units
            )
        except Exception as e:
            print(f"Error starting project version: {e}")
            raise
    
    def stop_project_version(self, project_version_arn: str) -> Dict:
        """
        Direct access to AWS Rekognition stop_project_version operation
        
        Args:
            project_version_arn: ARN of the project version
            
        Returns:
            Response from AWS Rekognition API
        """
        try:
            return self.rekognition_client.stop_project_version(
                ProjectVersionArn=project_version_arn
            )
        except Exception as e:
            print(f"Error stopping project version: {e}")
            raise
    
    def detect_custom_labels(
        self, 
        project_version_arn: str, 
        image_bytes: bytes = None,
        s3_object: Dict = None,
        max_results: int = 50,
        min_confidence: float = 50.0
    ) -> Dict:
        """
        Direct access to AWS Rekognition detect_custom_labels operation
        
        Args:
            project_version_arn: ARN of the project version
            image_bytes: Image bytes
            s3_object: S3 object information
            max_results: Maximum number of results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Response from AWS Rekognition API
        """
        params = {
            'ProjectVersionArn': project_version_arn,
            'MaxResults': max_results,
            'MinConfidence': min_confidence,
            'Image': {}
        }
        
        if image_bytes:
            params['Image']['Bytes'] = image_bytes
        elif s3_object:
            params['Image']['S3Object'] = s3_object
        else:
            raise ValueError("Either image_bytes or s3_object must be provided")
            
        try:
            return self.rekognition_client.detect_custom_labels(**params)
        except Exception as e:
            print(f"Error detecting custom labels: {e}")
            raise
    
    # Helper methods that use the direct API access methods
    
    def get_custom_model_project_name(self, brand_id: str) -> str:
        """Generate a consistent project name for a brand's custom model"""
        return f"brand-protection-{brand_id[:8]}"
    
    async def get_or_create_custom_label_project(self, brand_id: str, brand_name: str) -> str:
        """Get existing project ARN or create a new custom label project"""
        project_name = self.get_custom_model_project_name(brand_id)
        
        try:
            # Check if project already exists
            response = self.describe_projects()
            for project in response.get('ProjectDescriptions', []):
                if project['ProjectName'] == project_name:
                    return project['ProjectArn']
            
            # If not found, create new project
            create_response = self.create_project(project_name)
            
            return create_response['ProjectArn']
            
        except Exception as e:
            print(f"Error creating custom label project: {e}")
            raise
    
    async def get_active_model_version(self, project_arn: str) -> Optional[str]:
        """Get ARN of active model version if available"""
        try:
            response = self.describe_project_versions(project_arn)
            
            for version in response.get('ProjectVersionDescriptions', []):
                if version['Status'] == 'RUNNING':
                    return version['ProjectVersionArn']
            
            return None
            
        except Exception as e:
            print(f"Error checking model versions: {e}")
            return None
    
    async def has_custom_label_model(self, brand_id: str) -> bool:
        """Check if a brand has an active custom label model"""
        project_name = self.get_custom_model_project_name(brand_id)
        
        try:
            # Get project ARN
            response = self.describe_projects()
            project_arn = None
            
            for project in response.get('ProjectDescriptions', []):
                if project['ProjectName'] == project_name:
                    project_arn = project['ProjectArn']
                    break
            
            if not project_arn:
                return False
            
            # Check for running model version
            active_version = await self.get_active_model_version(project_arn)
            return active_version is not None
            
        except Exception as e:
            print(f"Error checking custom label model: {e}")
            return False
    
    async def get_custom_model_arn(self, brand_id: str) -> Optional[str]:
        """Get ARN of running custom model for a brand"""
        project_name = self.get_custom_model_project_name(brand_id)
        
        try:
            # Get project ARN
            response = self.describe_projects()
            project_arn = None
            
            for project in response.get('ProjectDescriptions', []):
                if project['ProjectName'] == project_name:
                    project_arn = project['ProjectArn']
                    break
            
            if not project_arn:
                return None
            
            # Get running model version
            return await self.get_active_model_version(project_arn)
            
        except Exception as e:
            print(f"Error getting custom model ARN: {e}")
            return None
    
    async def get_model_status(self, model_version_arn: str) -> Optional[str]:
        """Get current status of a model version"""
        try:
            # Extract project ARN from model version ARN
            parts = model_version_arn.split("/version/")
            if len(parts) != 2:
                return None
                
            project_arn = parts[0]
            version_name = parts[1].split("/")[0]
            
            response = self.describe_project_versions(
                project_arn=project_arn,
                version_names=[version_name]
            )
            
            for version in response.get('ProjectVersionDescriptions', []):
                if version['ProjectVersionArn'] == model_version_arn:
                    return version['Status']
            
            return None
            
        except Exception as e:
            print(f"Error getting model status: {e}")
            return None
    
    async def start_model(self, model_version_arn: str, min_inference_units: int = 1) -> bool:
        """Start a trained model for inference"""
        try:
            self.start_project_version(
                project_version_arn=model_version_arn,
                min_inference_units=min_inference_units
            )
            return True
            
        except Exception as e:
            print(f"Error starting model: {e}")
            raise
    
    async def stop_model(self, model_version_arn: str) -> bool:
        """Stop a running model"""
        try:
            self.stop_project_version(
                project_version_arn=model_version_arn
            )
            return True
            
        except Exception as e:
            print(f"Error stopping model: {e}")
            raise
    
    async def create_and_train_custom_dataset(
        self,
        brand_id: str,
        logo_path: str,
        snapshots: List[Dict],
        anthropic_client,
        db_connection
    ) -> Optional[str]:
        """
        Create custom dataset for AWS Rekognition Custom Labels using Claude 3.5 Haiku
        and train a model for the brand
        """
        try:
            # Get brand name
            cursor = db_connection.cursor()
            cursor.execute(
                """
                SELECT name FROM brands WHERE id = %s
                """,
                (brand_id,)
            )
            brand = cursor.fetchone()
            if not brand:
                raise ValueError(f"Brand with ID {brand_id} not found")
                
            # Get project ARN
            project_arn = await self.get_or_create_custom_label_project(brand_id, brand[0])
            
            # 1. Create dataset directory
            dataset_dir = os.path.join(self.custom_datasets_path, brand_id)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 2. Collect images from snapshots
            images = []
            
            # Add logo as a reference image
            if logo_path and os.path.exists(logo_path):
                images.append((logo_path, "logo"))
            
            # Add screenshots
            for snapshot in snapshots:
                if snapshot.get("screenshot_path") and os.path.exists(snapshot["screenshot_path"]):
                    images.append((snapshot["screenshot_path"], "screenshot"))
                
                # Get website assets (images)
                cursor.execute(
                    """
                    SELECT file_path FROM website_assets
                    WHERE snapshot_id = %s AND asset_type = 'image'
                    """,
                    (snapshot["id"],)
                )
                
                for row in cursor.fetchall():
                    if row[0] and os.path.exists(row[0]):
                        images.append((row[0], "asset"))
            
            if not images:
                print(f"No images found for brand {brand_id}")
                return None
            
            # 3. Process images with Claude to create annotations
            train_manifest_entries = []
            test_manifest_entries = []
            
            # Split images 80/20 for training/testing
            train_images = images[:int(len(images) * 0.8)]
            test_images = images[int(len(images) * 0.8):]
            
            # Process training images
            for i, (image_path, image_type) in enumerate(train_images):
                try:
                    # Read image
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    
                    # Get image format
                    img_format = os.path.splitext(image_path)[1][1:].lower()
                    if img_format == "jpg":
                        img_format = "jpeg"
                    
                    # Convert to base64
                    image_b64 = base64.b64encode(image_data).decode("utf-8")
                    
                    # Get image dimensions
                    with Image.open(image_path) as img:
                        width, height = img.size
                    
                    # Use Claude to identify brand elements
                    # Only use logo image as reference if it's not the current image being processed
                    is_reference = image_type == "logo"
                    
                    # Prepare prompt based on image type
                    if is_reference:
                        system_prompt = """
                        You are an expert annotation assistant for computer vision models. 
                        This is a reference logo image. Create a bounding box annotation around the entire logo.
                        Return ONLY a JSON array with a single bounding box in the format:
                        [
                          {
                            "label": "logo", 
                            "top": float, 
                            "left": float, 
                            "width": float, 
                            "height": float
                          }
                        ]
                        
                        All values should be normalized between 0.0 and 1.0 (as a fraction of image dimensions).
                        The bounding box should encompass the entire logo tightly.
                        """
                        user_prompt = "This is the brand's logo image. Please create a bounding box annotation around the entire logo."
                    else:
                        system_prompt = """
                        You are an expert annotation assistant for computer vision models.
                        For the given image, identify instances of the brand's logo or distinctive brand elements.
                        Return ONLY a JSON array of bounding boxes in the format:
                        [
                          {
                            "label": "logo", 
                            "top": float, 
                            "left": float, 
                            "width": float, 
                            "height": float
                          }
                        ]
                        
                        All values should be normalized between 0.0 and 1.0 (as a fraction of image dimensions).
                        If no brand elements are found, return an empty array [].
                        """
                        user_prompt = f"This image is from the brand's website. Please annotate all instances of the brand's logos or visual elements if present."
                    
                    response = await anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": f"image/{img_format}",
                                            "data": image_b64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": user_prompt
                                    }
                                ]
                            }
                        ]
                    )
                    
                    # Parse Claude's response to get annotations
                    annotations_text = response.content[0].text
                    import re
                    match = re.search(r'\[.*?\]', annotations_text, re.DOTALL)
                    
                    if match:
                        annotations = json.loads(match.group(0))
                        
                        # Save image to dataset directory
                        image_filename = f"train_{i}_{uuid.uuid4()}{os.path.splitext(image_path)[1]}"
                        image_save_path = os.path.join(dataset_dir, image_filename)
                        
                        import shutil
                        shutil.copy(image_path, image_save_path)
                        
                        # Generate unique S3 key
                        s3_key = f"{brand_id}/train/{image_filename}"
                        
                        # Upload to S3
                        self.s3_client.upload_file(
                            image_save_path,
                            self.custom_labels_bucket,
                            s3_key
                        )
                        
                        # Add to manifest with annotations
                        if annotations:
                            # Create SageMaker Ground Truth format
                            manifest_entry = {
                                "source-ref": f"s3://{self.custom_labels_bucket}/{s3_key}",
                                "brand-detection": {
                                    "image_size": {
                                        "width": width,
                                        "height": height,
                                        "depth": 3
                                    },
                                    "annotations": [
                                        {
                                            "class_id": 0, 
                                            "class_name": "logo",
                                            "top": int(ann["top"] * height),
                                            "left": int(ann["left"] * width),
                                            "width": int(ann["width"] * width),
                                            "height": int(ann["height"] * height)
                                        } for ann in annotations
                                    ]
                                },
                                "brand-detection-metadata": {
                                    "objects": [
                                        {
                                            "confidence": 1.0
                                        } for _ in annotations
                                    ],
                                    "class-map": {
                                        "0": "logo"
                                    },
                                    "type": "groundtruth/object-detection",
                                    "human-annotated": "yes",
                                    "creation-date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
                                }
                            }
                            
                            train_manifest_entries.append(manifest_entry)
                except Exception as e:
                    print(f"Error processing image {image_path} for annotations: {e}")
            
            # Process test images - simpler as we don't need annotations for test set
            for i, (image_path, image_type) in enumerate(test_images):
                try:
                    # Save image to dataset directory
                    image_filename = f"test_{i}_{uuid.uuid4()}{os.path.splitext(image_path)[1]}"
                    image_save_path = os.path.join(dataset_dir, image_filename)
                    
                    import shutil
                    shutil.copy(image_path, image_save_path)
                    
                    # Generate unique S3 key
                    s3_key = f"{brand_id}/test/{image_filename}"
                    
                    # Upload to S3
                    self.s3_client.upload_file(
                        image_save_path,
                        self.custom_labels_bucket,
                        s3_key
                    )
                    
                    # Add to test manifest (no annotations needed for test set)
                    test_manifest_entries.append({
                        "source-ref": f"s3://{self.custom_labels_bucket}/{s3_key}"
                    })
                except Exception as e:
                    print(f"Error processing test image {image_path}: {e}")
            
            # 4. Create manifest files
            train_manifest_path = os.path.join(dataset_dir, "train_manifest.json")
            test_manifest_path = os.path.join(dataset_dir, "test_manifest.json")
            
            with open(train_manifest_path, "w") as f:
                for entry in train_manifest_entries:
                    f.write(json.dumps(entry) + "\n")
            
            with open(test_manifest_path, "w") as f:
                for entry in test_manifest_entries:
                    f.write(json.dumps(entry) + "\n")
            
            # 5. Upload manifest files to S3
            train_manifest_s3_key = f"{brand_id}/train_manifest.json"
            test_manifest_s3_key = f"{brand_id}/test_manifest.json"
            
            self.s3_client.upload_file(
                train_manifest_path,
                self.custom_labels_bucket,
                train_manifest_s3_key
            )
            
            self.s3_client.upload_file(
                test_manifest_path,
                self.custom_labels_bucket,
                test_manifest_s3_key
            )
            
            # 6. Create dataset in AWS Rekognition Custom Labels
            train_dataset_arn = self.rekognition_client.create_dataset(
                ProjectArn=project_arn,
                DatasetType='TRAIN',
                DatasetSource={
                    'GroundTruthManifest': {
                        'S3Object': {
                            'Bucket': self.custom_labels_bucket,
                            'Name': train_manifest_s3_key
                        }
                    }
                }
            )['DatasetArn']
            
            test_dataset_arn = self.rekognition_client.create_dataset(
                ProjectArn=project_arn,
                DatasetType='TEST',
                DatasetSource={
                    'GroundTruthManifest': {
                        'S3Object': {
                            'Bucket': self.custom_labels_bucket,
                            'Name': test_manifest_s3_key
                        }
                    }
                }
            )['DatasetArn']
            
            # 7. Start training
            version_name = f"v{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = self.rekognition_client.create_project_version(
                ProjectArn=project_arn,
                VersionName=version_name,
                OutputConfig={
                    'S3Bucket': self.custom_labels_bucket,
                    'S3KeyPrefix': f"{brand_id}/output"
                },
                TrainingData={
                    'Assets': [
                        {
                            'GroundTruthManifest': {
                                'S3Object': {
                                    'Bucket': self.custom_labels_bucket,
                                    'Name': train_manifest_s3_key
                                }
                            }
                        }
                    ]
                },
                TestingData={
                    'Assets': [
                        {
                            'GroundTruthManifest': {
                                'S3Object': {
                                    'Bucket': self.custom_labels_bucket,
                                    'Name': test_manifest_s3_key
                                }
                            }
                        }
                    ]
                }
            )
            
            model_version_arn = response['ProjectVersionArn']
            
            # 8. Store model information in database
            cursor.execute(
                """
                INSERT INTO brand_custom_models (brand_id, project_arn, model_version_arn, status, created_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (brand_id, project_arn, model_version_arn, "TRAINING", datetime.datetime.now())
            )
            
            db_connection.commit()
            
            print(f"Started training custom model for brand {brand_id}: {model_version_arn}")
            return model_version_arn
            
        except Exception as e:
            print(f"Error creating custom dataset for brand {brand_id}: {e}")
            return None

    async def analyze_image_with_rekognition(self, brand_info: Dict, image_path: str, evidence_path: str) -> Optional[Dict]:
        """
        Analyze an image using AWS Rekognition (custom labels if available, otherwise standard detection)
        """
        if not os.path.exists(image_path):
            return None
        
        try:
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
            
            # Try to use custom model if available
            model_arn = await self.get_custom_model_arn(brand_info["id"])
            if model_arn:
                response = self.detect_custom_labels(
                    project_version_arn=model_arn,
                    image_bytes=image_bytes,
                    max_results=50,
                    min_confidence=50.0
                )
                
                detections = response.get('CustomLabels', [])
                if detections:
                    # Create evidence
                    evidence_id = str(uuid.uuid4())
                    evidence_file = os.path.join(evidence_path, f"{evidence_id}_custom_detection.json")
                    
                    # Save detection data
                    evidence_data = {
                        "detections": detections,
                        "image_path": image_path,
                        "detection_method": "custom_labels"
                    }
                    
                    with open(evidence_file, 'w', encoding='utf-8') as f:
                        json.dump(evidence_data, f, indent=2)
                    
                    # Calculate overall confidence
                    max_confidence = max(d['Confidence'] for d in detections) / 100.0
                    
                    return {
                        "type": "custom_label_match",
                        "confidence": max_confidence,
                        "evidence": {
                            "type": "image",
                            "description": "Brand elements detected using custom model",
                            "file_path": evidence_file,
                            "metadata": {
                                "image_path": image_path,
                                "detections_count": len(detections),
                                "max_confidence": max_confidence,
                                "analysis": f"Custom trained model detected {len(detections)} brand elements with {max_confidence:.1%} confidence."
                            }
                        }
                    }
            
            # Fallback to standard Rekognition
            # Compare with logo
            logo_path = brand_info["logo"]["file_path"]
            if logo_path and os.path.exists(logo_path):
                with open(logo_path, 'rb') as logo_file:
                    logo_bytes = logo_file.read()
                
                # Try comparing the images
                try:
                    compare_response = self.rekognition_client.compare_faces(
                        SourceImage={'Bytes': logo_bytes},
                        TargetImage={'Bytes': image_bytes},
                        SimilarityThreshold=70.0
                    )
                    
                    matches = compare_response.get('FaceMatches', [])
                    if matches:
                        max_similarity = max(match['Similarity'] for match in matches) / 100.0
                        
                        # Create evidence
                        evidence_id = str(uuid.uuid4())
                        evidence_file = os.path.join(evidence_path, f"{evidence_id}_face_comparison.json")
                        
                        evidence_data = {
                            "matches": matches,
                            "logo_path": logo_path,
                            "image_path": image_path,
                            "detection_method": "compare_faces"
                        }
                        
                        with open(evidence_file, 'w', encoding='utf-8') as f:
                            json.dump(evidence_data, f, indent=2)
                        
                        return {
                            "type": "face_match",
                            "confidence": max_similarity,
                            "evidence": {
                                "type": "image",
                                "description": "Logo face matching in image",
                                "file_path": evidence_file,
                                "metadata": {
                                    "logo_path": logo_path,
                                    "image_path": image_path,
                                    "matches_count": len(matches),
                                    "max_similarity": max_similarity,
                                    "analysis": f"Found {len(matches)} instances matching the logo with {max_similarity:.1%} similarity."
                                }
                            }
                        }
                except Exception as e:
                    print(f"CompareFaces analysis failed: {e}")
            
            # Try label detection
            labels_response = self.rekognition_client.detect_labels(
                Image={'Bytes': image_bytes},
                MaxLabels=20,
                MinConfidence=70.0
            )
            
            # Check for brand-related labels
            found_labels = []
            brand_terms = set([word.lower() for word in brand_info["name"].split()])
            brand_terms.update([kw.lower() for kw in brand_info.get("keywords", [])])
            
            for label in labels_response.get('Labels', []):
                if label['Name'].lower() in brand_terms:
                    found_labels.append(label)
            
            if found_labels:
                # Create evidence
                evidence_id = str(uuid.uuid4())
                evidence_file = os.path.join(evidence_path, f"{evidence_id}_label_detection.json")
                
                evidence_data = {
                    "found_labels": found_labels,
                    "image_path": image_path,
                    "detection_method": "detect_labels"
                }
                
                with open(evidence_file, 'w', encoding='utf-8') as f:
                    json.dump(evidence_data, f, indent=2)
                
                max_confidence = max(label['Confidence'] for label in found_labels) / 100.0
                
                return {
                    "type": "label_match",
                    "confidence": max_confidence,
                    "evidence": {
                        "type": "image",
                        "description": "Brand-related labels in image",
                        "file_path": evidence_file,
                        "metadata": {
                            "image_path": image_path,
                            "labels_count": len(found_labels),
                            "max_confidence": max_confidence,
                            "analysis": f"Detected {len(found_labels)} brand-related elements with {max_confidence:.1%} confidence."
                        }
                    }
                }
            
            # Try text detection
            text_response = self.rekognition_client.detect_text(
                Image={'Bytes': image_bytes}
            )
            
            # Check for brand name in text
            detected_texts = text_response.get('TextDetections', [])
            brand_name_lower = brand_info["name"].lower()
            brand_terms = [brand_name_lower] + [kw.lower() for kw in brand_info.get("keywords", [])]
            
            brand_text_matches = []
            for text in detected_texts:
                text_lower = text['DetectedText'].lower()
                for term in brand_terms:
                    if term in text_lower:
                        brand_text_matches.append({
                            'text': text['DetectedText'],
                            'confidence': text['Confidence'],
                            'matching_term': term,
                            'type': text['Type']
                        })
            
            if brand_text_matches:
                # Create evidence
                evidence_id = str(uuid.uuid4())
                evidence_file = os.path.join(evidence_path, f"{evidence_id}_text_detection.json")
                
                evidence_data = {
                    "brand_text_matches": brand_text_matches,
                    "image_path": image_path,
                    "detection_method": "detect_text"
                }
                
                with open(evidence_file, 'w', encoding='utf-8') as f:
                    json.dump(evidence_data, f, indent=2)
                
                # Use highest confidence as score
                max_confidence = max(match['confidence'] for match in brand_text_matches) / 100.0
                
                return {
                    "type": "text_match",
                    "confidence": max_confidence,
                    "evidence": {
                        "type": "image",
                        "description": "Brand text detected in image",
                        "file_path": evidence_file,
                        "metadata": {
                            "image_path": image_path,
                            "matches_count": len(brand_text_matches),
                            "max_confidence": max_confidence,
                            "analysis": f"Detected {len(brand_text_matches)} instances of brand text with {max_confidence:.1%} confidence."
                        }
                    }
                }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing image with Rekognition: {e}")
            return None
