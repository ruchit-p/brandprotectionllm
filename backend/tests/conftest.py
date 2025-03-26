import os
import pytest
import psycopg2
import redis
from psycopg2.extras import RealDictCursor
from unittest.mock import MagicMock, patch

# Set test environment variables
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5432"
os.environ["DB_NAME"] = "brand_protection_test"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "postgres"
os.environ["AWS_ACCESS_KEY"] = "test_access_key"
os.environ["AWS_SECRET_KEY"] = "test_secret_key"
os.environ["AWS_REGION"] = "us-west-2"
os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/1"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/1"

# Import settings after environment variables are set
from app.config import get_settings
settings = get_settings()

@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing"""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.fetchone.return_value = {"id": "test_id", "name": "Test Name"}
    cursor.fetchall.return_value = [{"id": "test_id", "name": "Test Name"}]
    return conn

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    client = MagicMock()
    client.get.return_value = b'{"status": "SUCCESS", "result": "test result"}'
    client.keys.return_value = [b'celery-task-meta-123']
    return client

@pytest.fixture
def mock_rekognition_service():
    """Mock Rekognition service for testing"""
    service = MagicMock()
    service.describe_projects.return_value = {
        "ProjectDescriptions": [
            {"ProjectName": "test-project", "ProjectArn": "test-project-arn"}
        ]
    }
    service.describe_project_versions.return_value = {
        "ProjectVersionDescriptions": [
            {
                "ProjectVersionArn": "test-version-arn",
                "Status": "TRAINING_COMPLETED"
            }
        ]
    }
    service.create_project.return_value = {"ProjectArn": "test-project-arn"}
    service.create_project_version.return_value = {"ProjectVersionArn": "test-version-arn"}
    service.get_model_status.return_value = "RUNNING"
    return service

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Detected brand logo in image.")]
    )
    return client

@pytest.fixture
def boto3_mock():
    """Mock boto3 for testing AWS services"""
    with patch("boto3.client") as mock_client:
        # Set up mock responses for various AWS services
        rekognition_mock = MagicMock()
        rekognition_mock.describe_projects.return_value = {
            "ProjectDescriptions": [
                {"ProjectName": "test-project", "ProjectArn": "test-project-arn"}
            ]
        }
        
        s3_mock = MagicMock()
        s3_mock.list_buckets.return_value = {"Buckets": [{"Name": "test-bucket"}]}
        
        sts_mock = MagicMock()
        sts_mock.get_caller_identity.return_value = {"Account": "123456789012"}
        
        # Configure the mock client to return our service mocks
        def get_mock_service(service_name, **kwargs):
            if service_name == 'rekognition':
                return rekognition_mock
            elif service_name == 's3':
                return s3_mock
            elif service_name == 'sts':
                return sts_mock
            return MagicMock()
            
        mock_client.side_effect = get_mock_service
        yield mock_client 