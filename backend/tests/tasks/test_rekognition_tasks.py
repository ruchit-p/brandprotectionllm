import os
import pytest
from unittest.mock import patch, MagicMock

from app.tasks.rekognition_tasks import (
    create_and_train_custom_dataset,
    start_custom_model,
    stop_custom_model,
    check_model_status
)

@patch('app.tasks.rekognition_tasks.get_db')
@patch('app.tasks.rekognition_tasks.RekognitionService')
def test_check_model_status(mock_rekognition_service_class, mock_get_db):
    """Test check_model_status task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up rekognition service mock
    mock_service = MagicMock()
    mock_rekognition_service_class.return_value = mock_service
    mock_service.get_model_status.return_value = "RUNNING"
    
    # Set up mock return values
    mock_cursor.fetchone.return_value = ("model_id", "model_version_arn", "TRAINING_COMPLETED")
    
    # Call the task
    result = check_model_status("model_id")
    
    # Assert expected behavior
    assert mock_service.get_model_status.called
    assert "status" in result
    assert result["status"] == "success"
    
    # Test with no model found
    mock_cursor.fetchone.return_value = None
    result = check_model_status("nonexistent_model_id")
    assert "error" in result["status"]

@patch('app.tasks.rekognition_tasks.get_db')
@patch('app.tasks.rekognition_tasks.RekognitionService')
def test_start_custom_model(mock_rekognition_service_class, mock_get_db):
    """Test start_custom_model task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up rekognition service mock
    mock_service = MagicMock()
    mock_rekognition_service_class.return_value = mock_service
    mock_service.start_project_version.return_value = {"Status": "STARTING"}
    
    # Set up mock return values for db queries
    mock_cursor.fetchone.return_value = ("model_id", "model_version_arn", "TRAINING_COMPLETED")
    
    # Call the task
    result = start_custom_model("brand_id", "model_id", 1)
    
    # Assert expected behavior
    assert mock_service.start_project_version.called
    assert "status" in result
    assert result["status"] == "starting"
    
    # Test with no model found
    mock_cursor.fetchone.return_value = None
    result = start_custom_model("brand_id", "nonexistent_model_id", 1)
    assert "error" in result["message"]

@patch('app.tasks.rekognition_tasks.get_db')
@patch('app.tasks.rekognition_tasks.RekognitionService')
def test_stop_custom_model(mock_rekognition_service_class, mock_get_db):
    """Test stop_custom_model task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up rekognition service mock
    mock_service = MagicMock()
    mock_rekognition_service_class.return_value = mock_service
    mock_service.stop_project_version.return_value = {"Status": "STOPPING"}
    
    # Set up mock return values for db queries
    mock_cursor.fetchone.return_value = ("model_id", "model_version_arn", "RUNNING")
    
    # Call the task
    result = stop_custom_model("brand_id", "model_id")
    
    # Assert expected behavior
    assert mock_service.stop_project_version.called
    assert "status" in result
    assert result["status"] == "stopping"
    
    # Test with no model found
    mock_cursor.fetchone.return_value = None
    result = stop_custom_model("brand_id", "nonexistent_model_id")
    assert "error" in result["message"]

@patch('app.tasks.rekognition_tasks.get_db')
@patch('app.tasks.rekognition_tasks.RekognitionService')
def test_create_and_train_custom_dataset(mock_rekognition_service_class, mock_get_db):
    """Test create_and_train_custom_dataset task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up rekognition service mock
    mock_service = MagicMock()
    mock_rekognition_service_class.return_value = mock_service
    mock_service.get_or_create_custom_label_project.return_value = "project_arn"
    mock_service.create_project_version.return_value = {"ProjectVersionArn": "model_version_arn"}
    
    # Set up mock return values for db queries
    mock_cursor.fetchone.return_value = ("brand_name",)
    
    # Mock data
    brand_id = "test_brand_id"
    logo_path = "/path/to/logo.png"
    snapshots = [
        {
            "id": "snapshot_id",
            "screenshot_path": "/path/to/screenshot.png",
            "html_path": "/path/to/html.html"
        }
    ]
    model_id = "model_id"
    
    # Create test directories and files
    os.makedirs(os.path.dirname(logo_path), exist_ok=True)
    
    # Mock os.path.exists to return True for our test paths
    with patch('os.path.exists', return_value=True):
        # Call the task
        result = create_and_train_custom_dataset(brand_id, logo_path, snapshots, model_id)
    
        # Assert expected behavior
        assert mock_service.get_or_create_custom_label_project.called
        assert mock_service.create_project_version.called
        assert "status" in result
        assert result["status"] == "success"
        
        # Test error handling
        mock_service.get_or_create_custom_label_project.side_effect = Exception("Test error")
        result = create_and_train_custom_dataset(brand_id, logo_path, snapshots, model_id)
        assert "error" in result["status"] 