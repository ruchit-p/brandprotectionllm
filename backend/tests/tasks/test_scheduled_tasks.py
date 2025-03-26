import pytest
from unittest.mock import patch, MagicMock

from app.tasks.scheduled_tasks import (
    check_all_model_statuses,
    cleanup_old_task_results,
    recover_interrupted_tasks,
    update_database_stats,
    verify_aws_permissions
)

@patch('app.tasks.scheduled_tasks.get_db')
@patch('app.tasks.scheduled_tasks.RekognitionService')
def test_check_all_model_statuses(mock_rekognition_service_class, mock_get_db):
    """Test check_all_model_statuses task"""
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
    mock_cursor.fetchall.return_value = [
        ("model1", "model_version_arn1", "TRAINING"),
        ("model2", "model_version_arn2", "RUNNING")
    ]
    
    # Call the task
    result = check_all_model_statuses()
    
    # Assert expected behavior
    assert mock_service.get_model_status.call_count == 2
    assert "status" in result
    assert result["status"] == "success"
    assert result["checked_count"] == 2
    
    # Test with no models
    mock_cursor.fetchall.return_value = []
    result = check_all_model_statuses()
    assert result["status"] == "success"
    assert "No models to check" in result["message"]
    
    # Test error handling
    mock_cursor.fetchall.return_value = [("model1", "model_version_arn1", "TRAINING")]
    mock_service.get_model_status.side_effect = Exception("Test error")
    result = check_all_model_statuses()
    assert result["status"] == "success"  # Overall task succeeds even if individual updates fail
    assert result["updated_count"] == 0   # No models should be updated

@patch('app.tasks.scheduled_tasks.Redis')
def test_cleanup_old_task_results(mock_redis_class):
    """Test cleanup_old_task_results task"""
    # Set up Redis mock
    mock_redis = MagicMock()
    mock_redis_class.from_url.return_value = mock_redis
    
    # Set up mock return values
    mock_redis.keys.return_value = [b'celery-task-meta-123', b'celery-task-meta-456']
    mock_redis.get.return_value = b'{"status": "SUCCESS"}'
    
    # Call the task
    result = cleanup_old_task_results()
    
    # Assert expected behavior
    assert mock_redis.keys.called
    assert mock_redis.delete.call_count == 2
    assert "status" in result
    assert result["status"] == "success"
    assert result["total_keys"] == 2
    assert result["deleted_count"] == 2
    
    # Test with no keys
    mock_redis.keys.return_value = []
    result = cleanup_old_task_results()
    assert "No task results to clean up" in result["message"]
    
    # Test error handling
    mock_redis.keys.return_value = [b'celery-task-meta-123']
    mock_redis.get.side_effect = Exception("Test error")
    result = cleanup_old_task_results()
    assert result["status"] == "success"  # Overall task succeeds even if individual deletes fail
    assert result["deleted_count"] == 0   # No results should be deleted

@patch('app.tasks.scheduled_tasks.get_db')
def test_recover_interrupted_tasks(mock_get_db):
    """Test recover_interrupted_tasks task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up mock return values for stalled models
    mock_cursor.fetchall.side_effect = [
        # First call: stalled models
        [
            ("model1", "brand1", "task1", "TRAINING", "2023-01-01"),
            ("model2", "brand2", "task2", "STARTING", "2023-01-01")
        ],
        # Second call: stalled analyses
        [
            ("website1", "ANALYZING", "2023-01-01"),
            ("website2", "ANALYZING", "2023-01-01")
        ]
    ]
    
    # Mock AsyncResult for task status checks
    with patch('app.tasks.scheduled_tasks.AsyncResult') as mock_async_result:
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.state = 'FAILURE'
        mock_async_result.return_value = mock_result
        
        # Call the task
        result = recover_interrupted_tasks()
        
        # Assert expected behavior
        assert mock_cursor.execute.call_count >= 4  # At least 4 DB queries (2 selects, 2+ updates)
        assert "status" in result
        assert result["status"] == "success"
        assert result["stalled_models"] == 2
        assert result["stalled_analyses"] == 2
        assert result["recovered_count"] >= 2
        
        # Test with no stalled tasks
        mock_cursor.fetchall.side_effect = [[], []]
        result = recover_interrupted_tasks()
        assert result["status"] == "success"
        assert result["stalled_models"] == 0
        assert result["stalled_analyses"] == 0
        assert result["recovered_count"] == 0

@patch('app.tasks.scheduled_tasks.get_db')
def test_update_database_stats(mock_get_db):
    """Test update_database_stats task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Call the task
    result = update_database_stats()
    
    # Assert expected behavior
    assert mock_cursor.execute.call_count >= 7  # At least one ANALYZE per table
    assert "status" in result
    assert result["status"] == "success"
    assert result["tables_analyzed"] == 7
    
    # Test error handling with failed analysis
    mock_cursor.execute.side_effect = Exception("Test error")
    result = update_database_stats()
    assert result["status"] == "error"

@patch('app.tasks.scheduled_tasks.boto3')
def test_verify_aws_permissions(mock_boto3):
    """Test verify_aws_permissions task"""
    # Set up mocks
    mock_sts = MagicMock()
    mock_rekognition = MagicMock()
    mock_s3 = MagicMock()
    
    # Configure the mock client to return our service mocks
    def get_mock_service(service_name, **kwargs):
        if service_name == 'sts':
            return mock_sts
        elif service_name == 'rekognition':
            return mock_rekognition
        elif service_name == 's3':
            return mock_s3
        return MagicMock()
        
    mock_boto3.client.side_effect = get_mock_service
    
    # Set up mock return values
    mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_rekognition.describe_projects.return_value = {"ProjectDescriptions": []}
    mock_s3.list_buckets.return_value = {"Buckets": []}
    
    # Call the task
    result = verify_aws_permissions()
    
    # Assert expected behavior
    assert mock_sts.get_caller_identity.called
    assert mock_rekognition.describe_projects.called
    assert mock_s3.list_buckets.called
    assert "status" in result
    assert result["status"] == "success"
    assert result["aws_identity"] == "123456789012"
    
    # Test error handling with invalid credentials
    mock_sts.get_caller_identity.side_effect = Exception("Invalid credentials")
    result = verify_aws_permissions()
    assert result["status"] == "error"
    
    # Test bucket access error
    mock_sts.get_caller_identity.side_effect = None
    mock_s3.head_bucket.side_effect = Exception("Bucket access denied")
    result = verify_aws_permissions()
    assert result["status"] == "warning"  # Credentials valid but bucket access fails 