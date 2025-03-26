import pytest
from unittest.mock import patch, MagicMock

from app.tasks.analysis_tasks import (
    analyze_website_for_brand,
    search_similar_content
)

@patch('app.tasks.analysis_tasks.get_db')
@patch('app.tasks.analysis_tasks.EmbeddingService')
@patch('app.tasks.analysis_tasks.AnalysisService')
def test_analyze_website_for_brand(mock_analysis_service_class, mock_embedding_service_class, mock_get_db):
    """Test analyze_website_for_brand task"""
    # Set up mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Set up embedding service mock
    mock_embedding_service = MagicMock()
    mock_embedding_service_class.return_value = mock_embedding_service
    
    # Set up analysis service mock
    mock_analysis_service = MagicMock()
    mock_analysis_service_class.return_value = mock_analysis_service
    mock_analysis_service.analyze_website_for_brand.return_value = {
        "similarity_score": 0.8,
        "text_similarity": 0.7,
        "html_similarity": 0.6,
        "image_similarity": 0.9
    }
    
    # Call the task
    result = analyze_website_for_brand("brand_id", "website_id")
    
    # Assert expected behavior
    assert mock_cursor.execute.call_count >= 2  # At least 2 DB queries (update status before and after)
    assert mock_analysis_service.analyze_website_for_brand.called
    assert "status" in result
    assert result["status"] == "success"
    assert "similarity_score" in result["result"]
    
    # Test error handling
    mock_analysis_service.analyze_website_for_brand.side_effect = Exception("Test error")
    result = analyze_website_for_brand("brand_id", "website_id")
    assert "error" in result["status"]
    
    # Verify that status is updated in case of error
    assert "error" in [call[0][1].lower() for call in mock_cursor.execute.call_args_list if len(call[0]) > 1]

@patch('app.tasks.analysis_tasks.EmbeddingService')
def test_search_similar_content_text(mock_embedding_service_class):
    """Test search_similar_content task for text"""
    # Set up embedding service mock
    mock_embedding_service = MagicMock()
    mock_embedding_service_class.return_value = mock_embedding_service
    mock_embedding_service.search_similar_text.return_value = [
        {"document": "Sample text", "score": 0.9},
        {"document": "Another sample", "score": 0.8}
    ]
    
    # Call the task with text search
    result = search_similar_content("text", {"text": "query text"}, 10)
    
    # Assert expected behavior
    assert mock_embedding_service.search_similar_text.called
    assert "status" in result
    assert result["status"] == "success"
    assert "results" in result
    assert len(result["results"]) == 2
    
    # Test error handling for missing text
    result = search_similar_content("text", {}, 10)
    assert "error" in result["status"]
    
    # Test error in search
    mock_embedding_service.search_similar_text.side_effect = Exception("Test error")
    result = search_similar_content("text", {"text": "query text"}, 10)
    assert "error" in result["status"]

@patch('app.tasks.analysis_tasks.EmbeddingService')
def test_search_similar_content_image(mock_embedding_service_class):
    """Test search_similar_content task for image"""
    # Set up embedding service mock
    mock_embedding_service = MagicMock()
    mock_embedding_service_class.return_value = mock_embedding_service
    mock_embedding_service.search_similar_images.return_value = [
        {"file_path": "/path/to/image1.jpg", "score": 0.9},
        {"file_path": "/path/to/image2.jpg", "score": 0.8}
    ]
    
    # Call the task with image search
    result = search_similar_content("image", {"image_path": "/path/to/query.jpg"}, 10)
    
    # Assert expected behavior
    assert mock_embedding_service.search_similar_images.called
    assert "status" in result
    assert result["status"] == "success"
    assert "results" in result
    assert len(result["results"]) == 2
    
    # Test error handling for missing image path
    result = search_similar_content("image", {}, 10)
    assert "error" in result["status"]
    
    # Test invalid content type
    result = search_similar_content("invalid", {}, 10)
    assert "error" in result["status"] 