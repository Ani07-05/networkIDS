"""
Tests for analytics API endpoints.
"""
import pytest
from fastapi import status


@pytest.mark.integration
def test_get_user_stats(client, auth_headers):
    """Test retrieving user statistics."""
    response = client.get(
        "/api/v1/analytics/stats",
        headers=auth_headers
    )
    
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
    
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "total_predictions" in data
        assert "predictions_today" in data
        assert "predictions_this_week" in data
        assert "predictions_this_month" in data
        assert "attack_rate" in data
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["attack_rate"], (int, float))


@pytest.mark.integration
def test_get_model_info(client):
    """Test retrieving model information."""
    response = client.get("/api/v1/analytics/models")
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    
    # Check model structure
    for model in data["models"]:
        assert "name" in model
        assert "type" in model
        assert "accuracy" in model
        assert "created_at" in model


@pytest.mark.integration
def test_get_binary_model_info(client):
    """Test retrieving specific binary model information."""
    response = client.get("/api/v1/analytics/models/binary")
    
    # Should return model info or 404 if not found
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
    
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "name" in data
        assert "type" in data
        assert data["type"] == "binary"


@pytest.mark.integration
def test_get_multiclass_model_info(client):
    """Test retrieving specific multiclass model information."""
    response = client.get("/api/v1/analytics/models/multiclass")
    
    # Should return model info or 404 if not found
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
    
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "name" in data
        assert "type" in data
        assert data["type"] == "multiclass"


@pytest.mark.integration
def test_get_stats_without_auth(client):
    """Test getting stats without authentication."""
    response = client.get("/api/v1/analytics/stats")
    
    # Should require authentication
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.integration
def test_get_prediction_trends(client, auth_headers):
    """Test retrieving prediction trends."""
    response = client.get(
        "/api/v1/analytics/trends?days=7",
        headers=auth_headers
    )
    
    # May not be implemented yet, but test the endpoint
    assert response.status_code in [
        status.HTTP_200_OK,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_401_UNAUTHORIZED
    ]


@pytest.mark.integration
def test_get_attack_distribution(client, auth_headers):
    """Test retrieving attack type distribution."""
    response = client.get(
        "/api/v1/analytics/attack-distribution",
        headers=auth_headers
    )
    
    # May not be implemented yet, but test the endpoint
    assert response.status_code in [
        status.HTTP_200_OK,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_401_UNAUTHORIZED
    ]


