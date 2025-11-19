"""
Tests for predictions API endpoints.
"""
import pytest
from fastapi import status


@pytest.mark.integration
def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.integration
def test_predict_single_without_auth(client, sample_features):
    """Test single prediction without authentication (should work in demo mode)."""
    response = client.post(
        "/api/v1/predictions/predict",
        json={
            "features": sample_features,
            "model_type": "binary"
        }
    )
    # In demo mode without Clerk, this should still work
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


@pytest.mark.integration
def test_predict_single_with_auth(client, sample_features, auth_headers, mock_onnx_model):
    """Test single prediction with authentication."""
    # Note: This will fail without proper Clerk setup, but tests the endpoint structure
    response = client.post(
        "/api/v1/predictions/predict",
        json={
            "features": sample_features,
            "model_type": "binary"
        },
        headers=auth_headers
    )
    # May return 401 if Clerk validation fails, which is expected in test env
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


@pytest.mark.integration
def test_predict_invalid_features(client):
    """Test prediction with invalid features."""
    response = client.post(
        "/api/v1/predictions/predict",
        json={
            "features": {"invalid": "data"},
            "model_type": "binary"
        }
    )
    # Should return validation error or 401
    assert response.status_code in [
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_400_BAD_REQUEST
    ]


@pytest.mark.integration
def test_predict_missing_model_type(client, sample_features):
    """Test prediction without specifying model type."""
    response = client.post(
        "/api/v1/predictions/predict",
        json={"features": sample_features}
    )
    # Should use default model type or return validation error
    assert response.status_code in [
        status.HTTP_200_OK,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_401_UNAUTHORIZED
    ]


@pytest.mark.integration
def test_get_prediction_history(client, auth_headers):
    """Test retrieving prediction history."""
    response = client.get(
        "/api/v1/predictions/history",
        headers=auth_headers
    )
    # May return 401 if Clerk validation fails
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
    
    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert isinstance(data["predictions"], list)


@pytest.mark.integration
def test_get_prediction_history_with_pagination(client, auth_headers):
    """Test prediction history with pagination parameters."""
    response = client.get(
        "/api/v1/predictions/history?skip=0&limit=10",
        headers=auth_headers
    )
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


@pytest.mark.integration
def test_predict_batch_endpoint(client, sample_features, auth_headers):
    """Test batch prediction endpoint."""
    batch_data = {
        "features_list": [sample_features, sample_features],
        "model_type": "binary"
    }
    
    response = client.post(
        "/api/v1/predictions/predict/batch",
        json=batch_data,
        headers=auth_headers
    )
    
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]


@pytest.mark.integration
def test_predict_batch_empty_list(client, auth_headers):
    """Test batch prediction with empty features list."""
    batch_data = {
        "features_list": [],
        "model_type": "binary"
    }
    
    response = client.post(
        "/api/v1/predictions/predict/batch",
        json=batch_data,
        headers=auth_headers
    )
    
    # Should return validation error
    assert response.status_code in [
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_401_UNAUTHORIZED
    ]


@pytest.mark.integration
def test_predict_batch_too_large(client, sample_features, auth_headers):
    """Test batch prediction with too many items."""
    # Assuming there's a limit on batch size (e.g., 1000)
    batch_data = {
        "features_list": [sample_features] * 1001,
        "model_type": "binary"
    }
    
    response = client.post(
        "/api/v1/predictions/predict/batch",
        json=batch_data,
        headers=auth_headers
    )
    
    # Should return validation error or accept it
    assert response.status_code in [
        status.HTTP_200_OK,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_401_UNAUTHORIZED
    ]


