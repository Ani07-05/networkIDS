"""
Tests for ML service functionality.
"""
import pytest
import numpy as np
from app.services.ml_service import MLService


@pytest.mark.unit
def test_ml_service_initialization(mock_onnx_model):
    """Test that ML service initializes correctly."""
    ml_service = MLService()
    assert ml_service is not None


@pytest.mark.unit
def test_predict_single_normal_traffic(mock_onnx_model, sample_features):
    """Test prediction for normal network traffic."""
    ml_service = MLService()
    
    # Mock the model loading
    ml_service.binary_model = mock_onnx_model()
    ml_service.multiclass_model = mock_onnx_model()
    ml_service.is_loaded = True
    
    result = ml_service.predict(sample_features, model_type="binary")
    
    assert result is not None
    assert "prediction" in result
    assert "confidence" in result
    assert result["prediction"] in ["normal", "attack"]
    assert 0 <= result["confidence"] <= 1


@pytest.mark.unit
def test_predict_batch(mock_onnx_model, sample_features):
    """Test batch prediction."""
    ml_service = MLService()
    
    # Mock the model loading
    ml_service.binary_model = mock_onnx_model()
    ml_service.multiclass_model = mock_onnx_model()
    ml_service.is_loaded = True
    
    # Create batch of features
    batch_features = [sample_features] * 5
    
    results = ml_service.predict_batch(batch_features, model_type="binary")
    
    assert len(results) == 5
    for result in results:
        assert "prediction" in result
        assert "confidence" in result
        assert result["prediction"] in ["normal", "attack"]


@pytest.mark.unit
def test_predict_multiclass(mock_onnx_model, sample_features):
    """Test multiclass prediction."""
    ml_service = MLService()
    
    # Mock the model loading
    ml_service.binary_model = mock_onnx_model()
    ml_service.multiclass_model = mock_onnx_model()
    ml_service.is_loaded = True
    
    result = ml_service.predict(sample_features, model_type="multiclass")
    
    assert result is not None
    assert "prediction" in result
    assert "confidence" in result
    # Multiclass should return attack type
    assert isinstance(result["prediction"], str)


@pytest.mark.unit
def test_predict_invalid_features(mock_onnx_model):
    """Test prediction with invalid features."""
    ml_service = MLService()
    
    # Mock the model loading
    ml_service.binary_model = mock_onnx_model()
    ml_service.multiclass_model = mock_onnx_model()
    ml_service.is_loaded = True
    
    invalid_features = {"duration": 0}  # Missing most features
    
    with pytest.raises(Exception):
        ml_service.predict(invalid_features, model_type="binary")


@pytest.mark.unit
def test_predict_invalid_model_type(mock_onnx_model, sample_features):
    """Test prediction with invalid model type."""
    ml_service = MLService()
    
    # Mock the model loading
    ml_service.binary_model = mock_onnx_model()
    ml_service.multiclass_model = mock_onnx_model()
    ml_service.is_loaded = True
    
    with pytest.raises(Exception):
        ml_service.predict(sample_features, model_type="invalid_type")


