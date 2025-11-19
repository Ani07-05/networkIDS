"""
Predictions API router.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import time
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.prediction import Prediction
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionHistory,
    PredictionHistoryResponse
)
from app.middleware.clerk_auth import get_current_user
from app.services.ml_service import get_ml_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def predict_single(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service = Depends(get_ml_service)
):
    """
    Make a single prediction for network traffic.
    
    - **features**: 41 NSL-KDD features
    - Returns prediction with confidence scores
    """
    try:
        # Get features dict
        features_dict = request.features.model_dump()
        
        # Make prediction
        result = ml_service.predict(features_dict)
        
        # Save to database
        prediction = Prediction(
            user_id=current_user.id,
            features=features_dict,
            is_attack=result["is_attack"],
            binary_confidence=result["binary_confidence"],
            attack_type=result["attack_type"],
            multiclass_confidence=result["multiclass_confidence"],
            multiclass_probabilities=result["multiclass_probabilities"],
            inference_time_ms=result["inference_time_ms"]
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return prediction
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service = Depends(get_ml_service)
):
    """
    Make batch predictions for multiple network traffic samples.
    
    - **predictions**: List of feature sets (max 1000)
    - Returns list of predictions with total processing time
    """
    start_time = time.time()
    
    try:
        features_list = [features.model_dump() for features in request.predictions]
        
        # Make batch predictions
        results = ml_service.predict_batch(features_list)
        
        # Prepare predictions objects
        predictions = []
        for features_dict, result in zip(features_list, results):
            prediction = Prediction(
                user_id=current_user.id,
                features=features_dict,
                is_attack=result["is_attack"],
                binary_confidence=result["binary_confidence"],
                attack_type=result["attack_type"],
                multiclass_confidence=result["multiclass_confidence"],
                multiclass_probabilities=result["multiclass_probabilities"],
                inference_time_ms=result["inference_time_ms"]
            )
            predictions.append(prediction)
        
        # Save batch to database
        db.add_all(predictions)
        db.commit()
        
        # Refresh all predictions
        for prediction in predictions:
            db.refresh(prediction)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            total=len(predictions),
            predictions=predictions,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    page: int = 1,
    page_size: int = 50,
    attack_type: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get prediction history for current user.
    
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 50, max: 100)
    - **attack_type**: Filter by attack type (optional)
    - Returns paginated prediction history
    """
    # Validate pagination
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 50
    
    # Build query
    query = db.query(Prediction).filter(Prediction.user_id == current_user.id)
    
    # Apply attack type filter
    if attack_type:
        query = query.filter(Prediction.attack_type == attack_type)
    
    # Get total count
    total = query.count()
    
    # Get paginated results
    predictions = query.order_by(Prediction.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    return PredictionHistoryResponse(
        total=total,
        page=page,
        page_size=page_size,
        predictions=predictions
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific prediction by ID.
    
    - **prediction_id**: Prediction UUID
    - Returns prediction details
    """
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    return prediction


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(
    prediction_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific prediction.
    
    - **prediction_id**: Prediction UUID
    - Returns 204 No Content on success
    """
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    
    db.delete(prediction)
    db.commit()
    
    return None
