"""
Analytics and model info API router.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, Integer
from typing import Dict, List
from datetime import datetime, timedelta

from app.database import get_db
from app.models.user import User
from app.models.prediction import Prediction
from app.middleware.clerk_auth import get_current_user
from app.schemas.user import UserStats

router = APIRouter()


@router.get("/stats", response_model=UserStats)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get statistics for current user.
    
    Returns:
    - Total predictions
    - Attacks detected
    - Normal traffic
    - Recent predictions (last 7 days)
    """
    # Total predictions
    total_predictions = db.query(func.count(Prediction.id)).filter(
        Prediction.user_id == current_user.id
    ).scalar() or 0
    
    # Attacks detected
    attacks_detected = db.query(func.count(Prediction.id)).filter(
        Prediction.user_id == current_user.id,
        Prediction.is_attack == True
    ).scalar() or 0
    
    # Normal traffic
    normal_traffic = total_predictions - attacks_detected
    
    # Recent predictions (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_predictions = db.query(func.count(Prediction.id)).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= seven_days_ago
    ).scalar() or 0
    
    return UserStats(
        total_predictions=total_predictions,
        attacks_detected=attacks_detected,
        normal_traffic=normal_traffic,
        recent_predictions=recent_predictions
    )


@router.get("/attack-distribution")
async def get_attack_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, int]:
    """
    Get distribution of attack types for current user.
    
    Returns dictionary with attack type counts.
    """
    results = db.query(
        Prediction.attack_type,
        func.count(Prediction.id).label('count')
    ).filter(
        Prediction.user_id == current_user.id
    ).group_by(
        Prediction.attack_type
    ).all()
    
    return {attack_type: count for attack_type, count in results}


@router.get("/confidence-stats")
async def get_confidence_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get confidence score statistics for current user.
    
    Returns average confidence scores for binary and multiclass predictions.
    """
    stats = db.query(
        func.avg(Prediction.binary_confidence).label('avg_binary'),
        func.avg(Prediction.multiclass_confidence).label('avg_multiclass'),
        func.min(Prediction.binary_confidence).label('min_binary'),
        func.max(Prediction.binary_confidence).label('max_binary')
    ).filter(
        Prediction.user_id == current_user.id
    ).first()
    
    return {
        "average_binary_confidence": float(stats.avg_binary or 0),
        "average_multiclass_confidence": float(stats.avg_multiclass or 0),
        "min_binary_confidence": float(stats.min_binary or 0),
        "max_binary_confidence": float(stats.max_binary or 0)
    }


@router.get("/timeline")
async def get_prediction_timeline(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[Dict]:
    """
    Get prediction timeline for last N days.
    
    - **days**: Number of days to retrieve (default: 30, max: 365)
    
    Returns daily prediction counts.
    """
    if days < 1 or days > 365:
        days = 30
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Query daily counts
    results = db.query(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count'),
        func.sum(func.cast(Prediction.is_attack, Integer)).label('attacks')
    ).filter(
        Prediction.user_id == current_user.id,
        Prediction.created_at >= start_date
    ).group_by(
        func.date(Prediction.created_at)
    ).order_by(
        func.date(Prediction.created_at)
    ).all()
    
    return [
        {
            "date": str(result.date),
            "total": result.count,
            "attacks": result.attacks or 0,
            "normal": result.count - (result.attacks or 0)
        }
        for result in results
    ]


@router.get("/model-info")
async def get_model_info() -> Dict:
    """
    Get information about the ML models.
    
    Returns model architecture, version, and performance metrics.
    """
    return {
        "version": "1.0.0",
        "models": {
            "binary": {
                "name": "Binary Classifier",
                "description": "Classifies traffic as Normal or Attack",
                "architecture": "Hybrid CNN + Transformer with Attention",
                "parameters": 73666,
                "input_features": 122,
                "output_classes": 2,
                "training_accuracy": 0.9973,
                "test_accuracy": 0.7863,
                "f1_score": 0.7843
            },
            "multiclass": {
                "name": "Multiclass Classifier",
                "description": "Classifies attack types: Normal, DoS, Probe, R2L, U2R",
                "architecture": "Hybrid CNN + Transformer with Attention",
                "parameters": 73861,
                "input_features": 122,
                "output_classes": 5,
                "training_accuracy": 0.9793,
                "test_accuracy": 0.7900,
                "f1_score": 0.7679
            }
        },
        "dataset": {
            "name": "NSL-KDD",
            "training_samples": 125973,
            "test_samples": 22544,
            "features": 41,
            "attack_categories": ["Normal", "DoS", "Probe", "R2L", "U2R"]
        },
        "performance": {
            "avg_inference_time_ms": 5.2,
            "max_batch_size": 1000
        }
    }

