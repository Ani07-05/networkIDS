"""
Prediction database model.
"""

from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.database import Base


class Prediction(Base):
    """Network traffic prediction model."""
    
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Input features (stored as JSON)
    features = Column(JSON, nullable=False)
    
    # Binary classification results
    is_attack = Column(Boolean, nullable=False)
    binary_confidence = Column(Float, nullable=False)
    
    # Multiclass classification results
    attack_type = Column(String, nullable=False)  # Normal, DoS, Probe, R2L, U2R
    multiclass_confidence = Column(Float, nullable=False)
    multiclass_probabilities = Column(JSON, nullable=False)  # All class probabilities
    
    # Metadata
    inference_time_ms = Column(Float, nullable=True)
    model_version = Column(String, default="1.0.0")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<Prediction {self.id} - {self.attack_type}>"


