"""
User Pydantic schemas.
"""

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from uuid import UUID
from typing import Optional


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a user."""
    clerk_id: str = Field(..., description="Clerk user ID")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: UUID
    clerk_id: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """User statistics schema."""
    total_predictions: int
    attacks_detected: int
    normal_traffic: int
    recent_predictions: int







