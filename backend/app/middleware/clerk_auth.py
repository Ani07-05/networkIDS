"""
Clerk JWT authentication middleware.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.user import User
from app.database import get_db

settings = get_settings()
security = HTTPBearer()


class ClerkUser:
    """Clerk authenticated user."""
    
    def __init__(self, clerk_id: str, email: str, first_name: Optional[str] = None, last_name: Optional[str] = None):
        self.clerk_id = clerk_id
        self.email = email
        self.first_name = first_name
        self.last_name = last_name


async def verify_clerk_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> ClerkUser:
    """
    Verify Clerk JWT token and return user information.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        ClerkUser instance
        
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    try:
        # Decode JWT token using RS256 for Clerk's public key
        payload = jwt.decode(
            token,
            settings.CLERK_JWT_KEY,
            algorithms=["RS256"],
            audience=None,
            options={"verify_aud": False, "verify_nbf": False},
        )
        
        # Extract user information
        clerk_id: str = payload.get("sub")
        email: str = payload.get("email")
        first_name: Optional[str] = payload.get("first_name")
        last_name: Optional[str] = payload.get("last_name")
        
        if not clerk_id or not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return ClerkUser(
            clerk_id=clerk_id,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    clerk_user: ClerkUser = Security(verify_clerk_token),
    db: Session = Security(get_db)
) -> User:
    """
    Get or create current user from database.
    
    Args:
        clerk_user: Verified Clerk user
        db: Database session
        
    Returns:
        User model instance
    """
    # Try to find existing user
    user = db.query(User).filter(User.clerk_id == clerk_user.clerk_id).first()
    
    # Create user if not exists
    if not user:
        user = User(
            clerk_id=clerk_user.clerk_id,
            email=clerk_user.email,
            first_name=clerk_user.first_name,
            last_name=clerk_user.last_name
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user
