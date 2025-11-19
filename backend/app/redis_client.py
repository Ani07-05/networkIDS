"""
Redis client for caching and rate limiting.
"""

import redis
from typing import Optional
from app.config import get_settings

settings = get_settings()

# Create Redis client
redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)


def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    return redis_client


def cache_get(key: str) -> Optional[str]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None
    """
    try:
        return redis_client.get(key)
    except redis.RedisError:
        return None


def cache_set(key: str, value: str, expire: int = 3600):
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        expire: Expiration time in seconds
    """
    try:
        redis_client.setex(key, expire, value)
    except redis.RedisError:
        pass


def cache_delete(key: str):
    """
    Delete key from cache.
    
    Args:
        key: Cache key to delete
    """
    try:
        redis_client.delete(key)
    except redis.RedisError:
        pass







