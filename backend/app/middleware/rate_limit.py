"""
Rate limiting middleware for API protection.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
from collections import defaultdict
from typing import Dict, Tuple


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    In production, use Redis for distributed rate limiting.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls  # Number of calls allowed
        self.period = period  # Time period in seconds
        self.clients: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, time.time()))
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Get current request count and window start time
        count, window_start = self.clients[client_ip]
        current_time = time.time()
        
        # Reset window if period has passed
        if current_time - window_start > self.period:
            count = 0
            window_start = current_time
        
        # Check rate limit
        if count >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {int(self.period - (current_time - window_start))} seconds."
            )
        
        # Increment counter
        self.clients[client_ip] = (count + 1, window_start)
        
        # Clean old entries periodically (every 1000 requests)
        if len(self.clients) > 1000:
            self._cleanup_old_entries(current_time)
        
        response: Response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.calls - count - 1))
        response.headers["X-RateLimit-Reset"] = str(int(window_start + self.period))
        
        return response
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than the rate limit period."""
        expired_keys = [
            ip for ip, (_, window_start) in self.clients.items()
            if current_time - window_start > self.period * 2
        ]
        for key in expired_keys:
            del self.clients[key]




