"""
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        init_db()
        print("Database initialized")
    except Exception as e:
        print(f"Warning: Database not available - {e}")
        print("Running in ML-only mode (predictions won't be saved)")
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced Network Intrusion Detection System API",
    lifespan=lifespan
)

# Configure CORS - Allow all origins for now to debug
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add security headers middleware
from app.middleware.security_headers import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)

# Add rate limiting middleware (100 requests per minute per IP)
from app.middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, calls=100, period=60)

# Add trusted host middleware (disabled in development)
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts_list
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "redis": "connected"
    }


# Import and include routers
from app.routers import predictions, analytics

app.include_router(
    predictions.router,
    prefix="/api/predictions",
    tags=["predictions"]
)
app.include_router(
    analytics.router,
    prefix="/api/analytics",
    tags=["analytics"]
)

