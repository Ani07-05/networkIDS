"""
Prediction Pydantic schemas.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID
from typing import Dict, List, Optional


class NetworkFeatures(BaseModel):
    """Network traffic features (41 NSL-KDD features)."""
    
    # Connection basic features
    duration: float = Field(..., ge=0, description="Connection duration in seconds")
    protocol_type: str = Field(..., description="Protocol type (tcp, udp, icmp)")
    service: str = Field(..., description="Network service (http, ftp, etc)")
    flag: str = Field(..., description="Connection status flag")
    src_bytes: int = Field(..., ge=0, description="Bytes from source to destination")
    dst_bytes: int = Field(..., ge=0, description="Bytes from destination to source")
    land: int = Field(..., ge=0, le=1, description="1 if connection is from/to same host/port")
    wrong_fragment: int = Field(..., ge=0, description="Number of wrong fragments")
    urgent: int = Field(..., ge=0, description="Number of urgent packets")
    
    # Content features
    hot: int = Field(..., ge=0, description="Number of 'hot' indicators")
    num_failed_logins: int = Field(..., ge=0, description="Number of failed login attempts")
    logged_in: int = Field(..., ge=0, le=1, description="1 if successfully logged in")
    num_compromised: int = Field(..., ge=0, description="Number of 'compromised' conditions")
    root_shell: int = Field(..., ge=0, le=1, description="1 if root shell is obtained")
    su_attempted: int = Field(..., ge=0, description="Number of 'su root' commands attempted")
    num_root: int = Field(..., ge=0, description="Number of 'root' accesses")
    num_file_creations: int = Field(..., ge=0, description="Number of file creation operations")
    num_shells: int = Field(..., ge=0, description="Number of shell prompts")
    num_access_files: int = Field(..., ge=0, description="Number of operations on access control files")
    num_outbound_cmds: int = Field(..., ge=0, description="Number of outbound commands")
    is_host_login: int = Field(..., ge=0, le=1, description="1 if login belongs to host list")
    is_guest_login: int = Field(..., ge=0, le=1, description="1 if login is a guest login")
    
    # Traffic features (time-based)
    count: int = Field(..., ge=0, description="Number of connections to same host in past 2 seconds")
    srv_count: int = Field(..., ge=0, description="Number of connections to same service in past 2 seconds")
    serror_rate: float = Field(..., ge=0, le=1, description="% of connections with SYN errors")
    srv_serror_rate: float = Field(..., ge=0, le=1, description="% of connections with SYN errors (same service)")
    rerror_rate: float = Field(..., ge=0, le=1, description="% of connections with REJ errors")
    srv_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections with REJ errors (same service)")
    same_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to same service")
    diff_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to different services")
    srv_diff_host_rate: float = Field(..., ge=0, le=1, description="% of connections to different hosts (same service)")
    
    # Host-based features
    dst_host_count: int = Field(..., ge=0, description="Number of connections having same destination host")
    dst_host_srv_count: int = Field(..., ge=0, description="Number of connections having same destination host and service")
    dst_host_same_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to same service (destination host)")
    dst_host_diff_srv_rate: float = Field(..., ge=0, le=1, description="% of connections to different services (destination host)")
    dst_host_same_src_port_rate: float = Field(..., ge=0, le=1, description="% of connections from same source port (destination host)")
    dst_host_srv_diff_host_rate: float = Field(..., ge=0, le=1, description="% of connections to different hosts (same service, destination host)")
    dst_host_serror_rate: float = Field(..., ge=0, le=1, description="% of connections with SYN errors (destination host)")
    dst_host_srv_serror_rate: float = Field(..., ge=0, le=1, description="% of connections with SYN errors (same service, destination host)")
    dst_host_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections with REJ errors (destination host)")
    dst_host_srv_rerror_rate: float = Field(..., ge=0, le=1, description="% of connections with REJ errors (same service, destination host)")


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    features: NetworkFeatures = Field(..., description="Network traffic features")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    id: UUID
    is_attack: bool
    binary_confidence: float
    attack_type: str
    multiclass_confidence: float
    multiclass_probabilities: Dict[str, float]
    inference_time_ms: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    predictions: List[NetworkFeatures] = Field(..., max_length=1000, description="List of network features (max 1000)")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    total: int
    predictions: List[PredictionResponse]
    processing_time_ms: float


class PredictionHistory(BaseModel):
    """Schema for prediction history query."""
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=100, description="Items per page")
    attack_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class PredictionHistoryResponse(BaseModel):
    """Schema for prediction history response."""
    total: int
    page: int
    page_size: int
    predictions: List[PredictionResponse]







