"""
Pytest configuration and fixtures for backend tests.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.config import settings

# Test database setup (in-memory SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    """Create a test client with dependency overrides."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_onnx_model(monkeypatch):
    """Mock ONNX model for testing predictions."""
    class MockInferenceSession:
        def __init__(self, *args, **kwargs):
            self.input_name = "input"
            self.output_name = "output"
        
        def get_inputs(self):
            class Input:
                name = "input"
            return [Input()]
        
        def get_outputs(self):
            class Output:
                name = "output"
            return [Output()]
        
        def run(self, output_names, input_dict):
            # Return mock predictions
            batch_size = input_dict[self.input_name].shape[0]
            # Binary classification: [normal, attack]
            predictions = np.array([[0.8, 0.2]] * batch_size, dtype=np.float32)
            return [predictions]
    
    # Patch onnxruntime.InferenceSession
    import onnxruntime as ort
    monkeypatch.setattr(ort, "InferenceSession", MockInferenceSession)
    
    return MockInferenceSession


@pytest.fixture
def sample_features():
    """Sample feature dictionary for predictions."""
    return {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 215,
        "dst_bytes": 45076,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 1,
        "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }


@pytest.fixture
def mock_clerk_token():
    """Mock Clerk JWT token for authentication tests."""
    return "mock_clerk_token_12345"


@pytest.fixture
def auth_headers(mock_clerk_token):
    """Headers with mock authentication token."""
    return {"Authorization": f"Bearer {mock_clerk_token}"}


