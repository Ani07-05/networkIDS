# Network Intrusion Detection System (NIDS)

A production-ready Network Intrusion Detection System using machine learning for real-time network traffic analysis and threat detection.

## Overview

This system uses optimized neural network models trained on the NSL-KDD dataset to classify network traffic as Normal or Attack, and categorize attacks into DoS, Probe, R2L, and U2R types.

### Key Features

- **Optimized ML Models**: Binary and Multiclass classifiers exported to ONNX format
- **Dual Classification**: Binary (Normal/Attack) and Multiclass (attack type) predictions
- **Lightning Fast**: 0.6ms average inference time - processes 1,600+ predictions/second
- **RESTful API**: FastAPI backend following SOLID principles with OpenAPI documentation
- **Modern Frontend**: React 18+ with TypeScript, Tailwind CSS, and shadcn/ui components
- **Secure Authentication**: Clerk JWT (RS256) with stateless authentication
- **Flexible Database**: PostgreSQL for production, SQLite for development
- **Batch Processing**: Handle up to 1,000 records per batch request
- **Real-time Logs**: Live model activity monitoring and prediction history
- **Production Ready**: Type-safe, OWASP compliant, horizontally scalable


## Quick Start

### Prerequisites

- Python 3.11+ or 3.12
- Node.js 18+
- Docker & Docker Compose (for production deployment)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/network-IDS.git
cd network-IDS
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```bash
# Clerk Authentication
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_key_here
CLERK_SECRET_KEY=sk_test_your_key_here
CLERK_JWT_KEY="-----BEGIN PUBLIC KEY-----\nYour_RSA_Public_Key\n-----END PUBLIC KEY-----"
CLERK_ISSUER=https://your-clerk-instance.clerk.accounts.dev
ALGORITHM=RS256

# Database
DATABASE_URL=sqlite:///./nids.db
# For production: postgresql://user:password@localhost:5432/nids

# Redis (optional - for rate limiting)
REDIS_URL=redis://localhost:6379/0

# API Settings
DEBUG=True
SECRET_KEY=your-secret-key-here
```

Get free Clerk keys at: https://dashboard.clerk.com

### 3. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 4. Run Application

**Backend (Port 8000):**
```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend (Port 5173):**
```bash
cd frontend
npm run dev
```

Access the application at: http://localhost:5173

### 5. Docker Deployment (Optional)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
All endpoints require Clerk JWT authentication via Bearer token.

#### Predictions
- `POST /api/predictions/predict` - Single prediction
- `POST /api/predictions/predict/batch` - Batch predictions (max 1000)
- `GET /api/predictions/history` - Prediction history with pagination
- `GET /api/predictions/{id}` - Get specific prediction
- `DELETE /api/predictions/{id}` - Delete prediction

#### Analytics
- `GET /api/analytics/stats` - User statistics
- `GET /api/analytics/attack-distribution` - Attack type distribution
- `GET /api/analytics/confidence-stats` - Confidence score statistics
- `GET /api/analytics/timeline` - Prediction timeline
- `GET /api/analytics/model-info` - Model information and metrics

### Example Request

```bash
curl -X POST "http://localhost:8000/api/predictions/predict" \
  -H "Authorization: Bearer YOUR_CLERK_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "duration": 0,
      "protocol_type": "tcp",
      "service": "http",
      "flag": "SF",
      "src_bytes": 181,
      "dst_bytes": 5450,
      ...
    }
  }'
```

## ML Model Details

### Dataset: NSL-KDD

- **Training**: 125,973 samples
- **Testing**: 22,544 samples
- **Features**: 41 network traffic features
- **Classes**: Normal, DoS, Probe, R2L, U2R

### Model Architecture

**Hybrid Deep Learning**:
1. **Input Layer**: 122 features (after preprocessing)
2. **Feature Extraction**: CNN layers for local patterns
3. **Attention Mechanism**: Transformer-style attention for global context
4. **Dense Layers**: [256, 128, 64] with dropout (0.3)
5. **Output**: Binary (2 classes) or Multiclass (5 classes)

**Parameters**:
- Binary Model: 73,666 parameters
- Multiclass Model: 73,861 parameters

### Performance Metrics

**Binary Classification (Normal vs Attack)**:
- Test Accuracy: ~78% on NSL-KDD test set
- Baseline performance within acceptable range

**Multiclass Classification (Attack Types)**:
- Detects: DoS, Probe, R2L, U2R attacks
- Classification confidence scores provided

**Inference Performance**:
- **Average Inference Time**: 0.6ms per sample
- **Batch Throughput**: 1,600+ predictions/second
- **Max Batch Size**: 1,000 samples per request
- **Technology**: ONNX Runtime for optimized inference


## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

### Database Migrations

```bash
cd backend

# Generate migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Code Quality

```bash
# Backend linting
cd backend
ruff check app/
black app/
mypy app/

# Frontend linting
cd frontend
npm run lint
```

## Production Deployment

1. **Configure Environment**:
   - Set DEBUG=False
   - Use strong SECRET_KEY
   - Configure production DATABASE_URL
   - Set up proper CORS_ORIGINS

2. **Database**:
   - Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
   - Enable connection pooling
   - Set up regular backups

3. **Caching**:
   - Use managed Redis (AWS ElastiCache, Redis Cloud)
   - Configure TTL policies

4. **Security**:
   - Enable HTTPS
   - Set up WAF (Web Application Firewall)
   - Configure rate limiting
   - Enable audit logging

5. **Monitoring**:
   - Application logs (structured JSON)
   - Performance metrics (response times, error rates)
   - Resource monitoring (CPU, memory, disk)

## Troubleshooting

### Models Not Found
Ensure models are trained and located at:
- `ml/models/nids_model_binary.onnx`
- `ml/models/nids_model_multiclass.onnx`
- `ml/data/processed/preprocessor.json`

### Database Connection Error
Check PostgreSQL is running and DATABASE_URL is correct.

### Redis Connection Error
Check Redis is running and REDIS_URL is correct.

### Authentication Errors
Verify Clerk keys are set correctly in `.env`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue on GitHub.




