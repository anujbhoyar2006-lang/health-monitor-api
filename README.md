# Simple FastAPI Health Monitoring Backend

A minimal FastAPI backend for health risk assessment based on vital signs.

## Features

- **Health Check**: GET `/` endpoint to verify API status
- **Risk Prediction**: POST `/predict` endpoint for health risk assessment
- **Simple Logic**: Classifies risk as NORMAL or HIGH based on heart rate and SpO2
- **Input Validation**: Automatic validation using Pydantic models
- **Interactive Docs**: Automatic API documentation at `/docs`

## Project Structure

```
backend/
├── __init__.py          # Package initialization
├── main.py             # FastAPI application and endpoints
└── schemas.py          # Pydantic models for request/response
requirements.txt        # Python dependencies
README.md              # This file
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/

## API Endpoints

### Health Check
- **URL**: `GET /`
- **Response**: `{"status": "ok"}`

### Risk Prediction
- **URL**: `POST /predict`
- **Request Body**:
```json
{
  "heart_rate": 85.0,
  "spo2": 98.0,
  "temperature": 98.6
}
```
- **Response**:
```json
{
  "risk_level": "NORMAL",
  "heart_rate": 85.0,
  "spo2": 98.0,
  "temperature": 98.6
}
```

## Risk Assessment Logic

The API classifies health risk based on these rules:

- **HIGH Risk**: `heart_rate > 100` OR `spo2 < 92`
- **NORMAL Risk**: All other cases

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8000/

# Risk prediction - Normal case
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 75, "spo2": 98, "temperature": 98.6}'

# Risk prediction - High risk case
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"heart_rate": 120, "spo2": 88, "temperature": 101.2}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())  # {"status": "ok"}

# Risk prediction
health_data = {
    "heart_rate": 85.0,
    "spo2": 95.0,
    "temperature": 99.1
}

response = requests.post("http://localhost:8000/predict", json=health_data)
print(response.json())
# {"risk_level": "NORMAL", "heart_rate": 85.0, "spo2": 95.0, "temperature": 99.1}
```

## Input Validation

The API automatically validates input data:

- **heart_rate**: Must be between 0-300 BPM
- **spo2**: Must be between 0-100%
- **temperature**: Must be between 80-120°F

Invalid inputs will return a 422 validation error with details.

## Development

### Running with Auto-reload
```bash
uvicorn backend.main:app --reload
```

### Running on Different Port
```bash
uvicorn backend.main:app --port 8080
```

### Production Deployment
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Next Steps

This is a minimal implementation. For production use, consider adding:

- Database integration for storing predictions
- Authentication and authorization
- More sophisticated health risk algorithms
- Logging and monitoring
- Rate limiting
- CORS configuration for web frontends
- Docker containerization
