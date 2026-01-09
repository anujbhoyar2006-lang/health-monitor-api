from fastapi import FastAPI, HTTPException
from backend.schemas import HealthRequest, HealthResponse, HealthCheckResponse, AnomalyRequest, AnomalyResponse
from backend.ml.models import AnomalyDetector
import numpy as np
import os

app = FastAPI(
    title="Health Monitoring API",
    description="Simple health risk assessment based on vital signs",
    version="1.0.0"
)

# ML detector (lazy-loaded)
detector = AnomalyDetector(artifacts_path=os.path.join(os.path.dirname(__file__), "ml", "artifacts"))
# don't load models at import time; they will be loaded lazily on first request


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(status="ok")


@app.post("/predict", response_model=HealthResponse)
async def predict_risk(health_data: HealthRequest):
    try:
        heart_rate = health_data.heart_rate
        spo2 = health_data.spo2
        temperature = health_data.temperature

        if heart_rate > 100 or spo2 < 92:
            risk_level = "HIGH"
        else:
            risk_level = "NORMAL"

        return HealthResponse(
            risk_level=risk_level,
            heart_rate=heart_rate,
            spo2=spo2,
            temperature=temperature
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing health data: {str(e)}"
        )


@app.post("/anomaly", response_model=AnomalyResponse)
async def detect_anomaly(vitals: AnomalyRequest):
    try:
        arr = np.array([[
            vitals.heart_rate,
            vitals.respiratory_rate,
            vitals.spo2,
            vitals.temperature,
            vitals.glucose
        ]])

        # lazy-load models if needed
        if detector.model is None:
            detector.load_models()

        result = detector.predict_anomaly(arr)
        # result is a dict for single-sample
        return AnomalyResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing anomaly detection: {str(e)}")


# 🚀 REQUIRED FOR RAILWAY
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )

