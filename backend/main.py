from fastapi import FastAPI, HTTPException
# Import schemas and model module robustly to handle different import contexts in production
try:
    from backend.schemas import HealthRequest, HealthResponse, HealthCheckResponse, AnomalyRequest, AnomalyResponse
    from backend.ml.models import AnomalyDetector
except Exception:
    import sys, os
    # Add this file's directory (backend) to sys.path so sibling imports work when module is executed as top-level
    backend_dir = os.path.dirname(__file__)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from schemas import HealthRequest, HealthResponse, HealthCheckResponse, AnomalyRequest, AnomalyResponse
    from ml.models import AnomalyDetector

import numpy as np
import os
import asyncio
import logging

app = FastAPI(
    title="Health Monitoring API",
    description="Simple health risk assessment based on vital signs",
    version="1.0.0"
)

# configure basic logging (can control via LOG_LEVEL env var)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

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


@app.on_event("startup")
async def startup_event():
    """Non-blocking model warm-up on startup (runs in threadpool).

    This respects MODEL_WARMUP env var. Failures during loading/training are logged and do not crash the process.
    """
    def _safe_load():
        try:
            detector.load_models()
        except Exception:
            logging.exception("Model warm-up failed during startup; continuing without model.")

    model_warmup = os.environ.get("MODEL_WARMUP", "true").lower() in ("1", "true", "yes")
    if model_warmup:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _safe_load)
    else:
        logging.info("MODEL_WARMUP disabled; skipping model warm-up on startup.")


@app.get("/ready")
async def readiness():
    """Return readiness including whether model artifacts are loaded."""
    model_loaded = detector.model is not None
    return {"status": "ok", "model_loaded": model_loaded}


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

