from fastapi import FastAPI, HTTPException
from schemas import HealthRequest, HealthResponse, HealthCheckResponse

app = FastAPI(
    title="Health Monitoring API",
    description="Simple health risk assessment based on vital signs",
    version="1.0.0"
)


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

