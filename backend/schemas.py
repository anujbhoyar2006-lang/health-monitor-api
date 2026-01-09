from pydantic import BaseModel, Field
from typing import Literal


class HealthRequest(BaseModel):
    """Request model for health data prediction"""
    heart_rate: float = Field(..., description="Heart rate in beats per minute", ge=0, le=300)
    spo2: float = Field(..., description="Blood oxygen saturation percentage", ge=0, le=100)
    temperature: float = Field(..., description="Body temperature in Fahrenheit", ge=80, le=120)


class HealthResponse(BaseModel):
    """Response model for health prediction"""
    risk_level: Literal["NORMAL", "HIGH"] = Field(..., description="Predicted risk level")
    heart_rate: float = Field(..., description="Input heart rate")
    spo2: float = Field(..., description="Input SpO2")
    temperature: float = Field(..., description="Input temperature")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API status")
