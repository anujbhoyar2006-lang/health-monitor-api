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


class AnomalyRequest(BaseModel):
    """Request model for anomaly detection (5 features)"""
    heart_rate: float = Field(..., description="Heart rate in beats per minute", ge=0, le=300)
    respiratory_rate: float = Field(..., description="Respiratory rate in breaths per minute", ge=0, le=100)
    spo2: float = Field(..., description="Blood oxygen saturation percentage", ge=0, le=100)
    temperature: float = Field(..., description="Body temperature in Fahrenheit", ge=80, le=120)
    glucose: float = Field(..., description="Blood glucose mg/dL", ge=0, le=1000)


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection"""
    anomaly_score: float
    anomaly_detected: bool
    risk_level: Literal["NORMAL", "MODERATE", "HIGH", "CRITICAL"]
    isolation_forest_score: float
    isolation_forest_prediction: int
    scaled_features: list[float]


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="API status")
