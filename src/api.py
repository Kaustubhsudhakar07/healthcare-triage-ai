"""
Production REST API Microservice (FastAPI)
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Exposes high-performance RESTful endpoints for real-time and batch triage inference,
hard clinical safety guardrails, SHAP factor attributions, and health metrics.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath("."))

from src.predict import ClinicalInferenceService, PatientPayload

from contextlib import asynccontextmanager

# Global Inference Service Singleton
service: Optional[ClinicalInferenceService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline on container startup."""
    global service
    pipeline_path = "models/pipeline.joblib"
    train_path = "data/processed/train.csv"
    if os.path.exists(pipeline_path):
        service = ClinicalInferenceService(pipeline_path=pipeline_path, train_path=train_path)
    else:
        service = None
    yield


# Initialize FastAPI App
app = FastAPI(
    title="Pre-Hospital Patient Criticality & Triage Inference API",
    description="Production RESTful microservice providing AI-assisted emergency acuity scores (1.0-10.0), 5-tier triage classification, and local SHAP explanations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for frontend and mobile ambulance telemetry integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchPatientPayload(BaseModel):
    patients: List[PatientPayload]


class PredictionResponse(BaseModel):
    criticality_score: float = Field(..., description="Estimated acuity index (1.0 - 10.0)")
    urgency_tier: str = Field(..., description="Operational urgency tier ('Low', 'Moderate', 'Elevated', 'High', 'Critical')")
    raw_urgency_tier: str
    safety_guardrails_triggered: bool
    safety_override_applied: bool
    red_flags: List[str]
    clinical_routing_guidance: str
    explanation: Optional[Dict[str, Any]] = None
    inference_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_path: str
    version: str
    uptime_seconds: float


START_TIME = time.time()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and model loading status."""
    return HealthResponse(
        status="HEALTHY" if service is not None else "DEGRADED",
        model_loaded=service is not None,
        pipeline_path="models/pipeline.joblib",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2)
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_patient_criticality(payload: PatientPayload):
    """
    Predicts pre-hospital criticality score (1.0 - 10.0) and operational urgency tier for a single patient.
    """
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded. Ensure models/pipeline.joblib exists."
        )
    
    t0 = time.perf_counter()
    try:
        result = service.predict(payload.model_dump())
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["inference_latency_ms"] = latency_ms
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Inference error: {str(e)}")


@app.post("/batch-predict", response_model=List[PredictionResponse], tags=["Inference"])
def batch_predict(batch: BatchPatientPayload):
    """
    Batch inference endpoint for high-throughput ambulance telemetry feeds.
    """
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model pipeline is not loaded.")
        
    responses = []
    for item in batch.patients:
        t0 = time.perf_counter()
        res = service.predict(item.model_dump())
        res["inference_latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        responses.append(PredictionResponse(**res))
        
    return responses


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
