"""
Pydantic models for API request and response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class LinearRegressionRequest(BaseModel):
    """Request model for Linear Regression prediction"""
    features: List[List[float]] = Field(
        ..., 
        description="List of feature vectors for prediction",
        example=[[1.0], [2.0], [3.0]]
    )


class LinearRegressionTrainRequest(BaseModel):
    """Request model for training Linear Regression model"""
    X: List[List[float]] = Field(
        ..., 
        description="Training features",
        example=[[1.0], [2.0], [3.0], [4.0], [5.0]]
    )
    y: List[float] = Field(
        ..., 
        description="Training labels",
        example=[2.0, 4.0, 5.0, 4.0, 5.0]
    )


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float] = Field(
        ..., 
        description="Model predictions"
    )
    model_type: str = Field(
        ..., 
        description="Type of model used"
    )


class TrainResponse(BaseModel):
    """Response model for training endpoints"""
    message: str
    model_type: str
    coefficients: Optional[List[float]] = None
    intercept: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
