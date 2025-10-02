"""
FastAPI application for ML Model Simulator
Provides REST API endpoints to access ML models externally
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from api_models import (
    LinearRegressionRequest,
    LinearRegressionTrainRequest,
    PredictionResponse,
    TrainResponse,
    HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Simulator API",
    description="API to access ML models externally. Send JSON input and receive prediction output.",
    version="1.0.0"
)

# Constants
MODEL_LINEAR_REGRESSION = "Linear Regression"

"""Security note: CORS restricted to local development origins.
Adjust ALLOWED_ORIGINS when deploying to another host/domain."""
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",  # Streamlit default
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _init_state(application: FastAPI):
    """Initialize shared application state.

    Attributes added:
        linear_regression_model: Trained LinearRegression instance or None
    """
    application.state.linear_regression_model = None


@app.on_event("startup")
async def startup_event():  # pragma: no cover - simple log
    _init_state(app)
    # Simple visibility log
    print("[startup] ML Model Simulator API started. Docs at /docs")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="online",
        message="ML Model Simulator API is running. Visit /docs for API documentation."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )


@app.post("/api/v1/linear-regression/train", response_model=TrainResponse)
async def train_linear_regression(request: LinearRegressionTrainRequest):
    """
    Train a Linear Regression model
    
    Args:
        request: Training data with features (X) and labels (y)
    
    Returns:
        Training confirmation with model coefficients and intercept
    
    Example:
        ```json
        {
            "X": [[1.0], [2.0], [3.0], [4.0], [5.0]],
            "y": [2.0, 4.0, 5.0, 4.0, 5.0]
        }
        ```
    """
    try:
        # Convert input to numpy arrays
        X = np.array(request.X)
        y = np.array(request.y)

        # Validate dimensions
        if len(X) != len(y):
            raise HTTPException(
                status_code=400,
                detail=f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )

        # Decide whether to use stub model to avoid heavy sklearn import (set USE_STUB_LINEAR=1 for tests)
        use_stub = os.getenv("USE_STUB_LINEAR") == "1"

        if use_stub:
            class _StubLinearRegression:
                def __init__(self):
                    self.coef_ = np.zeros(X.shape[1]) if X.ndim > 1 else np.array([0.0])
                    self.intercept_ = 0.0

                def fit(self, x_fit, y_fit):
                    # Provide a simple deterministic coefficient for single-feature data
                    if x_fit.ndim == 2 and x_fit.shape[1] == 1:
                        x = x_fit[:, 0]
                        yloc = y_fit
                        denom = (x - x.mean()).sum() or 1.0
                        slope = ((x - x.mean()) * (yloc - yloc.mean())).sum() / denom
                        self.coef_ = np.array([slope])
                        self.intercept_ = float(yloc.mean() - slope * x.mean())
                    return self

                def predict(self, x_in):
                    x_in = np.array(x_in)
                    if x_in.ndim == 2:
                        return x_in[:, 0] * self.coef_[0] + self.intercept_
                    return x_in * self.coef_[0] + self.intercept_

            model = _StubLinearRegression()
        else:
            # Lazy import to speed API startup (heavy sklearn import deferred until training actually requested)
            from sklearn.linear_model import LinearRegression  # type: ignore
            model = LinearRegression()

        model.fit(X, y)
        app.state.linear_regression_model = model

        return TrainResponse(
            message="Linear Regression model trained successfully",
            model_type=MODEL_LINEAR_REGRESSION,
            coefficients=model.coef_.tolist(),
            intercept=float(model.intercept_)
    )
    except HTTPException:
        # Re-raise validation HTTPExceptions directly (e.g., length mismatch)
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/api/v1/linear-regression/predict", response_model=PredictionResponse)
async def predict_linear_regression(request: LinearRegressionRequest):
    """
    Make predictions using trained Linear Regression model
    
    Args:
        request: Features for prediction
    
    Returns:
        Predictions for input features
    
    Example:
        ```json
        {
            "features": [[1.0], [2.0], [3.0]]
        }
        ```
    """
    model = getattr(app.state, "linear_regression_model", None)
    # Ensure model is trained
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first using /api/v1/linear-regression/train endpoint",
        )

    try:
        X = np.array(request.features)
        predictions = model.predict(X)
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_type=MODEL_LINEAR_REGRESSION,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}",
        )
    except Exception as e:  # pragma: no cover - unexpected path
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/api/v1/linear-regression/train-and-predict", response_model=PredictionResponse)
async def train_and_predict_linear_regression(
    train_request: LinearRegressionTrainRequest,
):
    """
    Train model and make predictions in one call
    
    Args:
        train_request: Training data (X, y) - predictions will be made on X
    
    Returns:
        Predictions for training features
    
    Example:
        ```json
        {
            "X": [[1.0], [2.0], [3.0], [4.0], [5.0]],
            "y": [2.0, 4.0, 5.0, 4.0, 5.0]
        }
        ```
    """
    try:
        X = np.array(train_request.X)
        y = np.array(train_request.y)
        if len(X) != len(y):
            raise HTTPException(
                status_code=400,
                detail=f"X and y must have same length. Got X: {len(X)}, y: {len(y)}",
            )
        use_stub = os.getenv("USE_STUB_LINEAR") == "1"
        if use_stub:
            class _StubLinearRegression:
                def __init__(self):
                    self.coef_ = np.zeros(X.shape[1]) if X.ndim > 1 else np.array([0.0])
                    self.intercept_ = 0.0

                def fit(self, x_fit, y_fit):
                    if x_fit.ndim == 2 and x_fit.shape[1] == 1:
                        x = x_fit[:, 0]
                        yloc = y_fit
                        denom = (x - x.mean()).sum() or 1.0
                        slope = ((x - x.mean()) * (yloc - yloc.mean())).sum() / denom
                        self.coef_ = np.array([slope])
                        self.intercept_ = float(yloc.mean() - slope * x.mean())
                    return self

                def predict(self, x_in):
                    x_in = np.array(x_in)
                    if x_in.ndim == 2:
                        return x_in[:, 0] * self.coef_[0] + self.intercept_
                    return x_in * self.coef_[0] + self.intercept_

            model = _StubLinearRegression()
        else:
            from sklearn.linear_model import LinearRegression  # type: ignore
            model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_type=MODEL_LINEAR_REGRESSION,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}",
        )
    except Exception as e:  # pragma: no cover - unexpected path
        raise HTTPException(
            status_code=500,
            detail=f"Operation failed: {str(e)}",
        )


@app.get("/api/v1/models")
async def list_models():
    """
    List all available models and their status
    
    Returns:
        Dictionary of available models and their training status
    """
    model = getattr(app.state, "linear_regression_model", None)
    return {
        "models": [
            {
                "name": MODEL_LINEAR_REGRESSION,
                "endpoint": "/api/v1/linear-regression",
                "trained": model is not None,
                "methods": ["train", "predict", "train-and-predict"],
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
