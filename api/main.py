from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.linear_regression import LinearRegressionModel
from models.logistic_regression import LogisticRegressionModel

app = FastAPI(title="ML Simulator API", version="1.0")

# Example input format
class ModelInput(BaseModel):
    model_name: str
    features: list[float]

@app.post("/predict")
def predict(data: ModelInput):
    model_name = data.model_name.lower()
    features = data.features

    if model_name == "linear_regression":
        model = LinearRegressionModel()
    elif model_name == "logistic_regression":
        model = LogisticRegressionModel()
    else:
        raise HTTPException(status_code=400, detail="Model not found")

    try:
        prediction = model.predict(features)
        return {"model": model_name, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
