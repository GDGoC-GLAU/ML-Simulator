from fastapi import FastAPI
from pydantic import BaseModel
from models.decision_tree import DecisionTreeModel
from utils.data_helpers import generate_sample_regression_data
import pandas as pd
import uvicorn

app = FastAPI(title="ML Simulator API", version="1.0")

# ----- Request Schema -----
class ModelRequest(BaseModel):
    model: str
    task_type: str = "classification"  # or regression
    features: list

# ----- Endpoints -----
@app.get("/")
def home():
    return {"message": "Welcome to the ML Simulator API 🎯"}

@app.post("/predict")
def predict(req: ModelRequest):
    """
    Example Input:
    {
        "model": "decision_tree",
        "task_type": "classification",
        "features": [[5.1, 3.5, 1.4, 0.2]]
    }
    """
    model_name = req.model.lower()
    task_type = req.task_type.lower()

    # Load or initialize model
    if model_name == "decision_tree":
        model = DecisionTreeModel(task_type=task_type)
        # Generate training data just for demo
        if task_type == "classification":
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=100, n_features=len(req.features[0]), random_state=42)
        else:
            df = generate_sample_regression_data(n_samples=100, n_features=len(req.features[0]))
            X, y = df.iloc[:, :-1], df.iloc[:, -1]

        # Train model
        model.train(X, y)

        # Predict
        prediction = model.predict(pd.DataFrame(req.features))
        return {
            "model_used": f"Decision Tree ({task_type})",
            "prediction": prediction.tolist()
        }
    else:
        return {"error": "Model not implemented yet."}


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
