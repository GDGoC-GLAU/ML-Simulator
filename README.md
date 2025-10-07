# � ML Simulator

ML Simulator is an interactive web application built with **Streamlit** that allows users to **visualize, train, and understand popular Machine Learning algorithms** in a simple and intuitive way. It also provides a **REST API** for external model access.

## Features
-Interactive simulation of ML algorithms  
-Adjustable hyperparameters for each model  
-Visualization of predictions and decision boundaries  
-Evaluation metrics including Confusion Matrix and ROC Curve  
-REST API for external model access  
-Clean and modular code structure  
-Easy to extend with new models

## Supported Algorithms
| Category       | Algorithms                                                    |
| -------------- | -------------------------------------------------------------- |
| Classification | Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN) |
| Regression     | Linear Regression, Polynomial Regression                     |
| Clustering     | K-Means                                                       |
| Metrics        | Confusion Matrix, ROC Curve, Accuracy, AUC                    |

## How to Run

### Web Interface (Streamlit)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### REST API Server
```bash
pip install -r requirements.txt
python api.py
```
Or using uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`
- Interactive API documentation: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## API Usage

### Startup Note
The first request (or server start) can take ~40 seconds on some machines due to the initial `scikit-learn` import. Subsequent requests are fast. If you need faster cold starts, consider:
- Pre-importing `sklearn` in a separate warm-up script before starting Uvicorn
- Using a smaller environment or pruning unused sklearn modules
- Building a Docker image and letting the import happen at container build time

### Error Responses
Common error cases the API will return with JSON `detail` field:
- 400: Mismatched lengths between `X` and `y` during training
- 400: Calling predict before training the model
- 400: Malformed JSON (e.g., wrong shapes that cannot be converted to arrays)
- 500: Unexpected server errors during training or prediction

Example (predict before training):
```bash
curl -s -X POST "http://localhost:8000/api/v1/linear-regression/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0], [2.0]]}' | jq
```
Response:
```json
{
  "detail": "Model not trained. Please train the model first using /api/v1/linear-regression/train endpoint"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

### List Available Models
```bash
curl http://localhost:8000/api/v1/models
```

### Linear Regression - Train Model
```bash
curl -X POST "http://localhost:8000/api/v1/linear-regression/train" \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1.0], [2.0], [3.0], [4.0], [5.0]],
    "y": [2.0, 4.0, 5.0, 4.0, 5.0]
  }'
```

### Linear Regression - Make Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/linear-regression/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0], [2.0], [3.0]]
  }'
```

### Linear Regression - Train and Predict (One-Step)
```bash
curl -X POST "http://localhost:8000/api/v1/linear-regression/train-and-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1.0], [2.0], [3.0], [4.0], [5.0]],
    "y": [2.0, 4.0, 5.0, 4.0, 5.0]
  }'
```

### Python Example
```python
import requests

# Train the model
train_data = {
    "X": [[1.0], [2.0], [3.0], [4.0], [5.0]],
    "y": [2.0, 4.0, 5.0, 4.0, 5.0]
}
response = requests.post(
    "http://localhost:8000/api/v1/linear-regression/train",
    json=train_data
)
print(response.json())

# Make predictions
predict_data = {
    "features": [[6.0], [7.0], [8.0]]
}
response = requests.post(
    "http://localhost:8000/api/v1/linear-regression/predict",
    json=predict_data
)
print(response.json())
```

### JavaScript Example
```javascript
// Train the model
const trainData = {
    X: [[1.0], [2.0], [3.0], [4.0], [5.0]],
    y: [2.0, 4.0, 5.0, 4.0, 5.0]
};

fetch('http://localhost:8000/api/v1/linear-regression/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(trainData)
})
.then(response => response.json())
.then(data => console.log(data));

// Make predictions
const predictData = {
    features: [[6.0], [7.0], [8.0]]
};

fetch('http://localhost:8000/api/v1/linear-regression/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(predictData)
})
.then(response => response.json())
.then(data => console.log(data));
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API root and health status |
| GET | `/health` | Health check |
| GET | `/api/v1/models` | List all available models |
| POST | `/api/v1/linear-regression/train` | Train Linear Regression model |
| POST | `/api/v1/linear-regression/predict` | Make predictions with trained model |
| POST | `/api/v1/linear-regression/train-and-predict` | Train and predict in one call |

## Testing

Minimal automated tests are included for core functionality plus error cases.

Run tests:
```bash
python -m pip install -r requirements.txt
python tests/test_api.py
```
Expected output includes passing tests for:
- Health endpoint
- Train then predict flow
- Combined train-and-predict
- Predict before training (error 400)
- Mismatched `X` and `y` lengths (error 400)

If you want to run them with `pytest`, you can install pytest and run:
```bash
pip install pytest
pytest -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

See [LICENSE](LICENSE) for license information.
