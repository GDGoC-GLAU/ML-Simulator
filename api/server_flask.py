from flask import Flask, request, jsonify
from models.decision_tree import DecisionTreeModel
from utils.data_helpers import generate_sample_regression_data
import pandas as pd
from sklearn.datasets import make_classification

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to ML Simulator Flask API"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data.get('model', 'decision_tree')
    task_type = data.get('task_type', 'classification')
    features = data.get('features', [])

    if model_name == "decision_tree":
        model = DecisionTreeModel(task_type=task_type)
        if task_type == "classification":
            X, y = make_classification(n_samples=100, n_features=len(features[0]), random_state=42)
        else:
            df = generate_sample_regression_data(n_samples=100, n_features=len(features[0]))
            X, y = df.iloc[:, :-1], df.iloc[:, -1]

        model.train(X, y)
        prediction = model.predict(pd.DataFrame(features))
        return jsonify({
            "model_used": f"Decision Tree ({task_type})",
            "prediction": prediction.tolist()
        })
    else:
        return jsonify({"error": "Model not implemented"}), 400

if __name__ == '__main__':
    app.run(debug=True)
