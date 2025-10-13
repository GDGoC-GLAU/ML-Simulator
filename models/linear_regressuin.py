# models/linear_regression_model.py
from sklearn.linear_model import LinearRegression
import numpy as np

# Train a simple model for demonstration
model = LinearRegression()
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model.fit(X, y)

def predict(features):
    arr = np.array(features).reshape(1, -1)
    prediction = model.predict(arr)
    return prediction.tolist()
