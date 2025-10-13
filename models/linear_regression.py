import numpy as np

class LinearRegressionModel:
    def __init__(self):
        # Placeholder coefficients
        self.coefficients = [1.5, -0.7, 0.3]
        self.bias = 2.0

    def predict(self, features):
        x = np.array(features)
        return float(np.dot(self.coefficients, x) + self.bias)
