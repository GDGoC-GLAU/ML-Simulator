from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np

class KNNModel:
    def __init__(self, task_type="classification", k=5):
        self.task_type = task_type
        self.k = k
        self.model = None

    def initialize_model(self):
        """Initialize KNN model based on task type."""
        if self.task_type == "classification":
            self.model = KNeighborsClassifier(n_neighbors=self.k)
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.k)

    def train(self, X, y):
        """Train KNN model."""
        if self.model is None:
            self.initialize_model()
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def score(self, X, y):
        """Return accuracy (for classification) or R² (for regression)."""
        return self.model.score(X, y)
