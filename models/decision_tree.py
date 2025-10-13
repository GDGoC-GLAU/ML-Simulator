from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import numpy as np

class DecisionTreeModel:
    """
    Decision Tree model wrapper supporting both classification and regression.
    """

    def __init__(self, task_type="classification", max_depth=None, random_state=42):
        self.task_type = task_type
        self.max_depth = max_depth
        self.random_state = random_state

        if task_type == "classification":
            self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        elif task_type == "regression":
            self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        else:
            raise ValueError("Invalid task_type. Use 'classification' or 'regression'.")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        if self.task_type == "classification":
            return {"accuracy": accuracy_score(y, preds)}
        else:
            return {"mse": mean_squared_error(y, preds)}

    def get_model(self):
        return self.model
