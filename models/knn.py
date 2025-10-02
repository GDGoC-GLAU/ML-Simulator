from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNClassifier:
    """A wrapper for the scikit-learn K-Nearest Neighbors Classifier."""

    def __init__(self, n_neighbors):
        # Initialize the scikit-learn model with K neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        """Train the model on the input features (X) and labels (y)."""
        self.model.fit(X, y)
        # Store training data for potential visualization use
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Make class predictions for the given data points."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Calculates the probability estimates for each class. 
        Necessary for generating the smooth decision boundary plot.
        """
        return self.model.predict_proba(X)
