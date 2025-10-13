from sklearn.datasets import make_classification, make_regression
import numpy as np

def get_classification_data():
    X, y = make_classification(
        n_samples=150, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    return X, y

def get_regression_data():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1
    return X, y
