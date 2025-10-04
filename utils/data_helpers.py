import numpy as np
from sklearn.datasets import make_classification, make_regression

def sample_regression_data():
    X, y = make_regression(n_samples=50, n_features=1, noise=10, random_state=42)
    return X, y

def sample_classification_data():
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, class_sep=0.5, flip_y=0.1, random_state=42)
    return X, y
