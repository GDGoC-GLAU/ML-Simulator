import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_support_vectors(self):
        return self.model.support_vectors_


class SVMRegressor:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)  # RMSE
