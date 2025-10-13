from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

class DecisionTreeModelClassifier:
    def __init__(self, max_depth=None, criterion="gini"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class DecisionTreeModelRegressor:
    def __init__(self, max_depth=None, criterion="squared_error"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, criterion=self.criterion)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)  # RMSE
