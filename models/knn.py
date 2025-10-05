from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_dataset(name="Iris"):
    """Load a dataset by name."""
    if name.lower() == "iris":
        data = load_iris()
    elif name.lower() == "wine":
        data = load_wine()
    elif name.lower() == "digits":
        data = load_digits()
    else:
        raise ValueError("Unsupported dataset")
    return data.data, data.target, data.target_names

def train_knn(X, y, k=5, test_size=0.3):
    """Train and evaluate KNN model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, X_test, y_test, y_pred, accuracy
