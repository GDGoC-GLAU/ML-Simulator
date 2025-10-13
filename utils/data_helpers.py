# utils/data_helpers.py
from sklearn.datasets import make_classification
import pandas as pd

def generate_classification_dataset(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    n_classes: int = 2,
    random_state: int = 42
):
    """
    Generate a synthetic classification dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate (default=100).
    n_features : int, optional
        Total number of features (default=10).
    n_informative : int, optional
        Number of informative features (default=5).
    n_classes : int, optional
        Number of target classes (default=2).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame containing the generated features and target column ('target').
    """

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data["target"] = y

    return data
