from sklearn.datasets import make_classification
import pandas as pd

def generate_sample_classification_data(
    n_samples: int = 100,
    n_features: int = 4,
    n_informative: int = 2,
    n_classes: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic classification dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) to generate.
    n_features : int
        Total number of input features.
    n_informative : int
        Number of informative features used to build the target variable.
    n_classes : int
        Number of classes for classification.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame with feature columns (X1, X2, …) and target column 'y'.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=random_state
    )

    feature_cols = [f"X{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = y
    return df
