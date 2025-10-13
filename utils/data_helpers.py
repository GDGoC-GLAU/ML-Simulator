from sklearn.datasets import make_regression
import pandas as pd

def generate_sample_regression_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        Number of data samples (rows).
    n_features : int
        Number of input features (columns).
    noise : float
        Standard deviation of the Gaussian noise added to the output.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing feature columns (X1, X2, …) and a target column 'y'.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )

    # Convert to DataFrame
    feature_cols = [f"X{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = y

    return df
