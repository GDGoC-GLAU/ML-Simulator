# utils/data_helpers.py

from sklearn.datasets import make_classification, make_regression
import pandas as pd

def generate_sample_regression(n_samples=100, n_features=1, noise=0.0, random_state=None):
    """
    Generate a sample regression dataset.

    Parameters:
        n_samples (int): Number of data points.
        n_features (int): Number of features.
        noise (float): Standard deviation of Gaussian noise added to the output.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        X (pd.DataFrame): Feature dataframe of shape (n_samples, n_features)
        y (pd.Series): Target variable of shape (n_samples,)
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    # Convert to pandas for convenience
    X_df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def generate_sample_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    class_sep=1.0,
    flip_y=0.01,
    random_state=None,
):
    """
    Generate a sample classification dataset.

    Parameters:
        n_samples (int): Number of data points.
        n_features (int): Total number of features.
        n_informative (int): Number of informative features.
        n_redundant (int): Number of redundant features.
        class_sep (float): Separation between classes.
        flip_y (float): Fraction of labels to randomly flip.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        X (pd.DataFrame): Feature dataframe of shape (n_samples, n_features)
        y (pd.Series): Target variable of shape (n_samples,)
    """
    informative = min(n_informative, n_features)
    redundant = min(n_redundant, max(n_features - informative, 0))
    repeated = max(n_features - informative - redundant, 0)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=repeated,
        n_classes=2,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")

    return X_df, y_series
