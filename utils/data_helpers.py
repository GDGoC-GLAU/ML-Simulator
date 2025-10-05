# utils/data_helpers.py

from sklearn.datasets import make_regression
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
