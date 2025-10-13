"""
Data Helper Functions for ML Simulator
Author: Akshit
Date: October 13, 2025
Purpose: Generate and load sample datasets for ML training
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (make_regression, make_classification,
                              load_iris, load_diabetes, load_breast_cancer,
                              load_wine, load_boston)
from sklearn.preprocessing import StandardScaler

def generate_regression_dataset(n_samples=1000, n_features=10, noise=10.0,
                                n_informative=5, random_state=42, 
                                include_bias=True, return_dataframe=True):
    """
    Generate synthetic regression dataset for training and testing
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=10
        Total number of features
    noise : float, default=10.0
        Standard deviation of Gaussian noise added to output
        Higher values = more noise, harder to predict
    n_informative : int, default=5
        Number of informative features
        Remaining features will be noise
    random_state : int, default=42
        Random seed for reproducibility
    include_bias : bool, default=True
        If True, adds a bias/intercept term
    return_dataframe : bool, default=True
        If True, returns pandas DataFrame, else numpy arrays
    
    Returns:
    --------
    X : DataFrame or ndarray
        Feature matrix (n_samples, n_features)
    y : Series or ndarray
        Target variable (n_samples,)
    
    Examples:
    ---------
    >>> # Generate basic dataset
    >>> X, y = generate_regression_dataset(n_samples=500, n_features=5)
    >>> print(X.shape, y.shape)
    (500, 5) (500,)
    
    >>> # Generate noisy dataset
    >>> X, y = generate_regression_dataset(n_samples=1000, noise=50.0)
    
    >>> # Generate dataset with many features
    >>> X, y = generate_regression_dataset(n_features=20, n_informative=10)
    """
    
    # Validate parameters
    if n_informative > n_features:
        raise ValueError(f"n_informative ({n_informative}) cannot be greater than n_features ({n_features})")
    
    if n_samples < 10:
        raise ValueError(f"n_samples must be at least 10, got {n_samples}")
    
    # Generate regression data
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        bias=0.0 if not include_bias else 100.0,
        random_state=random_state,
        coef=True
    )
    
    if return_dataframe:
        # Create DataFrame with meaningful column names
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Add metadata
        X_df.attrs['n_informative'] = n_informative
        X_df.attrs['noise_level'] = noise
        X_df.attrs['true_coefficients'] = coef
        
        return X_df, y_series
    else:
        return X, y


def generate_regression_dataset_with_nonlinearity(n_samples=1000, n_features=5,
                                                  noise=10.0, degree=2,
                                                  random_state=42):
    """
    Generate regression dataset with non-linear relationships
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of base features
    noise : float
        Noise level
    degree : int
        Polynomial degree (1=linear, 2=quadratic, etc.)
    random_state : int
        Random seed
    
    Returns:
    --------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable with non-linear relationships
    """
    np.random.seed(random_state)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear target
    y = np.zeros(n_samples)
    for i in range(n_features):
        coef = np.random.randn()
        y += coef * (X[:, i] ** degree)
    
    # Add noise
    y += np.random.normal(0, noise, n_samples)
    
    # Convert to DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def generate_regression_dataset_with_outliers(n_samples=1000, n_features=10,
                                              outlier_fraction=0.1, noise=10.0,
                                              random_state=42):
    """
    Generate regression dataset with outliers
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    outlier_fraction : float
        Fraction of outliers (0.0 to 1.0)
    noise : float
        Base noise level
    random_state : int
        Random seed
    
    Returns:
    --------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable with outliers
    is_outlier : Series
        Boolean mask indicating outliers
    """
    X, y = generate_regression_dataset(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    
    # Add outliers
    np.random.seed(random_state)
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    # Make outliers deviate significantly
    y_mean = y.mean()
    y_std = y.std()
    y.iloc[outlier_indices] += np.random.choice([-1, 1], n_outliers) * (3 * y_std)
    
    # Create outlier mask
    is_outlier = pd.Series(False, index=range(n_samples), name='is_outlier')
    is_outlier.iloc[outlier_indices] = True
    
    return X, y, is_outlier


def generate_time_series_regression(n_samples=1000, trend='linear',
                                    seasonality=True, noise=5.0,
                                    random_state=42):
    """
    Generate time series regression dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of time points
    trend : str
        'linear', 'quadratic', or 'exponential'
    seasonality : bool
        Whether to include seasonal component
    noise : float
        Noise level
    random_state : int
        Random seed
    
    Returns:
    --------
    X : DataFrame
        Features including time, lagged values, etc.
    y : Series
        Target time series
    """
    np.random.seed(random_state)
    
    # Time index
    t = np.arange(n_samples)
    
    # Trend component
    if trend == 'linear':
        trend_component = 0.5 * t
    elif trend == 'quadratic':
        trend_component = 0.001 * (t ** 2)
    elif trend == 'exponential':
        trend_component = np.exp(0.001 * t)
    else:
        trend_component = np.zeros(n_samples)
    
    # Seasonal component
    if seasonality:
        seasonal_component = 10 * np.sin(2 * np.pi * t / 50)
    else:
        seasonal_component = np.zeros(n_samples)
    
    # Combine and add noise
    y = trend_component + seasonal_component + np.random.normal(0, noise, n_samples)
    
    # Create features
    X_data = {
        'time': t,
        'trend': trend_component,
        'sin_component': np.sin(2 * np.pi * t / 50),
        'cos_component': np.cos(2 * np.pi * t / 50),
    }
    
    # Add lagged features
    y_series = pd.Series(y)
    for lag in [1, 2, 5, 10]:
        X_data[f'lag_{lag}'] = y_series.shift(lag).fillna(0).values
    
    X_df = pd.DataFrame(X_data)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def load_sample_regression_datasets():
    """
    Load built-in regression datasets from sklearn
    
    Returns:
    --------
    dict : Dictionary of datasets with metadata
    """
    datasets = {}
    
    # Diabetes dataset
    diabetes = load_diabetes()
    datasets['diabetes'] = {
        'X': pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
        'y': pd.Series(diabetes.target, name='progression'),
        'description': 'Diabetes progression prediction (442 samples, 10 features)',
        'task': 'regression',
        'n_samples': 442,
        'n_features': 10
    }
    
    # California Housing (if available)
    try:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        datasets['california_housing'] = {
            'X': pd.DataFrame(housing.data, columns=housing.feature_names),
            'y': pd.Series(housing.target, name='median_house_value'),
            'description': 'California housing prices (20640 samples, 8 features)',
            'task': 'regression',
            'n_samples': 20640,
            'n_features': 8
        }
    except:
        pass
    
    return datasets


def get_dataset_by_name(name='diabetes', scaled=False):
    """
    Get dataset by name with optional scaling
    
    Parameters:
    -----------
    name : str
        Dataset name: 'diabetes', 'california_housing', 'generated'
    scaled : bool
        Whether to scale features
    
    Returns:
    --------
    X : DataFrame
        Features
    y : Series
        Target
    """
    if name == 'generated':
        X, y = generate_regression_dataset()
    elif name == 'diabetes':
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
    elif name == 'california_housing':
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if scaled:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y


def save_dataset_to_csv(X, y, filename='regression_dataset.csv'):
    """
    Save generated dataset to CSV file
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target
    filename : str
        Output filename
    """
    df = X.copy()
    df['target'] = y
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved to {filename}")
    return filename


# Test function
def test_data_helpers():
    """Test all dataset generation functions"""
    print("🧪 Testing data helper functions...\n")
    
    # Test 1: Basic regression dataset
    print("1. Basic regression dataset:")
    X, y = generate_regression_dataset(n_samples=100, n_features=5)
    print(f"   Shape: X={X.shape}, y={y.shape}")
    print(f"   Features: {list(X.columns)}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]\n")
    
    # Test 2: Noisy dataset
    print("2. Noisy regression dataset:")
    X, y = generate_regression_dataset(n_samples=100, noise=50.0)
    print(f"   Noise level: 50.0")
    print(f"   Target std: {y.std():.2f}\n")
    
    # Test 3: Non-linear dataset
    print("3. Non-linear regression dataset:")
    X, y = generate_regression_dataset_with_nonlinearity(n_samples=100, degree=2)
    print(f"   Polynomial degree: 2")
    print(f"   Shape: {X.shape}\n")
    
    # Test 4: Dataset with outliers
    print("4. Dataset with outliers:")
    X, y, outliers = generate_regression_dataset_with_outliers(n_samples=100, outlier_fraction=0.1)
    print(f"   Total outliers: {outliers.sum()}")
    print(f"   Outlier percentage: {(outliers.sum()/len(outliers))*100:.1f}%\n")
    
    # Test 5: Time series
    print("5. Time series regression:")
    X, y = generate_time_series_regression(n_samples=100)
    print(f"   Shape: {X.shape}")
    print(f"   Features: {list(X.columns)}\n")
    
    # Test 6: Load sample datasets
    print("6. Sample datasets:")
    datasets = load_sample_regression_datasets()
    for name, data in datasets.items():
        print(f"   - {name}: {data['description']}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_data_helpers()
 