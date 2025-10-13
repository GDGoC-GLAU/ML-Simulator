"""
Data Helper Functions
Author: Akshit
"""

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
import pandas as pd

def get_sample_dataset(name='iris'):
    """
    Load sample datasets
    
    Parameters:
    -----------
    name : str
        Dataset name: 'iris', 'diabetes', 'breast_cancer', 'wine'
    
    Returns:
    --------
    pd.DataFrame
    """
    if name == 'iris':
        data = load_iris()
    elif name == 'diabetes':
        data = load_diabetes()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'wine':
        data = load_wine()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    return df, data.feature_names, data.target_names if hasattr(data, 'target_names') else None
