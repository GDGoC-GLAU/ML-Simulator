"""
Decision Tree Model Implementation
Author: Akshit
Date: October 13, 2025
Purpose: Decision Tree classifier and regressor for ML Simulator
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            mean_squared_error, r2_score, mean_absolute_error)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DecisionTreeModel:
    """
    Decision Tree wrapper for classification and regression
    """
    
    def __init__(self, task='classification', max_depth=5, min_samples_split=2, 
                 min_samples_leaf=1, criterion='gini', random_state=42):
        """
        Initialize Decision Tree model
        
        Parameters:
        -----------
        task : str
            'classification' or 'regression'
        max_depth : int
            Maximum depth of the tree
        min_samples_split : int
            Minimum samples required to split
        min_samples_leaf : int
            Minimum samples required in leaf node
        criterion : str
            'gini' or 'entropy' for classification, 'mse' or 'mae' for regression
        random_state : int
            Random seed for reproducibility
        """
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        if task == 'classification':
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=random_state
            )
        else:
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion if criterion in ['mse', 'mae'] else 'squared_error',
                random_state=random_state
            )
    
    def train(self, X, y, test_size=0.2):
        """
        Train the Decision Tree model
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        y : array-like or Series
            Target variable
        test_size : float
            Proportion of test set
        
        Returns:
        --------
        dict : Training results including metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_train_pred = self.model.predict(X_train)
        
        # Calculate metrics based on task
        if self.task == 'classification':
            results = self._classification_metrics(
                y_train, y_train_pred, y_test, y_pred, X_test
            )
        else:
            results = self._regression_metrics(
                y_train, y_train_pred, y_test, y_pred
            )
        
        # Add feature importance
        results['feature_importance'] = self.model.feature_importances_
        results['n_nodes'] = self.model.tree_.node_count
        results['n_leaves'] = self.model.get_n_leaves()
        results['max_depth_achieved'] = self.model.get_depth()
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        
        return results
    
    def _classification_metrics(self, y_train, y_train_pred, y_test, y_pred, X_test):
        """Calculate classification metrics"""
        
        # Probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        
        return {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    def _regression_metrics(self, y_train, y_train_pred, y_test, y_pred):
        """Calculate regression metrics"""
        return {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    def plot_tree_structure(self, feature_names=None, class_names=None, max_depth_display=3):
        """
        Visualize the decision tree structure
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        class_names : list
            Names of classes (for classification)
        max_depth_display : int
            Maximum depth to display
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        
        plot_tree(self.model, 
                 filled=True,
                 feature_names=feature_names,
                 class_names=class_names,
                 rounded=True,
                 fontsize=10,
                 max_depth=max_depth_display,
                 ax=ax)
        
        plt.title(f'Decision Tree Structure (Max Depth: {self.max_depth})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance as DataFrame"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
