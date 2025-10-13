"""
Support Vector Machine (SVM) Model Implementation
Author: Akshit
Date: October 13, 2025
Purpose: SVM classifier and regressor with visualization for ML Simulator
"""

from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report,
                            mean_squared_error, r2_score, mean_absolute_error)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class SVMModel:
    """
    Support Vector Machine wrapper for classification and regression
    """
    
    def __init__(self, task='classification', kernel='rbf', C=1.0, gamma='scale',
                 degree=3, epsilon=0.1, random_state=42):
        """
        Initialize SVM model
        
        Parameters:
        -----------
        task : str
            'classification' or 'regression'
        kernel : str
            'linear', 'poly', 'rbf', 'sigmoid'
        C : float
            Regularization parameter (smaller = more regularization)
        gamma : str or float
            Kernel coefficient ('scale', 'auto', or float)
        degree : int
            Degree for polynomial kernel
        epsilon : float
            Epsilon in epsilon-SVR (for regression)
        random_state : int
            Random seed
        """
        self.task = task
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.epsilon = epsilon
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        if task == 'classification':
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                random_state=random_state,
                probability=True  # Enable probability estimates
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                epsilon=epsilon
            )
    
    def train(self, X, y, test_size=0.2, scale_features=True):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        y : array-like or Series
            Target variable
        test_size : float
            Proportion of test set
        scale_features : bool
            Whether to scale features (highly recommended for SVM)
        
        Returns:
        --------
        dict : Training results including metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features (critical for SVM)
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)
        
        # Calculate metrics
        if self.task == 'classification':
            results = self._classification_metrics(
                y_train, y_train_pred, y_test, y_pred, X_test_scaled
            )
        else:
            results = self._regression_metrics(
                y_train, y_train_pred, y_test, y_pred
            )
        
        # Add SVM-specific information
        results['n_support_vectors'] = self.model.n_support_
        results['support_vectors'] = self.model.support_vectors_
        
        if self.task == 'classification':
            results['n_support_per_class'] = dict(zip(
                self.model.classes_, 
                self.model.n_support_
            ))
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        return results
    
    def _classification_metrics(self, y_train, y_train_pred, y_test, y_pred, X_test):
        """Calculate classification metrics"""
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
    
    def plot_decision_boundary(self, feature_idx=[0, 1], resolution=100):
        """
        Plot decision boundary and support vectors (2D only)
        
        Parameters:
        -----------
        feature_idx : list
            Indices of two features to plot
        resolution : int
            Resolution of mesh grid
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.task != 'classification':
            raise ValueError("Decision boundary plot only available for classification")
        
        if len(feature_idx) != 2:
            raise ValueError("Can only plot 2D decision boundary")
        
        # Extract 2D data
        X_plot = self.X_train[:, feature_idx]
        y_plot = self.y_train
        
        # Create mesh
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Create full feature array for prediction
        n_features = self.X_train.shape[1]
        mesh_samples = np.zeros((xx.ravel().shape[0], n_features))
        mesh_samples[:, feature_idx] = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict on mesh
        Z = self.model.predict(mesh_samples)
        Z = Z.reshape(xx.shape)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot decision boundary
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='black', linewidths=1, alpha=0.5)
        
        # Plot data points
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot,
                           cmap=cmap_bold, edgecolor='black', s=80, alpha=0.8)
        
        # Plot support vectors
        support_vectors = self.model.support_vectors_[:, feature_idx]
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                  s=200, facecolors='none', edgecolors='black', linewidths=2.5,
                  label=f'Support Vectors (n={len(support_vectors)})')
        
        # Labels
        feature_names = self.feature_names if self.feature_names else ['Feature 1', 'Feature 2']
        ax.set_xlabel(feature_names[feature_idx[0]], fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_names[feature_idx[1]], fontsize=12, fontweight='bold')
        ax.set_title(f'SVM Decision Boundary ({self.kernel.upper()} kernel, C={self.C})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_support_vectors_detail(self, feature_idx=[0, 1]):
        """
        Detailed plot of support vectors and margins
        
        Parameters:
        -----------
        feature_idx : list
            Indices of two features to plot
        
        Returns:
        --------
        matplotlib.figure.Figure
        """
        if self.task != 'classification' or self.kernel != 'linear':
            raise ValueError("Margin plot only available for linear SVM classification")
        
        X_plot = self.X_train[:, feature_idx]
        y_plot = self.y_train
        
        # Get the separating hyperplane
        w = self.model.coef_[0]
        b = self.model.intercept_[0]
        
        # Calculate slope and intercept
        slope = -w[feature_idx[0]] / w[feature_idx[1]]
        intercept = -b / w[feature_idx[1]]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot data points
        for class_value in np.unique(y_plot):
            mask = y_plot == class_value
            ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                      label=f'Class {class_value}', s=80, alpha=0.7, edgecolor='black')
        
        # Plot support vectors
        support_vectors = self.model.support_vectors_[:, feature_idx]
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
                  s=250, facecolors='none', edgecolors='red', linewidths=3,
                  label=f'Support Vectors (n={len(support_vectors)})')
        
        # Plot decision boundary and margins
        x_vals = np.array(ax.get_xlim())
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, 'k-', linewidth=2, label='Decision Boundary')
        
        # Plot margins (parallel lines through support vectors)
        margin = 1 / np.sqrt(np.sum(w ** 2))
        ax.plot(x_vals, y_vals + margin, 'k--', linewidth=1.5, alpha=0.5, label='Positive Margin')
        ax.plot(x_vals, y_vals - margin, 'k--', linewidth=1.5, alpha=0.5, label='Negative Margin')
        
        # Styling
        feature_names = self.feature_names if self.feature_names else ['Feature 1', 'Feature 2']
        ax.set_xlabel(feature_names[feature_idx[0]], fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_names[feature_idx[1]], fontsize=12, fontweight='bold')
        ax.set_title(f'SVM Hyperplane and Support Vectors (C={self.C})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def predict(self, X, scale=True):
        """Make predictions on new data"""
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.model.predict(X_scaled)
    
    def get_model_info(self):
        """Get detailed model information"""
        info = {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'n_support_vectors': int(np.sum(self.model.n_support_)),
        }
        
        if self.task == 'classification':
            info['classes'] = self.model.classes_.tolist()
            info['support_per_class'] = self.model.n_support_.tolist()
        
        if self.kernel == 'poly':
            info['degree'] = self.degree
        
        if self.task == 'regression':
            info['epsilon'] = self.epsilon
        
        return info
