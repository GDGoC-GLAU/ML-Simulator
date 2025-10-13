"""
Plot Helper Functions for ML Simulator
Author: Akshit
Date: October 13, 2025
Purpose: Utility functions for creating matplotlib/seaborn plots for Streamlit
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import io
import base64

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", return_fig=True):
    """
    Plot ROC curve with AUC score
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred_proba : array-like
        Predicted probabilities for positive class
    title : str, optional
        Plot title (default: "ROC Curve")
    return_fig : bool, optional
        If True, returns matplotlib figure object
        If False, returns base64 encoded image string
    
    Returns:
    --------
    matplotlib.figure.Figure or str
        Figure object or base64 encoded PNG string
    
    Example:
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    >>> y_pred_proba = model.predict_proba(X)[:, 1]
    >>> fig = plot_roc_curve(y, y_pred_proba)
    >>> import streamlit as st
    >>> st.pyplot(fig)
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#0984e3', linewidth=3, 
            label=f'ROC Curve (AUC = {roc_auc:.3f})', marker='o', 
            markersize=4, markevery=20)
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='#d63031', linestyle='--', 
            linewidth=2, label='Random Classifier (AUC = 0.5)')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#74b9ff')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add AUC score annotation
    ax.text(0.6, 0.2, f'AUC Score\n{roc_auc:.4f}', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        # Convert to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        return img_str


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plot confusion matrix heatmap
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for Streamlit display
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                xticklabels=labels if labels else ['0', '1'],
                yticklabels=labels if labels else ['0', '1'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importance_scores, top_n=10, title="Feature Importance"):
    """
    Plot horizontal bar chart of feature importance
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_scores : array-like
        Importance scores for each feature
    top_n : int, optional
        Number of top features to display (default: 10)
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for Streamlit display
    """
    
    # Sort by importance
    indices = np.argsort(np.abs(importance_scores))[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
    
    # Color based on positive/negative
    colors = ['#00b894' if score > 0 else '#ff7675' for score in top_scores]
    
    # Plot horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_scores, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(score + 0.01 if score > 0 else score - 0.01, i, 
                f'{score:.3f}', va='center', 
                ha='left' if score > 0 else 'right',
                fontweight='bold', fontsize=10)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(y_pred_proba, y_true, title="Prediction Probability Distribution"):
    """
    Plot histogram of prediction probabilities by true class
    
    Parameters:
    -----------
    y_pred_proba : array-like
        Predicted probabilities
    y_true : array-like
        True labels
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for Streamlit display
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predictions by true class
    proba_class_0 = y_pred_proba[y_true == 0]
    proba_class_1 = y_pred_proba[y_true == 1]
    
    # Plot histograms
    ax.hist(proba_class_0, bins=30, alpha=0.6, color='#ff7675', 
            label='Actual Class 0', edgecolor='black')
    ax.hist(proba_class_1, bins=30, alpha=0.6, color='#74b9ff', 
            label='Actual Class 1', edgecolor='black')
    
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, 
               label='Decision Threshold (0.5)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="Residual Plot"):
    """
    Plot residuals for regression models
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for Streamlit display
    """
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual scatter plot
    ax1.scatter(y_pred, residuals, alpha=0.6, color='#6c5ce7', edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Residual distribution
    ax2.hist(residuals, bins=30, alpha=0.7, color='#00b894', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values for regression
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object for Streamlit display
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, color='#0984e3', 
               edgecolor='black', s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


# Example usage function for testing
def test_plot_helpers():
    """Test function to demonstrate usage"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create plots
    roc_fig = plot_roc_curve(y_test, y_pred_proba)
    cm_fig = plot_confusion_matrix(y_test, y_pred)
    fi_fig = plot_feature_importance(
        [f'Feature {i}' for i in range(10)],
        model.coef_[0]
    )
    
    print("✅ All plot functions working correctly!")
    return roc_fig, cm_fig, fi_fig


if __name__ == "__main__":
    test_plot_helpers()
