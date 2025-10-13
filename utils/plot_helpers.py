# utils/plot_helpers.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score):
    """
    Plot ROC curve for a classification model.

    Parameters
    ----------
    y_true : array-like
        True class labels (0 or 1).
    y_score : array-like
        Predicted probabilities or scores for the positive class.

    Returns
    -------
    fig : matplotlib.figure.Figure
        ROC curve figure object (for Streamlit display).
    """

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create figure
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

    ax.set_title("ROC Curve", fontsize=14)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    return fig
