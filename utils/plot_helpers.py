import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_score, model_name="Model"):
    """
    Plots the ROC curve given true labels and predicted scores.
    
    Parameters:
        y_true (array-like): True binary labels (0/1)
        y_score (array-like): Predicted probabilities or scores for the positive class
        model_name (str): Name of the model for labeling the plot

    Returns:
        fig (matplotlib.figure.Figure): ROC curve figure for Streamlit display
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")

    plt.tight_layout()
    return fig
