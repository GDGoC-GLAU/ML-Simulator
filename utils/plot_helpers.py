import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    """
    Plots the ROC curve given true labels and predicted scores.

    Parameters:
        y_true (array-like): Ground truth binary labels (0 or 1)
        y_scores (array-like): Predicted probabilities or decision scores

    Returns:
        fig (matplotlib.figure.Figure): The ROC curve figure object
    """

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_style("whitegrid")

    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess")

    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    return fig
