import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, cmap="Blues", annotate=True, normalize=False, model_name="Model"):
    """
    Plots a confusion matrix with optional annotations and color customization.

    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list): List of class labels for axes
        cmap (str): Color map for the plot (default: 'Blues')
        annotate (bool): Whether to show cell values
        normalize (bool): Normalize values to show percentages
        model_name (str): Name of the model for the plot title

    Returns:
        fig (matplotlib.figure.Figure): Confusion matrix figure for Streamlit display
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=annotate,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=labels if labels is not None else sorted(set(y_true)),
        yticklabels=labels if labels is not None else sorted(set(y_true)),
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )
    
    ax.set_xlabel("Predicted Labels", fontsize=11)
    ax.set_ylabel("True Labels", fontsize=11)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13)
    plt.tight_layout()
    
    return fig
