import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, annotate=True, cmap="Blues"):
    """
    Plots a customizable confusion matrix.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): Class names to display on axes.
        annotate (bool): Whether to show cell values.
        cmap (str): Colormap for heatmap (e.g. 'Blues', 'Greens', 'Oranges').

    Returns:
        fig (matplotlib.figure.Figure): The confusion matrix figure.
    """

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_style("whitegrid")

    sns.heatmap(
        cm,
        annot=annotate,
        fmt="d" if annotate else "",
        cmap=cmap,
        cbar=False,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_xlabel("Predicted Labels", fontsize=11)
    ax.set_ylabel("True Labels", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=12)

    plt.tight_layout()
    return fig
