import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_regression_line(X, y, model):
    plt.figure()
    plt.scatter(X, y, color="blue", label="Data")
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.legend()
    return plt

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    return plt

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    return plt


def plot_confusion_matrix(y_true, y_pred, 
                          labels=None, 
                          figsize=(8, 6), 
                          cmap='Blues', 
                          annot=True, 
                          fmt='d', 
                          title='Confusion Matrix'):
    """
    Plots a confusion matrix with optional annotations and label customization.
    
    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        labels (list, optional): List of labels to index the matrix. Defaults to unique labels in y_true.
        figsize (tuple, optional): Size of the figure. Defaults to (8, 6).
        cmap (str, optional): Colormap for the matrix. Defaults to 'Blues'.
        annot (bool, optional): Whether to annotate the cells. Defaults to True.
        fmt (str, optional): Format of annotations. Defaults to 'd' (integer).
        title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()