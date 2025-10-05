import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

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


def plot_roc_curve(y_true, y_scores):
    """
    Plot ROC curve using true labels and predicted scores.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Model predicted probabilities or scores.

    Returns:
        plt: Matplotlib plot object for Streamlit display.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    return plt

def plot_feature_importance(importances, feature_names):
    plt.figure()
    sns.barplot(x=importances, y=feature_names)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    return plt