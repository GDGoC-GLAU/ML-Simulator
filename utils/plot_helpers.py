import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

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


def plot_feature_importance(feature_names, importances):
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances[indices],
        y=np.array(feature_names)[indices],
        orient="h",
        palette="viridis",
    )
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    return plt
