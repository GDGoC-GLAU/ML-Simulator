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

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        labels: List of class labels (optional)
        title: Title for the plot (optional)
        
    Returns:
        matplotlib.pyplot: Figure object for Streamlit display
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    
    return plt

def plot_roc_curve(y_true, y_scores):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores or probabilities for the positive class
        
    Returns:
        matplotlib.pyplot: Figure object for Streamlit display
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    return plt
