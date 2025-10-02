# pages/Logistic_Regression.py

import numpy as np
import matplotlib.pyplot as plt

# ----- Mock Training Data -----
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# ----- Mock Model Coefficients -----
weights = np.array([0.5, 0.5])
bias = -2

# ----- Sigmoid Function -----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ----- Predictions -----
def predict(X):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

y_pred = predict(X)
y_class = (y_pred >= 0.5).astype(int)
print("Predicted probabilities:", y_pred)
print("Predicted classes:", y_class)

# ----- Confusion Matrix -----
def confusion_matrix(y_true, y_pred_class):
    TP = np.sum((y_true == 1) & (y_pred_class == 1))
    TN = np.sum((y_true == 0) & (y_pred_class == 0))
    FP = np.sum((y_true == 0) & (y_pred_class == 1))
    FN = np.sum((y_true == 1) & (y_pred_class == 0))
    return np.array([[TP, FP], [FN, TN]])

cm = confusion_matrix(y, y_class)
print("Confusion Matrix:\n", cm)

# ----- ROC Curve -----
thresholds = np.linspace(0, 1, 100)
tpr = []
fpr = []

for t in thresholds:
    y_th_class = (y_pred >= t).astype(int)
    TP = np.sum((y == 1) & (y_th_class == 1))
    TN = np.sum((y == 0) & (y_th_class == 0))
    FP = np.sum((y == 0) & (y_th_class == 1))
    FN = np.sum((y == 1) & (y_th_class == 0))
    tpr.append(TP / (TP + FN))
    fpr.append(FP / (FP + TN))

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Mock)")
plt.legend()
plt.show()
