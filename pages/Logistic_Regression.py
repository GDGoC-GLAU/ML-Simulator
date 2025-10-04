import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.plot_helpers import plot_roc_curve, plot_confusion_matrix
from utils.data_helpers import sample_classification_data

st.header("Logistic Regression Simulator")

# Sample classification data
X, y = sample_classification_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# 1. DATA VISUALIZATION
st.subheader("Data Visualization")

# Create scatter plot of the data
fig_data, ax = plt.subplots(figsize=(10, 6))
colors = ['red' if label == 0 else 'blue' for label in y]
scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=50)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Classification Data Visualization')
ax.legend(['Class 0', 'Class 1'], loc='upper right')

# Add decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', alpha=0.8)

st.pyplot(fig_data)

# 2. MODEL PERFORMANCE VISUALIZATION
st.subheader("Model Performance")

st.subheader("ROC Curve")
fig_roc = plot_roc_curve(y_test, y_scores)
st.pyplot(fig_roc)

st.subheader("Confusion Matrix")
fig_cm = plot_confusion_matrix(y_test, y_pred, labels=['Class 0', 'Class 1'])
st.pyplot(fig_cm)

# 3. PERFORMANCE METRICS
st.subheader("Performance Metrics")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

with metrics_col1:
    st.metric("Accuracy", f"{accuracy:.3f}")
    
with metrics_col2:
    st.metric("Precision", f"{precision:.3f}")
    
with metrics_col3:
    st.metric("Recall", f"{recall:.3f}")
    
with metrics_col4:
    st.metric("F1 Score", f"{f1:.3f}")

# 4. PREDICTION INTERFACE
st.subheader("Make Predictions")

# Input for new prediction
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    feature1 = st.number_input("Feature 1:", value=0.0, step=0.1, format="%.2f")
    
with col_pred2:
    feature2 = st.number_input("Feature 2:", value=0.0, step=0.1, format="%.2f")

# Make prediction
new_point = np.array([[feature1, feature2]])
prediction = model.predict(new_point)[0]
prediction_prob = model.predict_proba(new_point)[0]

# Display prediction results
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    st.metric("Predicted Class", f"Class {prediction}")
    
with pred_col2:
    st.metric("Probability Class 0", f"{prediction_prob[0]:.3f}")
    
with pred_col3:
    st.metric("Probability Class 1", f"{prediction_prob[1]:.3f}")

# Show prediction on plot
fig_pred, ax = plt.subplots(figsize=(10, 6))
colors = ['red' if label == 0 else 'blue' for label in y]
ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=50, label='Training Data')
ax.scatter([feature1], [feature2], color='green', s=200, marker='*', label=f'New Point (Class {prediction})')

# Add decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', alpha=0.8)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Logistic Regression with Prediction')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig_pred)

# 5. DATA INFORMATION
st.subheader("Dataset Information")
info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric("Training Samples", len(X_train))
    
with info_col2:
    st.metric("Test Samples", len(X_test))
    
with info_col3:
    st.metric("Features", X.shape[1])
    
with info_col4:
    st.metric("Classes", len(np.unique(y)))