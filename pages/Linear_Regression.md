# pages/Logistic_Regression.py

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import existing helpers
from utils.data_helpers import generate_classification_dataset
from utils.plot_helpers import plot_roc_curve

# -------------------------------
# 🏷️ Page Configuration
# -------------------------------
st.set_page_config(page_title="Logistic Regression Simulator", layout="wide")
st.title("🔹 Logistic Regression Model")

st.write("""
This page trains a **Logistic Regression** model on a generated dataset,  
displays **predictions**, a **confusion matrix**, and an **ROC curve**.
""")

# -------------------------------
# ⚙️ Sidebar Controls
# -------------------------------
st.sidebar.header("Dataset Configuration")
n_samples = st.sidebar.slider("Number of Samples", 50, 1000, 200, 50)
n_features = st.sidebar.slider("Number of Features", 2, 20, 5)
n_informative = st.sidebar.slider("Informative Features", 1, n_features, 3)
n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2)

# Generate dataset
data = generate_classification_dataset(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_classes=n_classes
)

st.subheader("📊 Sample of Generated Dataset")
st.dataframe(data.head())

# -------------------------------
# 🧠 Model Training
# -------------------------------
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

st.subheader("⚙️ Model Training")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.success("✅ Model trained successfully!")

# -------------------------------
# 🔮 Predictions
# -------------------------------
st.subheader("🔮 Predictions on Test Set")
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1] if n_classes == 2 else None

st.write("**Sample Predictions:**")
pred_df = X_test.copy()
pred_df["Actual"] = y_test.values
pred_df["Predicted"] = y_pred
st.dataframe(pred_df.head(10))

# -------------------------------
# 📉 Confusion Matrix
# -------------------------------
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# -------------------------------
# 📈 ROC Curve (only for binary classification)
# -------------------------------
if n_classes == 2:
    st.subheader("📈 ROC Curve")
    roc_fig = plot_roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
    st.pyplot(roc_fig)
else:
    st.info("ROC Curve is only available for binary classification.")







