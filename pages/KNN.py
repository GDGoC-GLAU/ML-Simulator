import streamlit as st
from models.knn import KNNModel
from utils.data_helpers import generate_classification_data, generate_sample_regression_data
from utils.plot_helpers import plot_classification_results, plot_regression_results
import pandas as pd

st.set_page_config(page_title="KNN Model", layout="wide")

st.title("🧠 K-Nearest Neighbors (KNN) Simulator")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
task_type = st.sidebar.selectbox("Select Task Type", ["classification", "regression"])
k_value = st.sidebar.slider("Select k (number of neighbors)", min_value=1, max_value=20, value=5, step=1)
n_samples = st.sidebar.slider("Number of samples", 50, 500, 200)
n_features = st.sidebar.slider("Number of features", 2, 10, 2)

# --- Generate Dataset ---
st.subheader("📊 Dataset Preview")

if task_type == "classification":
    X, y = generate_classification_data(n_samples=n_samples, n_features=n_features)
else:
    data = generate_sample_regression_data(n_samples=n_samples, n_features=n_features)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

df = pd.DataFrame(X)
df["target"] = y
st.dataframe(df.head())

# --- Train Model ---
model = KNNModel(task_type=task_type, k=k_value)
model.train(X, y)
score = model.score(X, y)

# --- Display Results ---
st.success(f"✅ Model trained successfully with k = {k_value}")
st.metric("Model Score", f"{score:.3f}")

# --- Visualizations ---
st.subheader("📈 Visualization")

if task_type == "classification":
    st.pyplot(plot_classification_results(X, y, model))
else:
    st.pyplot(plot_regression_results(X, y, model))
