import streamlit as st
import pandas as pd
from models.decision_tree import DecisionTreeModel
from utils.data_helpers import generate_sample_regression_data  # For regression data
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

st.title("🌳 Decision Tree Model Playground")

task_type = st.radio("Select Task Type", ["classification", "regression"])

# Sidebar parameters
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
n_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
n_features = st.sidebar.slider("Number of Features", 1, 5, 2)
noise = st.sidebar.slider("Noise (for regression)", 0.0, 10.0, 1.0)

st.write("### Generated Dataset Preview")

# Generate dataset
if task_type == "classification":
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_features,
        n_redundant=0, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_features)])
    df["y"] = y
else:
    df = generate_sample_regression_data(n_samples, n_features, noise)

st.dataframe(df.head())

# Train model
st.write("### Train Model")
model = DecisionTreeModel(task_type=task_type, max_depth=max_depth)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
model.train(X, y)

# Evaluate
metrics = model.evaluate(X, y)
st.write("### Evaluation Metrics")
st.json(metrics)

# Visualization
st.write("### Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(model.get_model(), filled=True, feature_names=X.columns)
st.pyplot(fig)
