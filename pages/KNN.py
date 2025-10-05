import streamlit as st
import matplotlib.pyplot as plt
from models.knn import get_dataset, train_knn

st.set_page_config(page_title="KNN Simulator", page_icon="🤖")

st.title("🔍 K-Nearest Neighbors (KNN) Simulator")
st.write("Experiment with the KNN algorithm — adjust `k`, choose datasets, and visualize predictions.")

# Sidebar controls
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Digits"))
k_value = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
test_ratio = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.3)

# Load dataset
X, y, target_names = get_dataset(dataset_name)

# Train model
model, X_test, y_test, y_pred, acc = train_knn(X, y, k=k_value, test_size=test_ratio)

# Display results
st.subheader("📊 Model Performance")
st.metric("Accuracy", f"{acc * 100:.2f}%")

# Visualization
st.subheader("🎨 Prediction Visualization")
if X_test.shape[1] >= 2:
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis", edgecolor="k", alpha=0.7
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"KNN Visualization (k={k_value})")
    st.pyplot(fig)
else:
    st.write("This dataset has less than 2 features — visualization unavailable.")

st.caption("Tip: Increase or decrease `k` to see how accuracy and classification patterns change.")