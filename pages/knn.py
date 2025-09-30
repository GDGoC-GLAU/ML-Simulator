import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from models.knn import KNNClassifier 

# NOTE: In a final project, these helpers should ideally be imported 
# from project-wide utility files (e.g., from utils.data_helpers import generate_data).

def generate_data(n_samples=200, n_classes=2):
    """Generates a synthetic non-linear dataset using make_moons."""
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return X, y

def plot_decision_boundary(model, X, y, title):
    """Plots the decision boundary based on model prediction probabilities."""
    
    # Define plot bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create mesh grid for contour plotting
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict class probabilities across the grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Initialize and plot figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the decision boundary contour
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)

    # Plot the original training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.close(fig) 
    return fig


def run_knn_simulator():
    st.title("K-Nearest Neighbors (KNN) Simulator")
    st.markdown("Interact with the slider to see how the number of neighbors ($K$) affects the decision boundary.")
    
    st.sidebar.header("Model Parameters")
    
    k_value = st.sidebar.slider(
        "Select number of neighbors ($K$)",
        min_value=1,
        max_value=25, 
        value=5,      
        step=2        # Ensures K is odd for binary classification
    )
    
    # Load data
    X, y = generate_data(n_samples=200, n_classes=2)
    
    # Train model
    model = KNNClassifier(n_neighbors=k_value)
    model.fit(X, y)
    
    # Display performance metric
    accuracy = model.model.score(X, y)
    st.sidebar.metric(label="Accuracy", value=f"{accuracy:.2f}")

    # Plotting section
    st.subheader(f"Decision Boundary with K = {k_value}")
    
    plot_figure = plot_decision_boundary(
        model, 
        X, 
        y, 
        title=f"KNN Decision Boundary (K={k_value})"
    )
    st.pyplot(plot_figure)

run_knn_simulator()