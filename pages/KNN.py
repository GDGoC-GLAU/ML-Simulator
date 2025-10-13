import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from models.knn import KNNClassifier, KNNRegressor
from utils.plot_helpers import plot_confusion_matrix

st.title("🧩 K-Nearest Neighbors (KNN) Simulator")

mode = st.radio("Choose Mode", ["Classification", "Regression"])

if mode == "Classification":
    st.subheader("KNN Classifier Visualization")

    # Load dataset
    X, y = datasets.make_classification(
        n_samples=150, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )

    k = st.slider("Number of Neighbors (k)", 1, 15, 3)
    model = KNNClassifier(n_neighbors=k)
    model.train(X, y)
    y_pred = model.predict(X)

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(f"KNN Classifier (k={k})")
    st.pyplot(plt)

    st.write(f"**Accuracy:** {model.evaluate(X, y):.2f}")

    # Confusion matrix
    fig = plot_confusion_matrix(y, y_pred, labels=["Class 0", "Class 1"], cmap="Purples")
    st.pyplot(fig)

else:
    st.subheader("KNN Regressor Visualization")

    # Generate regression dataset
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    k = st.slider("Number of Neighbors (k)", 1, 15, 3)
    model = KNNRegressor(n_neighbors=k)
    model.train(X, y)
    y_pred = model.predict(X)

    # Plot regression
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label=f"KNN Prediction (k={k})")
    plt.title("KNN Regression")
    plt.legend()
    st.pyplot(plt)

    st.write(f"**RMSE:** {model.evaluate(X, y):.3f}")
