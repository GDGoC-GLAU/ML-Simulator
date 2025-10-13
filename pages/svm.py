import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from models.svm import SVMClassifier, SVMRegressor

st.title("🔷 Support Vector Machine (SVM) Simulator")

option = st.radio("Choose Mode", ["Classification", "Regression"])

if option == "Classification":
    st.subheader("SVM Classifier Visualization")

    # Load toy dataset
    X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)

    model = SVMClassifier(kernel=kernel, C=C)
    model.train(X, y)

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.scatter(model.get_support_vectors()[:, 0], model.get_support_vectors()[:, 1],
                s=100, facecolors='none', edgecolors='yellow', label='Support Vectors')
    plt.legend()
    st.pyplot(plt)
    st.write(f"**Accuracy:** {model.evaluate(X, y):.2f}")

else:
    st.subheader("SVM Regressor Visualization")

    # Generate regression dataset
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
    epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1)

    model = SVMRegressor(kernel=kernel, C=C, epsilon=epsilon)
    model.train(X, y)
    y_pred = model.predict(X)

    # Plot regression curve
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="SVM Prediction")
    plt.title("SVM Regression")
    plt.legend()
    st.pyplot(plt)
    st.write(f"**RMSE:** {model.evaluate(X, y):.3f}")
