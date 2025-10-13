import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from models.decision_tree import DecisionTreeModelClassifier, DecisionTreeModelRegressor
from utils.plot_helpers import plot_confusion_matrix
from utils.data_helpers import get_classification_data, get_regression_data

st.title("🌳 Decision Tree Simulator")

mode = st.radio("Choose Mode", ["Classification", "Regression"])

if mode == "Classification":
    st.subheader("Decision Tree Classifier")

    # Load dataset from utils
    X, y = get_classification_data()

    max_depth = st.slider("Max Depth", 1, 10, 3)
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])

    model = DecisionTreeModelClassifier(max_depth=max_depth, criterion=criterion)
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
    plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(f"Decision Boundary (Depth={max_depth}, Criterion={criterion})")
    st.pyplot(plt)

    # Confusion matrix
    fig = plot_confusion_matrix(y, y_pred, labels=["Class 0", "Class 1"], cmap="Oranges")
    st.pyplot(fig)

    st.write(f"**Accuracy:** {model.evaluate(X, y):.2f}")

else:
    st.subheader("Decision Tree Regressor")

    # Load dataset from utils
    X, y = get_regression_data()

    max_depth = st.slider("Max Depth", 1, 10, 3)
    criterion = st.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])

    model = DecisionTreeModelRegressor(max_depth=max_depth, criterion=criterion)
    model.train(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.title(f"Decision Tree Regression (Depth={max_depth})")
    plt.legend()
    st.pyplot(plt)

    st.write(f"**RMSE:** {model.evaluate(X, y):.3f}")
