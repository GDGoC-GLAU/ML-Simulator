import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.plot_helpers import plot_regression_line

st.header("📈 Linear Regression Simulator")

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Show results
st.subheader("Regression Line")
fig = plot_regression_line(X, y, model)
st.pyplot(fig)
