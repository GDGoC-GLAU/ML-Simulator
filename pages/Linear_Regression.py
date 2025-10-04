import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.plot_helpers import plot_regression_line
from utils.data_helpers import sample_regression_data

st.header("Linear Regression Simulator")

# Use sample data from data_helpers for consistency
X, y = sample_regression_data()

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# 1. DATA VISUALIZATION
st.subheader("Data Visualization")

# Create scatter plot of the original data
fig_data, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color="blue", alpha=0.7, s=100, label="Data Points")
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.set_title('Linear Regression Data Visualization')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig_data)

# 2. REGRESSION LINE VISUALIZATION
st.subheader("Regression Line")

fig_regression = plot_regression_line(X, y, model)
st.pyplot(fig_regression)

# 3. MODEL PERFORMANCE METRICS
st.subheader("Model Performance Metrics")

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

with metrics_col1:
    st.metric("Mean Squared Error", f"{mse:.3f}")
    
with metrics_col2:
    st.metric("R² Score", f"{r2:.3f}")
    
with metrics_col3:
    st.metric("Mean Absolute Error", f"{mae:.3f}")

# 4. MODEL COEFFICIENTS
st.subheader("Model Parameters")

coef = model.coef_[0]
intercept = model.intercept_

param_col1, param_col2 = st.columns(2)

with param_col1:
    st.metric("Slope (Coefficient)", f"{coef:.3f}")
    
with param_col2:
    st.metric("Intercept", f"{intercept:.3f}")

# Display the equation
st.subheader("Regression Equation")
st.latex(f"y = {coef:.3f}x + {intercept:.3f}")

# 5. DATA INFORMATION
st.subheader("Dataset Information")
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("Total Samples", len(X))
    
with info_col2:
    st.metric("Features", X.shape[1])
    
with info_col3:
    st.metric("Data Range", f"{y.min():.1f} - {y.max():.1f}")

# 6. PREDICTION INTERFACE
st.subheader("Make Predictions")

# Input for new prediction
new_value = st.number_input("Enter a value to predict:", value=6.0, step=0.1)
prediction = model.predict([[new_value]])[0]

st.write(f"**Prediction for x = {new_value}:** y = {prediction:.3f}")

# Show prediction on plot
fig_pred, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color="blue", alpha=0.7, s=100, label="Training Data")
ax.scatter([new_value], [prediction], color="red", s=150, label=f"Prediction (x={new_value})")
y_line = model.predict(X)
ax.plot(X, y_line, color="green", linewidth=2, label="Regression Line")
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.set_title('Linear Regression with Prediction')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig_pred)