import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.header("📈 Linear Regression Simulator")

# User input for custom data points
st.subheader("Input your custom data points")
input_X = st.text_area(
    "Enter X values (comma-separated, e.g. 1,2,3)",
    value="1,2,3,4,5"
)
input_y = st.text_area(
    "Enter corresponding y values (comma-separated, e.g. 2,4,5,4,5)",
    value="2,4,5,4,5"
)

# Parse user input into numpy arrays safely
def parse_input(text):
    try:
        arr = np.array([float(i.strip()) for i in text.split(",")]).reshape(-1, 1)
        return arr
    except Exception as e:
        st.error(f"Invalid input: {e}")
        return None

X = parse_input(input_X)
y = parse_input(input_y)

def plot_regression_line(X, y, model):
    plt.figure(figsize=(8,5))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    fig = plt.gcf()
    return fig

if X is not None and y is not None and X.shape[0] == y.shape[0]:
    # Train model
    model = LinearRegression()
    model.fit(X, y.ravel())

    # Predict
    y_pred = model.predict(X)

    # Show results - Regression line
    st.subheader("Regression Line")
    fig1 = plot_regression_line(X, y, model)
    st.pyplot(fig1)

    # Show R² score
    r2_score = model.score(X, y)
    st.subheader(f"R² Score: {r2_score:.4f}")

    # Plot residuals
    residuals = y.ravel() - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    fig2 = plt.gcf()
    st.subheader("Residual Plot")
    st.pyplot(fig2)
else:
    st.warning("Please enter valid X and y values with the same number of points.")