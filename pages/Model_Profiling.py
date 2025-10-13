import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import plotly.figure_factory as ff

st.title("📊 Model Profiling & Evaluation")

st.write("Upload your trained ML model (pickle file) and test it with data.")

# Upload model
uploaded_model = st.file_uploader("Upload your model (.pkl)", type=["pkl"])

# Upload dataset
uploaded_data = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_model and uploaded_data:
    model = pickle.load(uploaded_model)
    data = pd.read_csv(uploaded_data)

    st.write("### Preview of uploaded data")
    st.dataframe(data.head())

    # Assuming last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Make predictions
    y_pred = model.predict(X)

    # Detect classification vs regression
    if len(np.unique(y)) < 20:
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.write("#### Confusion Matrix")
        st.dataframe(cm)
    else:
        mse = mean_squared_error(y, y_pred)
        st.metric("Mean Squared Error", f"{mse:.3f}")

    st.success("✅ Model evaluation completed!")

    # Add upvote/favorite
    if "votes" not in st.session_state:
        st.session_state["votes"] = 0

    if st.button("👍 Upvote this Model"):
        st.session_state["votes"] += 1

    st.write(f"⭐ Current Upvotes: {st.session_state['votes']}")
