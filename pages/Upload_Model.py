import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils.plot_helpers import plot_confusion_matrix
from utils.metrics_helpers import calculate_metrics
from utils.model_storage import save_uploaded_model

st.set_page_config(page_title="Model Upload", layout="wide")

st.title("🤖 Upload Your Machine Learning Model")

st.write("Upload your trained model file (`.pkl` or `.joblib`) and test it with sample data.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload model file", type=["pkl", "joblib"])

if uploaded_file:
    model = joblib.load(uploaded_file)
    model_name = uploaded_file.name
    save_uploaded_model(uploaded_file)

    st.success(f"✅ Model '{model_name}' uploaded successfully!")

    # --- Test data upload ---
    st.subheader("📊 Upload Test Data")
    test_file = st.file_uploader("Upload test CSV", type=["csv"])

    if test_file:
        df = pd.read_csv(test_file)
        st.write("Preview of uploaded test data:")
        st.dataframe(df.head())

        target_column = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target_column])
        y_true = df[target_column]

        try:
            y_pred = model.predict(X)

            # --- Calculate metrics ---
            metrics = calculate_metrics(y_true, y_pred)
            st.subheader("📈 Model Performance")
            st.json(metrics)

            # --- Plot confusion matrix for classifiers ---
            if len(np.unique(y_true)) <= 10:
                fig = plot_confusion_matrix(y_true, y_pred)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

    # --- Upvote / favorite section ---
    st.markdown("---")
    st.subheader("⭐ Like This Model?")
    if "votes" not in st.session_state:
        st.session_state["votes"] = 0

    if st.button("👍 Upvote"):
        st.session_state["votes"] += 1

    st.write(f"Total Upvotes: {st.session_state['votes']}")
