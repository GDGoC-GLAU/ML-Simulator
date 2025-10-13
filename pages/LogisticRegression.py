from utils.plot_helpers import plot_roc_curve
import streamlit as st

# Example
fig = plot_roc_curve(y_true, y_pred_proba)
st.pyplot(fig)
