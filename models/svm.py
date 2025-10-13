from utils.plot_helpers import plot_confusion_matrix
import streamlit as st

fig = plot_confusion_matrix(y_true, y_pred, labels=["Class 0", "Class 1"], cmap="Purples")
st.pyplot(fig)
