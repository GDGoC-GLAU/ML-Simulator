import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from models.random_forest import evaluate_random_forest, train_random_forest
from utils.data_helpers import generate_sample_classification
from utils.plot_helpers import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)

st.header("🌲 Random Forest Classifier Simulator")

st.write(
    "Experiment with a Random Forest classifier by tweaking the dataset and "
    "model hyperparameters. Evaluate performance metrics, inspect the confusion "
    "matrix, plot the ROC curve, and explore feature importances."
)

with st.sidebar:
    st.header("Dataset Settings")
    n_samples = st.slider("Samples", min_value=100, max_value=1000, value=400, step=50)
    n_features = st.slider("Features", min_value=2, max_value=15, value=6, step=1)
    n_informative = st.slider(
        "Informative Features",
        min_value=1,
        max_value=n_features,
        value=min(4, n_features),
        step=1,
    )
    max_redundant = max(n_features - n_informative, 0)
    n_redundant = st.slider(
        "Redundant Features",
        min_value=0,
        max_value=max_redundant,
        value=min(1, max_redundant),
        step=1,
    )
    class_sep = st.slider(
        "Class Separation", min_value=0.5, max_value=3.0, value=1.2, step=0.1
    )
    flip_y = st.slider(
        "Label Noise", min_value=0.0, max_value=0.3, value=0.02, step=0.01
    )
    test_size = st.slider(
        "Test Split", min_value=0.1, max_value=0.5, value=0.25, step=0.05
    )
    random_state = st.number_input(
        "Random Seed", min_value=0, max_value=10_000, value=42, step=1
    )

    st.header("Model Hyperparameters")
    n_estimators = st.slider(
        "Number of Trees", min_value=10, max_value=500, value=200, step=10
    )
    max_depth_choice = st.selectbox(
        "Maximum Depth", options=["None", 3, 5, 7, 10, 15, 20], index=0
    )
    max_depth = None if max_depth_choice == "None" else int(max_depth_choice)
    criterion = st.selectbox("Split Criterion", options=["gini", "entropy"], index=0)

X_df, y_series = generate_sample_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=n_redundant,
    class_sep=class_sep,
    flip_y=flip_y,
    random_state=int(random_state),
)

st.subheader("Dataset Preview")
st.caption("First five rows of the generated dataset.")
st.dataframe(pd.concat([X_df, y_series], axis=1).head())

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_series,
    test_size=test_size,
    stratify=y_series,
    random_state=int(random_state),
)

model = train_random_forest(
    X_train,
    y_train,
    n_estimators=n_estimators,
    max_depth=max_depth,
    criterion=criterion,
    random_state=int(random_state),
)
y_pred, y_proba, metrics = evaluate_random_forest(model, X_test, y_test)

st.subheader("Performance Metrics")
metrics_df = (
    pd.DataFrame([metrics])
    .rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
            "roc_auc": "ROC AUC",
        }
    )
    .round(3)
)
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Confusion Matrix")
cm_plot = plot_confusion_matrix(
    y_test, y_pred, labels=[str(cls) for cls in model.classes_]
)
st.pyplot(cm_plot)

st.subheader("Feature Importance")
fi_plot = plot_feature_importance(X_df.columns.tolist(), model.feature_importances_)
st.pyplot(fi_plot)

if y_proba is not None:
    st.subheader("ROC Curve")
    roc_plot = plot_roc_curve(y_test, y_proba)
    st.pyplot(roc_plot)
