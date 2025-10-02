import json
import os
import uuid
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from utils.data_helpers import sample_classification_data, sample_regression_data
from utils.plot_helpers import plot_confusion_matrix

PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PAGES_DIR)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_models")
REGISTRY_FILE = os.path.join(UPLOAD_DIR, "registry.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            st.warning("Registry file is corrupted. Starting with an empty registry.")
            return []
    return []


def save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)


def persist_registry(registry):
    save_registry(registry)
    st.session_state.model_registry = registry


def ensure_session_state():
    if "model_registry" not in st.session_state:
        st.session_state.model_registry = load_registry()


def sort_records(records):
    def sort_group(group):
        return sorted(
            group,
            key=lambda item: (item.get("upvotes", 0), item.get("uploaded_at", "")),
            reverse=True,
        )

    favorites = [item for item in records if item.get("is_favorite")]
    others = [item for item in records if not item.get("is_favorite")]
    return sort_group(favorites) + sort_group(others)


@st.cache_data(show_spinner=False)
def get_sample_data(task_type):
    if task_type == "Classification":
        X, y = sample_classification_data()
    else:
        X, y = sample_regression_data()
    return X, y


def evaluate_model(model, task_type):
    X, y = get_sample_data(task_type)

    try:
        y_pred = model.predict(X)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("Model prediction failed.") from exc

    result = {"metrics": {}, "extra": {}}

    if task_type == "Classification":
        accuracy = accuracy_score(y, y_pred)
        labels = [str(label) for label in np.unique(y)]
        result["metrics"].update({"Accuracy": float(accuracy)})
        result["extra"]["labels"] = labels
        result["extra"]["confusion"] = plot_confusion_matrix(y, y_pred, labels)
    else:
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y, y_pred)
        result["metrics"].update(
            {
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": rmse,
                "R²": float(r2),
            }
        )

    return result


st.header("📥 Upload and Benchmark Your Model")
st.markdown(
    """
Share your own ML model to benchmark it against our built-in sample datasets.
Upload a pickled scikit-learn compatible estimator and choose whether it should be
scored as a classification or regression model.
"""
)

ensure_session_state()
registry = st.session_state.model_registry

with st.form("model-upload-form"):
    cols = st.columns(2)
    model_name = cols[0].text_input("Model name", help="A friendly display name for your model.")
    task_type = cols[1].selectbox("Problem type", ["Classification", "Regression"])

    uploaded_file = st.file_uploader(
        "Upload pickled model",
        type=["pkl", "joblib", "pickle", "sav"],
        help="Upload a pickle/joblib dump of a scikit-learn compatible estimator.",
    )
    submitted = st.form_submit_button("Evaluate model")

if submitted:
    if uploaded_file is None:
        st.error("Please upload a model file before submitting.")
    else:
        raw_bytes = uploaded_file.getvalue()
        model_id = str(uuid.uuid4())
        safe_name = model_name.strip() if model_name else uploaded_file.name
        filename = f"{model_id}.model"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as handle:
            handle.write(raw_bytes)

        try:
            model = joblib.load(file_path)
            evaluation = evaluate_model(model, task_type)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"We couldn't evaluate that model: {exc}")
            os.remove(file_path)
        else:
            record = {
                "id": model_id,
                "name": safe_name,
                "original_filename": uploaded_file.name,
                "saved_path": filename,
                "task_type": task_type,
                "metrics": evaluation["metrics"],
                "uploaded_at": datetime.utcnow().isoformat(),
                "upvotes": 0,
                "is_favorite": False,
            }
            registry.append(record)
            persist_registry(registry)

            st.success(f"Model '{safe_name}' evaluated successfully!")
            metric_cols = st.columns(len(record["metrics"]) or 1)
            for idx, (metric_name, metric_value) in enumerate(record["metrics"].items()):
                if record["task_type"] == "Classification" and metric_name == "Accuracy":
                    display_value = f"{metric_value:.2%}"
                else:
                    display_value = f"{metric_value:.4f}"
                metric_cols[idx].metric(metric_name, display_value)

            if record["task_type"] == "Classification":
                confusion_plot = evaluation["extra"].get("confusion")
                if confusion_plot is not None:
                    st.pyplot(confusion_plot)

if registry:
    st.divider()
    st.subheader("📊 Uploaded Models")
    sorted_registry = sort_records(registry)
    for item in sorted_registry:
        favorite_prefix = "⭐ " if item.get("is_favorite") else ""
        title = f"{favorite_prefix}{item['name']} · {item['task_type']}"
        with st.expander(title):
            meta_cols = st.columns(2)
            meta_cols[0].markdown(f"**Uploaded:** {item['uploaded_at'][:19].replace('T', ' ')} UTC")
            meta_cols[0].markdown(f"**Original file:** `{item['original_filename']}`")
            meta_cols[1].markdown(f"**Stored as:** `{item['saved_path']}`")
            meta_cols[1].markdown(f"**Upvotes:** {item['upvotes']}")

            metric_cols = st.columns(len(item["metrics"]) or 1)
            for idx, (metric_name, metric_value) in enumerate(item["metrics"].items()):
                if item["task_type"] == "Classification" and metric_name == "Accuracy":
                    display_value = f"{metric_value:.2%}"
                else:
                    display_value = f"{metric_value:.4f}"
                metric_cols[idx].metric(metric_name, display_value)

            action_cols = st.columns([1, 1, 2])

            upvote_key = f"upvote-{item['id']}"
            if action_cols[0].button("👍 Upvote", key=upvote_key):
                item["upvotes"] += 1
                persist_registry(registry)
                st.toast(f"Upvoted {item['name']}")

            favorite_key = f"favorite-{item['id']}"
            if favorite_key not in st.session_state:
                st.session_state[favorite_key] = item.get("is_favorite", False)
            favorite_value = action_cols[1].checkbox("⭐ Favorite", key=favorite_key)
            if favorite_value != item.get("is_favorite", False):
                item["is_favorite"] = favorite_value
                persist_registry(registry)
                icon = "⭐" if favorite_value else "✳️"
                st.toast(f"{icon} Updated favorite for {item['name']}")

            action_cols[2].caption(
                "Favorited models stay pinned to the top of this list."
            )
else:
    st.info("No models have been uploaded yet. Be the first to add one!")
