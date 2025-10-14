"""
Random Forest utilities.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def train_random_forest(
    X,
    y,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    criterion: str = "gini",
    random_state: Optional[int] = 42,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier with the provided data and hyperparameters.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def evaluate_random_forest(
    model: RandomForestClassifier, X, y_true
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Compute predictions and performance metrics for a trained model.
    """
    y_pred = model.predict(X)
    unique_classes = np.unique(y_true)
    average = "binary" if len(unique_classes) == 2 else "macro"

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    y_proba: Optional[np.ndarray] = None
    if len(unique_classes) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return y_pred, y_proba, metrics
