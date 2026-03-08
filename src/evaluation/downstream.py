"""
Downstream: use embeddings H as features for classification.
StandardScaler on train, then MLP or linear classifier; report precision, recall, F1, confusion matrix.
"""

from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


def run_downstream_classification(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    classifier: str = "mlp",
    random_state: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Args:
        X: (N, dim) embedding matrix.
        y: (N,) labels (any hashable type).
        n_splits: K for stratified K-fold.
        classifier: "mlp" or "linear".
        random_state: for reproducibility.
        **kwargs: passed to classifier constructor.
    Returns:
        dict with mean metrics, per-fold metrics, and optional confusion matrix.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=500,
            random_state=random_state,
            **kwargs,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=random_state, **kwargs)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["precision_weighted", "recall_weighted", "f1_weighted"]
    scores = cross_validate(clf, X_scaled, y_enc, cv=cv, scoring=scoring, return_train_score=False)
    mean_metrics = {
        "precision": np.mean(scores["test_precision_weighted"]),
        "recall": np.mean(scores["test_recall_weighted"]),
        "f1": np.mean(scores["test_f1_weighted"]),
    }
    return {
        "mean_metrics": mean_metrics,
        "cv_scores": scores,
        "label_encoder": le,
        "scaler": scaler,
    }


def classification_report_from_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> str:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    reports = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y_enc)):
        clf.fit(X_scaled[train_idx], y_enc[train_idx])
        y_pred = clf.predict(X_scaled[test_idx])
        reports.append(classification_report(y_enc[test_idx], y_pred, output_dict=False))
    return "\n".join(reports)
