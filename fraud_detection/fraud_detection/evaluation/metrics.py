"""Fraud-appropriate evaluation metrics.

The original notebook (`FraudDetectionUSingGAN.ipynb`, cell 15) reports only
accuracy/precision/recall/F1/ROC-AUC on a threshold-0.5 prediction. At ~3% fraud
prevalence, accuracy is close to meaningless (predicting "not fraud" for
everything scores ~97%) and ROC-AUC can look good while precision at any
usable operating threshold is poor. This module adds the metrics that matter
for a rare-event problem: AUPRC (area under precision-recall curve, the
standard metric for imbalanced classification), recall at a fixed false-positive
rate (an investigations team's alert-budget framing), and precision/recall at
a fixed alert count "k" (top-k transactions actually reviewed per period).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> Dict[str, float]:
    k = min(k, len(y_score))
    if k == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}
    top_k_idx = np.argsort(-y_score)[:k]
    n_positive = float(y_true.sum())
    n_hits = float(y_true[top_k_idx].sum())
    precision = n_hits / k
    recall = n_hits / n_positive if n_positive > 0 else 0.0
    return {"precision_at_k": precision, "recall_at_k": recall}


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    target_fpr: float = 0.01,
    k: int = 100,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        f"recall_at_fpr_{target_fpr}": recall_at_fpr(y_true, y_score, target_fpr)
        if len(np.unique(y_true)) > 1
        else float("nan"),
    }
    metrics.update(precision_recall_at_k(y_true, y_score, k))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"true_negatives": int(tn), "false_positives": int(fp), "false_negatives": int(fn), "true_positives": int(tp)})
    return metrics
