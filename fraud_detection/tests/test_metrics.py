import numpy as np

from fraud_detection.evaluation.metrics import (
    compute_metrics,
    precision_recall_at_k,
    recall_at_fpr,
)


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_score)
    assert metrics["auprc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_recall_at_fpr_bounds():
    y_true = np.array([0] * 90 + [1] * 10)
    y_score = np.linspace(0, 1, 100)
    r = recall_at_fpr(y_true, y_score, target_fpr=0.1)
    assert 0.0 <= r <= 1.0


def test_precision_recall_at_k():
    y_true = np.array([0, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.9, 0.8, 0.2])
    result = precision_recall_at_k(y_true, y_score, k=2)
    assert result["precision_at_k"] == 1.0  # top 2 scores (0.9, 0.8) are both fraud
    assert result["recall_at_k"] == 1.0


def test_compute_metrics_handles_single_class():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3])
    metrics = compute_metrics(y_true, y_score)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["auprc"])
