"""Unified evaluation harness -- one call to score any model that exposes
`predict_proba`-style output against the fraud metrics in `metrics.py`.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .metrics import compute_metrics


@dataclass
class FraudEvaluationHarness:
    target_fpr: float = 0.01
    k: int = 100
    threshold: float = 0.5

    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray, model_name: str = "model") -> Dict:
        metrics = compute_metrics(
            y_true, y_score, threshold=self.threshold, target_fpr=self.target_fpr, k=self.k
        )
        return {"model": model_name, **metrics}

    def compare(self, results_by_model: Dict[str, Dict]) -> str:
        """Render a small markdown table comparing multiple models' metrics."""
        if not results_by_model:
            return ""
        cols = ["model", "auprc", "roc_auc", f"recall_at_fpr_{self.target_fpr}", "precision_at_k", "recall_at_k", "f1"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for name, m in results_by_model.items():
            row = [name] + [f"{m.get(c, float('nan')):.4f}" if isinstance(m.get(c), (int, float)) else str(m.get(c)) for c in cols[1:]]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
