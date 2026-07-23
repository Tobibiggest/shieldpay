"""Cheap classical complement to the autoencoder (`autoencoder.py`) --
sklearn's IsolationForest. Unlike the reconstruction-based autoencoder, it
doesn't need non-fraud-only training data, and it's fast enough to serve as
a sanity check that the (much more expensive) autoencoder is earning its
cost on a given dataset.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestAnomalyDetector:
    n_estimators: int = 200
    contamination: str = "auto"
    random_state: int = 42
    model: Optional[IsolationForest] = None

    def fit(self, X: np.ndarray) -> "IsolationForestAnomalyDetector":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: higher = more "normal" -- negate so higher = more anomalous,
        # consistent with every other base model's "higher = more fraud-like" convention.
        return -self.model.score_samples(X)
