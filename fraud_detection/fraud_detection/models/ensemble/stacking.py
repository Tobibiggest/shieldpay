"""Stacking meta-learner: a small logistic-regression model trained on
out-of-fold base-model scores (predictions from a base model on rows it was
NOT trained on), which is what avoids the leakage a naive stack (fitting the
meta-learner on in-sample base-model predictions) would introduce -- an
in-sample base model is overconfident on its own training rows, and the
meta-learner would just learn to trust that overconfidence.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StackingMetaLearner:
    base_model_names: List[str] = field(default_factory=list)
    pipeline: Optional[Pipeline] = None

    def fit(self, meta_X: np.ndarray, meta_y: np.ndarray) -> "StackingMetaLearner":
        self.pipeline = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        self.pipeline.fit(meta_X, meta_y)
        return self

    def predict_proba(self, meta_X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(meta_X)[:, 1]
