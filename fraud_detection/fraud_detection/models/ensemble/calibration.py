"""Probability calibration for the stacked ensemble score.

A stacked logistic-regression output is not automatically a well-calibrated
probability -- it's just a score that ranks well. Fraud decisions need
calibrated probabilities because operational thresholds ("auto-block above
0.9", "queue for review above 0.5") only mean something consistent if "0.8"
reliably corresponds to roughly an 80% chance of fraud. Isotonic regression
is the default here: unlike Platt scaling (a fixed logistic-sigmoid shape),
it makes no parametric assumption about the miscalibration curve's shape,
which matters when the underlying score comes from a stack of very
different model types (tree, GNN, HGT, reconstruction error, isolation
score) rather than one well-behaved classifier.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class IsotonicCalibrator:
    model: Optional[IsotonicRegression] = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(raw_scores, y_true)
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.model.predict(raw_scores)


@dataclass
class PlattCalibrator:
    """Logistic-sigmoid calibration -- a simpler, lower-variance alternative
    to isotonic when the holdout set used to fit the calibrator is small."""

    model: Optional[LogisticRegression] = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression()
        self.model.fit(raw_scores.reshape(-1, 1), y_true)
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
