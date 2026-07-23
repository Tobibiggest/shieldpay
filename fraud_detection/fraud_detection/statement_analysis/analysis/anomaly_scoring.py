"""Feature engineering + fresh (per-statement) IsolationForest/
TabularAutoencoder anomaly scoring, purely from a normalized statement
DataFrame -- no pretrained model, no entity IDs, no labels.

This is a deliberate departure from how `TabularAutoencoder` is used in
Phase 6 (there, it's trained on non-fraud-ONLY labeled rows): here there are
no labels at all, so both models are fit on the WHOLE statement under the
standard unsupervised-anomaly assumption that most rows are normal and a
minority are outliers. `autoencoder.py` itself is unmodified and still used
correctly for its original, labeled purpose elsewhere.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ...models.anomaly.autoencoder import TabularAutoencoder
from ...models.anomaly.isolation_forest import IsolationForestAnomalyDetector

MIN_ROWS_FOR_ML_SCORING = 30
LOW_CONFIDENCE_MESSAGE = "low - insufficient transactions for anomaly scoring"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    features = pd.DataFrame(index=df.index)

    features["amount"] = df["amount"]
    features["log_amount"] = np.log1p(df["amount"])
    features["is_credit"] = (df["direction"] == "credit").astype(float)
    features["day_of_week"] = df["date"].dt.dayofweek.astype(float)
    features["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(float)
    features["days_since_previous"] = df["date"].diff().dt.days.fillna(0.0).clip(lower=0)
    features["rolling_7day_count"] = df.set_index("date")["amount"].rolling("7D").count().to_numpy()
    features["is_round_amount"] = ((df["amount"] % 100 == 0) & (df["amount"] > 0)).astype(float)

    amount_mean, amount_std = df["amount"].mean(), df["amount"].std()
    features["z_score_amount"] = (df["amount"] - amount_mean) / amount_std if amount_std > 0 else 0.0

    recipient_mean = df.groupby("description")["amount"].transform("mean")
    recipient_std = df.groupby("description")["amount"].transform("std").fillna(0)
    deviation = (df["amount"] - recipient_mean) / recipient_std.replace(0, np.nan)
    features["deviation_from_recipient_avg"] = deviation.fillna(0.0)

    return features.fillna(0.0)


@dataclass
class AnomalyScoringResult:
    scores: np.ndarray  # per-row combined anomaly score in [0, 1], aligned to df's (post-sort) row order
    confidence: str  # "ok" or a low-confidence explanation


def score_statement_anomalies(df: pd.DataFrame) -> AnomalyScoringResult:
    if len(df) < MIN_ROWS_FOR_ML_SCORING:
        return AnomalyScoringResult(scores=np.zeros(len(df)), confidence=LOW_CONFIDENCE_MESSAGE)

    features = _engineer_features(df)
    # MinMaxScaler, not StandardScaler: TabularAutoencoder's decoder ends in
    # nn.Sigmoid(), so its reconstruction targets are implicitly assumed to
    # be in [0, 1] -- matching how data/preprocessing.py scales elsewhere.
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features.to_numpy()).astype(np.float32)

    isolation_forest = IsolationForestAnomalyDetector(n_estimators=200)
    isolation_forest.fit(X)
    iso_scores = isolation_forest.anomaly_score(X)

    autoencoder = TabularAutoencoder(input_dim=X.shape[1], epochs=40, batch_size=min(64, len(X)))
    autoencoder.fit(X, verbose=False)
    ae_scores = autoencoder.anomaly_score(X)

    # Percentile-rank normalize each model's scores before averaging --
    # more robust to one model's skewed score distribution than plain
    # min-max averaging (a single extreme outlier can otherwise compress
    # every other score toward 0).
    iso_ranks = pd.Series(iso_scores).rank(pct=True).to_numpy()
    ae_ranks = pd.Series(ae_scores).rank(pct=True).to_numpy()
    combined = (iso_ranks + ae_ranks) / 2.0

    return AnomalyScoringResult(scores=combined, confidence="ok")
