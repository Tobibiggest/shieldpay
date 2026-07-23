import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fraud_detection.statement_analysis.analysis.anomaly_scoring import (  # noqa: E402
    LOW_CONFIDENCE_MESSAGE,
    score_statement_anomalies,
)


def _make_statement_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    descriptions = rng.choice(["Grocery", "Rent", "Utility", "Coffee", "Salary"], size=n)
    amounts = rng.exponential(50, size=n)
    directions = rng.choice(["debit", "credit"], size=n, p=[0.8, 0.2])
    return pd.DataFrame({"date": dates, "description": descriptions, "amount": amounts, "direction": directions})


def test_score_statement_anomalies_low_confidence_below_min_rows():
    df = _make_statement_df(n=10)
    result = score_statement_anomalies(df)
    assert result.confidence == LOW_CONFIDENCE_MESSAGE
    assert np.all(result.scores == 0)
    assert len(result.scores) == len(df)


def test_score_statement_anomalies_ok_confidence_above_min_rows():
    df = _make_statement_df(n=40)
    result = score_statement_anomalies(df)
    assert result.confidence == "ok"
    assert len(result.scores) == len(df)
    assert np.all(result.scores >= 0) and np.all(result.scores <= 1)


def test_score_statement_anomalies_flags_extreme_outlier_highly():
    df = _make_statement_df(n=40)
    # Inject one wildly large transaction -- should rank near the top of anomaly scores
    df = df.reset_index(drop=True)
    df.loc[len(df)] = [pd.Timestamp("2025-03-01"), "Huge Transfer", 1_000_000.0, "debit"]

    result = score_statement_anomalies(df)
    outlier_score = result.scores[df["description"] == "Huge Transfer"][0]
    assert outlier_score > np.median(result.scores)
