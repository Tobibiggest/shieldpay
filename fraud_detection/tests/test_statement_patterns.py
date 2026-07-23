import numpy as np
import pandas as pd

from fraud_detection.statement_analysis.analysis.patterns import (
    detect_amount_outliers,
    detect_benfords_law_deviation,
    detect_dormant_then_burst,
    detect_duplicate_transactions,
    detect_recipient_concentration,
    detect_round_number_bias,
    detect_structuring,
    detect_velocity_spikes,
    run_all_pattern_detectors,
)


def _make_df(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    if "balance" not in df.columns:
        df["balance"] = np.nan
    return df


def test_detect_velocity_spikes_fires_on_burst_day():
    rows = []
    for day in range(1, 15):
        rows.append({"date": f"2025-01-{day:02d}", "description": "shop", "amount": 20.0, "direction": "debit"})
    for i in range(10):  # burst day: far more transactions than the baseline ~1/day
        rows.append({"date": "2025-01-20", "description": f"burst_{i}", "amount": 20.0, "direction": "debit"})
    df = _make_df(rows)

    findings = detect_velocity_spikes(df)
    assert any(f["pattern_type"] == "velocity_spike" for f in findings)


def test_detect_benfords_law_deviation_fires_on_uniform_leading_digits():
    # Force many amounts starting with digit 9 -- strongly violates Benford's Law
    rows = [
        {"date": "2025-01-01", "description": f"t{i}", "amount": 900.0 + i, "direction": "debit"}
        for i in range(60)
    ]
    df = _make_df(rows)
    findings = detect_benfords_law_deviation(df)
    assert any(f["pattern_type"] == "benfords_law_deviation" for f in findings)


def test_detect_benfords_law_deviation_silent_on_too_few_rows():
    rows = [{"date": "2025-01-01", "description": "t", "amount": 900.0, "direction": "debit"}]
    df = _make_df(rows)
    assert detect_benfords_law_deviation(df) == []


def test_detect_round_number_bias_fires_when_mostly_round():
    rows = [
        {"date": f"2025-01-{i+1:02d}", "description": f"t{i}", "amount": 100.0 * (i + 1), "direction": "debit"}
        for i in range(15)
    ]
    df = _make_df(rows)
    findings = detect_round_number_bias(df)
    assert any(f["pattern_type"] == "round_number_bias" for f in findings)


def test_detect_structuring_fires_on_cluster_below_threshold():
    rows = [
        {"date": f"2025-01-{i+1:02d}", "description": f"t{i}", "amount": 9500.0, "direction": "debit"}
        for i in range(4)
    ]
    df = _make_df(rows)
    findings = detect_structuring(df)
    assert any(f["pattern_type"] == "structuring" for f in findings)


def test_detect_duplicate_transactions_fires_on_same_amount_and_description():
    rows = [
        {"date": "2025-01-01", "description": "Coffee Shop", "amount": 4.50, "direction": "debit"},
        {"date": "2025-01-01", "description": "Coffee Shop", "amount": 4.50, "direction": "debit"},
    ]
    df = _make_df(rows)
    findings = detect_duplicate_transactions(df)
    assert any(f["pattern_type"] == "duplicate_transaction" for f in findings)


def test_detect_dormant_then_burst_fires_after_long_gap():
    rows = [{"date": "2025-01-01", "description": "t0", "amount": 10.0, "direction": "debit"}]
    for i in range(4):
        rows.append({"date": f"2025-03-{i+1:02d}", "description": f"t{i+1}", "amount": 10.0, "direction": "debit"})
    df = _make_df(rows)
    findings = detect_dormant_then_burst(df)
    assert any(f["pattern_type"] == "dormant_then_burst" for f in findings)


def test_detect_recipient_concentration_fires_on_dominant_recipient():
    rows = [{"date": "2025-01-01", "description": "Big Recipient", "amount": 5000.0, "direction": "debit"}]
    rows += [
        {"date": f"2025-01-{i+2:02d}", "description": f"small_{i}", "amount": 10.0, "direction": "debit"}
        for i in range(5)
    ]
    df = _make_df(rows)
    findings = detect_recipient_concentration(df)
    assert any(f["pattern_type"] == "recipient_concentration" for f in findings)


def test_detect_amount_outliers_fires_on_extreme_value():
    rows = [
        {"date": f"2025-01-{i+1:02d}", "description": f"t{i}", "amount": 20.0 + i, "direction": "debit"}
        for i in range(12)
    ]
    rows.append({"date": "2025-02-01", "description": "huge", "amount": 100000.0, "direction": "debit"})
    df = _make_df(rows)
    findings = detect_amount_outliers(df)
    assert any(f["pattern_type"] == "amount_outlier" for f in findings)


def test_run_all_pattern_detectors_returns_list_of_dicts():
    rows = [
        {"date": f"2025-01-{i+1:02d}", "description": f"t{i}", "amount": 15.0 + i, "direction": "debit"}
        for i in range(20)
    ]
    df = _make_df(rows)
    findings = run_all_pattern_detectors(df)
    assert isinstance(findings, list)
    for f in findings:
        assert {"pattern_type", "description", "severity", "affected_row_indices"} <= f.keys()
