import pandas as pd

from fraud_detection.domains.aml.aml_patterns import (
    detect_layering_chains,
    detect_rapid_pass_through,
    run_all_aml_pattern_detectors,
)

BASE_TS = pd.Timestamp("2025-01-01T00:00:00")


def _chain_df():
    rows = []
    accounts = ["acct_A", "acct_B", "acct_C", "acct_D"]
    amount = 5000.0
    for i in range(3):
        rows.append({
            "sender_id": accounts[i],
            "recipient_id": accounts[i + 1],
            "transaction_amount": amount,
            "timestamp": BASE_TS + pd.Timedelta(hours=i * 10),
        })
        amount *= 0.97
    # unrelated noise transaction, far away in time and unconnected accounts
    rows.append({
        "sender_id": "acct_X", "recipient_id": "acct_Y",
        "transaction_amount": 50.0, "timestamp": BASE_TS + pd.Timedelta(days=5),
    })
    return pd.DataFrame(rows)


def _pass_through_df():
    rows = [
        {"sender_id": "acct_A", "recipient_id": "acct_mule", "transaction_amount": 3000.0, "timestamp": BASE_TS},
        {"sender_id": "acct_mule", "recipient_id": "acct_B", "transaction_amount": 2900.0, "timestamp": BASE_TS + pd.Timedelta(hours=2)},
        {"sender_id": "acct_X", "recipient_id": "acct_Y", "transaction_amount": 20.0, "timestamp": BASE_TS + pd.Timedelta(days=3)},
    ]
    return pd.DataFrame(rows)


def _no_pattern_df():
    rows = [
        {"sender_id": f"acct_{i}", "recipient_id": f"acct_{i + 100}", "transaction_amount": 30.0 + i,
         "timestamp": BASE_TS + pd.Timedelta(days=i)}
        for i in range(10)
    ]
    return pd.DataFrame(rows)


def test_detect_layering_chains_fires_on_planted_chain():
    df = _chain_df()
    findings = detect_layering_chains(df)
    assert any(f["pattern_type"] == "layering_chain" for f in findings)
    chain_finding = next(f for f in findings if f["pattern_type"] == "layering_chain")
    assert len(chain_finding["affected_row_indices"]) >= 3


def test_detect_layering_chains_silent_on_unconnected_transactions():
    df = _no_pattern_df()
    findings = detect_layering_chains(df)
    assert findings == []


def test_detect_rapid_pass_through_fires():
    df = _pass_through_df()
    findings = detect_rapid_pass_through(df)
    assert any(f["pattern_type"] == "rapid_pass_through" for f in findings)


def test_detect_rapid_pass_through_silent_on_unconnected_transactions():
    df = _no_pattern_df()
    findings = detect_rapid_pass_through(df)
    assert findings == []


def test_run_all_aml_pattern_detectors_returns_list_of_dicts():
    df = pd.concat([_chain_df(), _pass_through_df()], ignore_index=True)
    findings = run_all_aml_pattern_detectors(df)
    assert isinstance(findings, list)
    for f in findings:
        assert {"pattern_type", "description", "severity", "affected_row_indices"} <= f.keys()
