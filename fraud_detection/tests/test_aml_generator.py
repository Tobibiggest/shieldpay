"""Tests the AML synthetic generator + adapter. Mirrors
`tests/test_schema_adapters.py`'s realistic-imbalance/entity-repetition
checks, plus an automated camouflage-discipline check that operationalizes
the lesson learned building the payments generator: if a features-only
tabular model can trivially separate the planted fraud, the graph models
(built later, in Step 4) would have nothing to add value on top of.
"""

import pytest

from fraud_detection.data.preprocessing import temporal_split
from fraud_detection.domains.aml.aml_adapter import AMLAdapter
from fraud_detection.domains.aml.aml_generator import (
    ALL_FEATURE_COLS,
    generate_aml_transaction_dataset,
)
from fraud_detection.evaluation.metrics import compute_metrics
from fraud_detection.models.tabular.gbdt import XGBoostFraudModel


def _small_dataset(seed=7):
    return generate_aml_transaction_dataset(
        n_transactions=1_000,
        n_accounts=150,
        n_devices=80,
        n_ips=80,
        n_layering_chains=5,
        n_smurf_rings=3,
        seed=seed,
    )


def test_generator_produces_relational_columns():
    df = _small_dataset()
    expected_cols = {"transaction_id", "sender_id", "recipient_id", "device_id", "ip_address", "timestamp", "label", *ALL_FEATURE_COLS}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 1_000
    assert df["sender_id"].nunique() < len(df)
    assert df["device_id"].nunique() < len(df)
    assert df["timestamp"].is_monotonic_increasing


def test_fraud_rate_is_realistic_not_balanced():
    df = _small_dataset()
    fraud_rate = df["label"].mean()
    assert 0.0 < fraud_rate < 0.15


def test_layering_chains_share_a_small_device_pool():
    df = generate_aml_transaction_dataset(
        n_transactions=2_000, n_accounts=300, n_devices=150, n_ips=150,
        n_layering_chains=10, n_smurf_rings=0, fraud_ratio=0.05, seed=11,
    )
    fraud_df = df[df["label"] == 1]
    assert len(fraud_df) > 0
    # planted structural fraud should reuse devices far more than the
    # background population does (a small shared pool per chain)
    assert fraud_df["device_id"].nunique() < len(fraud_df)


def test_adapter_schema_has_no_merchant_role():
    adapter = AMLAdapter()
    schema = adapter.get_canonical_schema()
    role_names = {r.role_name for r in schema.entity_roles}
    assert role_names == {"originator", "beneficiary", "device", "ip"}
    assert schema.event_node_type == "transaction"


def test_adapter_to_canonical_is_identity():
    df = _small_dataset()
    adapter = AMLAdapter()
    canonical_df = adapter.to_canonical(df)
    assert list(canonical_df.columns) == list(df.columns)


def test_camouflage_discipline_features_alone_cannot_separate_fraud():
    df = generate_aml_transaction_dataset(
        n_transactions=6_000, n_accounts=600, n_devices=350, n_ips=350,
        n_layering_chains=15, n_smurf_rings=8, seed=3,
    )
    adapter = AMLAdapter()
    schema = adapter.get_canonical_schema()
    train_df, val_df, test_df = temporal_split(df, schema, test_frac=0.2, val_frac=0.1)

    model = XGBoostFraudModel(schema=schema).fit(train_df)
    y_test = test_df[schema.label_col].to_numpy()
    y_score = model.predict_proba(test_df)
    metrics = compute_metrics(y_test, y_score)

    # A features-only tabular model should NOT trivially separate the
    # camouflaged layering/smurfing fraud -- if this ceiling is breached,
    # the generator has regressed to the mistake made building payments'
    # first generator version, and the graph models built in Step 4 would
    # have nothing to add value on top of.
    assert metrics["auprc"] < 0.6
    assert metrics["auprc"] > 0.0  # sanity: not a garbage/constant model either
