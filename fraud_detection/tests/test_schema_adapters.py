import pandas as pd
import pytest

from fraud_detection.data.adapters import SyntheticRelationalAdapter, get_adapter
from fraud_detection.data.generators.relational_synthetic_generator import (
    ALL_FEATURE_COLS,
    generate_relational_fraud_dataset,
)
from fraud_detection.schema import FraudDatasetSchema


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory) -> pd.DataFrame:
    return generate_relational_fraud_dataset(
        n_transactions=1_000,
        n_accounts=150,
        n_devices=80,
        n_ips=80,
        n_merchants=30,
        n_fraud_rings=5,
        seed=7,
    )


def test_generator_produces_relational_columns(small_dataset: pd.DataFrame):
    expected_cols = {
        "transaction_id",
        "sender_id",
        "recipient_id",
        "device_id",
        "ip_address",
        "merchant_id",
        "timestamp",
        "label",
        *ALL_FEATURE_COLS,
    }
    assert expected_cols.issubset(set(small_dataset.columns))
    assert len(small_dataset) == 1_000
    # entity pools must be smaller than row count, i.e. IDs actually repeat
    assert small_dataset["sender_id"].nunique() < len(small_dataset)
    assert small_dataset["device_id"].nunique() < len(small_dataset)
    # sorted by timestamp for downstream temporal splitting
    assert small_dataset["timestamp"].is_monotonic_increasing


def test_fraud_rate_is_realistic_not_balanced(small_dataset: pd.DataFrame):
    fraud_rate = small_dataset["label"].mean()
    assert 0.0 < fraud_rate < 0.15  # nowhere near the old 50/50 oversampled ratio


def test_adapter_round_trip(tmp_path, small_dataset: pd.DataFrame):
    csv_path = tmp_path / "synthetic.csv"
    small_dataset.to_csv(csv_path, index=False)

    adapter = SyntheticRelationalAdapter()
    df = adapter.load(csv_path)
    schema = adapter.get_schema()

    assert isinstance(schema, FraudDatasetSchema)
    assert schema.has_relational_fields()
    assert schema.label_col in df.columns
    for col in schema.feature_cols:
        assert col in df.columns

    canonical_df = adapter.to_canonical(df)
    for canonical_name in FraudDatasetSchema.CANONICAL_NAMES.values():
        assert canonical_name in canonical_df.columns


def test_registry_lookup():
    adapter = get_adapter("synthetic_relational")
    assert isinstance(adapter, SyntheticRelationalAdapter)

    with pytest.raises(KeyError):
        get_adapter("does_not_exist")
