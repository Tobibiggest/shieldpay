"""Tests the IEEE-CIS adapter against small, synthetic fixture CSVs shaped
like the real Kaggle files (same column names, tiny row count) -- this repo
can't download the real dataset (see docs/DATASETS.md), but the adapter's
column-mapping logic and its portability into the shared graph pipeline can
still be verified without it.
"""

import numpy as np
import pandas as pd

from fraud_detection.data.adapters import IEEECISAdapter, get_adapter
from fraud_detection.data.graph.build_graph import build_hetero_graph
from fraud_detection.data.preprocessing import fit_preprocessor
from fraud_detection.schema import FraudDatasetSchema


def _make_fake_ieee_cis_files(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    n_cards, n_devices, n_addrs, n_domains = 30, 15, 10, 8

    transaction_ids = np.arange(1, n + 1)
    card1 = rng.integers(1000, 1000 + n_cards, size=n)
    addr1 = rng.integers(100, 100 + n_addrs, size=n)
    domains = [f"domain{i}.com" for i in range(n_domains)]

    transactions = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "isFraud": (rng.random(n) < 0.05).astype(int),
            "TransactionDT": np.sort(rng.integers(0, 1_000_000, size=n)),
            "TransactionAmt": rng.exponential(100, size=n),
            "ProductCD": rng.choice(["W", "C", "R", "H"], size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": addr1,
            "P_emaildomain": rng.choice(domains, size=n),
            "C1": rng.poisson(2, size=n).astype(float),
            "C2": rng.poisson(1, size=n).astype(float),
            "C13": rng.poisson(3, size=n).astype(float),
            "C14": rng.poisson(1, size=n).astype(float),
            "D1": rng.exponential(10, size=n),
            "D2": rng.exponential(5, size=n),
            "D4": rng.exponential(5, size=n),
            "D15": rng.exponential(20, size=n),
            "M1": rng.choice(["T", "F"], size=n),
            "M2": rng.choice(["T", "F"], size=n),
            "M3": rng.choice(["T", "F"], size=n),
            "M4": rng.choice(["M0", "M1", "M2"], size=n),
            "M6": rng.choice(["T", "F"], size=n),
            "card1": card1,
        }
    )

    devices = [f"device_{i}" for i in range(n_devices)]
    identity = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "DeviceInfo": rng.choice(devices, size=n),
            "DeviceType": rng.choice(["mobile", "desktop"], size=n),
        }
    )

    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path


def test_ieee_cis_adapter_load_and_schema(tmp_path):
    data_dir = _make_fake_ieee_cis_files(tmp_path)
    adapter = IEEECISAdapter()
    df = adapter.load(data_dir)
    schema = adapter.get_schema()

    assert isinstance(schema, FraudDatasetSchema)
    assert schema.has_relational_fields()
    assert schema.label_col in df.columns
    for col in schema.feature_cols:
        assert col in df.columns
    assert not df[schema.numeric_feature_cols].isna().any().any()
    assert not df[schema.categorical_feature_cols].isna().any().any()


def test_get_canonical_schema_matches_renamed_columns(tmp_path):
    """Regression test: `get_schema()` describes the dataset's native column
    names (isFraud, card1, ...); `load_canonical()` renames those columns to
    the canonical ones (label, sender_id, ...). Using `get_schema()` together
    with `load_canonical()`'s output is a mismatch -- `get_canonical_schema()`
    is what must be paired with it instead. IEEE-CIS is the adapter that
    actually exercises this (its native names differ from canonical); the
    synthetic adapter's native names already equal the canonical ones, so a
    bug here wouldn't have shown up there."""
    data_dir = _make_fake_ieee_cis_files(tmp_path)
    adapter = IEEECISAdapter()
    native_schema = adapter.get_schema()
    canonical_schema = adapter.get_canonical_schema()

    assert native_schema.label_col == "isFraud"
    assert canonical_schema.label_col == "label"
    assert native_schema.sender_id_col == "card1"
    assert canonical_schema.sender_id_col == "sender_id"

    canonical_df = adapter.load_canonical(data_dir)
    assert canonical_schema.label_col in canonical_df.columns
    assert canonical_schema.sender_id_col in canonical_df.columns
    assert native_schema.label_col not in canonical_df.columns


def test_ieee_cis_adapter_registry_and_canonical(tmp_path):
    data_dir = _make_fake_ieee_cis_files(tmp_path)
    adapter = get_adapter("ieee_cis")
    schema = adapter.get_schema()
    df = adapter.load(data_dir)
    canonical_df = adapter.to_canonical(df)

    # IEEE-CIS has no IP concept (ip_col is None) -- only check canonical
    # names for fields this adapter actually sets.
    for attr, canonical_name in FraudDatasetSchema.CANONICAL_NAMES.items():
        if getattr(schema, attr, None) is not None:
            assert canonical_name in canonical_df.columns
    assert "ip_address" not in canonical_df.columns

    # entity IDs must repeat -- otherwise there's no graph to build
    assert canonical_df["sender_id"].nunique() < len(canonical_df)
    assert canonical_df["device_id"].nunique() < len(canonical_df)


def test_ieee_cis_portable_to_same_graph_pipeline(tmp_path):
    """Portability proof: the same graph-building code that runs on the
    synthetic dataset also runs on IEEE-CIS-shaped data, with no model-code
    changes -- only the adapter differs."""
    data_dir = _make_fake_ieee_cis_files(tmp_path, n=300)
    adapter = get_adapter("ieee_cis")
    schema = adapter.get_canonical_schema()
    df = adapter.load_canonical(data_dir)

    preprocessor = fit_preprocessor(df, schema)
    data, maps, _ = build_hetero_graph(df, schema, preprocessor=preprocessor)

    assert set(data.node_types) == {"transaction", "account", "device", "ip", "merchant"}
    assert data["transaction"].x.shape[0] == len(df)
    assert data["account"].num_nodes < len(df)
    assert data["device"].num_nodes < len(df)
