import pytest

from fraud_detection.data.adapters import SyntheticRelationalAdapter
from fraud_detection.data.generators.relational_synthetic_generator import (
    generate_relational_fraud_dataset,
)
from fraud_detection.data.graph.build_graph import build_hetero_graph
from fraud_detection.data.graph.homogeneous import build_transaction_projection_graph


@pytest.fixture(scope="module")
def small_graph_inputs():
    df = generate_relational_fraud_dataset(
        n_transactions=800,
        n_accounts=100,
        n_devices=50,
        n_ips=50,
        n_merchants=20,
        n_fraud_rings=4,
        seed=11,
    )
    adapter = SyntheticRelationalAdapter()
    schema = adapter.get_schema()
    canonical_df = adapter.to_canonical(df)
    return canonical_df, schema


def test_build_hetero_graph_node_and_edge_types(small_graph_inputs):
    df, schema = small_graph_inputs
    data, maps, preprocessor = build_hetero_graph(df, schema)

    assert set(data.node_types) == {"transaction", "account", "device", "ip", "merchant"}
    expected_edge_types = {
        ("account", "sends", "transaction"),
        ("transaction", "rev_sends", "account"),
        ("transaction", "sent_to", "account"),
        ("account", "rev_sent_to", "transaction"),
        ("transaction", "uses_device", "device"),
        ("device", "rev_uses_device", "transaction"),
        ("transaction", "uses_ip", "ip"),
        ("ip", "rev_uses_ip", "transaction"),
        ("transaction", "at_merchant", "merchant"),
        ("merchant", "rev_at_merchant", "transaction"),
        ("account", "shares_device", "account"),
        ("account", "shares_ip", "account"),
    }
    assert expected_edge_types.issubset(set(data.edge_types))

    assert data["transaction"].x.shape[0] == len(df)
    assert data["transaction"].y.shape[0] == len(df)
    assert data["account"].num_nodes == len(maps.account)
    assert data["account"].num_nodes < len(df)  # accounts repeat across transactions

    # every transaction has exactly one sender edge
    assert data["account", "sends", "transaction"].edge_index.shape[1] == len(df)


def test_shares_device_edges_are_symmetric(small_graph_inputs):
    df, schema = small_graph_inputs
    data, _, _ = build_hetero_graph(df, schema)
    edge_index = data["account", "shares_device", "account"].edge_index
    if edge_index.shape[1] > 0:
        # for every (a, b) edge there should be a (b, a) edge
        pairs = set(map(tuple, edge_index.t().tolist()))
        for a, b in list(pairs)[:50]:
            assert (b, a) in pairs


def test_build_hetero_graph_raises_without_relational_fields():
    from fraud_detection.schema import FraudDatasetSchema
    import pandas as pd

    df = pd.DataFrame({"label": [0, 1], "amount": [1.0, 2.0]})
    flat_schema = FraudDatasetSchema(label_col="label", numeric_feature_cols=["amount"])
    with pytest.raises(ValueError):
        build_hetero_graph(df, flat_schema)


def test_build_transaction_projection_graph(small_graph_inputs):
    df, schema = small_graph_inputs
    data, preprocessor = build_transaction_projection_graph(df, schema)

    assert data.x.shape[0] == len(df)
    assert data.y.shape[0] == len(df)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.max().item() < len(df)

    # edges should be undirected: every (a, b) has a (b, a)
    pairs = set(map(tuple, data.edge_index.t().tolist()))
    for a, b in list(pairs)[:50]:
        assert (b, a) in pairs

    # fraud ring members share device/ip -- some transactions must be connected
    assert data.edge_index.shape[1] > 0
