import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fraud_detection.data.adapters import SyntheticRelationalAdapter  # noqa: E402
from fraud_detection.data.generators.relational_synthetic_generator import (  # noqa: E402
    generate_relational_fraud_dataset,
)
from fraud_detection.data.graph.hypergraph import build_transaction_hypergraph  # noqa: E402
from fraud_detection.models.hypergraph.hypergraph_conv import HypergraphFraudModel  # noqa: E402
from fraud_detection.schema import FraudDatasetSchema  # noqa: E402


@pytest.fixture(scope="module")
def small_graph_inputs():
    df = generate_relational_fraud_dataset(
        n_transactions=800, n_accounts=100, n_devices=50, n_ips=50, n_merchants=20, n_fraud_rings=4, seed=11
    )
    adapter = SyntheticRelationalAdapter()
    schema = adapter.get_canonical_schema()
    canonical_df = adapter.to_canonical(df)
    return canonical_df, schema


def test_build_transaction_hypergraph(small_graph_inputs):
    df, schema = small_graph_inputs
    data, preprocessor = build_transaction_hypergraph(df, schema)

    assert data.x.shape[0] == len(df)
    assert data.y.shape[0] == len(df)
    assert data.hyperedge_index.shape[0] == 2
    assert data.num_hyperedges > 0
    # every incident node index must be a valid transaction row
    assert data.hyperedge_index[0].max().item() < len(df)
    # every incident hyperedge index must be a valid hyperedge id
    assert data.hyperedge_index[1].max().item() < data.num_hyperedges


def test_build_transaction_hypergraph_raises_on_missing_columns():
    import pandas as pd

    df = pd.DataFrame({"label": [0, 1], "amount": [1.0, 2.0]})
    flat_schema = FraudDatasetSchema(label_col="label", numeric_feature_cols=["amount"])
    with pytest.raises(ValueError):
        build_transaction_hypergraph(df, flat_schema)


def test_hypergraph_model_forward_and_predict_proba_smoke():
    rng = np.random.default_rng(0)
    n_nodes, n_hyperedges, in_channels = 30, 6, 8
    x = torch.tensor(rng.random((n_nodes, in_channels)), dtype=torch.float)
    node_ids = rng.integers(0, n_nodes, size=60)
    hyperedge_ids = rng.integers(0, n_hyperedges, size=60)
    hyperedge_index = torch.tensor(np.stack([node_ids, hyperedge_ids]), dtype=torch.long)

    model = HypergraphFraudModel(in_channels=in_channels, hidden_channels=16)
    logits = model(x, hyperedge_index)
    assert logits.shape == (n_nodes, 2)

    proba = model.predict_proba(x, hyperedge_index)
    assert proba.shape == (n_nodes,)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)
