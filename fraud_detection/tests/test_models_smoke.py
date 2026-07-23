"""Forward-pass / tiny-training smoke tests for each model family -- fast
sanity checks that the model runs end-to-end on tiny random data, not
accuracy claims.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch_geometric.data import HeteroData  # noqa: E402

from fraud_detection.models.gan.cgan_wgan_gp import CWGANGPConfig, CWGANGPTrainer  # noqa: E402
from fraud_detection.models.gnn.gat import GATFraudModel  # noqa: E402
from fraud_detection.models.gnn.sage import GraphSAGEFraudModel  # noqa: E402
from fraud_detection.models.hgnn.hgt import HGTFraudModel  # noqa: E402
from fraud_detection.models.hgnn.rgcn import RGCNFraudModel  # noqa: E402


def test_cwgan_gp_fit_and_generate_smoke():
    rng = np.random.default_rng(0)
    n, dim = 200, 12
    X = rng.random((n, dim)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[:20] = 1  # imbalanced, like real fraud data

    config = CWGANGPConfig(epochs=2, batch_size=32, n_critic=2, noise_dim=8, hidden_dim=16)
    trainer = CWGANGPTrainer(input_dim=dim, config=config)
    trainer.fit(X, y, verbose=False)

    synthetic = trainer.generate(label=1, n_samples=10)
    assert synthetic.shape == (10, dim)
    assert np.all(synthetic >= 0.0) and np.all(synthetic <= 1.0)
    assert np.isfinite(synthetic).all()


def _tiny_random_graph(n_nodes=30, n_edges=60, in_channels=8, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.random((n_nodes, in_channels)), dtype=torch.float)
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return x, edge_index


def test_graphsage_forward_and_predict_proba_smoke():
    x, edge_index = _tiny_random_graph()
    model = GraphSAGEFraudModel(in_channels=x.shape[1], hidden_channels=16)
    logits = model(x, edge_index)
    assert logits.shape == (x.shape[0], 2)

    proba = model.predict_proba(x, edge_index)
    assert proba.shape == (x.shape[0],)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)


def test_gat_forward_and_predict_proba_smoke():
    x, edge_index = _tiny_random_graph()
    model = GATFraudModel(in_channels=x.shape[1], hidden_channels=16, heads=2)
    logits = model(x, edge_index)
    assert logits.shape == (x.shape[0], 2)

    proba = model.predict_proba(x, edge_index)
    assert proba.shape == (x.shape[0],)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)


def _tiny_hetero_data(n_transactions=20, n_accounts=6, n_devices=4, in_channels=8, seed=0) -> HeteroData:
    rng = np.random.default_rng(seed)
    data = HeteroData()
    data["transaction"].x = torch.tensor(rng.random((n_transactions, in_channels)), dtype=torch.float)
    data["transaction"].y = torch.tensor(rng.integers(0, 2, size=n_transactions), dtype=torch.long)
    data["account"].num_nodes = n_accounts
    data["device"].num_nodes = n_devices

    def _edges(n_tx, n_entity, n_edges):
        tx = rng.integers(0, n_tx, size=n_edges)
        ent = rng.integers(0, n_entity, size=n_edges)
        return torch.tensor(np.stack([tx, ent]), dtype=torch.long)

    sender_edges = _edges(n_transactions, n_accounts, n_transactions)
    device_edges = _edges(n_transactions, n_devices, n_transactions)

    data["account", "sends", "transaction"].edge_index = sender_edges[[1, 0]]
    data["transaction", "rev_sends", "account"].edge_index = sender_edges
    data["transaction", "uses_device", "device"].edge_index = device_edges
    data["device", "rev_uses_device", "transaction"].edge_index = device_edges[[1, 0]]
    return data


def test_hgt_forward_and_predict_proba_smoke():
    data = _tiny_hetero_data()
    metadata = data.metadata()
    num_nodes_dict = {nt: data[nt].num_nodes for nt in metadata[0]}
    model = HGTFraudModel(
        metadata, transaction_in_channels=data["transaction"].x.shape[1], num_nodes_dict=num_nodes_dict,
        hidden_channels=16, num_heads=2,
    )
    logits = model(data)
    assert logits.shape == (data["transaction"].x.shape[0], 2)

    proba = model.predict_proba(data)
    assert proba.shape == (data["transaction"].x.shape[0],)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)


def test_rgcn_forward_and_predict_proba_smoke():
    data = _tiny_hetero_data()
    metadata = data.metadata()
    num_nodes_dict = {nt: data[nt].num_nodes for nt in metadata[0]}
    model = RGCNFraudModel(
        metadata, transaction_in_channels=data["transaction"].x.shape[1], num_nodes_dict=num_nodes_dict,
        hidden_channels=16,
    )
    logits = model(data)
    assert logits.shape == (data["transaction"].x.shape[0], 2)

    proba = model.predict_proba(data)
    assert proba.shape == (data["transaction"].x.shape[0],)
    assert torch.all(proba >= 0) and torch.all(proba <= 1)
