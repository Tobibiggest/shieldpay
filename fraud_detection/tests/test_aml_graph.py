"""First real stress test of `build_domain_hetero_graph` against a schema
that ISN'T the converted payments one -- multi-role account merges, an
AML-specific `shares_beneficiary` collusion spec (payments has no
equivalent), and no merchant node type at all.
"""

import pytest

torch = pytest.importorskip("torch")

from fraud_detection.data.graph.build_domain_graph import build_domain_hetero_graph  # noqa: E402
from fraud_detection.domains.aml.aml_adapter import AMLAdapter  # noqa: E402
from fraud_detection.domains.aml.aml_generator import generate_aml_transaction_dataset  # noqa: E402


@pytest.fixture(scope="module")
def aml_graph_inputs():
    df = generate_aml_transaction_dataset(
        n_transactions=1_000, n_accounts=150, n_devices=80, n_ips=80,
        n_layering_chains=6, n_smurf_rings=4, seed=17,
    )
    adapter = AMLAdapter()
    schema = adapter.get_canonical_schema()
    canonical_df = adapter.to_canonical(df)
    return canonical_df, schema


def test_build_domain_hetero_graph_node_and_edge_types(aml_graph_inputs):
    df, schema = aml_graph_inputs
    data, node_index_maps, preprocessor = build_domain_hetero_graph(df, schema)

    assert set(data.node_types) == {"transaction", "account", "device", "ip"}  # no merchant in AML
    # All AML roles use default edge naming (entity_to_event=role_name,
    # event_to_entity=f"rev_{role_name}") -- uniform, unlike payments' hardcoded
    # asymmetric names (see domain_schema.from_fraud_dataset_schema).
    expected_edge_types = {
        ("account", "originator", "transaction"),
        ("transaction", "rev_originator", "account"),
        ("account", "beneficiary", "transaction"),
        ("transaction", "rev_beneficiary", "account"),
        ("device", "device", "transaction"),
        ("transaction", "rev_device", "device"),
        ("ip", "ip", "transaction"),
        ("transaction", "rev_ip", "ip"),
        ("account", "shares_device", "account"),
        ("account", "shares_ip", "account"),
        ("account", "shares_beneficiary", "account"),
    }
    assert expected_edge_types.issubset(set(data.edge_types))

    assert data["transaction"].x.shape[0] == len(df)
    assert data["transaction"].y.shape[0] == len(df)
    # originator + beneficiary both merge into the "account" node type, and
    # accounts repeat across transactions, so the merged node count is well
    # below 2x the transaction count.
    assert data["account"].num_nodes < 2 * len(df)
    assert data["account"].num_nodes > 0


def test_shares_beneficiary_edges_exist_and_are_symmetric(aml_graph_inputs):
    df, schema = aml_graph_inputs
    data, _, _ = build_domain_hetero_graph(df, schema)
    edge_index = data["account", "shares_beneficiary", "account"].edge_index
    assert edge_index.shape[1] > 0  # smurfing rings should produce some of these
    pairs = set(map(tuple, edge_index.t().tolist()))
    for a, b in list(pairs)[:50]:
        assert (b, a) in pairs


def test_build_domain_hetero_graph_raises_without_relational_fields():
    import pandas as pd

    from fraud_detection.data.graph.domain_schema import DomainGraphSchema

    df = pd.DataFrame({"label": [0, 1], "amount": [1.0, 2.0]})
    flat_schema = DomainGraphSchema(
        domain_name="flat", event_node_type="event", label_col="label", numeric_feature_cols=["amount"]
    )
    with pytest.raises(ValueError):
        build_domain_hetero_graph(df, flat_schema)
