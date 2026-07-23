"""Proves `build_domain_hetero_graph` (the new generic builder) reproduces
`build_hetero_graph` (the existing hardcoded payments builder) exactly on the
same data -- the evidence for the "generalizing didn't change payments'
behavior" claim, before any new domain relies on the generic path.

Comparison strategy: node integer assignment order is an implementation
detail (both builders number nodes by first-seen order, but iterate entity
roles in a different order), not a correctness property -- so edges are
remapped through each builder's own node-index map back to original entity-ID
strings before comparing, rather than comparing raw edge_index tensors.
"""

from typing import Dict, Set, Tuple

import torch

from fraud_detection.data.adapters import SyntheticRelationalAdapter
from fraud_detection.data.generators.relational_synthetic_generator import (
    generate_relational_fraud_dataset,
)
from fraud_detection.data.graph.build_domain_graph import build_domain_hetero_graph
from fraud_detection.data.graph.build_graph import build_hetero_graph
from fraud_detection.data.graph.domain_schema import from_fraud_dataset_schema


def _entity_event_pairs(edge_index: torch.Tensor, entity_idx_to_id: Dict[int, object], entity_is_source: bool) -> Set[Tuple[object, int]]:
    entity_positions = (edge_index[0] if entity_is_source else edge_index[1]).tolist()
    event_positions = (edge_index[1] if entity_is_source else edge_index[0]).tolist()
    return {(entity_idx_to_id[e], p) for e, p in zip(entity_positions, event_positions)}


def _sample_payments_data():
    df = generate_relational_fraud_dataset(
        n_transactions=600, n_accounts=80, n_devices=40, n_ips=40, n_merchants=15, n_fraud_rings=4, seed=21
    )
    adapter = SyntheticRelationalAdapter()
    schema = adapter.get_canonical_schema()
    canonical_df = adapter.to_canonical(df)
    return canonical_df, schema


def test_domain_graph_has_matching_node_and_edge_types():
    df, schema = _sample_payments_data()
    old_data, old_maps, preprocessor = build_hetero_graph(df, schema)
    domain_schema = from_fraud_dataset_schema(schema)
    new_data, new_maps, _ = build_domain_hetero_graph(df, domain_schema, preprocessor=preprocessor)

    assert set(old_data.node_types) == set(new_data.node_types)
    assert set(old_data.edge_types) == set(new_data.edge_types)

    for node_type in old_data.node_types:
        if node_type == "transaction":
            continue
        assert old_data[node_type].num_nodes == new_data[node_type].num_nodes, node_type


def test_domain_graph_edges_match_old_builder_exactly():
    df, schema = _sample_payments_data()
    old_data, old_maps, preprocessor = build_hetero_graph(df, schema)
    domain_schema = from_fraud_dataset_schema(schema)
    new_data, new_maps, _ = build_domain_hetero_graph(df, domain_schema, preprocessor=preprocessor)

    # role_name -> (node_type, entity_to_event_edge_name, old_maps_attr)
    role_checks = [
        ("sender", "account", "sends", "account"),
        ("recipient", "account", "rev_sent_to", "account"),
        ("device", "device", "rev_uses_device", "device"),
        ("ip", "ip", "rev_uses_ip", "ip"),
        ("merchant", "merchant", "rev_at_merchant", "merchant"),
    ]

    for role_name, node_type, entity_to_event_edge, old_attr in role_checks:
        old_map = getattr(old_maps, old_attr)
        new_map = new_maps[node_type]
        old_idx_to_id = {v: k for k, v in old_map.items()}
        new_idx_to_id = {v: k for k, v in new_map.items()}

        old_edge_index = old_data[node_type, entity_to_event_edge, "transaction"].edge_index
        new_edge_index = new_data[node_type, entity_to_event_edge, "transaction"].edge_index

        old_pairs = _entity_event_pairs(old_edge_index, old_idx_to_id, entity_is_source=True)
        new_pairs = _entity_event_pairs(new_edge_index, new_idx_to_id, entity_is_source=True)

        assert old_pairs == new_pairs, f"mismatch for role={role_name}"


def test_domain_graph_x_and_y_match_old_builder():
    df, schema = _sample_payments_data()
    old_data, _, preprocessor = build_hetero_graph(df, schema)
    domain_schema = from_fraud_dataset_schema(schema)
    new_data, _, _ = build_domain_hetero_graph(df, domain_schema, preprocessor=preprocessor)

    assert torch.equal(old_data["transaction"].x, new_data["transaction"].x)
    assert torch.equal(old_data["transaction"].y, new_data["transaction"].y)


def test_full_existing_suite_still_covers_original_builder():
    """Sanity check that this new module doesn't shadow or interfere with
    the original build_hetero_graph import path used throughout the rest of
    the codebase."""
    from fraud_detection.data.graph import build_hetero_graph as reexported

    assert reexported is build_hetero_graph
