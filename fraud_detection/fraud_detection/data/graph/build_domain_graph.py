"""Generalized graph builder driven by a `DomainGraphSchema` -- the
parameterized counterpart to `build_graph.py::build_hetero_graph`, which is
hardcoded to payments' 5 fixed entity types. This module and `domain_schema.py`
are additive: `build_hetero_graph` and `schema.py` are untouched, and
`tests/test_domain_graph_equivalence.py` proves this builder reproduces the
old one exactly on payments data (via `domain_schema.from_fraud_dataset_schema`)
before any new domain relies on it.

Reuses `data.preprocessing.fit_preprocessor` unmodified -- it only reads
`.numeric_feature_cols`/`.categorical_feature_cols`, which `DomainGraphSchema`
also provides, so no change was needed there either.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..preprocessing import FittedPreprocessor, fit_preprocessor
from .domain_schema import CollusionEdgeSpec, DomainGraphSchema, EntityRole

# Safety cap on how many rows within one shared-value group get pairwise-scanned
# for timed collusion edges, to bound worst-case cost if a single shared value
# (e.g. a beneficiary account) appears on a pathologically large number of events.
MAX_TIMED_GROUP_SCAN = 1000


def _build_index(values: pd.Series) -> Dict[object, int]:
    unique_vals = pd.unique(values.dropna())
    return {val: i for i, val in enumerate(unique_vals)}


def _group_roles_by_node_type(entity_roles: List[EntityRole]) -> Dict[str, List[EntityRole]]:
    grouped: Dict[str, List[EntityRole]] = {}
    for role in entity_roles:
        grouped.setdefault(role.node_type, []).append(role)
    return grouped


def _event_to_entity_edges(df: pd.DataFrame, col: str, node_map: Dict[object, int]) -> np.ndarray:
    """Returns edge_index [2, E] as (event_row_idx, entity_node_idx)."""
    if col not in df.columns or not node_map:
        return np.zeros((2, 0), dtype=np.int64)
    mapped = df[col].map(node_map)
    valid = mapped.notna().to_numpy()
    event_idx = np.nonzero(valid)[0]
    entity_idx = mapped.to_numpy()[valid].astype(np.int64)
    return np.stack([event_idx, entity_idx])


def _untimed_collusion_pairs(group_df: pd.DataFrame, anchor_col: str, max_group_size: int) -> List[Tuple[object, object]]:
    from itertools import combinations

    anchors = pd.unique(group_df[anchor_col].dropna())
    if len(anchors) < 2:
        return []
    if len(anchors) > max_group_size:
        anchors = anchors[:max_group_size]
    return list(combinations(anchors, 2))


def _timed_collusion_pairs(
    group_df: pd.DataFrame, anchor_col: str, timestamp_col: str, max_gap: pd.Timedelta, max_group_size: int
) -> List[Tuple[object, object]]:
    sub = group_df[[anchor_col, timestamp_col]].dropna().sort_values(timestamp_col)
    if len(sub) > MAX_TIMED_GROUP_SCAN:
        sub = sub.iloc[:MAX_TIMED_GROUP_SCAN]
    anchors = sub[anchor_col].tolist()
    times = sub[timestamp_col].tolist()

    pairs = set()
    n = len(anchors)
    for i in range(n):
        for j in range(i + 1, n):
            if times[j] - times[i] > max_gap:
                break
            if anchors[i] != anchors[j]:
                pairs.add(tuple(sorted((anchors[i], anchors[j]), key=str)))

    involved = {v for pair in pairs for v in pair}
    if len(involved) > max_group_size:
        keep = set(list(involved)[:max_group_size])
        pairs = {p for p in pairs if p[0] in keep and p[1] in keep}
    return list(pairs)


def _collusion_edges(
    df: pd.DataFrame,
    spec: CollusionEdgeSpec,
    shared_col: str,
    anchor_col: str,
    anchor_map: Dict[object, int],
    timestamp_col: Optional[str],
) -> np.ndarray:
    if shared_col not in df.columns or anchor_col not in df.columns:
        return np.zeros((2, 0), dtype=np.int64)

    all_pairs: List[Tuple[object, object]] = []
    for _, group_df in df.groupby(shared_col):
        if spec.max_gap is not None and timestamp_col and timestamp_col in df.columns:
            all_pairs.extend(_timed_collusion_pairs(group_df, anchor_col, timestamp_col, spec.max_gap, spec.max_group_size))
        else:
            all_pairs.extend(_untimed_collusion_pairs(group_df, anchor_col, spec.max_group_size))

    idx_pairs = [(anchor_map[a], anchor_map[b]) for a, b in all_pairs if a in anchor_map and b in anchor_map]
    if not idx_pairs:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(idx_pairs, dtype=np.int64).T
    return np.concatenate([arr, arr[[1, 0]]], axis=1)  # undirected: both directions


def build_domain_hetero_graph(
    df: pd.DataFrame,
    domain_schema: DomainGraphSchema,
    preprocessor: Optional[FittedPreprocessor] = None,
) -> Tuple[HeteroData, Dict[str, Dict[object, int]], FittedPreprocessor]:
    if not domain_schema.has_relational_fields():
        raise ValueError(
            f"DomainGraphSchema '{domain_schema.domain_name}' has no entity roles -- "
            "cannot build a graph from a purely flat dataset."
        )

    df = df.reset_index(drop=True)

    roles_by_node_type = _group_roles_by_node_type(domain_schema.entity_roles)
    node_index_maps: Dict[str, Dict[object, int]] = {}
    for node_type, roles in roles_by_node_type.items():
        series_list = [df[role.id_col] for role in roles if role.id_col in df.columns]
        combined = pd.concat(series_list) if series_list else pd.Series(dtype=object)
        node_index_maps[node_type] = _build_index(combined)

    if preprocessor is None:
        preprocessor = fit_preprocessor(df, domain_schema)
    x_event = torch.tensor(preprocessor.transform(df), dtype=torch.float)
    y_event = torch.tensor(df[domain_schema.label_col].to_numpy(), dtype=torch.long)

    data = HeteroData()
    data[domain_schema.event_node_type].x = x_event
    data[domain_schema.event_node_type].y = y_event
    for node_type, index_map in node_index_maps.items():
        data[node_type].num_nodes = len(index_map)

    for role in domain_schema.entity_roles:
        node_map = node_index_maps[role.node_type]
        event_to_entity = _event_to_entity_edges(df, role.id_col, node_map)  # (event_idx, entity_idx)
        entity_to_event = event_to_entity[[1, 0]]  # (entity_idx, event_idx)

        # Every role always gets both directions -- skipping the reverse edge
        # would leave that node type unreachable as a message-passing target
        # in some layer, silently breaking HGTConv/HeteroConv updates for it.
        data[role.node_type, role.resolved_entity_to_event_edge(), domain_schema.event_node_type].edge_index = (
            torch.tensor(entity_to_event, dtype=torch.long)
        )
        data[domain_schema.event_node_type, role.resolved_event_to_entity_edge(), role.node_type].edge_index = (
            torch.tensor(event_to_entity, dtype=torch.long)
        )

    role_by_name = {role.role_name: role for role in domain_schema.entity_roles}
    for spec in domain_schema.collusion_edges:
        anchor_role = role_by_name[spec.anchor_role_name]
        shared_role = role_by_name[spec.shared_role_name]
        anchor_map = node_index_maps[anchor_role.node_type]
        edges = _collusion_edges(
            df, spec, shared_role.id_col, anchor_role.id_col, anchor_map, domain_schema.timestamp_col
        )
        data[anchor_role.node_type, spec.edge_name, anchor_role.node_type].edge_index = torch.tensor(
            edges, dtype=torch.long
        )

    return data, node_index_maps, preprocessor
