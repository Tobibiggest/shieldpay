"""Builds a PyTorch Geometric HeteroData transaction graph from a canonical
dataframe (as produced by any adapter's `to_canonical`).

Node types: `transaction` (carries the real, preprocessed features) and
`account`/`device`/`ip`/`merchant` (ID-only -- entity embeddings are created
at the model level in Phase 4/5, keyed by the node-index mappings returned
here). `account` is a single node type shared by senders and recipients,
since real accounts both send and receive across different transactions.

Edge types model transaction structure (`sends`, `sent_to`, `uses_device`,
`uses_ip`, `at_merchant`, plus their reverses for two-way message passing)
and derived `shares_device`/`shares_ip` account-account edges. The derived
edges are what exposes planted fraud rings (see
`relational_synthetic_generator`) to a heterogeneous GNN: ring members share
a device/IP, so message passing along these edges can propagate a
"suspicious neighborhood" signal even to a ring member whose own
per-transaction features look unremarkable.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ...schema import FraudDatasetSchema
from ..preprocessing import FittedPreprocessor, fit_preprocessor

# Caps combinatorial edge blowup for a device/IP shared by an unusually large
# number of distinct accounts (e.g. a popular public Wi-Fi IP).
MAX_SHARED_ENTITY_GROUP_SIZE = 25


@dataclass
class NodeIndexMaps:
    transaction: Dict[object, int] = field(default_factory=dict)
    account: Dict[object, int] = field(default_factory=dict)
    device: Dict[object, int] = field(default_factory=dict)
    ip: Dict[object, int] = field(default_factory=dict)
    merchant: Dict[object, int] = field(default_factory=dict)


def _build_index(values: pd.Series) -> Dict[object, int]:
    unique_vals = pd.unique(values.dropna())
    return {val: i for i, val in enumerate(unique_vals)}


def _edge_transaction_to_entity(df: pd.DataFrame, col: str, node_map: Dict[object, int]) -> np.ndarray:
    """Returns edge_index [2, E] as (transaction_row_idx, entity_node_idx)."""
    if col not in df.columns or not node_map:
        return np.zeros((2, 0), dtype=np.int64)
    mapped = df[col].map(node_map)
    valid = mapped.notna().to_numpy()
    tx_idx = np.nonzero(valid)[0]
    entity_idx = mapped.to_numpy()[valid].astype(np.int64)
    return np.stack([tx_idx, entity_idx])


def _shared_entity_edges(df: pd.DataFrame, entity_col: str, account_map: Dict[object, int]) -> np.ndarray:
    """Account-account edges for accounts (as senders) that share the same
    device_id/ip_address -- the collusion / fraud-ring signal."""
    if entity_col not in df.columns or "sender_id" not in df.columns:
        return np.zeros((2, 0), dtype=np.int64)

    edges = []
    for _, group in df.groupby(entity_col)["sender_id"]:
        accounts = pd.unique(group.dropna())
        if len(accounts) < 2:
            continue
        if len(accounts) > MAX_SHARED_ENTITY_GROUP_SIZE:
            accounts = accounts[:MAX_SHARED_ENTITY_GROUP_SIZE]
        idxs = [account_map[a] for a in accounts if a in account_map]
        edges.extend(combinations(idxs, 2))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edges, dtype=np.int64).T
    return np.concatenate([arr, arr[[1, 0]]], axis=1)  # undirected: both directions


def build_hetero_graph(
    df: pd.DataFrame,
    schema: FraudDatasetSchema,
    preprocessor: Optional[FittedPreprocessor] = None,
) -> Tuple[HeteroData, NodeIndexMaps, FittedPreprocessor]:
    if not schema.has_relational_fields():
        raise ValueError(
            "Schema has no relational fields (sender/recipient/device/ip/merchant) "
            "-- cannot build a graph from a purely flat dataset."
        )

    df = df.reset_index(drop=True)

    account_series = pd.concat(
        [df.get("sender_id", pd.Series(dtype=object)), df.get("recipient_id", pd.Series(dtype=object))]
    )
    maps = NodeIndexMaps(
        transaction={i: i for i in range(len(df))},
        account=_build_index(account_series),
        device=_build_index(df["device_id"]) if "device_id" in df.columns else {},
        ip=_build_index(df["ip_address"]) if "ip_address" in df.columns else {},
        merchant=_build_index(df["merchant_id"]) if "merchant_id" in df.columns else {},
    )

    if preprocessor is None:
        preprocessor = fit_preprocessor(df, schema)
    x_transaction = torch.tensor(preprocessor.transform(df), dtype=torch.float)
    y_transaction = torch.tensor(df[schema.label_col].to_numpy(), dtype=torch.long)

    data = HeteroData()
    data["transaction"].x = x_transaction
    data["transaction"].y = y_transaction
    data["account"].num_nodes = len(maps.account)
    data["device"].num_nodes = len(maps.device)
    data["ip"].num_nodes = len(maps.ip)
    data["merchant"].num_nodes = len(maps.merchant)

    sender_edges = _edge_transaction_to_entity(df, "sender_id", maps.account)  # (tx, account)
    recipient_edges = _edge_transaction_to_entity(df, "recipient_id", maps.account)  # (tx, account)
    device_edges = _edge_transaction_to_entity(df, "device_id", maps.device)  # (tx, device)
    ip_edges = _edge_transaction_to_entity(df, "ip_address", maps.ip)  # (tx, ip)
    merchant_edges = _edge_transaction_to_entity(df, "merchant_id", maps.merchant)  # (tx, merchant)

    data["account", "sends", "transaction"].edge_index = torch.tensor(sender_edges[[1, 0]], dtype=torch.long)
    data["transaction", "rev_sends", "account"].edge_index = torch.tensor(sender_edges, dtype=torch.long)

    data["transaction", "sent_to", "account"].edge_index = torch.tensor(recipient_edges, dtype=torch.long)
    data["account", "rev_sent_to", "transaction"].edge_index = torch.tensor(recipient_edges[[1, 0]], dtype=torch.long)

    data["transaction", "uses_device", "device"].edge_index = torch.tensor(device_edges, dtype=torch.long)
    data["device", "rev_uses_device", "transaction"].edge_index = torch.tensor(device_edges[[1, 0]], dtype=torch.long)

    data["transaction", "uses_ip", "ip"].edge_index = torch.tensor(ip_edges, dtype=torch.long)
    data["ip", "rev_uses_ip", "transaction"].edge_index = torch.tensor(ip_edges[[1, 0]], dtype=torch.long)

    data["transaction", "at_merchant", "merchant"].edge_index = torch.tensor(merchant_edges, dtype=torch.long)
    data["merchant", "rev_at_merchant", "transaction"].edge_index = torch.tensor(merchant_edges[[1, 0]], dtype=torch.long)

    data["account", "shares_device", "account"].edge_index = torch.tensor(
        _shared_entity_edges(df, "device_id", maps.account), dtype=torch.long
    )
    data["account", "shares_ip", "account"].edge_index = torch.tensor(
        _shared_entity_edges(df, "ip_address", maps.account), dtype=torch.long
    )

    return data, maps, preprocessor
