"""Builds a transaction hypergraph for `HypergraphConv` (Phase 9, stretch):
models n-ary collusion -- N accounts sharing the same device+IP+recipient
simultaneously -- as one hyperedge connecting all their transactions, rather
than the O(k^2) pairwise `shares_device`/`shares_ip` edges the heterogeneous
graph builder (`build_graph.py`) uses. A hyperedge is the more faithful
representation of "this whole group acted together," but PyG's
`HypergraphConv` needs an explicit incidence structure
(`hyperedge_index: [2, sum(|hyperedge|)]`, row 0 = node index, row 1 =
hyperedge index) rather than a normal `edge_index`, and constructing
meaningful hyperedges needs its own grouping heuristic -- this is why it's
kept separate from, and optional relative to, the core heterogeneous graph.

Grouping heuristic: one hyperedge per distinct (device_id, ip_address,
recipient_id) triple shared by 2+ transactions -- exactly the structure
`relational_synthetic_generator` plants for fraud rings (shared device+IP,
common drop-account recipient).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ...schema import FraudDatasetSchema
from ..preprocessing import FittedPreprocessor, fit_preprocessor

MAX_HYPEREDGE_SIZE = 25
GROUP_COLS = ("device_id", "ip_address", "recipient_id")


def build_transaction_hypergraph(
    df: pd.DataFrame,
    schema: FraudDatasetSchema,
    preprocessor: Optional[FittedPreprocessor] = None,
    group_cols: Tuple[str, ...] = GROUP_COLS,
    max_hyperedge_size: int = MAX_HYPEREDGE_SIZE,
) -> Tuple[Data, FittedPreprocessor]:
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Hypergraph grouping columns missing from dataframe: {missing}")

    df = df.reset_index(drop=True)
    if preprocessor is None:
        preprocessor = fit_preprocessor(df, schema)
    x = torch.tensor(preprocessor.transform(df), dtype=torch.float)
    y = torch.tensor(df[schema.label_col].to_numpy(), dtype=torch.long)

    node_ids, hyperedge_ids = [], []
    hyperedge_counter = 0
    for _, positions in df.groupby(list(group_cols)).indices.items():
        idxs = list(positions)
        if len(idxs) < 2:
            continue
        if len(idxs) > max_hyperedge_size:
            idxs = idxs[:max_hyperedge_size]
        node_ids.extend(idxs)
        hyperedge_ids.extend([hyperedge_counter] * len(idxs))
        hyperedge_counter += 1

    if node_ids:
        hyperedge_index = torch.tensor(np.stack([node_ids, hyperedge_ids]), dtype=torch.long)
    else:
        hyperedge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, y=y, hyperedge_index=hyperedge_index, num_hyperedges=hyperedge_counter)
    return data, preprocessor
