"""Builds a homogeneous transaction-transaction graph by projecting the
relational structure down onto the transaction node set: two transactions
are connected if they share a sender, device, or IP. This is the graph
GraphSAGE/GAT (Phase 4) train on -- a simpler, single-node-type baseline
that exercises the same planted fraud-ring structure as the heterogeneous
graph builder (`build_graph.build_hetero_graph`) without needing entity
embeddings, which are introduced only for the heterogeneous model (Phase 5).
"""

from itertools import combinations
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from ...schema import FraudDatasetSchema
from ..preprocessing import FittedPreprocessor, fit_preprocessor

# Caps combinatorial edge blowup for an entity value shared by an unusually
# large number of transactions (e.g. a popular public Wi-Fi IP).
MAX_SHARED_ENTITY_GROUP_SIZE = 25

DEFAULT_SHARE_COLS = ("sender_id", "device_id", "ip_address")


def _group_projection_edges(df: pd.DataFrame, col: str, cap: int) -> np.ndarray:
    if col not in df.columns:
        return np.zeros((2, 0), dtype=np.int64)

    edges = []
    for _, positions in df.groupby(col).indices.items():
        idxs = list(positions)
        if len(idxs) < 2:
            continue
        if len(idxs) > cap:
            idxs = idxs[:cap]
        edges.extend(combinations(idxs, 2))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edges, dtype=np.int64).T
    return np.concatenate([arr, arr[[1, 0]]], axis=1)  # undirected


def build_transaction_projection_graph(
    df: pd.DataFrame,
    schema: FraudDatasetSchema,
    preprocessor: Optional[FittedPreprocessor] = None,
    share_cols: Iterable[str] = DEFAULT_SHARE_COLS,
    max_group_size: int = MAX_SHARED_ENTITY_GROUP_SIZE,
) -> Tuple[Data, FittedPreprocessor]:
    df = df.reset_index(drop=True)

    if preprocessor is None:
        preprocessor = fit_preprocessor(df, schema)
    x = torch.tensor(preprocessor.transform(df), dtype=torch.float)
    y = torch.tensor(df[schema.label_col].to_numpy(), dtype=torch.long)

    edge_arrays = [np.zeros((2, 0), dtype=np.int64)]
    for col in share_cols:
        edge_arrays.append(_group_projection_edges(df, col, max_group_size))
    edge_index = np.concatenate(edge_arrays, axis=1)
    if edge_index.shape[1] > 0:
        edge_index = np.unique(edge_index, axis=1)

    data = Data(x=x, y=y, edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data, preprocessor
