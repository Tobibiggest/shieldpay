"""Relational-GCN-style fallback: a cheaper heterogeneous baseline than HGT
(`hgt.py`), giving every relation type its own SAGEConv combined via PyG's
`HeteroConv`, rather than HGT's shared type-aware attention mechanism. Useful
as a sanity check that HGT's added attention machinery is actually earning
its extra cost on a given dataset.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class RGCNFraudModel(nn.Module):
    def __init__(
        self,
        metadata: Tuple[list, list],
        transaction_in_channels: int,
        num_nodes_dict: Dict[str, int],
        hidden_channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        node_types, edge_types = metadata

        self.transaction_proj = nn.Linear(transaction_in_channels, hidden_channels)
        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(max(num_nodes_dict.get(node_type, 0), 1), hidden_channels)
                for node_type in node_types
                if node_type != "transaction"
            }
        )

        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(hidden_channels, 2)

    def _init_node_features(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {"transaction": self.transaction_proj(data["transaction"].x)}
        device = data["transaction"].x.device
        for node_type, embedding in self.entity_embeddings.items():
            num_nodes = data[node_type].num_nodes
            idx = torch.arange(num_nodes, device=device)
            x_dict[node_type] = embedding(idx)
        return x_dict

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._init_node_features(data)
        edge_index_dict = data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {node_type: F.relu(x) for node_type, x in x_dict.items()}
        return self.classifier(x_dict["transaction"])

    @torch.no_grad()
    def predict_proba(self, data: HeteroData) -> torch.Tensor:
        self.eval()
        logits = self.forward(data)
        return F.softmax(logits, dim=1)[:, 1]
