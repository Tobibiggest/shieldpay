"""Heterogeneous Graph Transformer -- the primary "HNN" (heterogeneous
neural network) of this project, over the full transaction/account/device/
ip/merchant graph from `data.graph.build_graph.build_hetero_graph`.

Chosen over RGCN (`rgcn.py`, kept as a cheaper fallback) because this graph's
relation counts are very uneven -- a handful of `at_merchant` edges per
transaction vs. many `uses_device`/`shares_device` edges -- and HGT's
per-relation-type attention weights adapt to that imbalance, whereas RGCN's
fixed per-relation weight matrices treat every relation with equal capacity
regardless of how much signal it carries. Attention weights also double as
an explainability signal (e.g. "flagged mainly via the shares_device edge
into a known ring").

`account`/`device`/`ip`/`merchant` node types carry no raw features (only an
ID) -- unlike the transaction node, which has real preprocessed features.
Their representations start as a learned `nn.Embedding` per node type,
looked up by node index, and the HGT layers refine all node types jointly
through message passing.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


class HGTFraudModel(nn.Module):
    def __init__(
        self,
        metadata: Tuple[list, list],
        transaction_in_channels: int,
        num_nodes_dict: Dict[str, int],
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        node_types, _ = metadata

        self.transaction_proj = nn.Linear(transaction_in_channels, hidden_channels)
        self.entity_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(max(num_nodes_dict.get(node_type, 0), 1), hidden_channels)
                for node_type in node_types
                if node_type != "transaction"
            }
        )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads) for _ in range(num_layers)]
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
