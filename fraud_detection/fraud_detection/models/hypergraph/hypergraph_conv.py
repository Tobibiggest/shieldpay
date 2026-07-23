"""Hypergraph convolution fraud classifier (Phase 9, stretch) -- see
`data.graph.hypergraph.build_transaction_hypergraph` for how hyperedges are
constructed. `HypergraphConv` aggregates each node's features through the
hyperedges it belongs to in one step, which is the more natural
representation for "this whole group of N transactions acted together" (a
fraud ring), versus the pairwise `shares_device`/`shares_ip` edges used
elsewhere in this project, which only capture two-way relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv


class HypergraphFraudModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([HypergraphConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, hyperedge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, hyperedge_index)
        return F.softmax(logits, dim=1)[:, 1]
