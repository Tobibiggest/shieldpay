"""GAT fraud classifier -- attention-based ablation/explainability variant of
GraphSAGE (`sage.py`). Per-edge attention weights double as a "why was this
flagged" signal (e.g. high attention on a `shares_device` edge into a known
fraud ring member), which GraphSAGE's uniform-aggregation doesn't provide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATFraudModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]
