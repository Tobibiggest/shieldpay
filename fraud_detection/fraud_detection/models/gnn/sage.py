"""GraphSAGE fraud classifier over the transaction-projection graph (see
`data.graph.homogeneous.build_transaction_projection_graph`).

Chosen as the primary homogeneous GNN because it's inductive: unlike a
transductive spectral GCN, GraphSAGE learns an aggregation function over a
node's neighborhood rather than a fixed per-node embedding, so it can score a
brand-new transaction/account at serving time (via neighbor sampling)
without retraining -- the realistic setting for a production fraud system,
where new entities appear constantly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEFraudModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]
