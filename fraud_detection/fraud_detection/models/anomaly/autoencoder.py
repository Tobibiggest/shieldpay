"""Unsupervised anomaly detector: a small autoencoder trained ONLY on
non-fraud transactions, so reconstruction error becomes a proxy fraud score
-- it can flag transactions that look nothing like normal traffic even if
their specific fraud pattern never appeared (or was too rare) in the labeled
training data, which a purely supervised model like the GBDT/GNN classifiers
can't do by construction.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # preprocessed features are MinMax/one-hot scaled to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@dataclass
class TabularAutoencoder:
    input_dim: int
    hidden_dim: int = 32
    bottleneck_dim: int = 8
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    net: Optional[_AutoencoderNet] = None

    def fit(self, X_non_fraud: np.ndarray, verbose: bool = False) -> "TabularAutoencoder":
        self.net = _AutoencoderNet(self.input_dim, self.hidden_dim, self.bottleneck_dim)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_non_fraud, dtype=torch.float)),
            batch_size=min(self.batch_size, len(X_non_fraud)),
            shuffle=True,
        )

        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.net(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose and (epoch % max(self.epochs // 5, 1) == 0 or epoch == self.epochs - 1):
                print(f"    [autoencoder] epoch {epoch + 1}/{self.epochs} loss={total_loss / len(loader):.4f}")
        return self

    @torch.no_grad()
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float)
        recon = self.net(X_t)
        errors = ((recon - X_t) ** 2).mean(dim=1).numpy()
        self.net.train()
        return errors
