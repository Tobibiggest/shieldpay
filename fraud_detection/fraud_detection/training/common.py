"""Shared setup used by every train_*.py script: load a dataset via its
adapter, split it temporally, and return everything a training script needs.
Also holds the node-classifier training loops shared by `train_gnn.py`
(homogeneous graph) and `train_hgnn.py` (heterogeneous graph), so both use
the identical class-weighted cross-entropy training procedure and the same
validation-based early stopping / model selection.
"""

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from ..data.adapters import get_adapter
from ..data.preprocessing import temporal_split
from ..schema import FraudDatasetSchema


@dataclass
class LoadedDataset:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    schema: FraudDatasetSchema


def load_and_split(
    dataset_path: str,
    dataset_name: str = "synthetic_relational",
    test_frac: float = 0.2,
    val_frac: float = 0.1,
) -> LoadedDataset:
    adapter = get_adapter(dataset_name)
    df = adapter.load_canonical(dataset_path)
    schema = adapter.get_canonical_schema()
    train_df, val_df, test_df = temporal_split(df, schema, test_frac=test_frac, val_frac=val_frac)
    return LoadedDataset(train_df=train_df, val_df=val_df, test_df=test_df, schema=schema)


def _class_weights(y: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    class_counts = np.bincount(y[train_mask].cpu().numpy(), minlength=2).astype(np.float32)
    return torch.tensor(class_counts.sum() / (2 * np.clip(class_counts, 1, None)), dtype=torch.float)


def _train_with_early_stopping(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
    val_mask: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    verbose: bool = True,
):
    """Shared training loop. `forward_fn()` must return this epoch's logits
    for every node (the caller closes over whatever inputs `model` needs).

    Node-entity embeddings (used by the heterogeneous models in
    `models/hgnn/`) give a model enough capacity to memorize specific
    training-set device/account IDs rather than learning generalizable
    patterns -- observed here as train loss collapsing to near-zero within
    ~20 epochs while held-out AUPRC stayed mediocre. Tracking validation
    AUPRC each epoch and keeping the best-performing checkpoint (instead of
    just the final one) corrects for that without changing model capacity.
    """
    class_weights = _class_weights(y, train_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_auprc = -1.0
    best_state = None
    epochs_without_improvement = 0
    val_auprc = float("nan")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = forward_fn()
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                val_logits = forward_fn()
            val_proba = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_y = y[val_mask].cpu().numpy()
            if len(np.unique(val_y)) > 1:
                val_auprc = average_precision_score(val_y, val_proba[val_mask.cpu().numpy()])
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if verbose and (epoch % max(epochs // 5, 1) == 0 or epoch == epochs - 1):
            msg = f"    epoch {epoch + 1}/{epochs} loss={loss.item():.4f}"
            if val_mask is not None:
                msg += f" val_auprc={val_auprc:.4f} (best={best_val_auprc:.4f})"
            print(msg)

        if patience is not None and val_mask is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"    early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_node_classifier_homogeneous(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
    val_mask: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    verbose: bool = True,
):
    """Training loop for single-node-type graphs (GraphSAGE, GAT)."""
    return _train_with_early_stopping(
        model, lambda: model(x, edge_index), y, train_mask, epochs, lr, val_mask, patience, verbose
    )


def train_node_classifier_hetero(
    model,
    data,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
    val_mask: Optional[torch.Tensor] = None,
    patience: Optional[int] = None,
    verbose: bool = True,
):
    """Training loop for heterogeneous graphs (HGT, RGCN) -- `model(data)`
    must return transaction-node logits."""
    return _train_with_early_stopping(
        model, lambda: model(data), y, train_mask, epochs, lr, val_mask, patience, verbose
    )
