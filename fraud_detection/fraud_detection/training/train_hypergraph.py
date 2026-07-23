"""Trains the stretch-goal HypergraphConv model and compares it against
XGBoost/GraphSAGE -- see `data/graph/hypergraph.py` for why this is optional
relative to the core heterogeneous graph (HGT, `train_hgnn.py`).

    python -m fraud_detection.training.train_hypergraph --data data/synthetic_relational.csv
"""

import argparse

import numpy as np
import torch

from ..data.adapters import get_adapter
from ..data.graph.homogeneous import build_transaction_projection_graph
from ..data.graph.hypergraph import build_transaction_hypergraph
from ..data.preprocessing import fit_preprocessor, temporal_split_indices
from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.gnn.sage import GraphSAGEFraudModel
from ..models.hypergraph.hypergraph_conv import HypergraphFraudModel
from ..models.tabular.gbdt import XGBoostFraudModel
from ..utils.seed import set_global_seed
from .common import train_node_classifier_homogeneous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--report-dir", default="artifacts/hypergraph")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cpu")

    adapter = get_adapter(args.dataset_name)
    df = adapter.load_canonical(args.data)
    schema = adapter.get_canonical_schema()
    if schema.timestamp_col and schema.timestamp_col in df.columns:
        df = df.sort_values(schema.timestamp_col).reset_index(drop=True)

    train_idx, val_idx, test_idx = temporal_split_indices(df, schema)
    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask_np = np.zeros(n, dtype=bool)
    test_mask_np[test_idx] = True

    preprocessor = fit_preprocessor(df.iloc[train_idx], schema)
    y_test = df[schema.label_col].to_numpy()[test_idx]
    harness = FraudEvaluationHarness()
    results = {}

    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor)
    xgb.fit_arrays(
        preprocessor.transform(df.iloc[train_idx]), df[schema.label_col].to_numpy()[train_idx]
    )
    results["xgboost"] = harness.evaluate(y_test, xgb.predict_proba(df.iloc[test_idx]), model_name="xgboost")

    homo_data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    x, edge_index, y_homo = homo_data.x.to(device), homo_data.edge_index.to(device), homo_data.y.to(device)
    sage = GraphSAGEFraudModel(in_channels=x.shape[1]).to(device)
    train_node_classifier_homogeneous(
        sage, x, edge_index, y_homo, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    results["graphsage"] = harness.evaluate(
        y_test, sage.predict_proba(x, edge_index).cpu().numpy()[test_mask_np], model_name="graphsage"
    )

    print("Training HypergraphConv...")
    hyper_data, _ = build_transaction_hypergraph(df, schema, preprocessor=preprocessor)
    hx = hyper_data.x.to(device)
    hyperedge_index = hyper_data.hyperedge_index.to(device)
    hy = hyper_data.y.to(device)
    hypergraph_model = HypergraphFraudModel(in_channels=hx.shape[1]).to(device)
    train_node_classifier_homogeneous(
        hypergraph_model, hx, hyperedge_index, hy, train_mask, args.epochs, args.lr,
        val_mask=val_mask, patience=20,
    )
    results["hypergraph"] = harness.evaluate(
        y_test,
        hypergraph_model.predict_proba(hx, hyperedge_index).cpu().numpy()[test_mask_np],
        model_name="hypergraph",
    )

    print()
    print(harness.compare(results))
    for name, result in results.items():
        write_report(result, args.report_dir, run_name=name)


if __name__ == "__main__":
    main()
