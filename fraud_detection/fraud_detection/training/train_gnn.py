"""Trains GraphSAGE and GAT on the transaction-projection graph and compares
them against the XGBoost baseline, using the identical preprocessor and
temporal split for a fair comparison.

    python -m fraud_detection.training.train_gnn --data data/synthetic_relational.csv
"""

import argparse

import numpy as np
import torch

from ..data.adapters import get_adapter
from ..data.graph.homogeneous import build_transaction_projection_graph
from ..data.preprocessing import fit_preprocessor, temporal_split_indices
from ..evaluation.evaluate import FraudEvaluationHarness
from ..evaluation.report import write_report
from ..models.gnn.gat import GATFraudModel
from ..models.gnn.sage import GraphSAGEFraudModel
from ..models.tabular.gbdt import XGBoostFraudModel
from ..utils.seed import set_global_seed
from .common import train_node_classifier_homogeneous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="synthetic_relational")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--report-dir", default="artifacts/gnn")
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
    data, _ = build_transaction_projection_graph(df, schema, preprocessor=preprocessor)
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)

    y_test = df[schema.label_col].to_numpy()[test_idx]
    harness = FraudEvaluationHarness()
    results = {}

    # Re-fit XGBoost on the identical preprocessor/split for a fair comparison.
    xgb = XGBoostFraudModel(schema=schema, preprocessor=preprocessor)
    xgb.fit_arrays(
        preprocessor.transform(df.iloc[train_idx]), df[schema.label_col].to_numpy()[train_idx]
    )
    results["xgboost"] = harness.evaluate(y_test, xgb.predict_proba(df.iloc[test_idx]), model_name="xgboost")

    in_channels = x.shape[1]

    print("Training GraphSAGE...")
    sage = GraphSAGEFraudModel(in_channels=in_channels).to(device)
    train_node_classifier_homogeneous(
        sage, x, edge_index, y, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    sage_scores = sage.predict_proba(x, edge_index).cpu().numpy()
    results["graphsage"] = harness.evaluate(y_test, sage_scores[test_mask_np], model_name="graphsage")

    print("Training GAT...")
    gat = GATFraudModel(in_channels=in_channels).to(device)
    train_node_classifier_homogeneous(
        gat, x, edge_index, y, train_mask, args.epochs, args.lr, val_mask=val_mask, patience=20
    )
    gat_scores = gat.predict_proba(x, edge_index).cpu().numpy()
    results["gat"] = harness.evaluate(y_test, gat_scores[test_mask_np], model_name="gat")

    print()
    print(harness.compare(results))

    for name, result in results.items():
        write_report(result, args.report_dir, run_name=name)


if __name__ == "__main__":
    main()
